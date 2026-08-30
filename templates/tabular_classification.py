# %% [markdown]
# # Tabular 분류 템플릿 (이진/다중 자동 대응)
#
# 교재: docs/01-tabular.md
#
# NUM_CLASSES 를 데이터에서 자동으로 읽으므로 이진분류든 다중분류든 코드가 같습니다.
#
# ── 데이터 흐름 ──
# df_train / df_test  (로드)
#   ↓ 전처리 (train 에서 fit, test 는 transform)
# X, y  /  X_test
#   ↓ 분할
# X_train, X_val  (학습/검증)
#   ↓ 학습 → 검증 → 제출
# submission.csv
#
# ★ 표시가 있는 곳만 문제에 맞게 고치면 됩니다.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# 0. 제출 설정  ★ 시험 시작하면 가장 먼저 채울 것
# ============================================================
# 모델과 예측 결과는 "휴대폰번호_문제번호" 로 저장해 제출합니다.
#   01012345678_1.csv   (예측 결과)
#   01012345678_1.pkl   (모델)
# 파일명을 매번 손으로 치면 오타가 납니다. 여기서 한 번만 정하고 변수를 쓰세요.
PHONE = '01012345678'      # ★ 본인 휴대폰 번호 (하이픈 없이)
PROB_NO = 1                # ★ 문제 번호 (문제가 바뀌면 이 줄만 수정)

SUBMIT_CSV = f'{PHONE}_{PROB_NO}.csv'
SUBMIT_MODEL = f'{PHONE}_{PROB_NO}.pkl'
print(f"제출 파일: {SUBMIT_CSV} / {SUBMIT_MODEL}")

# ============================================================
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 ---
# df_train = pd.read_csv('train.csv')
# df_test  = pd.read_csv('test.csv')
# TARGET   = 'target'          # ★ 실제 타겟 컬럼명

# --- 연습용: Titanic (이진분류) ---
df_all = sns.load_dataset('titanic')
TARGET = 'survived'
df_train, df_test = train_test_split(
    df_all, test_size=0.2, random_state=RANDOM_SEED, stratify=df_all[TARGET]
)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

NUM_CLASSES = df_train[TARGET].nunique()
print(f"Train {df_train.shape} / Test {df_test.shape}")
print(f"클래스 {NUM_CLASSES}개 ({'이진분류' if NUM_CLASSES == 2 else '다중분류'})")
print(df_train[TARGET].value_counts().sort_index())
df_train.head()

# ============================================================
# 2. EDA — 전처리 계획을 세우기 위한 최소한의 확인
# ============================================================
# %%
# [컬럼 제거 판단 기준]
#   miss_pct     >= 70   정보량 부족          → 제거
#   nunique_pct  >= 95   ID/이름 등 식별자     → 제거
#   nunique      == 1    분산 0               → 제거
#   타겟 상관    >= 0.95 데이터 누수 의심      → 제거
#   의미 중복            class vs pclass 등    → 하나만 남김
col_summary = pd.DataFrame({
    'dtype': df_train.dtypes,
    'missing': df_train.isnull().sum(),
    'miss_pct': (df_train.isnull().mean() * 100).round(1),
    'nunique': df_train.nunique(),
    'nunique_pct': (df_train.nunique() / len(df_train) * 100).round(1),
    'sample': df_train.iloc[0],
})
print(col_summary.to_string())

# %%
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
if TARGET in numeric_cols:
    print("[타겟과의 상관관계 - 0.95 이상이면 누수 의심]")
    print(df_train[numeric_cols].corr()[TARGET].abs().sort_values(ascending=False))

# ============================================================
# 3. 전처리 — train 에서 fit, test 는 transform
# ============================================================
# %%
# --- 3-1. 불필요 컬럼 제거 (train, test 동시에) ★
drop_cols = ['alive', 'pclass', 'deck']       # ★ 위 표를 보고 결정
df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')
print(f"제거: {drop_cols}")

# --- 3-2. Feature / Target 분리
X = df_train.drop(columns=[TARGET])
y = df_train[TARGET].copy()

has_test_label = TARGET in df_test.columns
X_test = df_test.drop(columns=[TARGET]) if has_test_label else df_test.copy()
y_test = df_test[TARGET].copy() if has_test_label else None

# 타겟이 문자열이면 숫자로 (제출 시 inverse_transform 으로 복원)
target_encoder = None
if y.dtype == 'object' or y.dtype.name == 'category':
    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(y))
    if y_test is not None:
        y_test = pd.Series(target_encoder.transform(y_test))
    print(f"타겟 클래스: {target_encoder.classes_}")

# --- 3-3. 수치형 / 범주형 자동 분리
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"수치형 {len(num_cols)}: {num_cols}")
print(f"범주형 {len(cat_cols)}: {cat_cols}")

# %%
# --- 3-4. 결측치 (수치형=중앙값, 범주형=최빈값)
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

if num_cols:
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
if cat_cols:
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

print(f"결측 남은 개수 - train {X.isnull().sum().sum()} / test {X_test.isnull().sum().sum()}")

# %%
# --- 3-5. [선택] 이상치 클리핑 (IQR) — 선형 모델에서만 효과적
# for col in num_cols:
#     Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
#     lower, upper = Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1)
#     X[col] = X[col].clip(lower, upper)
#     X_test[col] = X_test[col].clip(lower, upper)   # train 기준으로 test 도 클리핑

# --- 3-6. [선택] 피처 엔지니어링 — train/test 에 똑같이 적용할 것
# X['family_size'] = X['sibsp'] + X['parch'] + 1
# X_test['family_size'] = X_test['sibsp'] + X_test['parch'] + 1

# %%
# --- 3-7. 인코딩 (LabelEncoder: 트리 모델에 충분하고 빠름)
# test 에만 있는 미지의 값은 -1 로 보내 에러를 막는다
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = X_test[col].astype(str).map(
        lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_ else -1
    )
    label_encoders[col] = le

# --- [선택] 순서 없는 범주 + 선형 모델이면 OneHot 이 유리
# ohe_cols = [c for c in cat_cols if X[c].nunique() <= 10]
# X = pd.get_dummies(X, columns=ohe_cols, drop_first=True)
# X_test = pd.get_dummies(X_test, columns=ohe_cols, drop_first=True)
# X_test = X_test.reindex(columns=X.columns, fill_value=0)   # 컬럼 불일치 방지

# %%
# --- 3-8. 스케일링 (트리 모델엔 불필요하지만 선형 모델 비교를 위해 적용)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# %%
# --- 3-9. 클래스 불균형 확인
minority_ratio = y.value_counts(normalize=True).min()
print(f"소수 클래스 비율: {minority_ratio:.2%}")
if minority_ratio < 0.2:
    print("  -> 불균형. class_weight='balanced' 또는 SMOTE 검토")
# from imblearn.over_sampling import SMOTE       # 분할 이후 X_train 에만 적용할 것
# X_train, y_train = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)

# ============================================================
# 4. Train / Validation 분할
# ============================================================
# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"Train {X_train.shape} / Val {X_val.shape} / Test(제출용) {X_test_scaled.shape}")

# ============================================================
# 5. 모델 비교 — 5개를 같은 조건으로
# ============================================================
# %%
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                             eval_metric='mlogloss', random_state=RANDOM_SEED),
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                               random_state=RANDOM_SEED, verbose=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    if NUM_CLASSES == 2:
        auc = roc_auc_score(y_val, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')

    results[name] = {'Accuracy': acc, 'F1_macro': f1, 'AUC': auc, 'CV_Acc': cv}
    print(f"[{name}] Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f} CV={cv:.4f}")

results_df = pd.DataFrame(results).T.sort_values('Accuracy', ascending=False)
print("\n[모델 비교]")
print(results_df)

# Accuracy 와 CV_Acc 차이가 크면 val 분할 운이 좋았던 것 → CV 높은 쪽을 선택
best_model_name = results_df.index[0]
print(f"\n1위: {best_model_name}")

# ============================================================
# 6. 하이퍼파라미터 튜닝 — 1위 모델 하나만
# ============================================================
# %%
param_grids = {
    'RandomForest': (
        RandomForestClassifier(random_state=RANDOM_SEED),
        {'n_estimators': [100, 200, 300], 'max_depth': [6, 10, 15, None],
         'min_samples_split': [2, 5]},
    ),
    'XGBoost': (
        XGBClassifier(eval_metric='mlogloss', random_state=RANDOM_SEED),
        {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8],
         'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0]},
    ),
    'LightGBM': (
        LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
        {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8],
         'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0]},
    ),
}

tune_name = best_model_name if best_model_name in param_grids else 'XGBoost'
est, params = param_grids[tune_name]
print(f"튜닝 대상: {tune_name}")

# scoring 은 문제가 요구하는 지표로 ★ (accuracy / f1_macro / roc_auc ...)
grid = GridSearchCV(est, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print(f"최적 파라미터: {grid.best_params_}")
print(f"최적 CV 점수: {grid.best_score_:.4f}")

# ============================================================
# 7. 최종 평가 (Validation)
# ============================================================
# %%
best_model = grid.best_estimator_
y_val_pred = best_model.predict(X_val)

acc_final = accuracy_score(y_val, y_val_pred)
f1_final = f1_score(y_val, y_val_pred, average='macro')
print(f"Accuracy {acc_final:.4f} / F1(macro) {f1_final:.4f}")
print(classification_report(y_val, y_val_pred))

# 성능 등급: Acc 0.95↑ 매우우수 / 0.90↑ 우수 / 0.85↑ 양호 / 0.80↑ 보통 / 미만 미흡
grade = next((g for t, g in [(0.95, '매우 우수'), (0.90, '우수'),
                             (0.85, '양호'), (0.80, '보통')] if acc_final >= t), '미흡')
print(f"-> {grade}")
if acc_final - f1_final > 0.1:
    print("Accuracy >> F1 : 소수 클래스를 못 맞히는 중 -> class_weight/SMOTE 검토")

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6 + NUM_CLASSES, 5 + NUM_CLASSES * 0.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (Validation)')
plt.tight_layout(); plt.show()

# ============================================================
# 8. 특성 중요도 — 한 컬럼이 압도적이면 누수를 의심할 것
# ============================================================
# %%
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(fi.head(10))
    fi.head(15).plot(kind='barh', figsize=(8, 5))
    plt.title('Feature Importance'); plt.tight_layout(); plt.show()

# ============================================================
# 9. 제출 파일 생성  ★
# ============================================================
# %%
test_predictions = best_model.predict(X_test_scaled)
if target_encoder:
    test_pred_labels = target_encoder.inverse_transform(test_predictions)
else:
    test_pred_labels = test_predictions

# ★ 문제가 요구하는 컬럼 구성으로 맞출 것
submission = df_test.copy()
submission['predicted'] = test_pred_labels

if has_test_label:
    y_cmp = target_encoder.transform(df_test[TARGET]) if target_encoder else df_test[TARGET]
    print(f"[Test] Acc={accuracy_score(y_cmp, test_predictions):.4f} "
          f"F1={f1_score(y_cmp, test_predictions, average='macro'):.4f}")

submission.to_csv(SUBMIT_CSV, index=False)
print(f"[저장] {SUBMIT_CSV} ({len(submission)}건)")
print(submission.head())

# 제출 전 점검: 행 수 == len(df_test), 컬럼명 일치, index=False, 결측 없음
assert len(submission) == len(df_test), "행 수 불일치!"

# %%
joblib.dump(best_model, SUBMIT_MODEL)
print(f"모델 저장 완료: {SUBMIT_MODEL}")

import os
print("현재 폴더:", [f for f in os.listdir('.') if f.startswith(PHONE)])
# 파일이 실제로 만들어졌는지 눈으로 확인. to_csv 가 조용히 다른 경로에 쓰는 일이 있습니다.
