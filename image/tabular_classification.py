# %% [markdown]
# # Tabular - 분류 (Classification) 통합 템플릿
# AICE Professional 시험 대비
#
# NUM_CLASSES = 2 → 이진분류 / NUM_CLASSES >= 3 → 다중분류
# 코드 변경 없이 자동 대응
#
# ── 데이터 흐름 ──
# df_train / df_test (로드)
#   ↓ 전처리
# X, y (학습 전체) / X_test, y_test (제출용)
#   ↓ 스케일링
# X_scaled / X_test_scaled
#   ↓ 분할
# X_train, X_val, y_train, y_val (학습/검증)
#   ↓ 학습 → 검증 → 제출
# model.fit(X_train) → predict(X_val) → predict(X_test_scaled) → submission.csv
#
# 평가지표: Accuracy, F1-Score (macro), AUC-ROC

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
# 1. 데이터 로드
# ============================================================
# %%
# ==============================================
# [데이터셋 선택] 아래 중 하나만 주석 해제
# ==============================================

# --- (A) 시험 환경 (train.csv / test.csv 별도 제공) ---
# df_train = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')
# TARGET = 'target'  # 실제 타겟 컬럼명으로 변경

# --- (B) Titanic (이진분류) ---
df_all = sns.load_dataset('titanic')
TARGET = 'survived'
df_train, df_test = train_test_split(
    df_all, test_size=0.2, random_state=RANDOM_SEED, stratify=df_all[TARGET]
)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# --- (C) Wine (다중분류) ---
# from sklearn.datasets import load_wine
# data = load_wine()
# df_all = pd.DataFrame(data.data, columns=data.feature_names)
# df_all['target'] = data.target
# TARGET = 'target'
# df_train, df_test = train_test_split(
#     df_all, test_size=0.2, random_state=RANDOM_SEED, stratify=df_all['target']
# )
# df_train = df_train.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

# --- (D) Iris (다중분류) ---
# from sklearn.datasets import load_iris
# data = load_iris()
# df_all = pd.DataFrame(data.data, columns=data.feature_names)
# df_all['target'] = data.target
# TARGET = 'target'
# df_train, df_test = train_test_split(
#     df_all, test_size=0.2, random_state=RANDOM_SEED, stratify=df_all['target']
# )
# df_train = df_train.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

NUM_CLASSES = df_train[TARGET].nunique()

print(f"Train 크기: {df_train.shape}")
print(f"Test  크기: {df_test.shape}")
print(f"클래스 수: {NUM_CLASSES} ({'이진분류' if NUM_CLASSES == 2 else '다중분류'})")
print(f"클래스 분포 (train):\n{df_train[TARGET].value_counts().sort_index()}")
df_train.head()

# ============================================================
# 2. EDA (탐색적 데이터 분석) - Train 데이터 기준
# ============================================================
# %%
print("=" * 50)
print("[타겟 분포]")
print(df_train[TARGET].value_counts())
print(f"비율: {df_train[TARGET].value_counts(normalize=True).values}")

print("\n[결측치]")
missing = df_train.isnull().sum()
print(missing[missing > 0])

print("\n[데이터 타입]")
print(df_train.dtypes)

print("\n[기술 통계]")
df_train.describe()

# %%
print("[수치형 변수 상관관계 - 타겟 기준]")
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
if TARGET in numeric_cols:
    corr_with_target = df_train[numeric_cols].corr()[TARGET].abs().sort_values(ascending=False)
    print(corr_with_target)

# %%
# -------------------------------------------------------
# 불필요 컬럼 판단을 위한 컬럼 요약 정보
# -------------------------------------------------------
# [제거 기준]
# 결측률 70% 이상          → 정보량 부족, 제거 권장
# 고유값 비율 95% 이상      → ID/이름 등 식별자 컬럼, 제거
# 타겟 상관관계 0.95 이상   → 데이터 누수(leakage) 의심, 제거
# 고유값 1개 (분산 0)       → 정보 없음, 제거
# 다른 컬럼과 의미 중복     → 하나만 남기고 제거 (예: class vs pclass)
print("\n[컬럼별 요약 - 제거 여부 판단용]")
col_summary = pd.DataFrame({
    'dtype': df_train.dtypes,
    'missing': df_train.isnull().sum(),
    'miss_pct': (df_train.isnull().mean() * 100).round(1),
    'nunique': df_train.nunique(),
    'nunique_pct': (df_train.nunique() / len(df_train) * 100).round(1),
    'sample': df_train.iloc[0]
})
print(col_summary.to_string())

# ============================================================
# 3. 전처리 (Preprocessing) - Train 기준 fit, Test는 transform만
# ============================================================
# %%
# -------------------------------------------------------
# 3-1. 불필요한 컬럼 제거 (train, test 동시)
# -------------------------------------------------------
# 위 요약 표를 보고 직접 판단하여 리스트에 추가
# 예: drop_cols = ['alive', 'class', 'deck']
drop_cols = ['alive', 'pclass', 'deck']
if drop_cols:
    print(f"제거할 컬럼: {drop_cols}")
    df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
    df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')

# -------------------------------------------------------
# 3-2. Feature / Target 분리
# -------------------------------------------------------
X = df_train.drop(columns=[TARGET])
y = df_train[TARGET].copy()

has_test_label = TARGET in df_test.columns
if has_test_label:
    X_test = df_test.drop(columns=[TARGET])
    y_test = df_test[TARGET].copy()
else:
    X_test = df_test.copy()
    y_test = None

# 타겟이 문자열이면 숫자로 변환
target_encoder = None
if y.dtype == 'object' or y.dtype.name == 'category':
    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(y))
    if y_test is not None:
        y_test = pd.Series(target_encoder.transform(y_test))
    print(f"타겟 클래스: {target_encoder.classes_}")

# -------------------------------------------------------
# 3-3. 수치형 / 범주형 자동 분리
# -------------------------------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"수치형 ({len(num_cols)}): {num_cols}")
print(f"범주형 ({len(cat_cols)}): {cat_cols}")

# %%
# -------------------------------------------------------
# 3-4. 결측치 처리 (train에서 fit, test는 transform)
# -------------------------------------------------------
# 수치형: median(중앙값) - 이상치에 강건함. mean(평균)도 가능
# 범주형: most_frequent(최빈값)
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

if num_cols:
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
if cat_cols:
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

print(f"결측치 처리 후 (train): {X.isnull().sum().sum()}")
print(f"결측치 처리 후 (test) : {X_test.isnull().sum().sum()}")

# %%
# -------------------------------------------------------
# 3-5. 이상치 처리 (선택 - 성능 안 나올 때 적용)
# -------------------------------------------------------
# IQR(사분위범위) 방식: Q1-1.5*IQR ~ Q3+1.5*IQR 범위 밖을 경계값으로 클리핑
# 주의: 트리 모델(RF, XGB, LGBM)은 이상치에 강건하여 보통 불필요
#       선형 모델(LogisticRegression, Ridge)에서는 효과적
# 주의: train 기준 IQR로 test도 클리핑해야 일관성 유지
# for col in num_cols:
#     Q1 = X[col].quantile(0.25)
#     Q3 = X[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
#     before = ((X[col] < lower) | (X[col] > upper)).sum()
#     X[col] = X[col].clip(lower, upper)
#     X_test[col] = X_test[col].clip(lower, upper)
#     if before > 0:
#         print(f"  {col}: 이상치 {before}개 클리핑 ({lower:.2f} ~ {upper:.2f})")

# %%
# -------------------------------------------------------
# 3-6. 피처 엔지니어링 (선택 - 도메인 지식 기반)
# -------------------------------------------------------
# 기존 컬럼을 조합하여 새로운 특성을 만들면 성능이 오를 수 있음
# 예시 (Titanic):
#   - 가족 크기: sibsp + parch + 1
#   - 혼자 탑승 여부: 가족 크기 == 1
#   - 요금 구간: pd.qcut(fare, 4)
# 주의: train과 test 모두에 동일하게 적용
# X['family_size'] = X['sibsp'] + X['parch'] + 1
# X_test['family_size'] = X_test['sibsp'] + X_test['parch'] + 1

# %%
# -------------------------------------------------------
# 3-7. 인코딩 (범주형 → 숫자)
# -------------------------------------------------------
# LabelEncoder: 순서가 있는 범주에 적합 (예: low/mid/high → 0/1/2)
# OneHotEncoder: 순서가 없는 범주에 적합 (예: 색상, 도시)
#   - 트리 모델은 LabelEncoder로 충분 (분기 기준으로 자동 처리)
#   - 선형 모델은 OneHotEncoder가 더 적합
# 주의: train에서 fit, test는 transform (unseen 레이블 대비 필요)

# [기본] LabelEncoder (빠르고 간단, 트리 모델에 충분)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = X_test[col].astype(str).map(
        lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
    )
    label_encoders[col] = le
    print(f"  {col}: {le.classes_} → {list(range(len(le.classes_)))}")

# [선택] OneHotEncoder - 범주가 순서 없고 & 고유값이 적을 때 (10개 이하 권장)
# ohe_cols = [col for col in cat_cols if X[col].nunique() <= 10]
# X = pd.get_dummies(X, columns=ohe_cols, drop_first=True)
# X_test = pd.get_dummies(X_test, columns=ohe_cols, drop_first=True)
# X_test = X_test.reindex(columns=X.columns, fill_value=0)

# %%
# -------------------------------------------------------
# 3-8. 스케일링 (train에서 fit, test는 transform)
# -------------------------------------------------------
# StandardScaler: 평균=0, 표준편차=1 (일반적으로 가장 무난)
# MinMaxScaler: 0~1 범위 (신경망 입력 시 주로 사용)
# 주의: 트리 모델(RF, XGB, LGBM)은 스케일링 불필요하지만, 여러 모델을
#       동시에 비교할 때는 적용해도 트리 모델 성능에 영향 없으므로 안전
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# %%
# -------------------------------------------------------
# 3-9. 클래스 불균형 처리 (선택 - 불균형 심할 때 적용)
# -------------------------------------------------------
# 소수 클래스 비율이 20% 이하이면 고려
# 방법 1: 모델 파라미터로 처리 (가장 간단)
#   - XGBoost: scale_pos_weight = (음성 수 / 양성 수)
#   - LightGBM: is_unbalance=True
#   - RandomForest: class_weight='balanced'
# 방법 2: SMOTE 오버샘플링 (pip install imbalanced-learn 필요)
#   - 학습 데이터에만 적용, 검증/테스트 데이터에는 절대 적용하지 않음
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=RANDOM_SEED)
# X_train, y_train = smote.fit_resample(X_train, y_train)
# print(f"SMOTE 적용 후: {pd.Series(y_train).value_counts()}")

minority_ratio = y.value_counts(normalize=True).min()
print(f"소수 클래스 비율: {minority_ratio:.2%}")
if minority_ratio < 0.2:
    print("  → 불균형 감지! class_weight='balanced' 또는 SMOTE 고려")

# ============================================================
# 4. Train / Validation 분할
# ============================================================
# %%
# X(학습 전체) → X_train(학습) / X_val(검증)
# X_test_scaled는 최종 제출용으로 별도 보관
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test(제출용): {X_test_scaled.shape}")

# ============================================================
# 5. 모델 학습 및 비교
# ============================================================
# %%
models = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_SEED
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_SEED
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        random_state=RANDOM_SEED, verbose=-1
    ),
}

results = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
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
    print(f"[{name}] Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | CV={cv:.4f}")

results_df = pd.DataFrame(results).T.sort_values('Accuracy', ascending=False)
print("\n[모델 비교 결과]")
print(results_df)

best_model_name = results_df.index[0]
print(f"\n1위 모델: {best_model_name}")

# ============================================================
# 6. 하이퍼파라미터 튜닝
# ============================================================
# %%
param_grids = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_SEED),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 10, 15, None],
            'min_samples_split': [2, 5],
        }
    },
    'XGBoost': {
        'model': XGBClassifier(
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_SEED
        ),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
        }
    },
}

tune_name = best_model_name if best_model_name in param_grids else 'XGBoost'
print(f"하이퍼파라미터 튜닝 대상: {tune_name}")

grid = GridSearchCV(
    param_grids[tune_name]['model'],
    param_grids[tune_name]['params'],
    cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print(f"최적 파라미터: {grid.best_params_}")
print(f"최적 CV 점수: {grid.best_score_:.4f}")

# ============================================================
# 7. 최종 평가 (Validation 기준)
# ============================================================
# %%
best_model = grid.best_estimator_
y_val_pred = best_model.predict(X_val)

acc_final = accuracy_score(y_val, y_val_pred)
f1_final = f1_score(y_val, y_val_pred, average='macro')

print("[최종 모델 성능 - Validation]")
print(f"Accuracy:   {acc_final:.4f}")
print(f"F1 (macro): {f1_final:.4f}")
print("\n[분류 보고서]")
print(classification_report(y_val, y_val_pred))

# %%
# -------------------------------------------------------
# 성능 평가 기준
# -------------------------------------------------------
# [Accuracy 기준]
#   0.95 이상 : 매우 우수 (Excellent)
#   0.90 ~ 0.95: 우수 (Good)
#   0.85 ~ 0.90: 양호 (Acceptable)
#   0.80 ~ 0.85: 보통 (Moderate) - 추가 전처리/피처 엔지니어링 필요
#   0.80 미만 : 미흡 (Poor) - 모델/데이터 재검토 필요
#
# [F1 Score (macro) 기준]
#   0.90 이상 : 매우 우수
#   0.80 ~ 0.90: 우수
#   0.70 ~ 0.80: 양호
#   0.60 ~ 0.70: 보통
#   0.60 미만 : 미흡
#
# [Accuracy vs F1 비교]
#   Accuracy ≈ F1 → 클래스 간 성능 균형 (불균형 없음)
#   Accuracy >> F1 → 소수 클래스 성능 부족 (불균형 문제 → SMOTE/가중치 적용)
# -------------------------------------------------------

acc_grades = [(0.95, '매우 우수'), (0.90, '우수'), (0.85, '양호'), (0.80, '보통')]
acc_grade = '미흡'
for threshold, grade in acc_grades:
    if acc_final >= threshold:
        acc_grade = grade
        break

f1_grades = [(0.90, '매우 우수'), (0.80, '우수'), (0.70, '양호'), (0.60, '보통')]
f1_grade = '미흡'
for threshold, grade in f1_grades:
    if f1_final >= threshold:
        f1_grade = grade
        break

print(f"\n[성능 평가]")
print(f"Accuracy = {acc_final:.4f} → {acc_grade}")
print(f"F1 Score = {f1_final:.4f} → {f1_grade}")
gap = acc_final - f1_final
if gap > 0.1:
    print(f"  Accuracy-F1 gap = {gap:.3f} → 클래스 불균형 의심, SMOTE/가중치 적용 검토")

# 혼동 행렬
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6 + NUM_CLASSES, 5 + NUM_CLASSES * 0.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Validation)')
plt.tight_layout()
plt.show()

# ============================================================
# 8. 특성 중요도
# ============================================================
# %%
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns)
    fi = fi.sort_values(ascending=False)
    print("[특성 중요도 Top 10]")
    print(fi.head(10))

    fi.head(15).plot(kind='barh', figsize=(8, 5))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# ============================================================
# 9. 결과 CSV 파일 생성
# ============================================================
# %%
# --- 9-1. 모델 비교 결과 저장 ---
#results_df.to_csv('classification_model_comparison.csv')
#print("[저장] classification_model_comparison.csv")
print(results_df)

# --- 9-2. Validation 결과 (원본 + 예측) 저장 ---
X_val_original = pd.DataFrame(
    scaler.inverse_transform(X_val),
    columns=X.columns,
    index=X_val.index
)
val_result_df = X_val_original.copy()
val_result_df['actual'] = y_val.values
val_result_df['predicted'] = y_val_pred
val_result_df['correct'] = (y_val.values == y_val_pred).astype(int)

#val_result_df.to_csv('classification_result.csv', index=False)
#print(f"\n[저장] classification_result.csv ({len(val_result_df)}건)")
print(val_result_df.head(10))

# --- 9-3. 최종 성능 요약 저장 ---
summary_df = pd.DataFrame([{
    'model': tune_name,
    'best_params': str(grid.best_params_),
    'accuracy': acc_final,
    'f1_macro': f1_final,
    'accuracy_grade': acc_grade,
    'f1_grade': f1_grade,
}])
#summary_df.to_csv('classification_summary.csv', index=False)
#print(f"\n[저장] classification_summary.csv")
print(summary_df)

# ============================================================
# 10. 제출 파일 생성 (Test 데이터 예측)
# ============================================================
# %%
test_predictions = best_model.predict(X_test_scaled)

if target_encoder:
    test_pred_labels = target_encoder.inverse_transform(test_predictions)
else:
    test_pred_labels = test_predictions

submission = df_test.copy()
submission['predicted'] = test_pred_labels

if has_test_label:
    if target_encoder:
        y_test_compare = target_encoder.transform(df_test[TARGET])
    else:
        y_test_compare = df_test[TARGET]
    test_acc = accuracy_score(y_test_compare, test_predictions)
    test_f1 = f1_score(y_test_compare, test_predictions, average='macro')
    submission['correct'] = (y_test_compare.values == test_predictions).astype(int)
    print(f"[Test 성능] Acc={test_acc:.4f} | F1={test_f1:.4f}")

submission.to_csv('submission.csv', index=False)
print(f"\n[저장] submission.csv ({len(submission)}건)")
print(submission.head(10))

# %%
joblib.dump(best_model, 'tabular_classification_model.pkl')
print("\n모델 저장 완료: tabular_classification_model.pkl")

# %%
