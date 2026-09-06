# %% [markdown]
# # Tabular 분류 템플릿 (이진/다중 자동 대응)
#
# 교재: docs/01-tabular.md
#
# p1_NUM_CLASSES 를 데이터에서 자동으로 읽으므로 이진분류든 다중분류든 코드가 같습니다.
#
# ── 데이터 흐름 ──
# p1_df_train / p1_df_test  (로드)
#   ↓ 전처리 (train 에서 fit, test 는 transform)
# p1_X, p1_y  /  p1_X_test
#   ↓ 분할
# p1_X_train, p1_X_val  (학습/검증)
#   ↓ 학습 → 검증 → 제출
# p1_submission.csv
#
# ★ 표시가 있는 곳만 문제에 맞게 고치면 됩니다.
#
# ── 변수 이름 규칙 ──
# 커널 하나에서 세 문제를 풀므로, 전역 변수가 서로 덮어쓰지 않도록
# 템플릿마다 시험 순서대로 접두사를 붙여 두었습니다.
#   1번 Tabular = p1_ · 2번 Text = p2_ · 3번 Image = p3_
# 실제 순서가 다르게 나오면 Ctrl+H 전체 바꾸기로 접두사를 맞추고 PROB_NO 도 함께 고치세요.
# 안 고치면 다른 문제의 제출물을 덮어씁니다.
# PHONE, RANDOM_SEED, check_submission 은 세션 공용이라 접두사가 없습니다.

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
# 0. 문제 설정
# ============================================================
# GPU 확인과 PHONE 은 세션 시작 셀(00_session_start.py)에서 합니다. 여기서 다시 하지 않습니다.
# 단, 세션이 튕기면 GPU 연결이 끊기고 변수가 전부 사라지므로 그 셀부터 다시 실행해야 합니다.
p1_PROB_NO = 1                # ★ 시험 순서: Tabular = 1번

# 세션 시작 셀을 실행했다면 PHONE 은 이미 정의돼 있습니다.
# 이 파일만 단독으로 연습할 때를 위한 기본값입니다.
try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

# 지문 ④: 제출 파일명 형식을 지문 표기 그대로 (밑줄·확장자까지) 확인
p1_SUBMIT_CSV = f'{PHONE}_{p1_PROB_NO}.csv'
p1_SUBMIT_MODEL = f'{PHONE}_{p1_PROB_NO}.pkl'
# 지문 ③: 결과물 컬럼명 — 지문이 지정한 이름 그대로 (임의로 지으면 채점 불가)
p1_PRED_COL = 'pred'                              # ★ 지문의 컬럼명으로 교체
print(f"제출 파일: {p1_SUBMIT_CSV} / {p1_SUBMIT_MODEL} / 결과 컬럼: {p1_PRED_COL}")

# ============================================================
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 (실제 문항 형식) ---
# 지문 4단계: ① 데이터 설명 (train, test — test 는 정답 없음)
#             ② 무엇을 맞출지  ③ 결과물 형식 (컬럼명·값. 예: 0/1 이 아니라 Y/N)
#             ④ 제출 파일명 (모델 = 휴대폰번호_문제번호.확장자 / 결과 = 휴대폰번호_문제번호.csv)
# p1_df_train = pd.read_csv('train.csv')
# p1_df_test  = pd.read_csv('test.csv')    # ★ 정답 없음 — 예측해서 제출하는 대상
# p1_TARGET   = 'target'                   # ★ 지문 ②의 타겟 컬럼명

# --- 연습용: Titanic (이진분류) — train 을 쪼개 test 역할을 만든다 ---
p1_df_all = sns.load_dataset('titanic')
p1_TARGET = 'survived'
p1_df_train, p1_df_test = train_test_split(
    p1_df_all, test_size=0.2, random_state=RANDOM_SEED, stratify=p1_df_all[p1_TARGET]
)
p1_df_train = p1_df_train.reset_index(drop=True)
p1_df_test = p1_df_test.reset_index(drop=True)

# ── 용어 주의 ──
# df_test / X_test = 시험이 준 파일(정답 없음). 예측해서 제출하는 대상.
# X_val            = train 을 우리가 쪼갠 내부 검증. 점수 확인용.

p1_NUM_CLASSES = p1_df_train[p1_TARGET].nunique()
print(f"Train {p1_df_train.shape} / Test(제출 대상·정답 없음) {p1_df_test.shape}")
print(f"클래스 {p1_NUM_CLASSES}개 ({'이진분류' if p1_NUM_CLASSES == 2 else '다중분류'})")
print(p1_df_train[p1_TARGET].value_counts().sort_index())
p1_df_train.head()

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
p1_col_summary = pd.DataFrame({
    'dtype': p1_df_train.dtypes,
    'missing': p1_df_train.isnull().sum(),
    'miss_pct': (p1_df_train.isnull().mean() * 100).round(1),
    'nunique': p1_df_train.nunique(),
    'nunique_pct': (p1_df_train.nunique() / len(p1_df_train) * 100).round(1),
    'sample': p1_df_train.iloc[0],
})
print(p1_col_summary.to_string())

# %%
p1_numeric_cols = p1_df_train.select_dtypes(include=[np.number]).columns
if p1_TARGET in p1_numeric_cols:
    print("[타겟과의 상관관계 - 0.95 이상이면 누수 의심]")
    print(p1_df_train[p1_numeric_cols].corr()[p1_TARGET].abs().sort_values(ascending=False))

# ============================================================
# 3. 전처리 — train 에서 fit, test 는 transform
# ============================================================
# %%
# --- 3-1. 불필요 컬럼 제거 (train, test 동시에) ★
p1_drop_cols = ['alive', 'pclass', 'deck']       # ★ 위 표를 보고 결정
p1_df_train = p1_df_train.drop(columns=[c for c in p1_drop_cols if c in p1_df_train.columns], errors='ignore')
p1_df_test = p1_df_test.drop(columns=[c for c in p1_drop_cols if c in p1_df_test.columns], errors='ignore')
print(f"제거: {p1_drop_cols}")

# --- 3-2. Feature / Target 분리
p1_X = p1_df_train.drop(columns=[p1_TARGET])
p1_y = p1_df_train[p1_TARGET].copy()

p1_has_test_label = p1_TARGET in p1_df_test.columns
p1_X_test = p1_df_test.drop(columns=[p1_TARGET]) if p1_has_test_label else p1_df_test.copy()
p1_y_test = p1_df_test[p1_TARGET].copy() if p1_has_test_label else None

# 타겟이 문자열이면 숫자로 (제출 시 inverse_transform 으로 복원)
p1_target_encoder = None
if p1_y.dtype == 'object' or p1_y.dtype.name == 'category':
    p1_target_encoder = LabelEncoder()
    p1_y = pd.Series(p1_target_encoder.fit_transform(p1_y))
    if p1_y_test is not None:
        p1_y_test = pd.Series(p1_target_encoder.transform(p1_y_test))
    print(f"타겟 클래스: {p1_target_encoder.classes_}")

# --- 3-3. 수치형 / 범주형 자동 분리
p1_num_cols = p1_X.select_dtypes(include=[np.number]).columns.tolist()
p1_cat_cols = p1_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"수치형 {len(p1_num_cols)}: {p1_num_cols}")
print(f"범주형 {len(p1_cat_cols)}: {p1_cat_cols}")

# %%
# --- 3-4. 결측치 (수치형=중앙값, 범주형=최빈값)
p1_num_imputer = SimpleImputer(strategy='median')
p1_cat_imputer = SimpleImputer(strategy='most_frequent')

if p1_num_cols:
    p1_X[p1_num_cols] = p1_num_imputer.fit_transform(p1_X[p1_num_cols])
    p1_X_test[p1_num_cols] = p1_num_imputer.transform(p1_X_test[p1_num_cols])
if p1_cat_cols:
    p1_X[p1_cat_cols] = p1_cat_imputer.fit_transform(p1_X[p1_cat_cols])
    p1_X_test[p1_cat_cols] = p1_cat_imputer.transform(p1_X_test[p1_cat_cols])

print(f"결측 남은 개수 - train {p1_X.isnull().sum().sum()} / test {p1_X_test.isnull().sum().sum()}")

# %%
# [선택] 이상치 클리핑·피처 엔지니어링은 점수가 부족할 때만 — 코드는 docs/01-tabular.md
# --- 3-5. 인코딩 (LabelEncoder: 트리 모델에 충분하고 빠름)
# train 에서 배운 값을 숫자 사전으로 만들고, test 는 사전으로 변환.
# test 에만 있는 값은 사전에 없으므로 NaN → -1 이 된다 (에러 대신 안전한 기본값)
p1_label_encoders = {}
for p1_col in p1_cat_cols:
    p1_le = LabelEncoder()
    p1_X[p1_col] = p1_le.fit_transform(p1_X[p1_col].astype(str))
    p1_code = {v: i for i, v in enumerate(p1_le.classes_)}
    p1_X_test[p1_col] = p1_X_test[p1_col].astype(str).map(p1_code).fillna(-1).astype(int)
    p1_label_encoders[p1_col] = p1_le

# [선택] 선형 모델 주력이면 OneHot — 코드는 docs/01-tabular.md

# %%
# --- 3-6. 스케일링 (트리 모델엔 불필요하지만 선형 모델 비교를 위해 적용)
p1_scaler = StandardScaler()
p1_X_scaled = pd.DataFrame(p1_scaler.fit_transform(p1_X), columns=p1_X.columns)
p1_X_test_scaled = pd.DataFrame(p1_scaler.transform(p1_X_test), columns=p1_X_test.columns)

# %%
# --- 3-7. 클래스 불균형 확인
p1_minority_ratio = p1_y.value_counts(normalize=True).min()
print(f"소수 클래스 비율: {p1_minority_ratio:.2%}")
if p1_minority_ratio < 0.2:
    print("  -> 불균형. class_weight='balanced' 검토 (SMOTE 는 분할 뒤 train 에만)")

# ============================================================
# 4. Train / Validation 분할
# ============================================================
# %%
p1_X_train, p1_X_val, p1_y_train, p1_y_val = train_test_split(
    p1_X_scaled, p1_y, test_size=0.2, random_state=RANDOM_SEED, stratify=p1_y
)
print(f"내부: Train {p1_X_train.shape} / Val {p1_X_val.shape}   제출 대상: Test {p1_X_test_scaled.shape}")

# ============================================================
# 5. 모델 비교 — 5개를 같은 조건으로
# ============================================================
# %%
p1_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                             eval_metric='mlogloss', random_state=RANDOM_SEED),
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                               random_state=RANDOM_SEED, verbose=-1),
}

p1_results = {}
for p1_name, p1_model in p1_models.items():
    p1_model.fit(p1_X_train, p1_y_train)
    p1_y_pred = p1_model.predict(p1_X_val)
    p1_y_proba = p1_model.predict_proba(p1_X_val)

    p1_acc = accuracy_score(p1_y_val, p1_y_pred)
    p1_f1 = f1_score(p1_y_val, p1_y_pred, average='macro')
    p1_cv = cross_val_score(p1_model, p1_X_train, p1_y_train, cv=5, scoring='accuracy').mean()
    if p1_NUM_CLASSES == 2:
        p1_auc = roc_auc_score(p1_y_val, p1_y_proba[:, 1])
    else:
        p1_auc = roc_auc_score(p1_y_val, p1_y_proba, multi_class='ovr', average='macro')

    p1_results[p1_name] = {'Accuracy': p1_acc, 'F1_macro': p1_f1, 'AUC': p1_auc, 'CV_Acc': p1_cv}
    print(f"[{p1_name}] Acc={p1_acc:.4f} F1={p1_f1:.4f} AUC={p1_auc:.4f} CV={p1_cv:.4f}")

p1_results_df = pd.DataFrame(p1_results).T.sort_values('Accuracy', ascending=False)
print("\n[모델 비교]")
print(p1_results_df)

# Accuracy 와 CV_Acc 차이가 크면 val 분할 운이 좋았던 것 → CV 높은 쪽을 선택
p1_best_model_name = p1_results_df.index[0]
print(f"\n1위: {p1_best_model_name}")

# ============================================================
# 6. 하이퍼파라미터 튜닝 — 1위 모델 하나만
# ============================================================
# %%
p1_param_grids = {
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

p1_tune_name = p1_best_model_name if p1_best_model_name in p1_param_grids else 'XGBoost'
p1_est, p1_params = p1_param_grids[p1_tune_name]
print(f"튜닝 대상: {p1_tune_name}")

# scoring 은 문제가 요구하는 지표로 ★ (accuracy / f1_macro / roc_auc ...)
# Colab CPU 는 2코어 — 그리드가 크거나 데이터가 10만 행+ 이면 오래 걸립니다.
# 5분 넘게 돌면 중단하고 RandomizedSearchCV(..., n_iter=20) 로 바꾸세요.
p1_grid = GridSearchCV(p1_est, p1_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
p1_grid.fit(p1_X_train, p1_y_train)
print(f"최적 파라미터: {p1_grid.best_params_}")
print(f"최적 CV 점수: {p1_grid.best_score_:.4f}")

# ============================================================
# 7. 최종 평가 (Validation)
# ============================================================
# %%
p1_best_model = p1_grid.best_estimator_
p1_y_val_pred = p1_best_model.predict(p1_X_val)

p1_acc_final = accuracy_score(p1_y_val, p1_y_val_pred)
p1_f1_final = f1_score(p1_y_val, p1_y_val_pred, average='macro')
print(f"Accuracy {p1_acc_final:.4f} / F1(macro) {p1_f1_final:.4f}")
print(classification_report(p1_y_val, p1_y_val_pred))

if p1_acc_final - p1_f1_final > 0.1:
    print("Accuracy >> F1 : 소수 클래스를 못 맞히는 중 -> class_weight/SMOTE 검토")

p1_cm = confusion_matrix(p1_y_val, p1_y_val_pred)
plt.figure(figsize=(6 + p1_NUM_CLASSES, 5 + p1_NUM_CLASSES * 0.5))
sns.heatmap(p1_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (Validation)')
plt.tight_layout(); plt.show()

# ============================================================
# 8. 특성 중요도 — 한 컬럼이 압도적이면 누수를 의심할 것
# ============================================================
# %%
if hasattr(p1_best_model, 'feature_importances_'):
    p1_fi = pd.Series(p1_best_model.feature_importances_, index=p1_X.columns).sort_values(ascending=False)
    print(p1_fi.head(10))
    p1_fi.head(15).plot(kind='barh', figsize=(8, 5))
    plt.title('Feature Importance'); plt.tight_layout(); plt.show()

# ============================================================
# 9. 제출 파일 생성  ★
# ============================================================
# %%
p1_test_predictions = p1_best_model.predict(p1_X_test_scaled)

# ★ label 복원 필수 — 인코딩된 숫자(0/1/2) 그대로 내면 그 문제는 0점
if p1_target_encoder:
    p1_test_pred_labels = p1_target_encoder.inverse_transform(p1_test_predictions)
else:
    p1_test_pred_labels = p1_test_predictions

# 지문 ③이 지정한 컬럼만 담는다. 값은 복원된 label (예: 0/1 이 아니라 Y/N)
p1_submission = pd.DataFrame({p1_PRED_COL: p1_test_pred_labels})
# 지문이 ID 류 컬럼도 요구하면 지문 컬럼명 그대로 추가:
# p1_submission.insert(0, 'ID', p1_df_test['ID'].values)

if p1_has_test_label:
    p1_y_cmp = p1_target_encoder.transform(p1_df_test[p1_TARGET]) if p1_target_encoder else p1_df_test[p1_TARGET]
    print(f"[Test] Acc={accuracy_score(p1_y_cmp, p1_test_predictions):.4f} "
          f"F1={f1_score(p1_y_cmp, p1_test_predictions, average='macro'):.4f}")

p1_submission.to_csv(p1_SUBMIT_CSV, index=False)
print(f"[저장] {p1_SUBMIT_CSV} ({len(p1_submission)}건)")
print(p1_submission.head())

# %%
joblib.dump(p1_best_model, p1_SUBMIT_MODEL)
print(f"모델 저장 완료: {p1_SUBMIT_MODEL}")

# ============================================================
# 10. 제출물 검증  ★ 문제를 끝낼 때마다
# ============================================================
# %%
# check_submission 은 세션 시작 셀(00_session_start.py)에서 정의됩니다.
# 행 수 / 요구 컬럼 / 결측 / index=False / label 미복원(0점) / 모델 파일을 한 번에 점검.
p1_allowed = p1_target_encoder.classes_ if p1_target_encoder else p1_y.unique()
try:
    check_submission(p1_SUBMIT_CSV, n_rows=len(p1_df_test),
                     required_cols=[p1_PRED_COL],
                     label_col=p1_PRED_COL, allowed_labels=p1_allowed,
                     model_path=p1_SUBMIT_MODEL)
except NameError:
    # 세션 시작 셀 없이 단독 연습 중일 때의 최소 점검
    import os
    p1__sub = pd.read_csv(p1_SUBMIT_CSV)
    assert len(p1__sub) == len(p1_df_test), "행 수 불일치!"
    assert not p1__sub[p1_PRED_COL].isnull().any(), "예측 컬럼에 결측!"
    assert set(p1__sub[p1_PRED_COL]) <= set(p1_allowed), "label 미복원 의심 — 0점 위험!"
    assert os.path.getsize(p1_SUBMIT_MODEL) > 0, "모델 파일 없음!"
    print("최소 점검 통과 (전체 검증은 세션 시작 셀의 check_submission)")
