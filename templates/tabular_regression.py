# %% [markdown]
# # Tabular 회귀 템플릿
#
# 교재: docs/01-tabular.md (6절)
#
# 분류 템플릿과 전처리·분할이 동일합니다. 달라지는 곳은 세 군데뿐입니다.
#   1) 모델        ...Classifier → ...Regressor
#   2) 분할        stratify 없음 (연속값이라 불가)
#   3) 평가지표    Accuracy/F1 → RMSE / MAE / R2
#
# ★ 표시가 있는 곳만 문제에 맞게 고치면 됩니다.
#
# ── 변수 이름 규칙: 왜 p1_ 이 붙어 있나 ──
# 시험은 주피터 커널 하나에서 여러 문제를 풉니다. 전역 변수는 커널 전체가 공유하므로,
# 접두사가 없으면 2번 문제의 X_train 이 1번 문제의 X_train 을 그대로 덮어씁니다.
# 1번 문제로 돌아와 셀 하나만 다시 돌리는 순간 조용히 틀린 결과가 나옵니다.
#
#   2번 문제에 쓸 때 (Colab: 편집 > 찾기 및 바꾸기, Ctrl+H)
#     p1_  ->  p2_        전체 바꾸기
#     p2_PROB_NO = 2      숫자도 함께 수정   ← 이 한 줄을 가장 자주 놓칩니다
#
# PHONE, RANDOM_SEED 는 세션 공용이라 접두사가 없습니다.

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# 0. 문제 설정  ★ 이 문제에서 바꾸는 것은 p1_PROB_NO 한 줄뿐
# ============================================================
# GPU 확인과 PHONE 은 세션 시작 셀(00_session_start.py)에서 합니다. 여기서 다시 하지 않습니다.
# 단, 세션이 튕기면 GPU 연결이 끊기고 변수가 전부 사라지므로 그 셀부터 다시 실행해야 합니다.
p1_PROB_NO = 1                # ★ 문제 번호

# 세션 시작 셀을 실행했다면 PHONE 은 이미 정의돼 있습니다.
# 이 파일만 단독으로 연습할 때를 위한 기본값입니다.
try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

p1_SUBMIT_CSV = f'{PHONE}_{p1_PROB_NO}.csv'
p1_SUBMIT_MODEL = f'{PHONE}_{p1_PROB_NO}.pkl'
print(f"제출 파일: {p1_SUBMIT_CSV} / {p1_SUBMIT_MODEL}")

# ============================================================
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 ---
# p1_df_train = pd.read_csv('train.csv')
# p1_df_test  = pd.read_csv('test.csv')
# p1_TARGET   = 'target'          # ★ 실제 타겟 컬럼명

# --- 연습용: California Housing ---
from sklearn.datasets import fetch_california_housing
p1_data = fetch_california_housing()
p1_df_all = pd.DataFrame(p1_data.data, columns=p1_data.feature_names)
p1_df_all['target'] = p1_data.target
p1_TARGET = 'target'
p1_df_train, p1_df_test = train_test_split(p1_df_all, test_size=0.2, random_state=RANDOM_SEED)
p1_df_train = p1_df_train.reset_index(drop=True)
p1_df_test = p1_df_test.reset_index(drop=True)

print(f"Train {p1_df_train.shape} / Test {p1_df_test.shape}")
print(p1_df_train[p1_TARGET].describe())
p1_df_train.head()

# ============================================================
# 2. EDA
# ============================================================
# %%
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
print("[타겟과의 상관관계 - 0.95 이상이면 누수 의심]")
print(p1_df_train[p1_numeric_cols].corr()[p1_TARGET].abs().sort_values(ascending=False))

# 타겟 분포 확인: 심하게 치우쳐 있으면 로그 변환을 검토
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); p1_df_train[p1_TARGET].hist(bins=50); plt.title('target')
plt.subplot(1, 2, 2); np.log1p(p1_df_train[p1_TARGET]).hist(bins=50); plt.title('log1p(target)')
plt.tight_layout(); plt.show()

# ============================================================
# 3. 전처리 — 분류 템플릿과 동일
# ============================================================
# %%
p1_drop_cols = []                                  # ★ 위 표를 보고 결정
if p1_drop_cols:
    p1_df_train = p1_df_train.drop(columns=p1_drop_cols, errors='ignore')
    p1_df_test = p1_df_test.drop(columns=p1_drop_cols, errors='ignore')

p1_X = p1_df_train.drop(columns=[p1_TARGET])
p1_y = p1_df_train[p1_TARGET].copy()

p1_has_test_label = p1_TARGET in p1_df_test.columns
p1_X_test = p1_df_test.drop(columns=[p1_TARGET]) if p1_has_test_label else p1_df_test.copy()
p1_y_test = p1_df_test[p1_TARGET].copy() if p1_has_test_label else None

p1_num_cols = p1_X.select_dtypes(include=[np.number]).columns.tolist()
p1_cat_cols = p1_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"수치형 {len(p1_num_cols)} / 범주형 {len(p1_cat_cols)}")

# %%
# 결측치
p1_num_imputer = SimpleImputer(strategy='median')
p1_cat_imputer = SimpleImputer(strategy='most_frequent')
if p1_num_cols:
    p1_X[p1_num_cols] = p1_num_imputer.fit_transform(p1_X[p1_num_cols])
    p1_X_test[p1_num_cols] = p1_num_imputer.transform(p1_X_test[p1_num_cols])
if p1_cat_cols:
    p1_X[p1_cat_cols] = p1_cat_imputer.fit_transform(p1_X[p1_cat_cols])
    p1_X_test[p1_cat_cols] = p1_cat_imputer.transform(p1_X_test[p1_cat_cols])

# 인코딩
for p1_col in p1_cat_cols:
    p1_le = LabelEncoder()
    p1_X[p1_col] = p1_le.fit_transform(p1_X[p1_col].astype(str))
    p1_X_test[p1_col] = p1_X_test[p1_col].astype(str).map(
        lambda v, _le=p1_le: _le.transform([v])[0] if v in _le.classes_ else -1
    )

# 스케일링
p1_scaler = StandardScaler()
p1_X_scaled = pd.DataFrame(p1_scaler.fit_transform(p1_X), columns=p1_X.columns)
p1_X_test_scaled = pd.DataFrame(p1_scaler.transform(p1_X_test), columns=p1_X_test.columns)

# %%
# --- [선택] 타겟 로그 변환 — 오른쪽으로 심하게 치우친 타겟에 효과적
# 학습은 log1p(p1_y) 로, 예측 후 expm1() 로 되돌린다
p1_USE_LOG_TARGET = False          # ★ 필요하면 True
p1_y_model = np.log1p(p1_y) if p1_USE_LOG_TARGET else p1_y

# ============================================================
# 4. Train / Validation 분할 — stratify 없음
# ============================================================
# %%
p1_X_train, p1_X_val, p1_y_train, p1_y_val = train_test_split(
    p1_X_scaled, p1_y_model, test_size=0.2, random_state=RANDOM_SEED
)
print(f"Train {p1_X_train.shape} / Val {p1_X_val.shape} / Test(제출용) {p1_X_test_scaled.shape}")

# ============================================================
# 5. 모델 비교
# ============================================================
# %%
p1_models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_SEED),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_SEED),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                            random_state=RANDOM_SEED),
    'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                              random_state=RANDOM_SEED, verbose=-1),
}

p1_results = {}
for p1_name, p1_model in p1_models.items():
    p1_model.fit(p1_X_train, p1_y_train)
    p1_y_pred = p1_model.predict(p1_X_val)

    p1_rmse = np.sqrt(mean_squared_error(p1_y_val, p1_y_pred))
    p1_mae = mean_absolute_error(p1_y_val, p1_y_pred)
    p1_r2 = r2_score(p1_y_val, p1_y_pred)
    p1_cv = cross_val_score(p1_model, p1_X_train, p1_y_train, cv=5, scoring='r2').mean()

    p1_results[p1_name] = {'RMSE': p1_rmse, 'MAE': p1_mae, 'R2': p1_r2, 'CV_R2': p1_cv}
    print(f"[{p1_name}] RMSE={p1_rmse:.4f} MAE={p1_mae:.4f} R2={p1_r2:.4f} CV={p1_cv:.4f}")

p1_results_df = pd.DataFrame(p1_results).T.sort_values('R2', ascending=False)
print("\n[모델 비교]")
print(p1_results_df)

p1_best_model_name = p1_results_df.index[0]
print(f"\n1위: {p1_best_model_name}")

# ============================================================
# 6. 하이퍼파라미터 튜닝 — 1위 모델 하나만
# ============================================================
# %%
p1_param_grids = {
    'RandomForest': (
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        {'n_estimators': [100, 200, 300], 'max_depth': [6, 10, 15, None],
         'min_samples_split': [2, 5]},
    ),
    'XGBoost': (
        XGBRegressor(random_state=RANDOM_SEED),
        {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8],
         'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0]},
    ),
    'LightGBM': (
        LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
        {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8],
         'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0]},
    ),
}

p1_tune_name = p1_best_model_name if p1_best_model_name in p1_param_grids else 'XGBoost'
p1_est, p1_params = p1_param_grids[p1_tune_name]
print(f"튜닝 대상: {p1_tune_name}")

# scoring: 'p1_r2' / 'neg_root_mean_squared_error' 중 문제 요구에 맞게 ★
p1_grid = GridSearchCV(p1_est, p1_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
p1_grid.fit(p1_X_train, p1_y_train)
print(f"최적 파라미터: {p1_grid.best_params_}")
print(f"최적 CV R2: {p1_grid.best_score_:.4f}")

# ============================================================
# 7. 최종 평가 (Validation)
# ============================================================
# %%
p1_best_model = p1_grid.best_estimator_
p1_y_val_pred = p1_best_model.predict(p1_X_val)

# 로그 변환을 썼다면 원래 스케일로 되돌려 평가
if p1_USE_LOG_TARGET:
    p1_y_val_eval, p1_y_pred_eval = np.expm1(p1_y_val), np.expm1(p1_y_val_pred)
else:
    p1_y_val_eval, p1_y_pred_eval = p1_y_val, p1_y_val_pred

p1_rmse_final = np.sqrt(mean_squared_error(p1_y_val_eval, p1_y_pred_eval))
p1_mae_final = mean_absolute_error(p1_y_val_eval, p1_y_pred_eval)
p1_r2_final = r2_score(p1_y_val_eval, p1_y_pred_eval)
print(f"RMSE {p1_rmse_final:.4f} / MAE {p1_mae_final:.4f} / R2 {p1_r2_final:.4f}")

# R2 등급: 0.9↑ 매우우수 / 0.8↑ 우수 / 0.7↑ 양호 / 0.6↑ 보통 / 미만 미흡
p1_grade = next((g for t, g in [(0.9, '매우 우수'), (0.8, '우수'),
                             (0.7, '양호'), (0.6, '보통')] if p1_r2_final >= t), '미흡')
print(f"-> {p1_grade}")
if p1_rmse_final > p1_mae_final * 1.5:
    print("RMSE >> MAE : 큰 오차(이상치)가 존재 -> 이상치 처리 또는 로그 변환 검토")

# 예측 vs 실제 산점도 (대각선에 가까울수록 좋음)
plt.figure(figsize=(6, 6))
plt.scatter(p1_y_val_eval, p1_y_pred_eval, alpha=0.3, s=10)
p1_lo, p1_hi = min(p1_y_val_eval.min(), p1_y_pred_eval.min()), max(p1_y_val_eval.max(), p1_y_pred_eval.max())
plt.plot([p1_lo, p1_hi], [p1_lo, p1_hi], 'r--')
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(f'Actual vs Predicted (R2={p1_r2_final:.3f})')
plt.tight_layout(); plt.show()

# ============================================================
# 8. 특성 중요도
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
if p1_USE_LOG_TARGET:
    p1_test_predictions = np.expm1(p1_test_predictions)

p1_submission = p1_df_test.copy()
p1_submission['predicted'] = p1_test_predictions        # ★ 요구 컬럼명으로

if p1_has_test_label:
    print(f"[Test] RMSE={np.sqrt(mean_squared_error(p1_df_test[p1_TARGET], p1_test_predictions)):.4f} "
          f"R2={r2_score(p1_df_test[p1_TARGET], p1_test_predictions):.4f}")

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
# 회귀는 label 복원이 없으므로 행 수 / 결측 / index / 모델 파일만 점검합니다.
try:
    check_submission(p1_SUBMIT_CSV, n_rows=len(p1_df_test),
                     model_path=p1_SUBMIT_MODEL)
except NameError:
    import os
    p1__sub = pd.read_csv(p1_SUBMIT_CSV)
    assert len(p1__sub) == len(p1_df_test), "행 수 불일치!"
    assert not p1__sub['predicted'].isnull().any(), "예측 컬럼에 결측!"
    assert os.path.getsize(p1_SUBMIT_MODEL) > 0, "모델 파일 없음!"
    print("최소 점검 통과 (전체 검증은 세션 시작 셀의 check_submission)")
