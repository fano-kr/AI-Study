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
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 ---
# df_train = pd.read_csv('train.csv')
# df_test  = pd.read_csv('test.csv')
# TARGET   = 'target'          # ★ 실제 타겟 컬럼명

# --- 연습용: California Housing ---
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df_all = pd.DataFrame(data.data, columns=data.feature_names)
df_all['target'] = data.target
TARGET = 'target'
df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=RANDOM_SEED)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Train {df_train.shape} / Test {df_test.shape}")
print(df_train[TARGET].describe())
df_train.head()

# ============================================================
# 2. EDA
# ============================================================
# %%
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
print("[타겟과의 상관관계 - 0.95 이상이면 누수 의심]")
print(df_train[numeric_cols].corr()[TARGET].abs().sort_values(ascending=False))

# 타겟 분포 확인: 심하게 치우쳐 있으면 로그 변환을 검토
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); df_train[TARGET].hist(bins=50); plt.title('target')
plt.subplot(1, 2, 2); np.log1p(df_train[TARGET]).hist(bins=50); plt.title('log1p(target)')
plt.tight_layout(); plt.show()

# ============================================================
# 3. 전처리 — 분류 템플릿과 동일
# ============================================================
# %%
drop_cols = []                                  # ★ 위 표를 보고 결정
if drop_cols:
    df_train = df_train.drop(columns=drop_cols, errors='ignore')
    df_test = df_test.drop(columns=drop_cols, errors='ignore')

X = df_train.drop(columns=[TARGET])
y = df_train[TARGET].copy()

has_test_label = TARGET in df_test.columns
X_test = df_test.drop(columns=[TARGET]) if has_test_label else df_test.copy()
y_test = df_test[TARGET].copy() if has_test_label else None

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"수치형 {len(num_cols)} / 범주형 {len(cat_cols)}")

# %%
# 결측치
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
if num_cols:
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
if cat_cols:
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# 인코딩
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = X_test[col].astype(str).map(
        lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_ else -1
    )

# 스케일링
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# %%
# --- [선택] 타겟 로그 변환 — 오른쪽으로 심하게 치우친 타겟에 효과적
# 학습은 log1p(y) 로, 예측 후 expm1() 로 되돌린다
USE_LOG_TARGET = False          # ★ 필요하면 True
y_model = np.log1p(y) if USE_LOG_TARGET else y

# ============================================================
# 4. Train / Validation 분할 — stratify 없음
# ============================================================
# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_model, test_size=0.2, random_state=RANDOM_SEED
)
print(f"Train {X_train.shape} / Val {X_val.shape} / Test(제출용) {X_test_scaled.shape}")

# ============================================================
# 5. 모델 비교
# ============================================================
# %%
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_SEED),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_SEED),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                            random_state=RANDOM_SEED),
    'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                              random_state=RANDOM_SEED, verbose=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CV_R2': cv}
    print(f"[{name}] RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} CV={cv:.4f}")

results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print("\n[모델 비교]")
print(results_df)

best_model_name = results_df.index[0]
print(f"\n1위: {best_model_name}")

# ============================================================
# 6. 하이퍼파라미터 튜닝 — 1위 모델 하나만
# ============================================================
# %%
param_grids = {
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

tune_name = best_model_name if best_model_name in param_grids else 'XGBoost'
est, params = param_grids[tune_name]
print(f"튜닝 대상: {tune_name}")

# scoring: 'r2' / 'neg_root_mean_squared_error' 중 문제 요구에 맞게 ★
grid = GridSearchCV(est, params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print(f"최적 파라미터: {grid.best_params_}")
print(f"최적 CV R2: {grid.best_score_:.4f}")

# ============================================================
# 7. 최종 평가 (Validation)
# ============================================================
# %%
best_model = grid.best_estimator_
y_val_pred = best_model.predict(X_val)

# 로그 변환을 썼다면 원래 스케일로 되돌려 평가
if USE_LOG_TARGET:
    y_val_eval, y_pred_eval = np.expm1(y_val), np.expm1(y_val_pred)
else:
    y_val_eval, y_pred_eval = y_val, y_val_pred

rmse_final = np.sqrt(mean_squared_error(y_val_eval, y_pred_eval))
mae_final = mean_absolute_error(y_val_eval, y_pred_eval)
r2_final = r2_score(y_val_eval, y_pred_eval)
print(f"RMSE {rmse_final:.4f} / MAE {mae_final:.4f} / R2 {r2_final:.4f}")

# R2 등급: 0.9↑ 매우우수 / 0.8↑ 우수 / 0.7↑ 양호 / 0.6↑ 보통 / 미만 미흡
grade = next((g for t, g in [(0.9, '매우 우수'), (0.8, '우수'),
                             (0.7, '양호'), (0.6, '보통')] if r2_final >= t), '미흡')
print(f"-> {grade}")
if rmse_final > mae_final * 1.5:
    print("RMSE >> MAE : 큰 오차(이상치)가 존재 -> 이상치 처리 또는 로그 변환 검토")

# 예측 vs 실제 산점도 (대각선에 가까울수록 좋음)
plt.figure(figsize=(6, 6))
plt.scatter(y_val_eval, y_pred_eval, alpha=0.3, s=10)
lo, hi = min(y_val_eval.min(), y_pred_eval.min()), max(y_val_eval.max(), y_pred_eval.max())
plt.plot([lo, hi], [lo, hi], 'r--')
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(f'Actual vs Predicted (R2={r2_final:.3f})')
plt.tight_layout(); plt.show()

# ============================================================
# 8. 특성 중요도
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
if USE_LOG_TARGET:
    test_predictions = np.expm1(test_predictions)

submission = df_test.copy()
submission['predicted'] = test_predictions        # ★ 요구 컬럼명으로

if has_test_label:
    print(f"[Test] RMSE={np.sqrt(mean_squared_error(df_test[TARGET], test_predictions)):.4f} "
          f"R2={r2_score(df_test[TARGET], test_predictions):.4f}")

submission.to_csv('submission.csv', index=False)
print(f"[저장] submission.csv ({len(submission)}건)")
print(submission.head())

assert len(submission) == len(df_test), "행 수 불일치!"

# %%
joblib.dump(best_model, 'tabular_regression_model.pkl')
print("모델 저장 완료")
