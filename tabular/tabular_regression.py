# %% [markdown]
# # Tabular - 회귀 (Regression) 템플릿
#
# 평가지표: RMSE, MAE, R² Score
# 대표 데이터셋: 주택 가격, BMI, 매출 예측 등
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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

# --- (B) California Housing (회귀) ---
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df_all = pd.DataFrame(data.data, columns=data.feature_names)
df_all['target'] = data.target
TARGET = 'target'
df_train, df_test = train_test_split(
    df_all, test_size=0.2, random_state=RANDOM_SEED
)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# --- (C) Boston Housing 대체 (회귀) ---
# from sklearn.datasets import fetch_openml
# data = fetch_openml(name='boston', version=1, as_frame=True)
# df_all = data.frame
# TARGET = 'MEDV'
# df_train, df_test = train_test_split(
#     df_all, test_size=0.2, random_state=RANDOM_SEED
# )
# df_train = df_train.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

# --- (D) Diabetes (회귀) ---
# from sklearn.datasets import load_diabetes
# data = load_diabetes()
# df_all = pd.DataFrame(data.data, columns=data.feature_names)
# df_all['target'] = data.target
# TARGET = 'target'
# df_train, df_test = train_test_split(
#     df_all, test_size=0.2, random_state=RANDOM_SEED
# )
# df_train = df_train.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

print(f"Train 크기: {df_train.shape}")
print(f"Test  크기: {df_test.shape}")
print(f"타겟 통계 (train):\n{df_train[TARGET].describe()}")
df_train.head()

# ============================================================
# 2. EDA (탐색적 데이터 분석) - Train 데이터 기준
# ============================================================
# %%
print("=" * 50)
print("[결측치]")
missing = df_train.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "결측치 없음")

print("\n[기술 통계]")
print(df_train.describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_train[TARGET].hist(bins=50, ax=axes[0])
axes[0].set_title('Target Distribution (Train)')
df_train.boxplot(column=TARGET, ax=axes[1])
axes[1].set_title('Target Boxplot (Train)')
plt.tight_layout()
plt.show()

# %%
print("[타겟과의 상관관계 (Train)]")
numeric_cols_eda = df_train.select_dtypes(include=[np.number]).columns
corr = df_train[numeric_cols_eda].corr()[TARGET].abs().sort_values(ascending=False)
print(corr)

# %%
# -------------------------------------------------------
# 불필요 컬럼 판단을 위한 컬럼 요약 정보
# -------------------------------------------------------
# [제거 기준]
# 결측률 70% 이상          → 정보량 부족, 제거 권장
# 고유값 비율 95% 이상      → ID/이름 등 식별자 컬럼, 제거
# 타겟 상관관계 0.95 이상   → 데이터 누수(leakage) 의심, 제거
# 고유값 1개 (분산 0)       → 정보 없음, 제거
# 다른 컬럼과 의미 중복     → 하나만 남기고 제거
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
drop_cols = []
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

# -------------------------------------------------------
# 3-3. 수치형 / 범주형 자동 분리
# -------------------------------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"수치형 ({len(num_cols)}): {num_cols}")
print(f"범주형 ({len(cat_cols)}): {cat_cols}")

# %%
# -------------------------------------------------------
# 3-4. 결측치 처리 (train에서 fit, test는 transform)
# -------------------------------------------------------
# 수치형: median(중앙값) - 이상치에 강건. mean(평균)도 가능
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
# 3-5. 이상치 처리 (선택 - 회귀에서 특히 효과적)
# -------------------------------------------------------
# 회귀 모델은 이상치에 민감하므로 분류보다 이상치 처리 효과가 큼
# IQR 방식: Q1-1.5*IQR ~ Q3+1.5*IQR 범위 밖을 경계값으로 클리핑
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

# 타겟 이상치 확인 (극단적인 타겟값은 모델 학습을 방해)
# 주의: train 데이터의 타겟에만 적용
# Q1_y, Q3_y = y.quantile(0.25), y.quantile(0.75)
# IQR_y = Q3_y - Q1_y
# mask = (y >= Q1_y - 1.5 * IQR_y) & (y <= Q3_y + 1.5 * IQR_y)
# X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
# print(f"  타겟 이상치 제거 후: {X.shape}")

# %%
# -------------------------------------------------------
# 3-6. 피처 엔지니어링 (선택 - 도메인 지식 기반)
# -------------------------------------------------------
# 예시 (주택 데이터):
#   - 방당 침실 비율: AveBedrms / AveRooms
#   - 인구 밀도: Population / AveOccup
# 주의: train과 test 모두에 동일하게 적용
# X['bedroom_ratio'] = X['AveBedrms'] / (X['AveRooms'] + 1e-8)
# X_test['bedroom_ratio'] = X_test['AveBedrms'] / (X_test['AveRooms'] + 1e-8)

# 다항 특성 (선형 모델에서 비선형 관계 포착)
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out())
# X_test = pd.DataFrame(poly.transform(X_test), columns=poly.get_feature_names_out())

# %%
# -------------------------------------------------------
# 3-7. 인코딩 (범주형 → 숫자)
# -------------------------------------------------------
# 주의: train에서 fit, test는 transform (unseen 레이블 대비 필요)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = X_test[col].astype(str).map(
        lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
    )
    label_encoders[col] = le
    print(f"  {col}: {le.classes_} → {list(range(len(le.classes_)))}")

# [선택] OneHotEncoder - 고유값 10개 이하인 범주형에 적합
# ohe_cols = [col for col in cat_cols if X[col].nunique() <= 10]
# X = pd.get_dummies(X, columns=ohe_cols, drop_first=True)
# X_test = pd.get_dummies(X_test, columns=ohe_cols, drop_first=True)
# X_test = X_test.reindex(columns=X.columns, fill_value=0)

# %%
# -------------------------------------------------------
# 3-8. 타겟 변환 (선택 - 타겟이 편향되었을 때)
# -------------------------------------------------------
# 타겟이 오른쪽으로 심하게 치우친 경우(skewness > 1) 로그 변환하면
# 모델 성능이 크게 올라갈 수 있음
# 주의: 예측 후에 반드시 np.expm1()으로 역변환 해야 함
TARGET_LOG_TRANSFORMED = False
skewness = y.skew()
print(f"타겟 skewness: {skewness:.3f}")
# if abs(skewness) > 1:
#     y = np.log1p(y)
#     if y_test is not None:
#         y_test = np.log1p(y_test)
#     TARGET_LOG_TRANSFORMED = True
#     print(f"  → 로그 변환 적용 (예측 시 np.expm1()으로 역변환 필요)")

# %%
# -------------------------------------------------------
# 3-9. 스케일링 (train에서 fit, test는 transform)
# -------------------------------------------------------
# StandardScaler: 평균=0, 표준편차=1 (일반적)
# 트리 모델(RF, XGB, LGBM)은 스케일링 불필요하지만,
# 선형 모델(Ridge, Lasso)과 함께 비교할 때는 적용이 안전
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ============================================================
# 4. Train / Validation 분할
# ============================================================
# %%
# X(학습 전체) → X_train(학습) / X_val(검증)
# X_test_scaled는 최종 제출용으로 별도 보관
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2, random_state=RANDOM_SEED
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test(제출용): {X_test_scaled.shape}")

# ============================================================
# 5. 모델 학습 및 비교
# ============================================================
# %%
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_SEED),
    'XGBoost': XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        random_state=RANDOM_SEED
    ),
    'LightGBM': LGBMRegressor(
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

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    cv_rmse = -cross_val_score(
        model, X_train, y_train, cv=5,
        scoring='neg_root_mean_squared_error'
    ).mean()

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CV_RMSE': cv_rmse}
    print(f"[{name}] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f} | CV_RMSE={cv_rmse:.4f}")

results_df = pd.DataFrame(results).T.sort_values('RMSE')
print("\n[모델 비교 결과]")
print(results_df)

best_model_name = results_df.index[0]
print(f"\n1위 모델: {best_model_name}")

# ============================================================
# 6. 하이퍼파라미터 튜닝 (1위 모델 자동 선택)
# ============================================================
# %%
param_grids = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=RANDOM_SEED),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 10, 15, None],
            'min_samples_split': [2, 5],
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=RANDOM_SEED),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
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
    cv=5, scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print(f"최적 파라미터: {grid.best_params_}")
print(f"최적 CV RMSE: {-grid.best_score_:.4f}")

# ============================================================
# 7. 최종 평가 (Validation 기준)
# ============================================================
# %%
best_model = grid.best_estimator_
y_val_pred = best_model.predict(X_val)

# 타겟 로그 변환을 적용한 경우 역변환
# if TARGET_LOG_TRANSFORMED:
#     y_val_pred = np.expm1(y_val_pred)
#     y_val = np.expm1(y_val)

rmse_final = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_final = mean_absolute_error(y_val, y_val_pred)
r2_final = r2_score(y_val, y_val_pred)

print("[최종 모델 성능 - Validation]")
print(f"RMSE: {rmse_final:.4f}")
print(f"MAE:  {mae_final:.4f}")
print(f"R²:   {r2_final:.4f}")

# %%
# -------------------------------------------------------
# 성능 평가 기준
# -------------------------------------------------------
# [R² Score 기준]
#   0.9 이상 : 매우 우수 (Excellent)
#   0.8 ~ 0.9: 우수 (Good)
#   0.7 ~ 0.8: 양호 (Acceptable)
#   0.5 ~ 0.7: 보통 (Moderate) - 추가 전처리/피처 엔지니어링 필요
#   0.5 미만 : 미흡 (Poor) - 모델/데이터 재검토 필요
#
# [RMSE 기준 - 타겟 범위 대비 상대 오차 (RMSE / (max-min))]
#   5% 미만  : 매우 우수
#   5~10%    : 우수
#   10~15%   : 양호
#   15~20%   : 보통
#   20% 이상 : 미흡
#
# [MAE vs RMSE 비교]
#   MAE ≈ RMSE  → 오차가 고르게 분포 (이상치 적음)
#   MAE << RMSE → 일부 큰 오차 존재 (이상치 확인 필요)
# -------------------------------------------------------

target_range = y_val.max() - y_val.min()
rmse_ratio = rmse_final / target_range * 100

r2_grades = [(0.9, '매우 우수'), (0.8, '우수'), (0.7, '양호'), (0.5, '보통')]
r2_grade = '미흡'
for threshold, grade in r2_grades:
    if r2_final >= threshold:
        r2_grade = grade
        break

rmse_grades = [(5, '매우 우수'), (10, '우수'), (15, '양호'), (20, '보통')]
rmse_grade = '미흡'
for threshold, grade in rmse_grades:
    if rmse_ratio < threshold:
        rmse_grade = grade
        break

print(f"\n[성능 평가]")
print(f"R² = {r2_final:.4f} → {r2_grade}")
print(f"RMSE 상대오차 = {rmse_ratio:.1f}% (타겟범위 대비) → {rmse_grade}")
print(f"MAE/RMSE 비율 = {mae_final/rmse_final:.3f} (1에 가까울수록 오차 균일)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_val, y_val_pred, alpha=0.5, s=10)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Actual vs Predicted (R²={r2_final:.4f})')

residuals = y_val - y_val_pred
axes[1].hist(residuals, bins=50, edgecolor='black')
axes[1].set_xlabel('Residual')
axes[1].set_title('Residual Distribution')
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
#results_df.to_csv('regression_model_comparison.csv')
#print("[저장] regression_model_comparison.csv")
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
val_result_df['residual'] = y_val.values - y_val_pred

#val_result_df.to_csv('regression_result.csv', index=False)
#print(f"\n[저장] regression_result.csv ({len(val_result_df)}건)")
print(val_result_df.head(10))

# --- 9-3. 최종 성능 요약 저장 ---
summary_df = pd.DataFrame([{
    'model': tune_name,
    'best_params': str(grid.best_params_),
    'RMSE': rmse_final,
    'MAE': mae_final,
    'R2': r2_final,
    'RMSE_ratio_pct': rmse_ratio,
    'R2_grade': r2_grade,
    'RMSE_grade': rmse_grade,
}])
#summary_df.to_csv('regression_summary.csv', index=False)
#print(f"\n[저장] regression_summary.csv")
print(summary_df)

# ============================================================
# 10. 제출 파일 생성 (Test 데이터 예측)
# ============================================================
# %%
test_predictions = best_model.predict(X_test_scaled)

# 로그 변환 적용했다면 역변환
# if TARGET_LOG_TRANSFORMED:
#     test_predictions = np.expm1(test_predictions)

submission = df_test.copy()
submission['predicted'] = test_predictions

if has_test_label:
    y_test_actual = df_test[TARGET]
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    test_r2 = r2_score(y_test_actual, test_predictions)
    submission['residual'] = y_test_actual.values - test_predictions
    print(f"[Test 성능] RMSE={test_rmse:.4f} | R²={test_r2:.4f}")

submission.to_csv('submission.csv', index=False)
print(f"\n[저장] submission.csv ({len(submission)}건)")
print(submission.head(10))

# %%
joblib.dump(best_model, 'tabular_regression_model.pkl')
print("\n모델 저장 완료: tabular_regression_model.pkl")

# %%
