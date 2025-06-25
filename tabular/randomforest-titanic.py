# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
df = sns.load_dataset('titanic')

# 1. 필요한 특성 선택
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'deck']
X = df[features]

# %%
df

# %%
# target 변수 인코딩
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(df['alive'])  # 'yes'/'no'를 1/0으로 변환

print("타겟 변수 인코딩 결과:")
print(f"원본 클래스: {y_encoder.classes_}")
print(f"변환된 값: {y_encoder.transform(y_encoder.classes_)}")

# %%


# %%
# 2. 결측치 처리
# 수치형 데이터 결측치
numeric_features = ['age', 'fare']
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])


# %%

# 범주형 데이터 결측치
categorical_features = ['embarked', 'deck', 'who', 'class', 'sex']
categorical_imputer = SimpleImputer(strategy='most_frequent')
X.loc[:, categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# 범주형 데이터 인코딩
label_encoders = {}
for feature in categorical_features:
    # 문자열로 변환 후 인코딩
    X[feature] = X[feature].astype(str)
    label_encoders[feature] = LabelEncoder()
    X.loc[:, feature] = label_encoders[feature].fit_transform(X[feature])

# 4. 특성 스케일링
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 교차 검증 수행
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"교차 검증 점수: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")


# %%
y_pred = rf_model.predict(X_test)
print("\n최종 테스트 정확도:", accuracy_score(y_test, y_pred))
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# %%
# 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"\n최적 파라미터: {grid_search.best_params_}")
print(f"최고 정확도: {grid_search.best_score_:.3f}")


# %%

# 최적화된 모델로 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\n최종 테스트 정확도:", accuracy_score(y_test, y_pred))
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# 특성 선택
selector = SelectFromModel(best_model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 선택된 특성으로 최종 모델 학습
final_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X_train_selected, y_train)



# %%
df[features].head()

# %%
X2 = df[features].copy()

# 범주형 데이터 결측치 처리
categorical_features = ['embarked', 'deck', 'who', 'class', 'sex']
categorical_imputer = SimpleImputer(strategy='most_frequent')
X2.loc[:, categorical_features] = categorical_imputer.fit_transform(X2[categorical_features])

# 범주형 데이터 인코딩
label_encoders = {}
for feature in categorical_features:
    # 문자열로 변환 후 인코딩
    X2[feature] = X2[feature].astype(str)
    label_encoders[feature] = LabelEncoder()
    encoded_values = label_encoders[feature].fit_transform(X2[feature])
    X2.loc[:, feature] = encoded_values

# 특성 스케일링
scaler = StandardScaler()
X2 = pd.DataFrame(
    scaler.fit_transform(X2),
    columns=X2.columns,
    index=X2.index
)


# %%
X2

# %%
y_pred2 = rf_model.predict(X2)

# %%
df['predicted_survived'] = y_pred2

# %%
df

# %%
df[df['survived'] != df['predicted_survived']]

# %%
y_encoder.inverse_transform(y_pred2)


