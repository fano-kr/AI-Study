# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
df = sns.load_dataset('titanic')

print("="*60)
print("타이타닉 생존 예측 - 체계적 특성 선택 적용")
print("="*60)
print(f"데이터 크기: {df.shape}")
print(f"전체 컬럼: {df.columns.tolist()}")

# %%
# =============================================================================
# 1단계: 결측치 분석 및 기본 데이터 탐색
# =============================================================================
print("\n" + "="*60)
print("1단계: 결측치 분석 및 기본 데이터 탐색")
print("="*60)

def analyze_missing_values(df):
    """결측치 상세 분석 함수"""
    missing_info = []
    
    for col in df.columns:
        if col not in ['survived', 'alive']:  # 타겟 변수 제외
            missing_count = df[col].isnull().sum()
            missing_pct = round(missing_count / len(df) * 100, 2)
            unique_count = df[col].nunique()
            unique_ratio = round(unique_count / len(df), 3)
            
            missing_info.append({
                'Feature': col,
                'Missing_Count': missing_count,
                'Missing_Pct': missing_pct,
                'Unique_Count': unique_count,
                'Unique_Ratio': unique_ratio,
                'Data_Type': str(df[col].dtype)
            })
    
    return pd.DataFrame(missing_info).sort_values('Missing_Pct', ascending=False)

missing_analysis = analyze_missing_values(df)
print("결측치 분석 결과:")
print(missing_analysis.to_string(index=False))

# 결측치가 많은 특성 확인
high_missing_features = missing_analysis[missing_analysis['Missing_Pct'] > 30]['Feature'].tolist()
print(f"\n⚠️ 높은 결측치 특성 (>30%): {high_missing_features}")

# %%
# =============================================================================
# 2단계: 상관관계 분석
# =============================================================================
print("\n" + "="*60)
print("2단계: 상관관계 분석")
print("="*60)

def prepare_correlation_analysis(df, target_col='survived'):
    """상관관계 분석을 위한 데이터 전처리"""
    df_corr = df.copy()
    
    # 범주형 변수 인코딩
    categorical_cols = df_corr.select_dtypes(include=['object', 'category']).columns
    label_encoders_temp = {}
    
    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
            label_encoders_temp[col] = le
    
    # boolean 타입 변환
    bool_cols = df_corr.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_corr[col] = df_corr[col].astype(int)
    
    return df_corr, label_encoders_temp

# 상관관계 분석용 데이터 준비
df_encoded, temp_encoders = prepare_correlation_analysis(df)

# 타겟 변수와의 상관관계 계산
target_corr = df_encoded.corr()['survived'].abs().sort_values(ascending=False)
print("타겟 변수(survived)와의 상관관계:")
target_corr_display = target_corr[target_corr.index != 'survived']
for feature, corr in target_corr_display.items():
    print(f"  {feature:<12}: {corr:.3f}")

# 높은 상관관계 특성 쌍 탐지 (다중공선성 확인)
correlation_matrix = df_encoded.corr()
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = abs(correlation_matrix.iloc[i, j])
        if corr_val > 0.8:  # 0.8 이상의 높은 상관관계
            high_corr_pairs.append({
                'Feature1': correlation_matrix.columns[i],
                'Feature2': correlation_matrix.columns[j],
                'Correlation': round(corr_val, 3)
            })

print(f"\n높은 상관관계 특성 쌍 (|r| > 0.8):")
for pair in high_corr_pairs:
    print(f"  {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Correlation']}")

# %%
# =============================================================================
# 3단계: 상호정보량 분석
# =============================================================================
print("\n" + "="*60)
print("3단계: 상호정보량 분석")
print("="*60)

# 상호정보량 계산을 위한 데이터 준비
X_temp = df_encoded.drop('survived', axis=1)
y_temp = df_encoded['survived']

# 결측치 임시 처리
X_filled = X_temp.copy()
for col in X_filled.columns:
    if X_filled[col].dtype in ['float64', 'int64']:
        X_filled[col].fillna(X_filled[col].mean(), inplace=True)
    else:
        X_filled[col].fillna(X_filled[col].mode()[0] if len(X_filled[col].mode()) > 0 else 0, inplace=True)

# 상호정보량 계산
mi_scores = mutual_info_classif(X_filled, y_temp, random_state=42)
mi_dict = dict(zip(X_temp.columns, mi_scores))

print("상호정보량 점수 (높은 순):")
for feature, score in sorted(mi_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature:<12}: {score:.3f}")

# %%
# =============================================================================
# 4단계: 특성 선택 기준 적용
# =============================================================================
print("\n" + "="*60)
print("4단계: 특성 선택 기준 적용")
print("="*60)

def apply_feature_selection_criteria(missing_df, target_corr, mi_dict):
    """체계적 특성 선택 기준 적용"""
    
    selection_results = []
    
    for _, row in missing_df.iterrows():
        feature = row['Feature']
        issues = []
        
        # 기준 1: 과도한 결측치 (50% 이상)
        if row['Missing_Pct'] > 50:
            issues.append(f"과도한 결측치 ({row['Missing_Pct']:.1f}%)")
        
        # 기준 2: 매우 낮은 변동성 (unique_ratio < 0.001, 거의 모든 값이 동일한 경우만)
        if row['Unique_Ratio'] < 0.001:
            issues.append(f"매우 낮은 변동성 ({row['Unique_Ratio']:.3f})")
        
        # 기준 3: 타겟과 매우 약한 관계
        corr_val = target_corr.get(feature, 0)
        mi_val = mi_dict.get(feature, 0)
        if corr_val < 0.05 and mi_val < 0.01:
            issues.append(f"타겟과 매우 약한 관계 (corr: {corr_val:.3f}, MI: {mi_val:.3f})")
        
        # 기준 4: 도메인 지식 기반 중복성 확인
        redundant_features = {
            'alive': 'survived와 완전 동일 (타겟 변수)',
            'class': 'pclass의 범주형 버전 (중복)',
            'embark_town': 'embarked의 다른 표현 (중복)',
            'adult_male': 'sex + age로부터 파생 가능',
            'alone': 'sibsp + parch로부터 계산 가능 (sibsp==0 and parch==0)',
            'who': 'sex + age 정보의 조합'
        }
        
        if feature in redundant_features:
            issues.append(f"중복성: {redundant_features[feature]}")
        
        # 최종 결정
        if len(issues) == 0:
            decision = "✅ 선택"
        elif len(issues) == 1 and any(keyword in issues[0] for keyword in ['높은 결측치', '약한 관계']):
            decision = "⚠️ 고려"  # 처리 가능한 문제
        else:
            decision = "❌ 제외"
        
        selection_results.append({
            'Feature': feature,
            'Target_Corr': round(corr_val, 3),
            'Mutual_Info': round(mi_val, 3),
            'Missing_Pct': row['Missing_Pct'],
            'Issues': '; '.join(issues) if issues else 'None',
            'Decision': decision
        })
    
    return pd.DataFrame(selection_results).sort_values('Target_Corr', ascending=False)

# 특성 선택 기준 적용
selection_df = apply_feature_selection_criteria(missing_analysis, target_corr, mi_dict)
print("특성 선택 분석 결과:")
print(selection_df.to_string(index=False))

# %%
# =============================================================================
# 5단계: 최종 특성 선택 결정
# =============================================================================
print("\n" + "="*60)
print("5단계: 최종 특성 선택 결정")
print("="*60)

# 선택된 특성들
selected_features = selection_df[selection_df['Decision'] == '✅ 선택']['Feature'].tolist()
consider_features = selection_df[selection_df['Decision'] == '⚠️ 고려']['Feature'].tolist()
excluded_features = selection_df[selection_df['Decision'] == '❌ 제외']['Feature'].tolist()

# 고려 특성 중에서 도메인 지식으로 중요한 특성 추가
important_consider = []
for feature in consider_features:
    # age는 결측치가 있지만 생존 예측에 중요한 특성
    if feature == 'age':
        important_consider.append(feature)
        print(f"도메인 지식으로 추가: {feature} (생존 예측에 중요)")

# 최종 선택된 특성 리스트
final_features = selected_features + important_consider

print(f"\n✅ 최종 선택된 특성 ({len(final_features)}개):")
for i, feature in enumerate(final_features, 1):
    corr_val = selection_df[selection_df['Feature'] == feature]['Target_Corr'].iloc[0]
    mi_val = selection_df[selection_df['Feature'] == feature]['Mutual_Info'].iloc[0]
    missing_pct = selection_df[selection_df['Feature'] == feature]['Missing_Pct'].iloc[0]
    print(f"  {i:2d}. {feature:<12} (상관관계: {corr_val:.3f}, MI: {mi_val:.3f}, 결측치: {missing_pct:4.1f}%)")

print(f"\n❌ 제외된 특성 ({len(excluded_features)}개):")
for feature in excluded_features:
    issues = selection_df[selection_df['Feature'] == feature]['Issues'].iloc[0]
    print(f"  - {feature:<12}: {issues}")

# 기존 코드와의 비교
original_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'deck']
print(f"\n📊 기존 vs 개선된 특성 선택 비교:")
print(f"  기존 특성 ({len(original_features)}개): {original_features}")
print(f"  개선 특성 ({len(final_features)}개): {final_features}")
print(f"  제거된 특성: {set(original_features) - set(final_features)}")
print(f"  추가된 특성: {set(final_features) - set(original_features)}")

# %%
# =============================================================================
# 6단계: 선택된 특성으로 모델 구축
# =============================================================================
print("\n" + "="*60)
print("6단계: 선택된 특성으로 모델 구축")
print("="*60)

# 최종 선택된 특성으로 데이터 준비
X = df[final_features].copy()

print(f"선택된 특성으로 모델 구축: {final_features}")
print(f"특성 데이터 크기: {X.shape}")

# target 변수 인코딩
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(df['alive'])  # 'yes'/'no'를 1/0으로 변환

print("타겟 변수 인코딩 결과:")
print(f"원본 클래스: {y_encoder.classes_}")
print(f"변환된 값: {y_encoder.transform(y_encoder.classes_)}")

# %%
# =============================================================================
# 7단계: 데이터 전처리 (결측치 처리 및 인코딩)
# =============================================================================
print("\n" + "="*60)
print("7단계: 데이터 전처리")
print("="*60)

# 결측치 처리를 위한 특성 분류
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"수치형 특성 ({len(numeric_features)}개): {numeric_features}")
print(f"범주형 특성 ({len(categorical_features)}개): {categorical_features}")

# 수치형 데이터 결측치 처리 (평균값으로 대체)
if numeric_features:
    numeric_imputer = SimpleImputer(strategy='mean')
    X.loc[:, numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    print(f"수치형 특성 결측치 처리 완료 (평균값 대체)")

# 범주형 데이터 결측치 처리 (최빈값으로 대체)
if categorical_features:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X.loc[:, categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    print(f"범주형 특성 결측치 처리 완료 (최빈값 대체)")

# 범주형 데이터 라벨 인코딩
label_encoders = {}
if categorical_features:
    print("\n범주형 특성 라벨 인코딩:")
    for feature in categorical_features:
        X[feature] = X[feature].astype(str)
        label_encoders[feature] = LabelEncoder()
        X.loc[:, feature] = label_encoders[feature].fit_transform(X[feature])
        unique_labels = label_encoders[feature].classes_
        print(f"  {feature}: {unique_labels} → {list(range(len(unique_labels)))}")

# 특성 스케일링 (StandardScaler 적용)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"\n특성 스케일링 완료 (StandardScaler 적용)")
print(f"전처리 완료된 데이터 크기: {X_scaled.shape}")

# %%
# =============================================================================
# 8단계: 모델 학습 및 평가
# =============================================================================
print("\n" + "="*60)
print("8단계: 모델 학습 및 평가")
print("="*60)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
print(f"타겟 분포 - 훈련: {np.bincount(y_train)}, 테스트: {np.bincount(y_test)}")

# 기본 RandomForest 모델 학습
print("\n🌟 기본 RandomForest 모델 학습:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 교차 검증 수행
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"교차 검증 점수: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")

# 테스트 성능 평가
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 정확도: {test_accuracy:.3f}")

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))

# %%
# =============================================================================
# 9단계: 하이퍼파라미터 튜닝
# =============================================================================
print("\n" + "="*60)
print("9단계: 하이퍼파라미터 튜닝")
print("="*60)

# 그리드 서치를 위한 파라미터 그리드 정의
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("그리드 서치 실행 중...")
print(f"탐색할 파라미터 조합 수: {np.prod([len(v) for v in param_grid.values()])}")

# 그리드 서치 실행
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n🎯 최적 파라미터: {grid_search.best_params_}")
print(f"최고 교차 검증 점수: {grid_search.best_score_:.3f}")

# %%
# =============================================================================
# 10단계: 최적화된 모델 성능 평가
# =============================================================================
print("\n" + "="*60)
print("10단계: 최적화된 모델 성능 평가")
print("="*60)

# 최적화된 모델로 예측
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)

print(f"🏆 최적화된 모델 테스트 정확도: {best_accuracy:.3f}")
print(f"기본 모델 대비 성능 향상: {best_accuracy - test_accuracy:+.3f}")

print("\n최적화된 모델 분류 보고서:")
print(classification_report(y_test, y_pred_best, target_names=y_encoder.classes_))

# 특성 중요도 확인
feature_importance = pd.DataFrame({
    'Feature': final_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 특성 중요도 (상위 10개):")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['Feature']:<12}: {row['Importance']:.3f}")

# %%
# =============================================================================
# 11단계: 특성 선택 기반 모델 최적화
# =============================================================================
print("\n" + "="*60)
print("11단계: 특성 선택 기반 모델 최적화")
print("="*60)

# SelectFromModel을 사용한 추가 특성 선택
selector = SelectFromModel(best_model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 선택된 특성 확인
selected_mask = selector.get_support()
selected_feature_names = [final_features[i] for i in range(len(final_features)) if selected_mask[i]]

print(f"모델 기반 특성 선택 결과:")
print(f"원본 특성 수: {len(final_features)} → 선택된 특성 수: {len(selected_feature_names)}")
print(f"선택된 특성: {selected_feature_names}")

# 선택된 특성으로 최종 모델 학습
final_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X_train_selected, y_train)

# 최종 모델 성능 평가
y_pred_final = final_model.predict(X_test_selected)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\n🌟 최종 모델 테스트 정확도: {final_accuracy:.3f}")
print(f"특성 선택 전후 성능 비교: {final_accuracy:.3f} vs {best_accuracy:.3f} ({final_accuracy - best_accuracy:+.3f})")

# %%
# =============================================================================
# 12단계: 전체 데이터셋에 대한 예측
# =============================================================================
print("\n" + "="*60)
print("12단계: 전체 데이터셋에 대한 예측")
print("="*60)

# 원본 데이터에 대한 예측을 위한 전처리 (훈련 시 사용한 전처리기 재사용)
X_full = df[final_features].copy()

print("전체 데이터셋 전처리 중...")

# 결측치 처리 (훈련 시 사용한 imputer 재사용)
if numeric_features:
    X_full.loc[:, numeric_features] = numeric_imputer.transform(X_full[numeric_features])

if categorical_features:
    X_full.loc[:, categorical_features] = categorical_imputer.transform(X_full[categorical_features])

# 라벨 인코딩 (훈련 시 사용한 encoder 재사용)
for feature in categorical_features:
    X_full[feature] = X_full[feature].astype(str)
    X_full.loc[:, feature] = label_encoders[feature].transform(X_full[feature])

# 스케일링 (훈련 시 사용한 scaler 재사용)
X_full_scaled = pd.DataFrame(
    scaler.transform(X_full),
    columns=X_full.columns,
    index=X_full.index
)

# 특성 선택 적용
X_full_selected = selector.transform(X_full_scaled)

# 전체 데이터에 대한 예측
y_pred_full = final_model.predict(X_full_selected)

# 예측 결과를 원본 데이터프레임에 추가
df['predicted_survived'] = y_pred_full
df['predicted_survived_label'] = y_encoder.inverse_transform(y_pred_full)

print(f"전체 데이터 예측 완료: {len(y_pred_full)}건")

# 실제 vs 예측 비교
df_comparison = df.copy()
df_comparison['actual_survived'] = y_encoder.transform(df['alive'])

# 예측 정확도 계산
full_accuracy = accuracy_score(df_comparison['actual_survived'], df_comparison['predicted_survived'])
print(f"전체 데이터 예측 정확도: {full_accuracy:.3f}")

# 오분류된 케이스 분석
misclassified = df_comparison[df_comparison['actual_survived'] != df_comparison['predicted_survived']]
print(f"오분류된 케이스: {len(misclassified)}건 ({len(misclassified)/len(df)*100:.1f}%)")

# %%
# =============================================================================
# 13단계: 결과 요약 및 시각화
# =============================================================================
print("\n" + "="*60)
print("13단계: 결과 요약")
print("="*60)

print("🎯 타이타닉 생존 예측 모델 구축 완료!")
print("\n📊 최종 결과 요약:")
print(f"  📈 사용된 특성 수: {len(selected_feature_names)}개")
print(f"  📋 선택된 특성: {selected_feature_names}")
print(f"  🎯 테스트 정확도: {final_accuracy:.3f}")
print(f"  📊 전체 데이터 정확도: {full_accuracy:.3f}")

print(f"\n🔍 특성 선택 효과:")
original_feature_count = len(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'deck'])
print(f"  원본 특성 수: {original_feature_count}개 → 최종 특성 수: {len(selected_feature_names)}개")
print(f"  특성 감소율: {(original_feature_count - len(selected_feature_names))/original_feature_count*100:.1f}%")

print(f"\n💡 모델링 인사이트:")
print(f"  1. 결측치 분석을 통해 deck(77% 결측치) 특성 제외")
print(f"  2. 상관관계 분석으로 중복 특성(class, embark_town) 제거")
print(f"  3. 도메인 지식 활용으로 파생 특성(adult_male, alone) 제외")
print(f"  4. 체계적 특성 선택으로 모델 성능 최적화")

print(f"\n🏆 최종 모델 파라미터:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

df.head(10)[['sex', 'age', 'pclass', 'fare', 'alive', 'predicted_survived_label']]

# %%
# =============================================================================
# 14단계: 모델 및 전처리기 저장
# =============================================================================
print("\n" + "="*60)
print("14단계: 모델 및 전처리기 저장")
print("="*60)

import joblib
import pickle
import os

# 저장할 디렉토리 생성
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

print("💾 모델 및 전처리기 저장 중...")

# 1. 최종 모델 저장
model_path = os.path.join(model_dir, 'titanic_randomforest_final_model.pkl')
joblib.dump(final_model, model_path)
print(f"✅ 최종 모델 저장: {model_path}")

# 2. 전처리기들 저장
preprocessors = {
    'numeric_imputer': numeric_imputer,
    'categorical_imputer': categorical_imputer,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'feature_selector': selector,
    'target_encoder': y_encoder
}

preprocessor_path = os.path.join(model_dir, 'titanic_preprocessors.pkl')
joblib.dump(preprocessors, preprocessor_path)
print(f"✅ 전처리기 저장: {preprocessor_path}")

# 3. 모델 메타데이터 저장
metadata = {
    'final_features': final_features,
    'selected_features': selected_feature_names,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'best_params': grid_search.best_params_,
    'test_accuracy': final_accuracy,
    'full_data_accuracy': full_accuracy,
    'model_type': 'RandomForestClassifier',
    'feature_count': len(selected_feature_names),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

metadata_path = os.path.join(model_dir, 'titanic_model_metadata.pkl')
joblib.dump(metadata, metadata_path)
print(f"✅ 모델 메타데이터 저장: {metadata_path}")

# 4. 예측 결과 CSV 저장
results_df = df[['sex', 'age', 'pclass', 'fare', 'alive', 'predicted_survived_label']].copy()
results_df.columns = ['성별', '나이', '등급', '요금', '실제_생존', '예측_생존']
results_path = os.path.join(model_dir, 'titanic_predictions.csv')
results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
print(f"✅ 예측 결과 저장: {results_path}")

print(f"\n📁 저장된 파일 목록:")
for file in os.listdir(model_dir):
    file_path = os.path.join(model_dir, file)
    file_size = os.path.getsize(file_path) / 1024  # KB
    print(f"  - {file:<35} ({file_size:.1f} KB)")

# %%
# =============================================================================
# 15단계: 모델 로드 및 예측 함수 생성
# =============================================================================
print("\n" + "="*60)
print("15단계: 모델 로드 및 예측 함수")
print("="*60)

def load_titanic_model(model_dir='saved_models'):
    """저장된 타이타닉 생존 예측 모델과 전처리기를 로드하는 함수"""
    
    # 모델 로드
    model_path = os.path.join(model_dir, 'titanic_randomforest_final_model.pkl')
    model = joblib.load(model_path)
    
    # 전처리기 로드
    preprocessor_path = os.path.join(model_dir, 'titanic_preprocessors.pkl')
    preprocessors = joblib.load(preprocessor_path)
    
    # 메타데이터 로드
    metadata_path = os.path.join(model_dir, 'titanic_model_metadata.pkl')
    metadata = joblib.load(metadata_path)
    
    return model, preprocessors, metadata

def predict_survival(passenger_data, model_dir='saved_models'):
    """
    새로운 승객 데이터에 대해 생존 예측을 수행하는 함수
    
    Args:
        passenger_data (dict): 승객 정보 딕셔너리
        model_dir (str): 모델이 저장된 디렉토리 경로
    
    Returns:
        dict: 예측 결과 딕셔너리
    """
    
    # 모델 및 전처리기 로드
    model, preprocessors, metadata = load_titanic_model(model_dir)
    
    # 입력 데이터를 DataFrame으로 변환
    input_df = pd.DataFrame([passenger_data])
    
    # 필요한 특성만 선택
    final_features = metadata['final_features']
    input_df = input_df[final_features]
    
    # 전처리 적용
    numeric_features = metadata['numeric_features']
    categorical_features = metadata['categorical_features']
    
    # 결측치 처리
    if numeric_features:
        input_df.loc[:, numeric_features] = preprocessors['numeric_imputer'].transform(input_df[numeric_features])
    
    if categorical_features:
        input_df.loc[:, categorical_features] = preprocessors['categorical_imputer'].transform(input_df[categorical_features])
    
    # 라벨 인코딩
    for feature in categorical_features:
        input_df[feature] = input_df[feature].astype(str)
        input_df.loc[:, feature] = preprocessors['label_encoders'][feature].transform(input_df[feature])
    
    # 스케일링
    input_scaled = preprocessors['scaler'].transform(input_df)
    
    # 특성 선택
    input_selected = preprocessors['feature_selector'].transform(input_scaled)
    
    # 예측 수행
    prediction = model.predict(input_selected)[0]
    prediction_proba = model.predict_proba(input_selected)[0]
    
    # 결과 변환
    survival_label = preprocessors['target_encoder'].inverse_transform([prediction])[0]
    
    return {
        'survival_prediction': survival_label,
        'survival_probability': {
            'no': prediction_proba[0],
            'yes': prediction_proba[1]
        },
        'confidence': max(prediction_proba),
        'selected_features': metadata['selected_features'],
        'model_accuracy': metadata['test_accuracy']
    }

# 예측 함수 테스트
print("🔮 예측 함수 테스트:")

# 테스트 승객 데이터 (타이타닉 영화의 잭과 로즈를 모델로)
test_passengers = [
    {
        'pclass': 3,      # 3등석
        'sex': 'male',    # 남성
        'age': 20,        # 20세
        'sibsp': 0,       # 형제자매/배우자 없음
        'parch': 0,       # 부모/자녀 없음
        'fare': 7.25,     # 저렴한 요금
        'embarked': 'S'   # Southampton 승선
    },
    {
        'pclass': 1,      # 1등석
        'sex': 'female',  # 여성
        'age': 17,        # 17세
        'sibsp': 1,       # 약혼자 있음
        'parch': 2,       # 부모 2명
        'fare': 100.0,    # 비싼 요금
        'embarked': 'S'   # Southampton 승선
    }
]

test_names = ['잭 (Jack)', '로즈 (Rose)']

for i, (name, passenger) in enumerate(zip(test_names, test_passengers)):
    try:
        result = predict_survival(passenger, model_dir)
        print(f"\n{i+1}. {name}:")
        print(f"   예측 결과: {result['survival_prediction']}")
        print(f"   생존 확률: {result['survival_probability']['yes']:.3f}")
        print(f"   신뢰도: {result['confidence']:.3f}")
    except Exception as e:
        print(f"   ❌ 예측 실패: {e}")

print(f"\n💡 사용법:")
print(f"   1. 모델 로드: model, preprocessors, metadata = load_titanic_model()")
print(f"   2. 예측 수행: result = predict_survival(passenger_data)")
print(f"   3. 결과 확인: result['survival_prediction'], result['survival_probability']")

print(f"\n🎯 모델 저장 및 로드 완료!")
print(f"   저장 위치: {os.path.abspath(model_dir)}")
print(f"   모델 정확도: {metadata['test_accuracy']:.3f}")
print(f"   사용된 특성: {len(metadata['selected_features'])}개")


