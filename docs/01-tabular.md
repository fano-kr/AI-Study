# 01. Tabular — 판단이 필요한 도메인

**소요 20분** · 이전: [00. 오리엔테이션](00-orientation.md) · 다음: [02. Text](02-text.md)
**실습:** [notebooks/01_tabular_quickstart.ipynb](../notebooks/01_tabular_quickstart.ipynb)
**템플릿:** [templates/tabular_classification.py](../templates/tabular_classification.py) · [templates/tabular_regression.py](../templates/tabular_regression.py)

---

정형 데이터는 코드가 어려운 게 아니라 **결정이 많은** 도메인입니다.
이 차시는 그 결정들을 "고민"이 아니라 "규칙"으로 바꾸는 데 집중합니다.

## 1. EDA는 감상이 아니라 전처리 계획서다

시험장에서 산점도를 예쁘게 그릴 시간은 없습니다. EDA의 목적은 단 하나,
**어느 컬럼을 어떻게 처리할지 정하는 것**입니다. 그래서 그림 대신 이 표 하나를 찍습니다.

```python
col_summary = pd.DataFrame({
    'dtype':       df_train.dtypes,
    'missing':     df_train.isnull().sum(),
    'miss_pct':   (df_train.isnull().mean() * 100).round(1),
    'nunique':     df_train.nunique(),
    'nunique_pct':(df_train.nunique() / len(df_train) * 100).round(1),
    'sample':      df_train.iloc[0],
})
print(col_summary.to_string())
```

이 표를 보고 컬럼 제거를 **기계적으로** 판단합니다.

| 신호 | 의미 | 조치 |
|---|---|---|
| `miss_pct` ≥ 70 | 정보량 부족 | 제거 |
| `nunique_pct` ≥ 95 | ID·이름 등 식별자 | 제거 |
| `nunique` == 1 | 분산 0, 정보 없음 | 제거 |
| 타겟 상관 ≥ 0.95 | 데이터 누수 의심 | 제거 |
| 다른 컬럼과 의미 중복 | 예: `class` vs `pclass` | 하나만 남김 |

마지막 두 줄이 특히 중요합니다.
Titanic의 `alive` 컬럼은 타겟 `survived`를 yes/no로 바꿔 놓은 것에 불과합니다.
이걸 그냥 두면 **정확도 100%가 나오고**, 그건 모델이 잘한 게 아니라 답을 보고 푼 것입니다.

> **시험 팁** — 정확도가 0.99를 넘으면 기뻐하기 전에 누수를 의심하세요.
> `feature_importances_`에서 한 컬럼이 압도적이면 거의 확실합니다.

## 2. 전처리 — 순서가 정해져 있다

순서를 매번 고민하지 않도록 외웁니다. **제거 → 분리 → 결측 → (이상치) → 인코딩 → 스케일링**

```python
# 1) 불필요 컬럼 제거 (train, test 동시에!)
drop_cols = ['alive', 'pclass', 'deck']
df_train = df_train.drop(columns=drop_cols, errors='ignore')
df_test  = df_test.drop(columns=drop_cols,  errors='ignore')

# 2) X / y 분리
X, y = df_train.drop(columns=[TARGET]), df_train[TARGET]

# 3) 수치형 / 범주형 자동 분리
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
```

3)처럼 컬럼을 자동 분리해 두면 이후 단계가 전부 `for col in cat_cols:` 형태로 정리됩니다.
데이터가 바뀌어도 코드를 안 고쳐도 되는 이유가 여기서 생깁니다.

### 결측치

```python
num_imputer = SimpleImputer(strategy='median')        # 중앙값: 이상치에 강건
cat_imputer = SimpleImputer(strategy='most_frequent') # 최빈값

X[num_cols]      = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])   # transform 만!
```

수치형에 `mean`이 아니라 `median`을 기본으로 두는 이유는 이상치가 섞여 있어도 흔들리지 않기 때문입니다.
분포가 정규에 가깝다는 확신이 있을 때만 `mean`으로 바꾸세요.

### 인코딩 — 모델에 따라 다르다

| | LabelEncoder | OneHotEncoder |
|---|---|---|
| 하는 일 | 범주 → 0,1,2… 정수 | 범주 → 컬럼 여러 개 (0/1) |
| 적합한 경우 | 순서가 있는 범주 (low/mid/high) | 순서가 없는 범주 (색상, 도시) |
| 트리 모델 | **충분함** (분기로 알아서 처리) | 불필요하게 차원만 늘어남 |
| 선형 모델 | 위험 (없는 순서를 만들어냄) | **적합** |

시험에서는 트리 모델(RF/XGB/LGBM)을 주력으로 쓰게 되므로 **LabelEncoder가 기본값**입니다. 빠르고 컬럼이 안 늘어납니다.

단, test에만 나타나는 미지의 값이 있으면 `LabelEncoder.transform`이 에러로 죽습니다. 이렇게 막습니다.

```python
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = X_test[col].astype(str).map(
        lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
    )
```

처음 보는 값은 `-1`로 보냅니다. 트리 모델은 이걸 그냥 하나의 분기로 처리하므로 안전합니다.

### 스케일링

```python
scaler = StandardScaler()
X_scaled      = pd.DataFrame(scaler.fit_transform(X),      columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),     columns=X_test.columns)
```

트리 모델에는 스케일링이 **필요 없습니다**(분기 기준이라 단위에 무관).
그런데도 템플릿이 항상 스케일링을 하는 이유는, LogisticRegression 같은 선형 모델을 같은 표에서 비교하기 위해서입니다.
트리 모델은 스케일링을 해도 성능이 변하지 않으므로 **모두에게 적용하는 쪽이 안전**합니다.

### 선택 단계 — 성능이 안 나올 때만

아래 세 가지는 기본 파이프라인에 넣지 말고, **점수가 목표에 못 미칠 때** 꺼내 쓰세요.

- **이상치 클리핑 (IQR)** — 선형 모델에는 효과적, 트리 모델에는 거의 무의미
- **피처 엔지니어링** — Titanic이면 `family_size = sibsp + parch + 1` 같은 조합. train/test에 똑같이 적용할 것
- **클래스 불균형 처리** — 소수 클래스 비율 20% 미만이면 고려

불균형은 SMOTE보다 **모델 파라미터가 먼저**입니다. 코드 한 줄이면 되고 누수 위험도 없습니다.

```python
RandomForestClassifier(class_weight='balanced')
XGBClassifier(scale_pos_weight=(음성 수 / 양성 수))
LGBMClassifier(is_unbalance=True)
```

SMOTE를 쓴다면 **분할 이후 학습 데이터에만** 적용하세요. 검증 데이터에 적용하면 점수가 거짓말을 합니다.

## 3. 분할 — test는 성역이다

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
```

`stratify=y`를 빼먹지 마세요. 클래스 비율이 유지되지 않으면 검증 점수가 흔들립니다.

용어를 확실히 해 둡니다. 시험에서 헷갈리면 바로 실수로 이어집니다.

| 이름 | 정체 | 쓰임 |
|---|---|---|
| `X_train` | train.csv의 80% | 학습 |
| `X_val` | train.csv의 20% | 성능 측정, 모델 선택 |
| `X_test` | **test.csv 전체** | 제출용 예측. 학습에도, 평가에도 쓰지 않음 |

## 4. 모델 비교 — 고민하지 말고 다 돌린다

어떤 모델이 좋을지 이론으로 예측하는 것보다, 5개를 같은 조건에 넣고 돌려 보는 게 빠르고 정확합니다.

```python
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'RandomForest':       RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    'GradientBoosting':   GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED),
    'XGBoost':            XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RANDOM_SEED),
    'LightGBM':           LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RANDOM_SEED, verbose=-1),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average='macro')
    cv  = cross_val_score(model, X_train, y_train, cv=5).mean()
    results[name] = {'Accuracy': acc, 'F1_macro': f1, 'CV_Acc': cv}
```

**`CV_Acc`를 함께 보는 이유**가 중요합니다.
`Accuracy`는 val 한 번의 결과라 운이 섞여 있지만, `CV_Acc`는 5번 나눠 학습한 평균입니다.
둘이 크게 차이 나면(예: Acc 0.88, CV 0.79) **val 분할이 유리하게 걸린 것**이지 모델이 좋은 게 아닙니다.
이럴 땐 CV가 높은 모델을 고르세요.

AUC는 이진/다중에서 호출법이 다릅니다. 이 한 줄로 통일합니다.

```python
if NUM_CLASSES == 2:
    auc = roc_auc_score(y_val, y_proba[:, 1])
else:
    auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
```

## 5. 튜닝 — 1위 모델 하나만

```python
grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

5개 모델을 전부 튜닝하면 시간이 남지 않습니다. **비교에서 1위 한 개만** 튜닝하세요.
그리드가 커질수록 시간이 곱으로 늘어나니, 파라미터 3개 × 값 3개 정도가 현실적인 상한입니다.
시간이 정말 없으면 `GridSearchCV` 대신 `RandomizedSearchCV(n_iter=20)`가 안전합니다.

## 6. 회귀는 무엇이 다른가

**전처리와 분할은 사실상 동일합니다.** 세 군데만 바뀝니다.

| | 분류 | 회귀 |
|---|---|---|
| 모델 | `...Classifier` | `...Regressor` |
| 분할 | `stratify=y` | `stratify` 없음 (연속값이라 불가) |
| 지표 | Accuracy, F1, AUC | RMSE, MAE, R² |

```python
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae  = mean_absolute_error(y_val, y_pred)
r2   = r2_score(y_val, y_pred)
```

지표 읽는 법:

- **R²** — 1에 가까울수록 좋음. 0.9↑ 매우 우수 / 0.8↑ 우수 / 0.6 미만 재검토. 음수면 평균으로 찍는 것만도 못하다는 뜻
- **RMSE** — 타겟과 같은 단위. 큰 오차에 더 민감
- **RMSE ≫ MAE** — 소수의 큰 오차(이상치)가 있다는 신호 → 이상치 처리나 로그 변환 검토

타겟이 심하게 치우쳐 있으면 `np.log1p(y)`로 학습하고, 예측 후 `np.expm1()`로 되돌리는 방법이 자주 먹힙니다.

## 7. 제출

```python
test_predictions = best_model.predict(X_test_scaled)

if target_encoder:                       # 타겟이 문자열이었다면 원래 라벨로 복원
    test_predictions = target_encoder.inverse_transform(test_predictions)

submission = pd.DataFrame({'id': df_test['id'], 'predicted': test_predictions})
submission.to_csv('submission.csv', index=False)
```

**제출 직전 3초 점검**

1. 행 수가 `len(df_test)`와 같은가
2. 컬럼명이 문제에서 요구한 이름과 같은가
3. `index=False`를 넣었는가 (안 넣으면 컬럼이 하나 더 생김)
4. 타겟이 문자열이었다면 라벨로 복원했는가

---

**정리** — Tabular에서 점수를 잃는 곳은 모델이 아니라 (1) 누수 컬럼을 못 걸러내기 (2) test에 `fit_transform` 쓰기 (3) 제출 형식 틀리기, 이 셋입니다.

다음: [02. Text — 두 갈래 길](02-text.md)
