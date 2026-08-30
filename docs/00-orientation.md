# 00. 오리엔테이션 — 하나의 파이프라인, 세 개의 도메인

**소요 6분** · 다음: [01. Tabular](01-tabular.md)

---

## 1. 이 시험의 성격

AICE Professional 실기는 주피터 노트북 환경에서 데이터를 받아 **모델을 만들고 예측 결과를 제출**하는 형식입니다.
정형(Tabular) · 텍스트(Text) · 이미지(Image) 중 하나 또는 둘이 출제되며, 요구사항은 대체로 이 형태입니다.

> "주어진 `train.csv`로 모델을 학습시키고, `test.csv`에 대한 예측 결과를 `submission.csv`로 저장하시오."

여기서 나오는 함정은 대부분 **모델 성능이 아니라 절차**에서 나옵니다.

- 전처리를 test에도 똑같이 적용했는가
- 스케일러를 test로 `fit` 하지 않았는가
- 제출 파일의 행 수와 컬럼명이 요구사항과 맞는가

즉 정확도 0.93을 0.95로 올리는 일보다, **끝까지 도는 코드를 시간 안에 완성하는 일**이 점수에 직결됩니다.
이 과정의 모든 선택은 그 기준으로 내려져 있습니다.

## 2. 시간을 어디에 쓸 것인가

이 60분 과정은 Tabular에 20분, Text 15분, Image 14분을 배정합니다. 균등 배분이 아닌 이유:

- **Tabular가 전처리 판단이 가장 많다.** 결측치·이상치·인코딩·스케일링·불균형 — 사람이 결정할 지점이 5개 이상입니다.
- **Text와 Image는 전처리가 거의 정해져 있다.** 텍스트는 정제 + 벡터화, 이미지는 리사이즈 + 증강으로 끝납니다. 대신 모델 쪽(BERT, 전이학습)에 정형화된 레시피가 있고, 그 레시피만 외우면 됩니다.

그래서 Tabular는 "판단하는 법"을, Text·Image는 "레시피"를 다룹니다.

## 3. 세 도메인을 관통하는 뼈대

```
 1. 로드        df_train / df_test 를 읽는다
 2. EDA        결측·타입·분포·타겟 상관을 확인한다  ← 여기서 전처리 계획이 결정된다
 3. 전처리      train 에서 fit, test 는 transform  ← 시험에서 가장 많이 틀리는 곳
 4. 분할        train → X_train / X_val  (test 는 절대 건드리지 않는다)
 5. 모델 비교    후보 3~5개를 같은 조건으로 돌려 1위를 고른다
 6. 튜닝        1위 모델만 GridSearchCV
 7. 평가        Accuracy / F1 (분류), RMSE / R² (회귀)
 8. 제출        X_test 예측 → submission.csv
```

세 도메인에서 **3번과 5번만 달라집니다.**

| 단계 | Tabular | Text | Image |
|---|---|---|---|
| 3. 전처리 | 결측/인코딩/스케일링 | 정제 → TF-IDF 또는 토크나이저 | 리사이즈 → 정규화 → 증강 |
| 5. 모델 | RF / XGB / LGBM | LinearSVC / LogisticRegression, BERT | MobileNetV2 전이학습 |

나머지 1·2·4·6·7·8번은 코드가 거의 그대로 재사용됩니다.
`templates/` 안의 네 파일이 서로 닮아 보인다면, 그건 실수가 아니라 의도입니다.

## 4. 세 가지 습관

이 과정에서 반복해서 강조할 세 가지입니다. 나머지를 다 잊어도 이건 남기세요.

### ① 시드를 고정한다

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)   # 딥러닝을 쓸 때
```

돌릴 때마다 점수가 달라지면 "방금 바꾼 게 효과가 있었는지" 판단할 수 없습니다.
튜닝의 전제 조건이 재현성입니다.

### ② train에서 fit, test는 transform

```python
scaler = StandardScaler()
X_scaled      = scaler.fit_transform(X)        # train: fit + transform
X_test_scaled = scaler.transform(X_test)       # test: transform 만
```

`X_test`에 `fit_transform`을 쓰면 test의 평균·분산이 모델에 새어 들어갑니다(데이터 누수).
검증 점수는 오히려 올라가기 때문에 **틀렸다는 사실조차 눈치채기 어렵습니다.** 그래서 위험합니다.

같은 원칙이 결측치 대치(`SimpleImputer`), 인코딩(`LabelEncoder`), TF-IDF(`TfidfVectorizer`)에 전부 적용됩니다.

### ③ 이진/다중을 코드로 분기하지 않는다

```python
NUM_CLASSES = df_train[TARGET].nunique()
```

클래스 수를 변수로 뽑아 두면, 이진분류든 7개 다중분류든 **같은 코드가 돕니다.**
딥러닝 쪽에서는 이 원칙이 특히 강력합니다. 출력층을 `sigmoid` + `binary_crossentropy`가 아니라
항상 `softmax` + `categorical_crossentropy`로 통일하면, 클래스 수만 바꿔 끼우면 끝입니다.

시험장에서 데이터를 열어 보고 "아 3개짜리네, 코드 고쳐야겠다" 하는 순간 5분이 날아갑니다.

## 5. 성능 판단 기준

모델을 돌린 뒤 "이 정도면 된 건가?"를 매번 고민하지 않도록, 기준선을 미리 정해 둡니다.

| Accuracy | 판정 | 행동 |
|---|---|---|
| 0.95 이상 | 매우 우수 | 제출하고 다음 문제로 |
| 0.90 ~ 0.95 | 우수 | 시간 남으면 튜닝 |
| 0.85 ~ 0.90 | 양호 | 피처 엔지니어링 검토 |
| 0.80 ~ 0.85 | 보통 | 전처리 재점검 |
| 0.80 미만 | 미흡 | 데이터/타겟 해석이 틀렸을 가능성 |

**Accuracy와 F1(macro)의 차이도 함께 보세요.**
둘이 비슷하면 클래스 균형이 맞는 것이고, `Accuracy >> F1`이면 소수 클래스를 못 맞히고 있다는 신호입니다
(→ `class_weight='balanced'` 또는 SMOTE 검토).

회귀에서는 R²를 같은 방식으로 씁니다: 0.9 이상 매우 우수, 0.8 이상 우수, 0.6 미만이면 재검토.

---

다음: [01. Tabular — 판단이 필요한 도메인](01-tabular.md)
