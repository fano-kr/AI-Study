# 04. 시험 당일 체크리스트

**소요 5분** · 이전: [03. Image](03-image.md)

---

이 문서 하나만 인쇄해서 옆에 두어도 되도록 만들었습니다.

## 1. 시작 5분 안에 하는 일

```python
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

print(df_train.shape, df_test.shape)
print(df_train.dtypes)
print(df_train.isnull().sum()[lambda s: s > 0])
df_train.head()
```

그리고 **문제를 다시 읽습니다.** 확인할 것:

- 타겟 컬럼 이름이 정확히 무엇인가
- 제출 파일 이름과 컬럼명이 무엇인가
- 요구 지표가 Accuracy인가 F1인가 RMSE인가 (튜닝의 `scoring`이 여기서 정해집니다)
- test에 라벨이 있는가 없는가

## 2. 시간 배분

3시간 기준의 권장 배분입니다. **핵심은 "끝까지 도는 코드"를 전반부에 확보하는 것**입니다.

| 구간 | 할 일 |
|---|---|
| 0–15분 | 문제 정독, 데이터 로드, EDA 표 출력, 제거할 컬럼 결정 |
| 15–40분 | 전처리 → 분할 → **기본 모델 1개로 끝까지 돌려 submission.csv 저장** |
| 40–90분 | 모델 비교, 1위 선정 |
| 90–140분 | 튜닝, 성능이 오르면 submission 갱신 |
| 140–170분 | 여유 문항 처리, 피처 엔지니어링 |
| 170–180분 | 최종 점검 및 제출 |

40분 시점에 이미 제출 가능한 파일이 있어야 합니다.
완벽한 모델을 만들다 시간이 끝나는 것이 가장 흔한 실패입니다.

## 3. 절대 하지 말 것

| 금지 | 왜 |
|---|---|
| `scaler.fit_transform(X_test)` | 데이터 누수. 검증 점수는 오히려 올라가서 발견이 어려움 |
| 분할 전에 SMOTE | 검증 데이터에 합성 샘플이 섞여 점수가 거짓이 됨 |
| test로 학습 | 실격 사유가 될 수 있음 |
| 시드 미고정 | 개선 여부를 판단할 수 없게 됨 |
| 타겟과 상관 0.99 컬럼 방치 | 누수. 정확도 100%는 축하가 아니라 경고 |
| 5개 모델 전부 GridSearch | 시간 초과 |
| BERT부터 시작 | 실패 시 제출물이 없음 |
| shuffle된 데이터셋으로 이미지 예측 | 파일명과 예측이 어긋남 |

## 4. 에러 응급처치

| 에러 | 원인 | 조치 |
|---|---|---|
| `could not convert string to float` | 범주형이 인코딩 안 됨 | `select_dtypes`로 남은 object 컬럼 확인 |
| `Input contains NaN` | 결측 처리 누락 | `X.isnull().sum().sum()` 확인 |
| `y contains previously unseen labels` | test에만 있는 범주 | 인코딩을 `map(...) else -1` 방식으로 |
| `Found input variables with inconsistent numbers of samples` | X와 y 행 수 불일치 | `reset_index(drop=True)` 후 재시도 |
| `Number of features of the model must match` | train/test 컬럼 불일치 | `X_test = X_test.reindex(columns=X.columns, fill_value=0)` |
| `ResourceExhaustedError` (OOM) | 배치가 큼 | BATCH_SIZE 32→16→8, MAX_LENGTH 128→64 |
| `loss: nan` | 학습률 과대 | BERT는 2e-5, 파인튜닝은 1e-5 |
| `ValueError: Shapes ... incompatible` | 출력층/손실 불일치 | `categorical` ↔ `categorical_crossentropy` 짝 확인 |

## 5. 도메인별 한 줄 요약

**Tabular**
```
컬럼 요약표 → 누수/식별자 제거 → median·최빈값 결측 → LabelEncoder → StandardScaler
→ stratify 분할 → 5모델 비교 → 1위만 GridSearch → submission
```

**Text**
```
clean_text → TfidfVectorizer → LinearSVC/LogReg/NB 비교 → submission 확보
             (한국어면 analyzer='char_wb', ngram_range=(2,4) — 공백 토큰화보다 확실히 낫다)
→ (시간 남으면) klue/bert-base, lr=2e-5, from_logits=True, from_pt=True
```

**Image**
```
image_dataset_from_directory(label_mode='categorical') → 증강 레이어 → MobileNetV2 동결
→ Phase1 lr=1e-3 → 상위 30층 해제 + 재컴파일 → Phase2 lr=1e-5
→ 순서 보장 데이터셋으로 예측 → submission
```

## 6. 제출 전 최종 점검

```python
sub = pd.read_csv('submission.csv')
print(sub.shape)          # 행 수 == len(df_test) 인가
print(sub.columns.tolist())  # 요구된 컬럼명과 같은가
print(sub.isnull().sum())    # 결측 없는가
sub.head()
```

- [ ] 행 수가 test와 일치
- [ ] 컬럼명이 문제 요구와 일치
- [ ] `index=False`로 저장
- [ ] 결측 없음
- [ ] 타겟이 문자열이었다면 원래 라벨로 복원
- [ ] 파일명이 요구된 이름과 정확히 일치

## 7. 마지막 한마디

이 과정에서 가르친 것은 결국 문장 하나입니다.

> **끝까지 도는 코드를 먼저 만들고, 남는 시간으로 점수를 올린다.**

세 도메인의 템플릿, 시드 고정, `fit`/`transform` 규칙, `NUM_CLASSES` 통일 — 전부
"막히지 않고 끝까지 가기 위한" 장치입니다.

시험장에서 새로운 걸 시도하지 마세요. [templates/](../templates/)의 네 파일을 손에 익혀 두는 것이,
알고리즘을 하나 더 아는 것보다 확실하게 점수가 됩니다.

---

돌아가기: [README](../README.md)
