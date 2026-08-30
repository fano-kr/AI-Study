# 04. 시험 당일 체크리스트

**소요 7분** · 이전: [03. Image](03-image.md)

---

이 문서 하나만 인쇄해서 옆에 두어도 되도록 만들었습니다.

## 1. 시작 3분 — 코드보다 먼저

### ① GPU 연결 (가장 먼저)

시험 환경의 GPU는 **자동으로 붙지 않습니다.** 연결 코드를 직접 실행해야 합니다.
성능은 Colab 무료 T4와 비슷한 수준입니다.

```python
!nvidia-smi

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus}")
assert gpus, "GPU 미연결! 런타임 설정을 확인하세요"
```

`assert`가 중요합니다. GPU가 없어도 코드는 **조용히 CPU로 돌아갑니다.**
이미지·BERT 학습이 수십 배 느려지고, 원인을 모른 채 시간을 다 씁니다.
멈춰 세워야 알아차립니다.

Tabular 문제만 푼다면 GPU가 없어도 되지만, **확인은 무조건 먼저** 하세요.
문제를 다 읽고 나서 "이미지 문제가 있네" 하고 되돌아가면 이미 늦습니다.

### ② 제출 정보 확정

```python
PHONE = '01012345678'    # 본인 휴대폰 번호 (하이픈 없이)
PROB_NO = 1              # 문제 번호

SUBMIT_CSV = f'{PHONE}_{PROB_NO}.csv'
SUBMIT_MODEL = f'{PHONE}_{PROB_NO}.pkl'
print(SUBMIT_CSV, SUBMIT_MODEL)
```

파일명을 코드 곳곳에 직접 타이핑하지 마세요. **오타 하나로 채점이 안 됩니다.**
맨 위에서 변수로 한 번만 정하고, 저장할 때는 항상 변수를 씁니다.

### ③ 분석 자료 정독

문제와 함께 **컬럼 설명·데이터 정의·요구사항 문서가 주어집니다.** 여기서 확인할 것:

- 타겟 컬럼이 정확히 무엇인가
- 각 컬럼의 의미 (→ **누수 컬럼이 여기서 드러납니다**)
- 제출 파일 형식·컬럼명, 모델 저장 형식
- 요구 지표가 Accuracy인가 F1인가 RMSE인가 (튜닝의 `scoring`이 여기서 정해집니다)
- test에 라벨이 있는가 없는가

자료에 "결과를 사후 기록한 컬럼"이라고 적혀 있으면, 상관계수를 볼 것도 없이 제거 대상입니다.
데이터를 뒤져서 알아내는 것보다 **읽어서 아는 쪽이 훨씬 빠릅니다.**

### ④ 데이터 로드

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

**여기까지 하고 `Ctrl+S`.** 다음 항목을 보세요.

## 2. 저장 — 안 하면 날아간다

작성한 코드는 **자동 저장되지 않습니다.** 세션이 끊기면 미저장분은 전부 사라집니다.
몇 시간의 작업이 한순간에 없어지는 사고가 실제로 일어납니다.

**저장 타이밍을 규칙으로 정해 두세요.**

- 셀 하나를 완성할 때마다 `Ctrl+S`
- 단계가 끝날 때마다 (로드 완료 → 저장, 전처리 완료 → 저장, 모델 비교 완료 → 저장)
- **긴 학습을 시작하기 직전에 반드시** (여기서 세션이 죽을 확률이 가장 높습니다)
- 제출 파일을 만든 직후

이건 시험장에서 새로 생기는 습관이 아닙니다. **연습할 때부터 손이 저장하게 만드세요.**

같은 이유로, 오래 걸리는 학습 결과는 파일로 떨궈 둡니다.
세션이 죽어도 다시 학습하지 않아도 됩니다.

```python
joblib.dump(best_model, SUBMIT_MODEL)      # sklearn
model.save(f'{PHONE}_{PROB_NO}.keras')     # 딥러닝
```

> **연습은 Colab에서.** 시험 환경이 Colab과 비슷합니다.
> 로컬 주피터에만 익숙하면 런타임 설정, 파일 업로드, 단축키에서 시간을 씁니다.

## 3. 제출 파일명 규칙

모델과 예측 결과를 **본인 휴대폰 번호 + 문제 번호**로 저장합니다.

| 대상 | 이름 | 예시 |
|---|---|---|
| 예측 결과 | `{휴대폰번호}_{문제번호}.csv` | `01012345678_1.csv` |
| 모델 | `{휴대폰번호}_{문제번호}.pkl` | `01012345678_1.pkl` |

- 휴대폰 번호는 **하이픈 없이** 붙여 씁니다
- 문제마다 번호가 바뀌므로, 새 문제를 시작할 때 `PROB_NO`만 바꿉니다
- 딥러닝 모델은 `.pkl`로 저장되지 않습니다. **문제에서 지정한 형식을 따르되**, 지정이 없으면 `.keras`(또는 `.h5`)를 쓰고 이름 규칙은 그대로 지키세요

```python
# 문제가 바뀌면 이 줄만 수정
PROB_NO = 2

SUBMIT_CSV = f'{PHONE}_{PROB_NO}.csv'
SUBMIT_MODEL = f'{PHONE}_{PROB_NO}.pkl'

submission.to_csv(SUBMIT_CSV, index=False)
joblib.dump(best_model, SUBMIT_MODEL)

import os
print(os.listdir('.'))     # 파일이 실제로 만들어졌는지 눈으로 확인
```

마지막 줄을 습관으로 만드세요. `to_csv`가 조용히 다른 디렉토리에 쓰는 일이 있습니다.

## 4. 시간 배분

3시간 기준의 권장 배분입니다. **핵심은 "끝까지 도는 코드"를 전반부에 확보하는 것**입니다.

| 구간 | 할 일 |
|---|---|
| 0–5분 | **GPU 연결**, `PHONE`/`PROB_NO` 설정, 저장 |
| 5–20분 | **분석 자료 정독**, 데이터 로드, EDA 표 출력, 제거할 컬럼 결정 |
| 20–40분 | 전처리 → 분할 → **기본 모델 1개로 끝까지 돌려 제출 파일 저장** |
| 40–90분 | 모델 비교, 1위 선정 |
| 90–140분 | 튜닝, 성능이 오르면 제출 파일 갱신 |
| 140–170분 | 남은 문항 처리, 피처 엔지니어링 |
| 170–180분 | 최종 점검 및 제출 |

40분 시점에 이미 제출 가능한 파일이 있어야 합니다.
완벽한 모델을 만들다 시간이 끝나는 것이 가장 흔한 실패입니다.

## 5. 절대 하지 말 것

| 금지 | 왜 |
|---|---|
| GPU 연결 확인 생략 | CPU로 조용히 돌아가 시간을 다 씀 |
| 저장 안 하고 긴 학습 시작 | 세션이 끊기면 코드가 전부 사라짐 |
| 파일명을 매번 손으로 타이핑 | 오타 하나로 채점 불가 |
| 분석 자료 안 읽고 코드부터 | 누수 컬럼을 놓치고, 요구 지표를 잘못 맞춤 |
| `scaler.fit_transform(X_test)` | 데이터 누수. 검증 점수는 오히려 올라가서 발견이 어려움 |
| 분할 전에 SMOTE | 검증 데이터에 합성 샘플이 섞여 점수가 거짓이 됨 |
| test로 학습 | 실격 사유가 될 수 있음 |
| 시드 미고정 | 개선 여부를 판단할 수 없게 됨 |
| 타겟과 상관 0.99 컬럼 방치 | 누수. 정확도 100%는 축하가 아니라 경고 |
| 5개 모델 전부 GridSearch | 시간 초과 |
| BERT부터 시작 | 실패 시 제출물이 없음 |
| shuffle된 데이터셋으로 이미지 예측 | 파일명과 예측이 어긋남 |

## 6. 에러 응급처치

| 에러 | 원인 | 조치 |
|---|---|---|
| 학습이 이상하게 느림 | GPU 미연결 | `tf.config.list_physical_devices('GPU')` 확인 |
| `could not convert string to float` | 범주형이 인코딩 안 됨 | `select_dtypes`로 남은 object 컬럼 확인 |
| `Input contains NaN` | 결측 처리 누락 | `X.isnull().sum().sum()` 확인 |
| `y contains previously unseen labels` | test에만 있는 범주 | 인코딩을 `map(...) else -1` 방식으로 |
| `Found input variables with inconsistent numbers of samples` | X와 y 행 수 불일치 | `reset_index(drop=True)` 후 재시도 |
| `Number of features of the model must match` | train/test 컬럼 불일치 | `X_test = X_test.reindex(columns=X.columns, fill_value=0)` |
| `ResourceExhaustedError` (OOM) | 배치가 큼 | BATCH_SIZE 32→16→8, MAX_LENGTH 128→64 |
| `loss: nan` | 학습률 과대 | BERT는 2e-5, 파인튜닝은 1e-5 |
| `ValueError: Shapes ... incompatible` | 출력층/손실 불일치 | `categorical` ↔ `categorical_crossentropy` 짝 확인 |

## 7. 도메인별 한 줄 요약

**Tabular**
```
컬럼 요약표 → 누수/식별자 제거 → median·최빈값 결측 → LabelEncoder → StandardScaler
→ stratify 분할 → 5모델 비교 → 1위만 GridSearch → {PHONE}_{PROB_NO}.csv
```

**Text**
```
clean_text → TfidfVectorizer → LinearSVC/LogReg/NB 비교 → 제출 파일 확보
             (한국어면 analyzer='char_wb', ngram_range=(2,4) — 공백 토큰화보다 확실히 낫다)
→ (시간 남으면) klue/bert-base, lr=2e-5, from_logits=True, from_pt=True
```

**Image**
```
GPU 확인 → image_dataset_from_directory(label_mode='categorical') → 증강 레이어
→ MobileNetV2 동결 Phase1 lr=1e-3 → 상위 30층 해제 + 재컴파일 → Phase2 lr=1e-5
→ 순서 보장 데이터셋으로 예측 → 제출 파일
```

## 8. 제출 전 최종 점검

```python
import os
print(os.listdir('.'))            # 파일이 실제로 있는가

sub = pd.read_csv(SUBMIT_CSV)
print(sub.shape)                  # 행 수 == len(df_test) 인가
print(sub.columns.tolist())       # 요구된 컬럼명과 같은가
print(sub.isnull().sum())         # 결측 없는가
sub.head()
```

- [ ] 파일명이 `{휴대폰번호}_{문제번호}` 규칙과 정확히 일치 (하이픈 없음)
- [ ] 모델 파일과 CSV 파일이 **둘 다** 저장됨
- [ ] 행 수가 test와 일치
- [ ] 컬럼명이 문제 요구와 일치
- [ ] `index=False`로 저장
- [ ] 결측 없음
- [ ] 타겟이 문자열이었다면 원래 라벨로 복원
- [ ] **노트북 저장 완료** (`Ctrl+S`)

## 9. 마지막 한마디

이 과정에서 가르친 것은 결국 문장 하나입니다.

> **끝까지 도는 코드를 먼저 만들고, 남는 시간으로 점수를 올린다.**

세 도메인의 템플릿, 시드 고정, `fit`/`transform` 규칙, `NUM_CLASSES` 통일 — 전부
"막히지 않고 끝까지 가기 위한" 장치입니다.
GPU 확인과 수시 저장도 같은 목적입니다. 실력을 점수로 바꾸는 데 필요한 것들이지, 실력 그 자체가 아닙니다.

시험장에서 새로운 걸 시도하지 마세요. [templates/](../templates/)의 네 파일을 손에 익혀 두는 것이,
알고리즘을 하나 더 아는 것보다 확실하게 점수가 됩니다.

---

돌아가기: [README](../README.md)
