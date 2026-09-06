# 04. 시험 당일 체크리스트

**소요 7분** · 이전: [03. Image](03-image.md)

---

인쇄해서 옆에 두고 쓰는 문서입니다.

**시험 시간은 총 180분입니다.** 아래 순서를 3분 안에 끝내고 문제로 들어가세요.

## 1. 세션 시작 셀

이 절의 ①②는 세션이 살아 있는 동안 **한 번**입니다. 문제가 몇 개든 다시 실행하지 않습니다.
**단, 세션이 튕기면 이 셀부터 다시 실행해야 합니다** (5절 복구 절차).
템플릿으로는 [00_session_start.py](../templates/00_session_start.py) 한 파일에 모여 있습니다.

### ① GPU 연결 (가장 먼저)

시험 환경의 GPU는 **자동으로 붙지 않습니다.** 연결 코드를 직접 실행해야 합니다.
성능은 Colab 무료 T4와 비슷한 수준입니다.

```python

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus}")
assert gpus, "GPU 미연결! 런타임 설정을 확인하세요"
```

GPU가 없어도 코드는 **에러 없이 CPU로 떨어집니다.**
이미지·BERT가 수십 배 느려지므로 `assert`로 멈춰 세웁니다.

**확인은 무조건 먼저** 하세요.
문제를 다 읽고 나서 "이미지 문제가 있네" 하고 되돌아가면 이미 늦습니다.

세션이 유지되는 동안에는 여기서 확인하면 끝입니다. 2번, 3번 문제에서 다시 확인하지 마세요.

**단, 세션이 튕기면 다시 연결해야 합니다.** 런타임이 재시작되면 GPU가 떨어지고,
재연결 시 CPU로 붙는 경우가 있습니다.

### ② 제출 정보 — PHONE은 한 번, PROB_NO는 문제마다

```python
# 세션 시작 셀에서 한 번
PHONE = '01012345678'    # 본인 휴대폰 번호 (하이픈 없이)

# 각 템플릿에 시험 순서대로 미리 적혀 있음 (1 Tabular · 2 Text · 3 Image)
PROB_NO = 1

SUBMIT_CSV = f'{PHONE}_{PROB_NO}.csv'
SUBMIT_MODEL = f'{PHONE}_{PROB_NO}.pkl'
print(SUBMIT_CSV, SUBMIT_MODEL)
```

파일명을 직접 타이핑하지 마세요. **오타 하나로 채점이 안 됩니다.**

남은 시간을 찍어 주는 함수를 두면 튜닝을 더 돌릴지 판단하기 쉽습니다.

```python
import time
EXAM_START, EXAM_MINUTES = time.time(), 180

def remaining():
    used = (time.time() - EXAM_START) / 60
    print(f"경과 {used:.0f}분 / 남음 {EXAM_MINUTES - used:.0f}분")
    return EXAM_MINUTES - used
```

## 2. 문제 순서 — GPU를 많이 쓰는 것부터

문제를 여러 개 받으면 **어느 것부터 손대는지가 점수를 가릅니다.**

| 푸는 순서 | 유형 (문제 번호) | 학습 시간 | 그 시간에 할 일 |
|---|---|---|---|
| 1 | **텍스트 BERT** (2번 문제) | 3만 건 3에폭 약 40분 | 다른 문제 코드 작성 |
| 2 | **이미지** (3번 문제) | 두 단계 15~25분 | 다른 문제 코드 작성 |
| 3 | 텍스트 TF-IDF (2번 문제) | 약 10초 | — |
| 4 | 정형 (1번 문제) | 1분 미만 | — |

*T4 실측 기준. **BERT 가 가장 오래 걸립니다** — 이미지가 아니라 BERT 를 먼저 거세요.
train 6만 건을 그대로 쓰면 3에폭에 80분이라 다른 문항을 잡아먹습니다.*

**배점(Tabular 30 · Text 35 · Image 35)도 순서에 반영하세요.**
Text·Image가 각 35점으로 크고 학습도 길므로 먼저 착수합니다. 정형(30점)은 몇 분이면
끝나니 학습 대기 중에 처리합니다. 배점이 큰 문제를 미완성으로 남기는 게 가장 큰 손실입니다.

문제 번호(1 Tabular · 2 Text · 3 Image)와 푸는 순서는 별개입니다.
번호는 제출 파일명과 변수 접두사에만 쓰이므로, 3번부터 풀어도 아무 문제 없습니다.

**학습이 도는 동안 손이 비기 때문**입니다.
이미지 `fit()`을 먼저 걸어 두면 그 30분이 정형 문제를 짜는 시간이 됩니다.
순서를 뒤집으면 정형을 다 푼 뒤에야 학습이 시작됩니다.

**커널이 하나입니다.** 셀은 순서대로 실행되므로 학습 중에 다른 셀을 *실행*할 수는 없고,
*작성*만 가능합니다. 대기열에 넣어 둔 셀은 학습이 끝나는 즉시 이어서 돌아갑니다.

```
18분  이미지 fit() 실행  ─┐
                          │  이 30분 동안 정형 문제 셀을 작성해 대기열에 넣는다
48분  학습 종료          ─┘
48분  대기 중이던 정형 셀이 자동으로 이어서 실행
```

## 3. 문제를 시작할 때

### ③ 분석 자료 정독

문제 지문은 **네 부분**으로 옵니다. 각각에서 뽑아낼 것이 정해져 있습니다.

| 지문 | 내용 | 코드로 옮길 것 |
|---|---|---|
| ① 데이터 설명 | `train`, `test` — **`test`에는 정답이 없습니다** | `test`는 예측해서 제출하는 대상. 학습·평가에 쓰지 않음 |
| ② 무엇을 맞출지 | 타겟 컬럼, 분류/회귀 | `TARGET`, 요구 지표(→ 튜닝 `scoring`) |
| ③ 결과물 형식 | **결과 컬럼명**과 **값** (예: 0/1 이 아니라 Y/N) | `PRED_COL`, label 복원 |
| ④ 제출 파일명 | 모델 = `휴대폰번호_문제번호.확장자`<br>결과 = `휴대폰번호_문제번호.csv` | `SUBMIT_CSV` / `SUBMIT_MODEL` |

③과 ④가 실점 지대입니다. 컬럼명을 임의로 짓거나 값을 인코딩된 숫자로 내면
모델이 아무리 좋아도 **0점**입니다. 지문 표기를 그대로 옮기세요.

자료에 "결과를 사후 기록한 컬럼"이라 적혀 있으면 상관계수를 볼 것도 없이 제거합니다.

**지문이 방법을 지정하면 그 방법이 정답입니다.** "딥러닝으로", "원-핫 인코딩으로",
"h5로 저장" — 지시형 문항에서는 이 교재의 기본값보다 지시가 우선입니다.

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

**여기까지 하고 중간저장.** 다음 항목을 보세요.

문제 하나의 전체 흐름입니다. **저장이 먼저고, 개선은 루프**입니다.

```
자료 정독 → 템플릿 붙여넣기 → 로드·전처리·분할 → 기본 모델 1개
  → label 복원 → 저장 (여기까지 40분) → check_submission + 눈 검증
         ↑                                  │ 실패: 수정 후 재저장
         └──────────────────────────────────┘
  → 점수 ≥ 목표?  ── 예 ──→ 다음 문제로
         │ 아니오
  → 시간 30분+ 남음?  ── 아니오 ──→ 지금 제출물로 확정
         │ 예
  → 튜닝 · 방법 B ── 점수 오르면 재저장 → 재검증 ↩
```

두 루프(실패 수정, 튜닝) 모두 **저장으로 되돌아옵니다.** 발표자료에 같은 그림이 흐름도로 들어 있습니다.

## 4. 변수 이름 — 커널은 하나다

시험은 **주피터 커널 하나에서 여러 문제를 풉니다.** 전역 변수는 커널 전체가 공유합니다.

```python
# 1번 문제 (이미지)
X_train = ...        # 이미지 배치

# 2번 문제 (정형) — 같은 커널
X_train = ...        # 1번 문제의 X_train 이 사라졌다
```

1번 문제로 돌아가 셀 하나만 다시 돌리면 **에러 없이 틀린 결과**가 나옵니다.
`TARGET`이 2번 문제의 컬럼명으로 바뀌어 있어도 코드는 그대로 돕니다.

시험 문제 순서는 **1번 Tabular · 2번 Text · 3번 Image**로 고정이므로,
템플릿마다 그 순서대로 접두사와 `PROB_NO`가 **미리 붙어 있습니다.**

```python
# 01_tabular_*.py        # 02_text_*.py          # 03_image_*.py
p1_PROB_NO = 1           p2_PROB_NO = 2          p3_PROB_NO = 3
p1_X_train = ...         p2_X_train = ...        p3_val_result = ...
```

표준 케이스에서는 **아무것도 바꿀 필요가 없습니다.** 템플릿을 그대로 붙여넣으면
변수가 충돌하지 않고 제출 파일도 `_1`, `_2`, `_3`으로 나뉩니다.

실제 순서가 예상과 다르게 나올 때만 `Ctrl+H`로 접두사를 전체 바꾸고
`PROB_NO` 숫자를 함께 고칩니다 — 숫자를 놓치면 다른 문제 제출물을 덮어씁니다.

접두사가 없는 것은 세션 공용 넷뿐입니다 — `PHONE`, `RANDOM_SEED`, `remaining()`, `check_submission()`.

## 5. 세션이 튕겼을 때 — 복구 절차

런타임이 재시작되면 **GPU 연결과 커널 변수가 전부 사라집니다.**
코드는 중간저장한 지점까지 남아 있으므로, 아래 순서로 복구합니다.

1. **런타임 유형이 T4인지 다시 확인** — 재연결 시 CPU로 붙는 경우가 있습니다
2. **세션 시작 셀을 처음부터 다시 실행** (GPU 연결 + `PHONE` + 시드)
3. **이미 끝낸 문제는 재학습하지 말고 파일에서 복구**
4. 진행 중이던 문제만 이어서

```python
import joblib
p1_best_model = joblib.load(f'{PHONE}_1.pkl')               # 1번 Tabular
p2_bundle = joblib.load(f'{PHONE}_2.pkl')                   # 2번 Text (모델+벡터라이저)
p2_best_ml_model, p2_tfidf = p2_bundle['model'], p2_bundle['vectorizer']
p3_model = tf.keras.models.load_model(f'{PHONE}_3.keras')   # 3번 Image
```

3번이 가능하려면 **학습이 끝날 때마다 모델을 저장해 뒀어야** 합니다.
안 해 뒀으면 30분짜리 학습을 처음부터 다시 돌려야 합니다.

> 튕김은 긴 학습 중에 가장 자주 일어납니다.
> "긴 학습 직전 저장"이 저장 규칙 중 가장 중요한 이유입니다.

## 6. 중간저장 — 무엇을 지켜 주고 무엇을 지켜 주지 않는가

시험 환경에는 **중간저장** 기능이 있습니다. 지켜 주는 범위가 정해져 있습니다.

| | 중간저장 | 세션이 튕기면 |
|---|---|---|
| 작성한 **코드** | 저장 지점까지 보존 | 저장 지점 이후만 소실 |
| 커널 **변수** (`p1_X_train`, 모델 …) | **보존 안 됨** | **전부 소실** |
| **GPU 연결** | 무관 | **끊김 — 다시 연결** |

중간저장은 **코드를 지키는 장치이지 런타임을 지키는 장치가 아닙니다.**

**이중 방어 — 코드는 중간저장으로, 학습 결과는 파일로.**

```python
joblib.dump(p1_best_model, p1_SUBMIT_MODEL)      # sklearn
p1_model.save(f'{PHONE}_{p1_PROB_NO}.keras')     # 딥러닝
```

이미지 템플릿의 `ModelCheckpoint`는 학습 **도중에도** 최고 가중치를 파일로 떨궈 둡니다.
60% 진행된 시점에 튕겨도 거기까지는 건집니다.

```python
ModelCheckpoint(f'{PHONE}_{p1_PROB_NO}_ckpt.keras', save_best_only=True, ...)
```

**저장 타이밍**

- 단계가 끝날 때마다 (로드 완료 → 저장, 전처리 완료 → 저장, 모델 비교 완료 → 저장)
- **긴 학습을 시작하기 직전에 반드시** (여기서 세션이 죽을 확률이 가장 높습니다)
- 학습이 끝나면 **즉시 모델을 파일로**
- 제출 파일을 만든 직후

이건 시험장에서 새로 생기는 습관이 아닙니다. **연습할 때부터 손이 저장하게 만드세요.**

> **연습은 Colab T4에서.** 시험 환경이 이와 비슷합니다.
> 다른 환경에 익숙해져 있으면 런타임 설정, 파일 업로드, 단축키에서 시간을 씁니다.

## 7. 제출 파일명 규칙

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

**CSV 내용도 채점 대상입니다.** 예측 컬럼명과 값은 요구된 label 그대로 —
인코딩된 숫자로 내면 **0점**입니다 (`inverse_transform` / `class_names[i]` / `label_map`).

## 8. 180분을 어떻게 쓸 것인가

**"끝까지 도는 코드"를 전반부에 확보하는 것**이 핵심입니다.

### 문제가 둘일 때 (이미지 + 정형)

| 시각 | 무엇을 |
|---|---|
| 0–3 | **세션 시작 셀** — GPU 연결(한 번), PHONE, 시드 |
| 3–18 | **이미지(3번)**: 분석 자료 정독 → 데이터 로드 → 모델 구축 |
| **18** | **`fit()` 실행** — 여기서부터 손이 빈다 |
| 18–48 | BERT 학습(약 40분) 도는 동안 **정형·이미지 셀을 작성** |
| 48–55 | 이미지 평가 → `{PHONE}_3.csv` / `.keras` 저장 |
| 55–75 | 대기하던 정형 셀 실행 → `{PHONE}_1.csv` / `.pkl` 저장 |
| 75–150 | **점수가 낮은 쪽부터** 튜닝, 오르면 제출 파일 갱신 |
| 150–170 | 여유분 (피처 엔지니어링, 재학습) |
| 170–180 | 최종 점검 및 제출 |

### 문제가 하나일 때

| 시각 | 무엇을 |
|---|---|
| 0–3 | 세션 시작 셀 |
| 3–20 | 분석 자료 정독, 데이터 로드, 컬럼 요약표, 제거할 컬럼 결정 |
| 20–40 | 전처리 → 분할 → **기본 모델 1개로 끝까지 돌려 제출 파일 저장** |
| 40–90 | 모델 비교, 1위 선정 |
| 90–140 | 튜닝, 성능이 오르면 제출 파일 갱신 |
| 140–170 | 피처 엔지니어링, 재학습 |
| 170–180 | 최종 점검 및 제출 |

**40분 시점에 이미 제출 가능한 파일이 있어야 합니다.**
완벽한 모델을 만들다 시간이 끝나는 것이 가장 흔한 실패입니다.

문제가 둘이면 앞 문제에 90분을 넘기지 마세요.
한 문제를 0.95까지 올리는 것보다, 두 문제를 0.85씩 내는 쪽이 거의 항상 유리합니다.

## 9. 절대 하지 말 것

| 금지 | 왜 |
|---|---|
| GPU 연결 확인 생략 | CPU로 조용히 돌아가 시간을 다 씀 |
| 튕긴 뒤 GPU 재연결 없이 계속 | 재연결 시 CPU로 붙어 있을 수 있음 |
| 학습 결과를 파일로 안 남김 | 튕기면 30분짜리 학습을 처음부터 다시 |
| 한 문제에 90분 초과 | 다른 문항이 통째로 날아감 |
| 정형 문제부터 풀기 | 이미지 학습 30분 동안 할 일이 없어짐 |
| 접두사 없이 여러 문제 풀기 | 앞 문제 변수를 덮어써 조용히 틀린 결과 |
| 접두사만 바꾸고 `PROB_NO` 안 고치기 | 다른 문제 제출물을 덮어씀 |
| 저장 안 하고 긴 학습 시작 | 세션이 끊기면 코드가 전부 사라짐 |
| 파일명을 매번 손으로 타이핑 | 오타 하나로 채점 불가 |
| 예측값을 인코딩 숫자 그대로 제출 | label 미복원 — **그 문제 0점** |
| 분석 자료 안 읽고 코드부터 | 누수 컬럼을 놓치고, 요구 지표를 잘못 맞춤 |
| `scaler.fit_transform(X_test)` | 데이터 누수. 검증 점수는 오히려 올라가서 발견이 어려움 |
| 분할 전에 SMOTE | 검증 데이터에 합성 샘플이 섞여 점수가 거짓이 됨 |
| test로 학습 | 실격 사유가 될 수 있음 |
| 시드 미고정 | 개선 여부를 판단할 수 없게 됨 |
| 타겟과 상관 0.99 컬럼 방치 | 누수. 정확도 100%는 축하가 아니라 경고 |
| 5개 모델 전부 GridSearch | 시간 초과 |
| BERT부터 시작 | 실패 시 제출물이 없음 |
| shuffle된 데이터셋으로 이미지 예측 | 파일명과 예측이 어긋남 |

## 10. 에러 응급처치

| 에러 | 원인 | 조치 |
|---|---|---|
| 학습이 이상하게 느림 | GPU 미연결 (튕긴 뒤 재연결 안 함) | 세션 시작 셀 재실행 |
| 튕긴 뒤 `NameError` 폭발 | 커널 변수 전부 소실 | 세션 시작 셀 → 저장된 모델 로드 → 이어서 |
| 결과가 이상한데 에러는 없음 | 앞 문제 변수가 살아 있음 | `p1_TARGET` 등 접두사가 맞는지 확인 |
| `NameError: p2_...` | 해당 템플릿 셀을 아직 실행 안 함 | 위에서부터 순서대로 실행 |
| `could not convert string to float` | 범주형이 인코딩 안 됨 | `select_dtypes`로 남은 object 컬럼 확인 |
| `Input contains NaN` | 결측 처리 누락 | `X.isnull().sum().sum()` 확인 |
| `y contains previously unseen labels` | test에만 있는 범주 | 인코딩을 사전 `map(...)` + `fillna(-1)` 방식으로 |
| `Found input variables with inconsistent numbers of samples` | X와 y 행 수 불일치 | `reset_index(drop=True)` 후 재시도 |
| `Number of features of the model must match` | train/test 컬럼 불일치 | `X_test = X_test.reindex(columns=X.columns, fill_value=0)` |
| `Keras is Keras 3, ... not yet supported in Transformers` | tf-keras 미설치 (BERT) | 첫 셀에서 `!pip install -q tf-keras` → 런타임 재시작 → 세션 시작 셀부터 |
| `ResourceExhaustedError` (OOM) | 배치가 큼 | BATCH_SIZE 32→16→8, MAX_LENGTH 128→64 |
| `loss: nan` | 학습률 과대 | BERT는 2e-5, 파인튜닝은 1e-5 |
| **`TypeError: ... unexpected keyword argument`** (콜백·데이터셋 생성 시 1에폭 근처에서 멈춤) | **Colab(Keras 3) 전용 인자를 시험장(Keras 2/TF 2.13)에서 사용** | 그 인자를 **빼라**. 대표: `image_dataset_from_directory(..., verbose=...)`, 최신 콜백 인자. 세션 시작 배너로 **Keras 버전 먼저 확인** |
| EarlyStopping 이 best 를 안 살리는 듯(끝까지 학습) | `restore_best_weights` 동작이 Keras 버전마다 다름 | 콜백에 의존 말고 **학습 후 `model.load_weights(CKPT)`** — 템플릿에 반영됨 |
| `ValueError: Shapes ... incompatible` | 출력층/손실 불일치 | `categorical` ↔ `categorical_crossentropy` 짝 확인 |

## 11. 도메인별 한 줄 요약

**Tabular**
```
컬럼 요약표 → 누수/식별자 제거 → median·최빈값 결측 → LabelEncoder → StandardScaler
→ stratify 분할 → 5모델 비교 → 1위만 GridSearch → {PHONE}_{PROB_NO}.csv
(딥러닝 지시 문항 → 01_tabular_dl.py: get_dummies + MinMax + Sequential + 학습 곡선)
```

**Text**
```
clean_text → TfidfVectorizer → LinearSVC/LogReg/NB 비교 → 제출 파일 확보
             (한국어면 analyzer='char_wb', ngram_range=(2,4) — 공백 토큰화보다 확실히 낫다)
→ (시간 남으면) klue/bert-base, lr=2e-5, from_logits=True, from_pt=True
             (BERT 전제: 첫 셀 !pip install -q tf-keras → 세션 시작 셀이 옛 Keras 로 전환)
```

**Image**
```
image_dataset_from_directory(label_mode='categorical') → 증강 레이어
→ MobileNetV2 동결 Phase1 lr=1e-3 → 상위 30층 해제 + 재컴파일 → Phase2 lr=1e-5
→ 순서 보장 데이터셋으로 예측 → 제출 파일
```
(GPU는 세션 시작 셀에서 이미 확인했습니다.
객체 위치(BBox)를 요구하면 → 03_image_localization.py: XML→xywh, 출력층 4+N, 커스텀 손실, IoU)

## 12. 제출 전 최종 점검

검증은 두 단계입니다 — **구조는 함수로, 내용은 눈으로.**
둘 다 각 도메인 템플릿의 마지막 셀에 들어 있습니다.

```python
# ① 구조 — 세션 시작 셀의 check_submission
#    행 수 · 요구 컬럼 · 결측 · index=False · label 미복원(0점) · 모델 파일
p1_allowed = p1_target_encoder.classes_ if p1_target_encoder else p1_y.unique()
check_submission(p1_SUBMIT_CSV, n_rows=len(p1_df_test),
                 label_col='predicted', allowed_labels=p1_allowed,
                 model_path=p1_SUBMIT_MODEL)
```

```python
# ② 내용 — 도메인별 눈 검증
# 텍스트: 원문과 예측을 나란히 출력해 직접 읽는다
for _, row in submit_df.sample(10).iterrows():
    print(f"[예측: {row['predicted_name']}]  {str(row[TEXT_COL])[:60]}")

# 이미지: 그림을 띄워 예측과 대조한다 (초록 = 일치, 빨강 = 불일치)
plt.imshow(tf.keras.utils.load_img(path)); plt.title(f"P:{pred} / T:{actual}")

# 정형: submission.head() 로 값이 label 인지 본다
```

구조 검증이 통과해도 내용이 엉뚱할 수 있습니다 — 예측이 전부 한 클래스로 쏠려 있거나,
명백한 문장을 반대로 찍고 있으면 제출 전에 원인부터 찾으세요.

- [ ] 파일명이 `{휴대폰번호}_{문제번호}` 규칙과 정확히 일치 (하이픈 없음)
- [ ] 모델 파일과 CSV 파일이 **둘 다** 저장됨
- [ ] 행 수가 test와 일치
- [ ] 컬럼명이 문제 요구와 일치
- [ ] `index=False`로 저장
- [ ] 결측 없음
- [ ] 예측 컬럼명·값이 요구된 label 그대로 (**미복원 = 0점**)
- [ ] 문제별 파일이 **각각** 있는가 (`_1.csv`, `_2.csv` …) — 덮어쓰지 않았는가
- [ ] **중간저장 완료**

## 13. 마지막 한마디

결국 문장 하나입니다.

> **끝까지 도는 코드를 먼저 만들고, 남는 시간으로 점수를 올린다.**

템플릿, 시드 고정, `fit`/`transform` 규칙, `NUM_CLASSES` 통일, 세션 시작 셀, 수시 저장 —
전부 막히지 않고 끝까지 가기 위한 장치입니다.

시험장에서 새로운 걸 시도하지 마세요. [templates/](../templates/)의 다섯 파일을 손에 익혀 두는 것이
알고리즘을 하나 더 아는 것보다 확실합니다.

---

돌아가기: [README](../README.md)
