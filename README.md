# AICE Professional 속성 과정 (2026)

Tabular · Text · Image 세 도메인의 **시험용 파이프라인 하나**를 익히는 과정입니다.
[fano-kr/AI-Study](https://github.com/fano-kr/AI-Study)의 실습 코드를 강의 순서에 맞게 재구성했습니다.

## 이 과정이 노리는 것

AICE Professional 실기는 새 모델을 발명하는 시험이 아니라
**정해진 순서를 시간 안에 정확히 밟는 시험**입니다. 그래서 알고리즘 이론 대신
아래 한 줄을 세 도메인에 반복 적용합니다.

```
로드 → EDA → 전처리 → 분할 → 모델 비교 → 튜닝 → 평가 → 제출 파일
```

세 도메인에서 바뀌는 것은 **전처리와 모델뿐**이고 나머지 뼈대는 그대로입니다.

## 타임라인 — 발표 45분 + Q&A 15분

| 시간 | 차시 | 교재 | 노트북 |
|---|---|---|---|
| 00–06 | 오리엔테이션 · 시험 환경 · 공통 파이프라인 | [00-orientation.md](docs/00-orientation.md) | — |
| 06–19 | **Tabular** 분류/회귀 | [01-tabular.md](docs/01-tabular.md) | [01_tabular.ipynb](notebooks/01_tabular_quickstart.ipynb) |
| 19–29 | **Text** 분류 (TF-IDF / BERT) | [02-text.md](docs/02-text.md) | [02_text.ipynb](notebooks/02_text_quickstart.ipynb) |
| 29–38 | **Image** 분류 (전이학습) | [03-image.md](docs/03-image.md) | [03_image.ipynb](notebooks/03_image_quickstart.ipynb) |
| 38–45 | 시험 당일 — 순서·변수·시간 배분 | [04-exam-day.md](docs/04-exam-day.md) | — |
| **45–60** | **Q&A** | — | — |

Tabular에 시간을 가장 많이 배정한 이유는 [00-orientation.md](docs/00-orientation.md)에 있습니다.

> **45분에서는 노트북을 라이브로 따라 치기 어렵습니다.**
> 발표 중에는 필요한 셀만 시연하고, `notebooks/`는 강의 후 각자 돌려 보는 자료로 안내하세요.
> 시간이 밀리면 회귀(01), BERT 함정(02), categorical(03) 슬라이드를 먼저 줄이세요.

## 시험 환경 여섯 가지

실력과 무관하게 점수를 잃는 지점입니다. 시험 시간은 **총 180분**.
자세한 내용은 [00-orientation.md](docs/00-orientation.md)와 [04-exam-day.md](docs/04-exam-day.md)에 있습니다.

| | 내용 |
|---|---|
| **지문은 4단계** | ①데이터(train·test, test 정답 없음) ②맞출 것 ③결과물 컬럼명·값 ④파일명. ③④ 어기면 0점 |
| **GPU는 자동으로 안 붙는다** | 시작할 때 한 번. **세션이 튕기면 다시** — 재연결 시 CPU로 붙기도 합니다 |
| **중간저장은 코드만 지킨다** | 변수와 GPU는 안 지켜집니다. 학습 결과는 **파일로** 따로 |
| **파일명은 휴대폰번호_문제번호** | `01012345678_1.csv` / `01012345678_1.pkl` — 하이픈 없이 |
| **CSV의 값은 label로 복원** | 인코딩 숫자(0/1/2) 그대로 내면 **0점**. 컬럼명도 요구 그대로 |
| **커널이 하나다** | GPU 많이 쓰는 문제부터. 변수는 `p1_`/`p2_`/`p3_`로 문제별 분리 (시험 순서대로 미리 세팅) |

세션 단위로 정하는 것과 문제마다 바꾸는 것을 분리해 두면 실수가 줄어듭니다.

```python
# ── 세션 시작 셀 (templates/00_session_start.py) — 시작할 때, 그리고 튕길 때마다 ──
!nvidia-smi
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus, "GPU 미연결!"          # 없으면 에러 없이 CPU로 떨어진다

PHONE = '01012345678'

# ── 각 문제 템플릿 맨 위 — 문제마다 이 한 줄만 ──
PROB_NO = 1
SUBMIT_CSV = f'{PHONE}_{PROB_NO}.csv'
SUBMIT_MODEL = f'{PHONE}_{PROB_NO}.pkl'
```

## 폴더 구조

```
docs/        강의 교재 (차시별 1개)
notebooks/   축약 실습 — 강의 후 각자 돌려 보는 자료
templates/   시험장에서 복사해 쓰는 완성 파이프라인
slides/      발표자료 (HTML)
kt-study/    KT 공식 실습 원본 (로컬 전용 — 용량·배포 문제로 git 미포함)
```

`notebooks/`는 개념 확인용 축약본, `templates/`는 시험장에서 통째로 붙여넣는 전체본입니다.
강의 후 손에 남겨야 하는 건 `templates/` 쪽입니다.

준비물은 이 네 벌이 전부입니다. **이진분류 템플릿은 없습니다** —
클래스 2개인 다중분류가 곧 이진분류라서, Text·Image는 다중 한 벌로 준비합니다.

시험 문제 순서는 **1번 Tabular · 2번 Text · 3번 Image**로 고정입니다.
파일명과 변수 접두사(`p1_`·`p2_`·`p3_`), `PROB_NO`가 이 순서에 미리 맞춰져 있어
표준 케이스에서는 치환 없이 그대로 붙여넣으면 됩니다.

| 템플릿 | 용도 |
|---|---|
| [00_session_start.py](templates/00_session_start.py) | GPU 연결, PHONE, 시드, **제출물 검증 `check_submission()`**, 문제 순서, 복구 절차 |
| [01_tabular_classification.py](templates/01_tabular_classification.py) | **문제 1** — 정형 분류 (이진/다중 자동 대응) |
| [01_tabular_regression.py](templates/01_tabular_regression.py) | **문제 1** — 정형 회귀 |
| [01_tabular_dl.py](templates/01_tabular_dl.py) | **문제 1 변형** — 정형 DNN (문제가 딥러닝을 지시할 때) |
| [02_text_classification.py](templates/02_text_classification.py) | **문제 2** — 텍스트 분류 (TF-IDF 우선, BERT 선택) |
| [03_image_classification.py](templates/03_image_classification.py) | **문제 3** — 이미지 분류 (MobileNetV2 전이학습) |
| [03_image_localization.py](templates/03_image_localization.py) | **문제 3 변형** — BBox 회귀+분류 (문제가 객체 위치를 요구할 때) |

발표자료는 [slides/index.html](slides/index.html)을 브라우저에서 열면 됩니다.
<kbd>←</kbd> <kbd>→</kbd> 이동, <kbd>O</kbd> 목차. 하단 막대는 슬라이드 수가 아니라
**발표 45분 중 지금 몇 분째인지**를 보여 줍니다.

## 시작하기

**Colab T4를 실행 환경으로 전제**합니다. 시험 환경이 이와 비슷하기 때문입니다.

1. [Colab](https://colab.research.google.com/)에서 `파일 → 노트북 업로드`로 `notebooks/`의 파일을 올립니다
2. **`런타임 → 런타임 유형 변경 → T4 GPU`** — 먼저 바꾸고 시작하세요
3. 첫 셀부터 순서대로 실행

노트북 3종은 Colab에 이미 있는 패키지만 씁니다.
텍스트 템플릿의 KLUE 예제(방법 A)나 BERT(방법 B)를 돌릴 때만 **첫 셀**에 한 줄 추가하세요.

```python
!pip install -q tf-keras "transformers>=4.40,<5" datasets
```

`tf-keras`가 필요한 이유: 요즘 Colab의 기본 Keras는 3인데, transformers의 TF BERT는
Keras 3를 지원하지 않습니다. `tf-keras`가 설치돼 있으면 세션 시작 셀
([templates/00_session_start.py](templates/00_session_start.py))이 자동으로 옛 Keras로
전환합니다. 반드시 **tensorflow를 import하기 전에** 설치하세요 —
이미 import한 뒤라면 런타임 재시작 후 처음부터 실행하면 됩니다.

이미지 노트북은 GPU가 없으면 `assert`로 멈춥니다. 시험장과 같은 동작입니다.

### T4 실측 (선택)

자료의 소요 시간은 **T4 기준 추정치**입니다. 실측하려면
[notebooks/99_colab_benchmark.ipynb](notebooks/99_colab_benchmark.ipynb)를
Colab T4에서 위에서부터 실행하세요. 마지막 셀이 도메인별 실제 시간을 표로 출력합니다.

## 사전 지식

- Python 문법, `pandas` DataFrame 조작 (`groupby`, `merge` 수준까지)
- `scikit-learn`의 `fit` / `transform` / `predict` 개념
- AICE Associate 수준의 머신러닝 용어 (과적합, 검증 데이터, 정확도)

셋 중 하나라도 흔들리면 45분 안에 소화하기 어렵습니다. 노트북을 먼저 혼자 돌려 보고 오세요.

## 주의

시험 시간을 **180분**으로 놓고 시간 배분을 설계했습니다.
문항 수·채점 기준 같은 나머지 운영 정보는 회차마다 바뀔 수 있으므로
**응시 전 [AICE 공식 사이트](https://aice.study/)에서 확인**하세요.
이 저장소가 보장하는 것은 운영 정보가 아니라 **코드 파이프라인**입니다.
