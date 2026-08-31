# AICE Professional 속성 과정 (2026)

Tabular · Text · Image 세 도메인의 **시험용 파이프라인 하나**를 몸에 익히는 60분짜리 압축 과정입니다.
[fano-kr/AI-Study](https://github.com/fano-kr/AI-Study)의 실습 코드를 강의 순서에 맞게 정리·재구성했습니다.

## 이 과정이 노리는 것

AICE Professional 실기는 "새로운 모델을 발명하는 시험"이 아니라
**정해진 순서를 시간 안에 정확히 밟는 시험**입니다. 그래서 이 과정은 알고리즘 이론 대신
아래 한 줄을 세 도메인에 반복 적용하는 데 시간을 씁니다.

```
로드 → EDA → 전처리 → 분할 → 모델 비교 → 튜닝 → 평가 → submission.csv
```

세 도메인에서 바뀌는 것은 **전처리와 모델뿐**이고, 나머지 뼈대는 그대로입니다.
그 사실을 체감하는 순간이 이 과정의 목표 지점입니다.

## 타임라인 — 발표 45분 + Q&A 15분

| 시간 | 차시 | 교재 | 노트북 |
|---|---|---|---|
| 00–06 | 오리엔테이션 · 시험 환경 · 공통 파이프라인 | [00-orientation.md](docs/00-orientation.md) | — |
| 06–19 | **Tabular** 분류/회귀 | [01-tabular.md](docs/01-tabular.md) | [01_tabular.ipynb](notebooks/01_tabular_quickstart.ipynb) |
| 19–29 | **Text** 분류 (TF-IDF / BERT) | [02-text.md](docs/02-text.md) | [02_text.ipynb](notebooks/02_text_quickstart.ipynb) |
| 29–38 | **Image** 분류 (전이학습) | [03-image.md](docs/03-image.md) | [03_image.ipynb](notebooks/03_image_quickstart.ipynb) |
| 38–45 | 시험 당일 — 순서·변수·시간 배분 | [04-exam-day.md](docs/04-exam-day.md) | — |
| **45–60** | **Q&A** | — | — |

Tabular에 시간을 가장 많이 배정한 이유는 [00-orientation.md](docs/00-orientation.md)에 적어 두었습니다.

> **45분에서는 노트북을 라이브로 따라 치기 어렵습니다.**
> 발표 중에는 필요한 셀만 시연하고, `notebooks/`는 **강의 후 각자 돌려 보는 자료**로 안내하세요.
> 시간이 밀리면 회귀(01), BERT 함정(02), categorical(03) 슬라이드를 먼저 줄이면 흐름이 덜 끊깁니다.

## 시험 환경에서 놓치기 쉬운 네 가지

모델 실력과 무관하게 점수를 잃는 지점입니다. 자세한 내용은 [00-orientation.md](docs/00-orientation.md)와
[04-exam-day.md](docs/04-exam-day.md)에 있습니다.

시험 시간은 **총 180분**입니다.

| | 내용 |
|---|---|
| **분석 자료가 주어진다** | 컬럼 설명·요구사항 문서를 먼저 읽으세요. 누수 컬럼이 거기서 드러납니다 |
| **GPU는 자동으로 안 붙는다** | 시작할 때 한 번. **세션이 튕기면 다시** — 재연결 시 CPU로 붙기도 합니다 |
| **중간저장은 코드만 지킨다** | 변수와 GPU는 안 지켜집니다. 학습 결과는 **파일로** 따로 |
| **파일명은 휴대폰번호_문제번호** | `01012345678_1.csv` / `01012345678_1.pkl` — 하이픈 없이 |
| **커널이 하나다** | GPU 많이 쓰는 문제부터. 변수는 `p1_` 접두사로 문제별 분리 |

세션 단위로 정하는 것과 문제마다 바꾸는 것을 분리해 두면 실수가 줄어듭니다.

```python
# ── 세션 시작 셀 (templates/00_session_start.py) — 시작할 때, 그리고 튕길 때마다 ──
!nvidia-smi
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus, "GPU 미연결!"          # 없으면 조용히 CPU로 도니 멈춰 세운다

PHONE = '01012345678'

# ── 각 문제 템플릿 맨 위 — 문제마다 이 한 줄만 ──
PROB_NO = 1
SUBMIT_CSV = f'{PHONE}_{PROB_NO}.csv'
SUBMIT_MODEL = f'{PHONE}_{PROB_NO}.pkl'
```

**연습은 Colab T4에서 하세요.** 시험 환경이 이와 비슷합니다.
다른 환경에 익숙해져 있으면 런타임 설정·파일 업로드·단축키에서 시간을 씁니다.

## 폴더 구조

```
docs/        강의 교재 (읽는 자료, 차시별 1개)
notebooks/   강의 중 함께 실행하는 축약 실습 (빠르게 끝나도록 데이터를 줄여 둠)
templates/   시험장에서 복사해 쓰는 완성 파이프라인 (길고, 주석이 많고, 빈칸이 표시되어 있음)
slides/      강의 진행용 발표자료 (HTML 23장, 브라우저로 바로 열림)
```

[slides/index.html](slides/index.html)을 브라우저에서 열면 발표 모드로 실행됩니다.
<kbd>←</kbd> <kbd>→</kbd> 로 이동, <kbd>O</kbd> 로 목차. 하단 막대는 슬라이드 수가 아니라
**60분 중 지금 몇 분째인지**를 보여 줍니다.

`notebooks/`와 `templates/`의 역할이 다릅니다.
노트북은 **강의 중에 따라 치는 축약본**이고, 템플릿은 **시험장에서 통째로 붙여넣는 전체본**입니다.
강의를 다 들은 뒤 실제로 손에 남겨야 하는 건 `templates/` 쪽입니다.

모든 도메인 템플릿의 변수에는 **`p1_` 접두사**가 붙어 있습니다.
커널 하나로 여러 문제를 풀기 때문입니다. 2번 문제는 `Ctrl+H`로 `p1_` → `p2_` 일괄 치환하고
`p2_PROB_NO = 2`만 고치면 됩니다.

| 템플릿 | 용도 |
|---|---|
| [00_session_start.py](templates/00_session_start.py) | GPU 연결, PHONE, 시드, 문제 순서 전략, **튕겼을 때 복구 절차** |
| [tabular_classification.py](templates/tabular_classification.py) | 정형 데이터 분류 (이진/다중 자동 대응) |
| [tabular_regression.py](templates/tabular_regression.py) | 정형 데이터 회귀 |
| [text_classification.py](templates/text_classification.py) | 텍스트 분류 (TF-IDF 우선, BERT 선택) |
| [image_classification.py](templates/image_classification.py) | 이미지 분류 (MobileNetV2 전이학습) |

## 시작하기

이 과정은 **Colab T4를 실행 환경으로 전제**합니다. 시험 환경이 이와 비슷하기 때문입니다.

1. [Colab](https://colab.research.google.com/)에서 `파일 → 노트북 업로드`로 `notebooks/`의 파일을 올립니다
2. **`런타임 → 런타임 유형 변경 → T4 GPU`** — 먼저 바꾸고 시작하세요
3. 첫 셀부터 순서대로 실행

실습 노트북 3종은 Colab 런타임에 이미 있는 패키지만 씁니다. 별도 설치가 필요 없습니다.
텍스트 템플릿의 KLUE 예제를 돌릴 때만 한 줄 추가하세요.

```python
!pip install -q datasets transformers
```

이미지 노트북은 GPU가 붙어 있지 않으면 `assert`로 멈춥니다. 시험장과 같은 동작입니다.

## 사전 지식

- Python 문법, `pandas` DataFrame 조작 (`groupby`, `merge` 수준까지)
- `scikit-learn`의 `fit` / `transform` / `predict` 개념
- AICE Associate 수준의 머신러닝 용어 (과적합, 검증 데이터, 정확도)

위 세 가지가 흔들리면 60분 안에 소화하기 어렵습니다. 그때는 노트북을 먼저 혼자 한 번 돌려 보고 오세요.

## 주의

이 교재는 시험 시간을 **180분**으로 놓고 시간 배분을 설계했습니다.
문항 수·채점 기준 같은 나머지 운영 정보는 회차마다 바뀔 수 있으므로,
**응시 전 [AICE 공식 사이트](https://aice.study/)에서 확인**하세요.
이 저장소가 보장하는 것은 운영 정보가 아니라 **코드 파이프라인**입니다.
