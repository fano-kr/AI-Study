# AICE Professional 1시간 속성 과정 (2026)

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

## 60분 타임라인

| 시간 | 차시 | 교재 | 실습 |
|---|---|---|---|
| 00–06 | 오리엔테이션 · 공통 파이프라인 | [00-orientation.md](docs/00-orientation.md) | — |
| 06–26 | **Tabular** 분류/회귀 | [01-tabular.md](docs/01-tabular.md) | [01_tabular.ipynb](notebooks/01_tabular_quickstart.ipynb) |
| 26–41 | **Text** 분류 (TF-IDF / BERT) | [02-text.md](docs/02-text.md) | [02_text.ipynb](notebooks/02_text_quickstart.ipynb) |
| 41–55 | **Image** 분류 (전이학습) | [03-image.md](docs/03-image.md) | [03_image.ipynb](notebooks/03_image_quickstart.ipynb) |
| 55–60 | 시험 당일 체크리스트 | [04-exam-day.md](docs/04-exam-day.md) | — |

Tabular에 시간을 가장 많이 배정한 이유는 [00-orientation.md](docs/00-orientation.md)에 적어 두었습니다.

## 폴더 구조

```
docs/        강의 교재 (읽는 자료, 차시별 1개)
notebooks/   강의 중 함께 실행하는 축약 실습 (빠르게 끝나도록 데이터를 줄여 둠)
templates/   시험장에서 복사해 쓰는 완성 파이프라인 (길고, 주석이 많고, 빈칸이 표시되어 있음)
```

`notebooks/`와 `templates/`의 역할이 다릅니다.
노트북은 **강의 중에 따라 치는 축약본**이고, 템플릿은 **시험장에서 통째로 붙여넣는 전체본**입니다.
강의를 다 들은 뒤 실제로 손에 남겨야 하는 건 `templates/` 쪽입니다.

| 템플릿 | 용도 |
|---|---|
| [tabular_classification.py](templates/tabular_classification.py) | 정형 데이터 분류 (이진/다중 자동 대응) |
| [tabular_regression.py](templates/tabular_regression.py) | 정형 데이터 회귀 |
| [text_classification.py](templates/text_classification.py) | 텍스트 분류 (TF-IDF 우선, BERT 선택) |
| [image_classification.py](templates/image_classification.py) | 이미지 분류 (MobileNetV2 전이학습) |

## 시작하기

### 로컬

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

`requirements.txt`는 Tabular/Text(TF-IDF)까지만 필수로 두었습니다.
TensorFlow와 transformers는 무겁기 때문에 주석으로 분리해 두었으니, Image·BERT 실습을 할 때만 설치하세요.

### Colab (Image·BERT 실습 권장)

이미지 실습과 BERT 실습은 GPU가 있는 편이 훨씬 빠릅니다.
Colab에서 `런타임 → 런타임 유형 변경 → GPU`로 바꾼 뒤 노트북을 업로드해 실행하세요.

## 사전 지식

- Python 문법, `pandas` DataFrame 조작 (`groupby`, `merge` 수준까지)
- `scikit-learn`의 `fit` / `transform` / `predict` 개념
- AICE Associate 수준의 머신러닝 용어 (과적합, 검증 데이터, 정확도)

위 세 가지가 흔들리면 60분 안에 소화하기 어렵습니다. 그때는 노트북을 먼저 혼자 한 번 돌려 보고 오세요.

## 주의

시험 시간·문항 수·채점 기준 같은 운영 정보는 회차마다 바뀝니다.
이 교재의 숫자는 학습 설계를 위한 기준일 뿐이므로, **응시 전 [AICE 공식 사이트](https://aice.study/)에서 반드시 확인**하세요.
이 저장소가 보장하는 것은 운영 정보가 아니라 **코드 파이프라인**입니다.
