# AI Study 속성 과정 (2026)

Tabular · Text · Image 세 도메인의 **파이프라인 하나**를 익히는 과정입니다.
[fano-kr/AI-Study_old](https://github.com/fano-kr/AI-Study_old)의 실습 코드를 강의 순서에 맞게 재구성했습니다.

## 이 과정이 노리는 것

AI Study는 새 모델을 발명하는 것이 아니라
**정해진 순서를 시간 안에 정확히 밟는 학습**입니다. 그래서 알고리즘 이론 대신
아래 한 줄을 세 도메인에 반복 적용합니다.

```
로드 → 데이터 확인 → 전처리 → 분할 → 모델 비교 → 눈으로 확인 → 제출 파일
```

세 도메인에서 바뀌는 것은 **전처리와 모델뿐**이고 나머지 뼈대는 그대로입니다.

| 시간 | 차시 | 배점 | 노트북 |
|---|---|---|---|
| 00–06 | 오리엔테이션 · 학습 환경 · 공통 파이프라인 | — | — |
| 06–19 | **Tabular** 분류/회귀 | 30 | [분류](notebooks/01_tabular_classification.ipynb) · [회귀](notebooks/01_tabular_regression.ipynb) |
| 19–29 | **Text** 분류 (TF-IDF) | 35 | [02_text](notebooks/02_text_classification.ipynb) |
| 29–38 | **Image** 분류 (전이학습) | 35 | [03_image](notebooks/03_image_classification.ipynb) |
| 38–45 | 학습 당일 — 순서·변수·시간 배분 | — | — |
| **45–60** | **Q&A** | — | — |

> **45분에서는 노트북을 라이브로 따라 치기 어렵습니다.**
> 발표 중에는 필요한 셀만 시연하고, `notebooks/`는 강의 후 각자 돌려 보는 자료로 안내하세요.

## 학습 환경 여섯 가지

**학습 사양(공식):** GPU T4급(16GB) · Python 3.10 · **TF 2.13(Keras 2)** · PyTorch 2.1.
**연습은 Colab T4 에서** 합니다(로컬 미사용). Colab 은 Keras 3 라 이미지 노트북은 **Keras 2 모드**로
맞춰 연습합니다 — 방법은 아래 '시작하기' 참고.

| | 내용 |
|---|---|
| **지문은 4단계** | ①데이터(train·test, test 정답 없음) ②맞출 것 ③결과물 컬럼명·값 ④파일명. ③④ 어기면 0점 |
| **GPU는 자동으로 안 붙는다** | 시작할 때 한 번. **세션이 튕기면 다시** — 재연결 시 CPU로 붙기도 합니다 |
| **중간저장은 코드만 지킨다** | 변수와 GPU는 안 지켜집니다. 학습 결과는 **파일로** 따로 |
| **파일명은 휴대폰번호_문제번호** | `01012345678_1.csv` / `01012345678_1.pkl` — 하이픈 없이 |
| **CSV의 값은 label로 복원** | 인코딩 숫자(0/1/2) 그대로 내면 **0점**. 컬럼명도 요구 그대로 |
| **커널이 하나다** | GPU 많이 쓰는 문제부터. 변수는 `p1_`/`p2_`/`p3_`로 문제별 분리 |

**세션당 한 번 정하는 것**(GPU 연결·`PHONE`)과 **문제마다 바꾸는 것**(`PROB_NO`)을
분리해 두면 실수가 줄어듭니다.

## 폴더 구조

```
notebooks/   완성형 실습 노트북 (학습에서 그대로 쓰는 전체본)
slides/      발표자료 (HTML)
pdf/         발표·실습 PDF 내보내기
```

`notebooks/`에는 네 개의 완성형 노트북이 있습니다.

| 노트북 | 용도 |
|---|---|
| [01_tabular_classification.ipynb](notebooks/01_tabular_classification.ipynb) | **문제 1** — 정형 분류 (이진/다중 자동 대응, 타겟 인코딩·복원) |
| [01_tabular_regression.ipynb](notebooks/01_tabular_regression.ipynb) | **문제 1** — 정형 회귀 |
| [02_text_classification.ipynb](notebooks/02_text_classification.ipynb) | **문제 2** — 텍스트 분류 (char_wb TF-IDF + 선형 모델) |
| [03_image_classification.ipynb](notebooks/03_image_classification.ipynb) | **문제 3** — 이미지 분류 (MobileNetV2 전이학습) |

각 노트북은 **설정 → 데이터 확인 → (맞출·버릴 컬럼 지정) → 전처리 → 모델 비교 → 눈으로 확인 →
제출 파일 저장** 순서로, 학습에서 **데이터 경로·타겟만 바꿔** 그대로 쓰도록 만들었습니다.

학습 문제 순서는 **1번 Tabular · 2번 Text · 3번 Image**로 고정입니다.
변수 접두사(`p1_`·`p2_`·`p3_`)를 이 순서에 맞춰 두면 커널 하나를 공유해도 서로 안 덮어씁니다.

## 시작하기

**Colab T4를 실행 환경으로 전제**합니다. 학습 환경이 이와 비슷하기 때문입니다.

1. [Colab](https://colab.research.google.com/)에서 `파일 → 노트북 업로드`로 `notebooks/`의 파일을 올립니다
2. **`런타임 → 런타임 유형 변경 → T4 GPU`** — 먼저 바꾸고 시작하세요
3. 첫 셀부터 순서대로 실행

노트북은 Colab에 이미 있는 패키지만 씁니다. 데이터도 노트북이 온라인에서 자동으로 내려받습니다.

**이미지 노트북**은 학습환겅(TF 2.13 = Keras 2)과 API를 맞추려고 첫 셀에서 `tf-keras`(Keras 2 모드)로
전환합니다. 반드시 **tensorflow import 전에** 실행되며, 늦었다면 런타임 재시작 후 처음부터.
이미지 노트북은 GPU가 없으면 `assert`로 멈춥니다 — 학습환경과 같은 동작입니다.

## 사전 지식

- Python 문법, `pandas` DataFrame 조작 (`groupby`, `merge` 수준까지)
- `scikit-learn`의 `fit` / `transform` / `predict` 개념
- 머신러닝 용어 (과적합, 검증 데이터, 정확도)

셋 중 하나라도 흔들리면 45분 안에 소화하기 어렵습니다. 노트북을 먼저 혼자 돌려 보고 오세요.

## 주의

시간을 **180분**으로 놓고 시간 배분을 설계했습니다.
문항 수·채점 기준 같은 나머지 운영 정보는 회차마다 바뀔 수 있으므로
**응시 전 [공식 사이트](https://aice.study/)에서 확인**하세요.
[샘플 문항](https://aice.study/certi/practice)도 미리 풀어 보세요 —
이 저장소가 보장하는 것은 운영 정보가 아니라 **코드 파이프라인**입니다.
