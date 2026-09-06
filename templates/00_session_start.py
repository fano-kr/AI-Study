# %% [markdown]
# # 00. 세션 시작 셀 — 시작할 때, 그리고 세션이 튕길 때마다
#
# 교재: docs/04-exam-day.md
#
# 세션이 살아 있는 동안에는 **한 번이면 됩니다.** 문제가 몇 개든 다시 실행하지 않습니다.
#
# 단, **세션이 튕기면 이 셀을 처음부터 다시 실행해야 합니다.**
# 런타임이 재시작되면 GPU 연결이 끊기고 커널 변수도 전부 사라집니다.
# 자세한 복구 절차는 이 파일 맨 아래 7절에 있습니다.
#
#   00_session_start.py           ← 지금 이 셀. 세션당 1회
#   01_tabular_classification.py  ← 1번 문제 (회귀면 01_tabular_regression.py)
#   02_text_classification.py     ← 2번 문제
#   03_image_classification.py    ← 3번 문제 (BBox 요구 시 03_image_localization.py)
#
# ── 문제 푸는 순서: GPU 를 많이 쓰는 것부터 ──
#
#   1순위  이미지 (3번 문제)        학습 10~30분
#   2순위  텍스트 BERT (2번 문제)   학습 20~40분
#   3순위  텍스트 TF-IDF (2번 문제) 몇 분
#   4순위  정형 (1번 문제)          몇 분
#
# 문제 번호(1 Tabular · 2 Text · 3 Image)와 푸는 순서는 별개입니다.
# 번호는 제출 파일명과 변수 접두사에만 쓰입니다.
#
# 이유는 학습이 도는 동안 손이 비기 때문입니다.
# 이미지 문제의 fit() 을 먼저 걸어 두고, 그 시간에 정형 문제의 전처리 코드를 작성합니다.
# 순서를 뒤집으면 정형 문제를 다 푼 뒤에야 이미지 학습이 시작되고,
# 그 30분 동안 화면만 보게 됩니다.
#
# 주의: 커널이 하나이므로 셀은 순서대로 실행됩니다.
#       학습이 도는 동안 다른 셀을 "실행"할 수는 없고, "작성"해 둘 수 있습니다.
#       작성해 둔 셀은 학습이 끝나는 즉시 이어서 돌아갑니다.
#
# ── 변수 이름 규칙 ──
#
# 커널 하나를 세 문제가 공유하므로 전역 변수가 서로 덮어씁니다.
# 템플릿마다 시험 순서대로 접두사가 미리 붙어 있습니다.
#   1번 Tabular = p1_ · 2번 Text = p2_ · 3번 Image = p3_
# 순서가 다르게 나올 때만 Ctrl+H 로 접두사와 PROB_NO 를 맞추면 됩니다.
#
# 이 셀에서 정한 PHONE, RANDOM_SEED, check_submission 은 세션 공용이라 접두사가 없습니다.

# ============================================================
# 1. GPU 연결 확인  ★ 여기서 한 번만
# ============================================================
# %%
# 시험 환경의 GPU 는 자동으로 붙지 않습니다. 연결 코드를 직접 실행해야 합니다.
# (성능은 Colab 무료 T4 와 비슷한 수준)
#
# GPU 가 없어도 코드는 에러 없이 "조용히" CPU 로 돌아갑니다.
# 이미지/BERT 학습이 수십 배 느려지고 원인을 모른 채 시간을 다 쓰게 되므로,
# assert 로 멈춰 세워 지금 알아차립니다.
#
# Tabular 문제만 풀 계획이어도 확인은 지금 하세요.
# 문제를 다 읽고 "이미지 문제가 있네" 하고 되돌아오면 이미 늦습니다.


# --- BERT(텍스트 방법 B)를 쓸 가능성이 있다면, 이 셀보다 먼저 설치 셀을 실행하세요 ---
#   !pip install -q tf-keras "transformers>=4.40,<5" datasets
#
# 이유: 요즘 환경(TF 2.16+)의 기본 Keras 는 3인데, transformers 의 TF BERT 는
# Keras 3 를 지원하지 않습니다. tf-keras(옛 Keras)를 설치해 두면 아래 두 줄이
# tensorflow 를 import 하기 "전에" 옛 Keras 로 전환해 줍니다. 코드 수정은 없습니다.
# (이미 tensorflow 를 import 한 뒤에 설치했다면: 런타임 재시작 → 이 셀부터 다시)
import os
import importlib.util
if importlib.util.find_spec('tf_keras'):       # tf-keras 가 설치된 경우에만 켠다
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
# 옛 Keras(tf-keras)로 전환된 상태에서는 __version__ 속성이 없으므로 getattr 로 읽는다
KERAS_VER = str(getattr(tf.keras, '__version__', '2 (legacy)'))
print(f"TensorFlow {tf.__version__} / Keras {KERAS_VER}")
print(f"GPU: {gpus}")
assert gpus, "GPU 미연결! 런타임 설정을 확인하세요"

if KERAS_VER.startswith('3'):
    print("참고: Keras 3 로 동작 중(요즘 Colab) — Tabular·TF-IDF·이미지는 문제없지만 BERT(방법 B)는 불가.")
    print("      BERT 를 쓰려면 !pip install -q tf-keras → 런타임 재시작 → 이 셀부터 다시.")
    print("      ★ 시험장은 Keras 2(TF 2.13). Colab 예제에서 본 '최신 인자'를 그대로 쓰면")
    print("        시험에서 TypeError: unexpected keyword argument 로 죽을 수 있음")
    print("        (대표: image_dataset_from_directory(..., verbose=...) — 2.13 엔 없음).")

# 확인 끝. 세션이 유지되는 동안에는 어떤 템플릿에서도 다시 확인하지 않습니다.
# 세션이 튕겨서 런타임이 재시작되면 CPU 로 붙을 수 있으므로 이 셀부터 다시 실행하세요.

# ============================================================
# 2. 제출 정보  ★ 세션이 유지되는 동안 한 번
# ============================================================
# %%
# 모델과 예측 결과는 "휴대폰번호_문제번호" 로 저장해 제출합니다.
#   01012345678_1.csv   (1번 문제 예측 결과)
#   01012345678_1.pkl   (1번 문제 모델)
#
# 파일명을 코드 곳곳에 손으로 치면 오타 하나로 채점이 안 됩니다.
# 번호는 여기서 한 번만 적고, 저장할 때는 항상 변수를 씁니다.

PHONE = '01012345678'      # ★ 본인 휴대폰 번호 (하이픈 없이)

print(f"제출 파일명 규칙: {PHONE}_[문제번호].csv / .pkl")

# ============================================================
# 3. 공통 설정
# ============================================================
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"시드 고정: {RANDOM_SEED}")

# ============================================================
# 4. 저장 규칙 (읽고 넘어갈 것)
# ============================================================
# %% [markdown]
# 시험 환경에는 **중간저장** 기능이 있습니다. 주기적으로 저장하세요.
#
# 다만 중간저장이 지켜 주는 것과 지켜 주지 않는 것을 정확히 구분해야 합니다.
#
# | | 중간저장 | 세션이 튕기면 |
# |---|---|---|
# | 작성한 **코드** | 저장한 지점까지 보존 | 저장 지점 이후만 소실 |
# | 커널 **변수** (`p1_X_train`, 모델 …) | **보존 안 됨** | **전부 소실** |
# | **GPU 연결** | 무관 | **끊김 — 다시 연결해야 함** |
#
# 즉 **코드는 중간저장으로, 학습 결과는 파일로** 이중 방어합니다.
# 30분 걸린 학습을 다시 돌리지 않으려면 모델을 파일로 떨궈 두는 수밖에 없습니다.
#
# 저장 타이밍:
#
# - 단계가 끝날 때마다 (로드 완료 → 저장, 전처리 완료 → 저장)
# - **긴 학습을 시작하기 직전에 반드시** (세션이 죽을 확률이 가장 높은 구간)
# - 학습이 끝나면 즉시 모델을 파일로
# - 제출 파일을 만든 직후
#
# 총 180분입니다. 여기까지 3분 안에 끝내고 분석 자료를 읽으러 가세요.

# %%
import time
EXAM_START = time.time()
EXAM_MINUTES = 180


def remaining():
    """남은 시간(분)을 출력. 튜닝을 더 돌릴지 판단할 때 쓴다."""
    used = (time.time() - EXAM_START) / 60
    print(f"경과 {used:.0f}분 / 남음 {EXAM_MINUTES - used:.0f}분")
    return EXAM_MINUTES - used


remaining()


# ============================================================
# 5. 제출물 검증 도구 — 문제를 끝낼 때마다 호출
# ============================================================
# %%
# 제출 직전 실수(행 수, 컬럼명, 결측, index, label 미복원)를 한 번에 잡습니다.
# 어긋나면 assert 로 즉시 멈추고, 통과하면 요약을 출력합니다.
# 세션 공용이므로 접두사가 없습니다. 각 도메인 템플릿 마지막 셀이 이 함수를 부릅니다.
def check_submission(csv_path, n_rows=None, required_cols=None,
                     label_col=None, allowed_labels=None, model_path=None):
    """제출 파일 검증.

    csv_path       : 제출 CSV 경로 (예: p1_SUBMIT_CSV)
    n_rows         : len(df_test) — 행 수가 이와 같아야 함
    required_cols  : 문제가 요구한 컬럼명 리스트 (지문에서 그대로 옮겨 적을 것)
    label_col      : 예측값이 든 컬럼명
    allowed_labels : 그 컬럼에 허용되는 label 목록 (train 타겟의 고유값)
                     — 인코딩 숫자(0/1/2)가 섞여 있으면 여기서 걸린다. 0점 방지.
    model_path     : 모델 파일도 함께 확인하려면 경로
    """
    import os

    assert os.path.exists(csv_path), f"CSV 없음: {csv_path} — to_csv 경로 확인"
    sub = pd.read_csv(csv_path)

    bad = [c for c in sub.columns if str(c).startswith('Unnamed')]
    assert not bad, f"index=False 누락 — {bad} 컬럼이 생겼다"

    if n_rows is not None:
        assert len(sub) == n_rows, f"행 수 불일치: {len(sub)} != {n_rows}"

    if required_cols is not None:
        missing = [c for c in required_cols if c not in sub.columns]
        assert not missing, f"요구 컬럼 누락: {missing} (있는 컬럼: {list(sub.columns)})"

    # 결측: 채점 대상 컬럼(required_cols·label_col)은 assert, 나머지는 경고만.
    # 지문이 여러 컬럼(ID 등)을 요구할 때, 채점 외 컬럼의 NaN 은 경고만 한다.
    graded = set(required_cols or []) | ({label_col} if label_col else set())
    graded = [c for c in graded if c in sub.columns]
    check_cols = graded if graded else list(sub.columns)
    null_cols = [c for c in check_cols if sub[c].isnull().any()]
    assert not null_cols, f"채점 대상 컬럼에 결측: {null_cols}"
    other_null = [c for c in sub.columns if c not in check_cols and sub[c].isnull().any()]
    if other_null:
        print(f"  (참고) 채점 외 컬럼에 결측 있음: {other_null} — 요구 컬럼만 제출한다면 무관")

    if label_col is not None and allowed_labels is not None:
        assert label_col in sub.columns, f"'{label_col}' 컬럼 없음"
        got = set(sub[label_col].unique())
        extra = got - set(allowed_labels)
        assert not extra, (f"label 미복원 의심 — 허용 밖 값 {sorted(extra, key=str)[:5]} "
                           f"(허용: {sorted(set(allowed_labels), key=str)[:5]}). 숫자 그대로면 0점!")

    if model_path is not None:
        assert os.path.exists(model_path), f"모델 파일 없음: {model_path}"
        assert os.path.getsize(model_path) > 0, f"모델 파일이 비어 있음: {model_path}"

    print(f"[검증 통과] {csv_path} — {len(sub)}행, 컬럼 {list(sub.columns)}")
    if label_col is not None:
        print(f"  {label_col} 값 예시: {sub[label_col].head(5).tolist()}")
    if model_path is not None:
        import os
        print(f"  모델: {model_path} ({os.path.getsize(model_path):,} bytes)")
    return sub

# %% [markdown]
# ## 6. 이제 문제를 연다
#
# 순서를 정하고 시작하세요. **GPU를 많이 쓰는 문제부터**입니다.
#
# | 순서 | 문제 유형 | 학습 시간 | 그 시간에 할 일 |
# |---|---|---|---|
# | 1 | 이미지 (3번 문제) | 10~30분 | 다음 문제의 코드 작성 |
# | 2 | 텍스트 BERT (2번) | 20~40분 | 정형 문제 코드 작성 |
# | 3 | 텍스트 TF-IDF (2번) | 몇 분 | — |
# | 4 | 정형 (1번) | 몇 분 | — |
#
# 정형 문제를 먼저 다 풀면, 이미지 학습 30분 동안 할 일이 없어집니다.
# 긴 학습을 **먼저 걸어 두는 것**이 180분을 버는 유일한 방법입니다.

# %% [markdown]
# ## 7. 세션이 튕겼을 때 — 복구 절차
#
# 런타임이 재시작되면 **GPU 연결과 커널 변수가 전부 사라집니다.**
# 코드는 중간저장한 지점까지 남아 있으므로, 아래 순서로 복구합니다.
#
# 1. **런타임 유형이 T4 인지 다시 확인** — 재연결 시 CPU 로 붙는 경우가 있습니다
# 2. **이 세션 시작 셀을 처음부터 다시 실행** (GPU 연결 + PHONE + 시드)
# 3. **이미 끝낸 문제는 재학습하지 말고 파일에서 복구** (아래 셀)
# 4. 진행 중이던 문제만 이어서
#
# 3번을 위해 학습이 끝날 때마다 모델을 저장해 두는 것입니다.
# 저장해 두지 않았다면 30분짜리 학습을 처음부터 다시 돌려야 합니다.

# %%
# --- 복구용: 이미 끝낸 문제의 결과를 파일에서 되살린다 ---
# 필요한 것만 주석을 풀어 실행하세요.

# import joblib
# p1_best_model = joblib.load(f'{PHONE}_1.pkl')                     # 1번 Tabular
# p2_bundle = joblib.load(f'{PHONE}_2.pkl')                         # 2번 Text (모델+벡터라이저)
# p2_best_ml_model, p2_tfidf = p2_bundle['model'], p2_bundle['vectorizer']
# p3_model = tf.keras.models.load_model(f'{PHONE}_3.keras')         # 3번 Image

import os
_done = sorted(f for f in os.listdir('.') if f.startswith(PHONE))
print(f"저장돼 있는 제출물: {_done if _done else '없음 (아직 푼 문제 없음)'}")