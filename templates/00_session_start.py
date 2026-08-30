# %% [markdown]
# # 00. 세션 시작 셀 — 시험 시작 후 딱 한 번만 실행
#
# 교재: docs/04-exam-day.md
#
# 이 셀은 **시험 세션당 한 번**입니다. 문제가 몇 개든 다시 실행하지 않습니다.
# 여기서 정한 GPU 연결과 PHONE 은 세션이 끝날 때까지 유효합니다.
#
# 문제마다 바꾸는 것은 각 도메인 템플릿 맨 위의 PROB_NO 한 줄뿐입니다.
#
#   00_session_start.py        ← 지금 이 셀. 세션당 1회
#   tabular_classification.py  ← 문제마다 붙여넣기 (PROB_NO 만 수정)
#   tabular_regression.py
#   text_classification.py
#   image_classification.py

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

# !nvidia-smi

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow {tf.__version__}")
print(f"GPU: {gpus}")
assert gpus, "GPU 미연결! 런타임 설정을 확인하세요"

# 확인 끝. 이후 어떤 템플릿에서도 GPU 를 다시 확인하지 않습니다.

# ============================================================
# 2. 제출 정보  ★ 세션당 한 번
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
# 작성한 코드는 **자동 저장되지 않습니다.** 세션이 끊기면 미저장분은 전부 사라집니다.
#
# 저장 타이밍을 규칙으로 정해 두세요.
#
# - 셀 하나를 완성할 때마다 `Ctrl+S`
# - 단계가 끝날 때마다 (로드 완료 → 저장, 전처리 완료 → 저장)
# - **긴 학습을 시작하기 직전에 반드시** (세션이 죽을 확률이 가장 높은 구간)
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
