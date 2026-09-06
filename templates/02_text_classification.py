# %% [markdown]
# # Text 분류 템플릿 (TF-IDF 우선 · BERT 선택)
#
# 교재: docs/02-text.md
#
# 전략: 방법 A 를 먼저 돌려 submission.csv 를 확보한 뒤,
#       시간이 남고 점수가 부족할 때만 방법 B 로 갱신합니다.
#
#   방법 A  TF-IDF + ML       1~5분    실패 위험 거의 없음
#   방법 B  BERT Fine-tuning  20~40분  성능은 높지만 실패 위험 있음
#   (소요 시간은 Colab T4 기준)
#
# ★ 표시가 있는 곳만 문제에 맞게 고치면 됩니다.
#
# ── 변수 이름 규칙 ──
# 커널 하나에서 세 문제를 풀므로, 전역 변수가 서로 덮어쓰지 않도록
# 템플릿마다 시험 순서대로 접두사를 붙여 두었습니다.
#   1번 Tabular = p1_ · 2번 Text = p2_ · 3번 Image = p3_
# 실제 순서가 다르게 나오면 Ctrl+H 전체 바꾸기로 접두사를 맞추고 PROB_NO 도 함께 고치세요.
# 안 고치면 다른 문제의 제출물을 덮어씁니다.
# PHONE, RANDOM_SEED, check_submission 은 세션 공용이라 접두사가 없습니다.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# 0. 문제 설정
# ============================================================
# GPU 확인과 PHONE 은 세션 시작 셀(00_session_start.py)에서 합니다. 여기서 다시 하지 않습니다.
# 단, 세션이 튕기면 GPU 연결이 끊기고 변수가 전부 사라지므로 그 셀부터 다시 실행해야 합니다.
p2_PROB_NO = 2                # ★ 시험 순서: Text = 2번

# 세션 시작 셀을 실행했다면 PHONE 은 이미 정의돼 있습니다.
# 이 파일만 단독으로 연습할 때를 위한 기본값입니다.
try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

p2_SUBMIT_CSV = f'{PHONE}_{p2_PROB_NO}.csv'
p2_SUBMIT_MODEL = f'{PHONE}_{p2_PROB_NO}.pkl'          # sklearn 계열
p2_SUBMIT_KERAS = f'{PHONE}_{p2_PROB_NO}.keras'        # 지문 ④의 확장자 그대로
p2_PRED_COL = 'pred'                                   # ★ 지문 ③의 결과 컬럼명 그대로 (공식 예: '카테고리')
p2_KEEP_TEST_COLS = True                               # ★ 공식 규격: test 원본 + 결과 컬럼(train 과 동일 구조). 단일 컬럼만 원하면 False
print(f"제출 파일: {p2_SUBMIT_CSV} / {p2_SUBMIT_MODEL} / 결과 컬럼: {p2_PRED_COL}")

# ============================================================
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 ---
# p2_df_train = pd.read_csv('train.csv')
# p2_df_test  = pd.read_csv('test.csv')
# p2_TEXT_COL = 'text'            # ★ 텍스트 컬럼명
# p2_TARGET   = 'label'           # ★ 타겟 컬럼명
# p2_target_names = None          # ★ 클래스 이름 리스트 (있으면)

# --- 연습용: NSMC 네이버 영화 감성 (2클래스, raw TSV — HF Hub 불필요) ---
# 시험 실데이터는 CSV 로 제공되므로 연습 로더는 버전 영향 없는 raw TSV 를 씁니다.
_NSMC = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings'
p2_df_train = pd.read_csv(f'{_NSMC}_train.txt', sep='	').dropna().sample(30000, random_state=RANDOM_SEED).reset_index(drop=True)
p2_df_test = pd.read_csv(f'{_NSMC}_test.txt', sep='	').dropna().sample(6000, random_state=RANDOM_SEED).reset_index(drop=True)
p2_target_names = ['부정', '긍정']
p2_TEXT_COL, p2_TARGET = 'document', 'label'

# --- (대안) KLUE-YNAT 7클래스: datasets 버전에 따라 로딩법이 다름 ---
# 최신 datasets 에서 load_dataset('klue','ynat') 가 HfUriError 로 깨지면 아래처럼:
#   from datasets import load_dataset
#   klue = load_dataset('klue/ynat')                       # 또는 trust_remote_code=True
#   p2_df_train = pd.DataFrame({'text': klue['train']['title'], 'label': klue['train']['label']})
#   p2_df_test  = pd.DataFrame({'text': klue['validation']['title'], 'label': klue['validation']['label']})
#   p2_target_names = ['IT과학','경제','사회','생활문화','세계','스포츠','정치']
#   p2_TEXT_COL, p2_TARGET = 'text', 'label'

p2_NUM_CLASSES = p2_df_train[p2_TARGET].nunique()
p2_label_map = {i: n for i, n in enumerate(p2_target_names)} if p2_target_names else None

print(f"Train {p2_df_train.shape} / Test {p2_df_test.shape}")
print(f"클래스 {p2_NUM_CLASSES}개")
print(p2_df_train[p2_TARGET].value_counts().sort_index())
p2_df_train.head()

# ============================================================
# 2. EDA — 텍스트 길이가 p2_MAX_LENGTH 를 정한다
# ============================================================
# %%
p2_lengths = p2_df_train[p2_TEXT_COL].astype(str).apply(len)
print(p2_lengths.describe())
print(f"95 퍼센타일: {p2_lengths.quantile(0.95):.0f}자")
print(f"결측: {p2_df_train[p2_TEXT_COL].isnull().sum()}")

# ============================================================
# 3. 텍스트 정제
# ============================================================
# %%
# 단순함을 위해 방법 A(TF-IDF)와 B(BERT)가 같은 정제를 공유합니다.
# 원칙은 "TF-IDF 는 세게, BERT 는 약하게"(문장부호도 문맥 정보) —
# B 점수를 더 짜내고 싶을 때만 특수문자 제거 줄을 빼고 다시 돌리세요.
def p2_clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)                 # HTML 태그
    text = re.sub(r'http\S+|www\.\S+', ' ', text)        # URL
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)     # 특수문자
    text = re.sub(r'\s+', ' ', text).strip()             # 다중 공백
    return text

p2_df_train[p2_TEXT_COL] = p2_df_train[p2_TEXT_COL].apply(p2_clean_text)
p2_df_test_orig = p2_df_test.copy()   # ★ 제출은 원본 컬럼 구조로 — 전처리 전 보존
p2_df_test[p2_TEXT_COL] = p2_df_test[p2_TEXT_COL].apply(p2_clean_text)

# 정제 후 빈 문자열이 생겼는지 확인 (벡터화 단계에서 조용히 문제를 일으킴)
p2_empty_train = (p2_df_train[p2_TEXT_COL].str.strip() == '').sum()
print(f"정제 후 빈 텍스트: train {p2_empty_train}건")
if p2_empty_train:
    p2_df_train = p2_df_train[p2_df_train[p2_TEXT_COL].str.strip() != ''].reset_index(drop=True)

# --- Train → Train/Val 분할 (방법 A, B 가 같은 분할을 공유) ---
p2_df_tr, p2_df_val = train_test_split(
    p2_df_train, test_size=0.2, random_state=RANDOM_SEED, stratify=p2_df_train[p2_TARGET]
)
p2_df_tr = p2_df_tr.reset_index(drop=True)
p2_df_val = p2_df_val.reset_index(drop=True)
print(f"내부: Train {len(p2_df_tr)} / Val {len(p2_df_val)}   제출 대상: Test {len(p2_df_test)}")

# ============================================================
# 방법 A — TF-IDF + ML   (먼저 이걸로 submission 을 확보한다)
# ============================================================
# %%
# max_features : 상위 N개 피처만 (메모리/속도)
# ngram_range  : (1,2) 면 단어 + 2단어 조합. "좋지 않다" 같은 표현을 잡는다
# min_df / max_df : 너무 희귀하거나 너무 흔한 단어 제거
# sublinear_tf : TF 에 log 스케일. 긴 문서의 과대평가 완화
#
# ★ 한국어라면 char_wb 쪽을 쓰세요.
#   기본 토큰화는 공백 기준이라 "영화가/영화는/영화를" 이 전부 다른 피처로 흩어집니다.
#   문자 n-gram 은 조사/어미가 붙어도 "재미없" 같은 조각을 잡아냅니다.
#   NSMC 3만 건 실측: 단어(1,2) 0.781 -> 문자(2,4) 0.847 (+6.6%p, 벡터화 시간 3배)
p2_USE_CHAR_NGRAM = True        # ★ 한국어면 True, 영어면 False

if p2_USE_CHAR_NGRAM:
    p2_tfidf = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=50000, min_df=2, sublinear_tf=True,
    )
else:
    p2_tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True,
    )

p2_X_train = p2_tfidf.fit_transform(p2_df_tr[p2_TEXT_COL])      # fit 은 train 에만
p2_X_val = p2_tfidf.transform(p2_df_val[p2_TEXT_COL])
p2_X_test = p2_tfidf.transform(p2_df_test[p2_TEXT_COL])
p2_y_train, p2_y_val = p2_df_tr[p2_TARGET], p2_df_val[p2_TARGET]
print(f"TF-IDF 행렬: Train {p2_X_train.shape} / Val {p2_X_val.shape}")

# %%
# 고차원 희소 행렬에서는 선형 모델이 강하다. RandomForest 는 느리고 성능도 밀려 제외.
p2_ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'LinearSVC': LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
    'MultinomialNB': MultinomialNB(alpha=0.1),
}

p2_results, p2_best_acc, p2_best_ml_name, p2_best_ml_model = {}, 0, '', None
for p2_name, p2_model in p2_ml_models.items():
    p2_model.fit(p2_X_train, p2_y_train)
    p2_y_pred = p2_model.predict(p2_X_val)
    p2_acc = accuracy_score(p2_y_val, p2_y_pred)
    p2_f1 = f1_score(p2_y_val, p2_y_pred, average='macro')
    p2_cv = cross_val_score(p2_model, p2_X_train, p2_y_train, cv=5, scoring='accuracy').mean()
    p2_results[p2_name] = {'Accuracy': p2_acc, 'F1_macro': p2_f1, 'CV_Acc': p2_cv}
    print(f"[{p2_name}] Acc={p2_acc:.4f} F1={p2_f1:.4f} CV={p2_cv:.4f}")
    if p2_acc > p2_best_acc:
        p2_best_acc, p2_best_ml_name, p2_best_ml_model = p2_acc, p2_name, p2_model

p2_results_df = pd.DataFrame(p2_results).T.sort_values('Accuracy', ascending=False)
print("\n[모델 비교]")
print(p2_results_df)
print(f"\n1위: {p2_best_ml_name}")

# %%
p2_y_pred_val = p2_best_ml_model.predict(p2_X_val)
p2_acc_final = accuracy_score(p2_y_val, p2_y_pred_val)
p2_f1_final = f1_score(p2_y_val, p2_y_pred_val, average='macro')
print(f"Accuracy {p2_acc_final:.4f} / F1(macro) {p2_f1_final:.4f}")
print(classification_report(p2_y_val, p2_y_pred_val, target_names=p2_target_names))

# Acc 는 높은데 F1 이 낮으면 다수 클래스만 맞히고 있다는 뜻
if p2_acc_final - p2_f1_final > 0.1:
    print("Accuracy >> F1 : 클래스 불균형 -> class_weight='balanced' 검토")

p2_grade = next((g for t, g in [(0.95, '매우 우수'), (0.90, '우수'),
                             (0.85, '양호'), (0.80, '보통')] if p2_acc_final >= t), '미흡')
print(f"-> {p2_grade}")

# %%
# --- 제출 파일 (방법 A) ★ ---
p2_y_pred_test = p2_best_ml_model.predict(p2_X_test)

# 지문 ③: 값은 지문이 지정한 label 로 (예: 0/1 이 아니라 '긍정'/'부정') — 숫자 그대로 내면 0점
if p2_label_map:
    p2_pred_out = pd.Series(p2_y_pred_test).map(p2_label_map)
else:
    p2_pred_out = pd.Series(p2_y_pred_test)
# ★ 제출 규격(공식 샘플문항): test 원본을 그대로 두고 결과 컬럼을 '마지막에' 추가
#    → 훈련셋과 동일한 컬럼 구조. 값은 복원된 label(숫자 그대로면 0점).
if p2_KEEP_TEST_COLS:
    p2_submit_df = p2_df_test_orig.drop(columns=[c for c in [p2_TARGET] if c in p2_df_test_orig.columns])
    p2_submit_df[p2_PRED_COL] = pd.Series(p2_pred_out).values
else:
    p2_submit_df = pd.DataFrame({p2_PRED_COL: p2_pred_out})

if p2_TARGET in p2_df_test.columns:   # 연습용에서만 (시험 test 엔 정답 없음)
    print(f"[연습 확인] Acc={accuracy_score(p2_df_test[p2_TARGET], p2_y_pred_test):.4f} "
          f"F1={f1_score(p2_df_test[p2_TARGET], p2_y_pred_test, average='macro'):.4f}")

p2_submit_df.to_csv(p2_SUBMIT_CSV, index=False, encoding='utf-8-sig')  # 한글 라벨 안전
print(f"[저장] {p2_SUBMIT_CSV} ({len(p2_submit_df)}건) 컬럼 {list(p2_submit_df.columns)}")

import joblib, os
# 벡터라이저와 모델을 함께 담아야 나중에 그대로 예측할 수 있습니다
joblib.dump({'model': p2_best_ml_model, 'vectorizer': p2_tfidf}, p2_SUBMIT_MODEL)
print(f"모델 저장 완료: {p2_SUBMIT_MODEL}")

# %%
# --- 제출물 검증 ★ (check_submission 은 세션 시작 셀에서 정의) ---
# label 미복원(0점) 방지: 예측값이 train 타겟의 label 집합 안에 있는지까지 확인
p2_allowed = p2_target_names if p2_label_map else p2_df_train[p2_TARGET].unique()
try:
    check_submission(p2_SUBMIT_CSV, n_rows=len(p2_df_test),
                     required_cols=[p2_PRED_COL],
                     label_col=p2_PRED_COL, allowed_labels=p2_allowed,
                     model_path=p2_SUBMIT_MODEL)
except NameError:
    p2__sub = pd.read_csv(p2_SUBMIT_CSV)
    assert len(p2__sub) == len(p2_df_test), "행 수 불일치!"
    assert not p2__sub[p2_PRED_COL].isnull().any(), "예측 컬럼에 결측!"
    assert set(p2__sub[p2_PRED_COL]) <= set(p2_allowed), "label 미복원 의심 — 0점 위험!"
    print("최소 점검 통과 (전체 검증은 세션 시작 셀의 check_submission)")

# %%
# --- 눈 검증 ★ 원문과 예측을 나란히 출력해 직접 읽어 본다 ---
# 구조 검증이 통과해도 내용이 엉뚱할 수 있습니다.
# 무작위 10건을 읽고 예측이 상식적으로 맞는지 확인하세요.
# (전부 한 클래스로 쏠려 있거나, 명백한 문장을 반대로 찍고 있으면 제출 전에 원인부터)
p2_rng = np.random.RandomState(RANDOM_SEED)
for p2__i in p2_rng.choice(len(p2_df_test), min(10, len(p2_df_test)), replace=False):
    print(f"[예측: {p2_submit_df[p2_PRED_COL].iloc[p2__i]}]  "
          f"{str(p2_df_test[p2_TEXT_COL].iloc[p2__i])[:60]}")

# %% [markdown]
# ---
# ## 여기까지가 안전 지대입니다.
# 점수가 목표에 도달했으면 멈추고 다음 문제로 가세요.
# 0.85 미만이고 30분 이상 남았을 때만 아래 방법 B 를 시도합니다.
# ---

# ============================================================
# 방법 B — BERT Fine-tuning  (GPU 필요)
# ============================================================
# %%
# ── 시작 전 확인 ──
# transformers 의 TF BERT 는 Keras 3(요즘 기본)를 지원하지 않습니다.
# 첫 셀에서 !pip install -q tf-keras "transformers>=4.40,<5" 를 실행했다면
# 00 세션 시작 셀이 자동으로 옛 Keras 로 전환해 두었습니다. 아래 assert 가 그걸 확인합니다.
import tensorflow as tf

# 옛 Keras 로 전환되면 __version__ 속성이 없으므로 '3으로 시작하지 않으면 통과'로 판정
assert not str(getattr(tf.keras, '__version__', '2')).startswith('3'), (
    "Keras 3 로 동작 중이라 BERT(TF)를 쓸 수 없습니다. "
    "!pip install -q tf-keras → 런타임 재시작 → 00 세션 시작 셀부터 다시 실행하세요. "
    "(시간이 없으면 방법 A 제출물로 충분합니다)"
)

from transformers import BertTokenizerFast, TFBertForSequenceClassification

tf.random.set_seed(RANDOM_SEED)
# GPU 는 세션 시작 셀에서 확인합니다. 세션이 튕겼다면 그 셀부터 다시 실행하세요.

# ★ 데이터 언어에 맞춰 선택할 것. 한국어에 영어 모델을 쓰면 성능이 무너진다.
p2_BERT_MODEL_NAME = 'klue/bert-base'        # 한국어
# p2_BERT_MODEL_NAME = 'bert-base-uncased'   # 영어
p2_MAX_LENGTH = 128        # 위 EDA 의 95 퍼센타일을 참고. 늘리면 시간도 비례해 늘어남
p2_BATCH_SIZE = 32         # OOM 나면 16 → 8
p2_EPOCHS = 5              # EarlyStopping 이 알아서 멈춘다
p2_LEARNING_RATE = 2e-5    # BERT 표준. 1e-5 ~ 5e-5 를 벗어나면 학습이 망가진다

p2_X_train_text = p2_df_tr[p2_TEXT_COL].tolist()
p2_X_val_text = p2_df_val[p2_TEXT_COL].tolist()
p2_X_test_text = p2_df_test[p2_TEXT_COL].tolist()
p2_y_train_b = p2_df_tr[p2_TARGET].tolist()
p2_y_val_b = p2_df_val[p2_TARGET].tolist()

# ★ T4 시간 안전장치 (실측) — 에폭당: 6천건 2.7분 / 3만건 13분 / 6만건 27분.
#   6만 건을 3에폭 돌리면 80분 — 한 문제에 이만큼은 치명적입니다.
#   train 이 3만 건을 넘으면 아래를 풀어 샘플링하세요.
# if len(p2_X_train_text) > 30000:
#     p2_df_tr = p2_df_tr.sample(30000, random_state=RANDOM_SEED).reset_index(drop=True)
#     p2_X_train_text = p2_df_tr[p2_TEXT_COL].tolist()
#     p2_y_train_b = p2_df_tr[p2_TARGET].tolist()
print(f"BERT 학습량: {len(p2_X_train_text):,}건 — T4 에폭당 약 {len(p2_X_train_text)/6000*2.7:.0f}분 (3만+ 이면 샘플링)")

# %%
p2_tokenizer = BertTokenizerFast.from_pretrained(p2_BERT_MODEL_NAME)
p2_train_enc = p2_tokenizer(p2_X_train_text, truncation=True, padding=True,
                      max_length=p2_MAX_LENGTH, return_tensors='tf')
p2_val_enc = p2_tokenizer(p2_X_val_text, truncation=True, padding=True,
                    max_length=p2_MAX_LENGTH, return_tensors='tf')
print(f"토크나이징 완료: {p2_train_enc['input_ids'].shape}")

p2_train_ds = (tf.data.Dataset.from_tensor_slices((dict(p2_train_enc), p2_y_train_b))
            .shuffle(1000, seed=RANDOM_SEED).batch(p2_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
p2_val_ds = (tf.data.Dataset.from_tensor_slices((dict(p2_val_enc), p2_y_val_b))
          .batch(p2_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# %%
# from_pt=True : HF 의 한국어 모델은 PyTorch 가중치만 있는 경우가 많다
#                (변환에 torch 가 필요합니다 — Colab 에는 기본 포함)
p2_bert_model = TFBertForSequenceClassification.from_pretrained(
    p2_BERT_MODEL_NAME, num_labels=p2_NUM_CLASSES, from_pt=True
)

# from_logits=True : 이 모델은 softmax 를 거치지 않은 logits 를 출력한다. 빠뜨리면 학습이 안 됨
p2_bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p2_LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

p2_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7),
]

p2_start = time.time()
p2_history = p2_bert_model.fit(p2_train_ds, validation_data=p2_val_ds, epochs=p2_EPOCHS,
                         callbacks=p2_callbacks, verbose=1)
print(f"학습 완료: {(time.time() - p2_start) / 60:.1f}분")

# %%
p2_fig, (p2_ax1, p2_ax2) = plt.subplots(1, 2, figsize=(12, 4))
p2_ax1.plot(p2_history.history['loss'], label='Train'); p2_ax1.plot(p2_history.history['val_loss'], label='Val')
p2_ax1.set_title('Loss'); p2_ax1.legend()
p2_ax2.plot(p2_history.history['accuracy'], label='Train'); p2_ax2.plot(p2_history.history['val_accuracy'], label='Val')
p2_ax2.set_title('Accuracy'); p2_ax2.legend()
plt.tight_layout(); plt.show()

# %%
# 예측 결과는 텐서가 아니라 객체 → .logits 에서 꺼낸다
p2_predictions = p2_bert_model.predict(p2_val_ds)
p2_pred_labels = np.argmax(p2_predictions.logits, axis=1)

p2_acc_bert = accuracy_score(p2_y_val_b, p2_pred_labels)
p2_f1_bert = f1_score(p2_y_val_b, p2_pred_labels, average='macro')
print(f"[BERT] Accuracy {p2_acc_bert:.4f} / F1(macro) {p2_f1_bert:.4f}")
print(classification_report(p2_y_val_b, p2_pred_labels, target_names=p2_target_names))

print(f"\n방법 A: {p2_acc_final:.4f}  vs  방법 B: {p2_acc_bert:.4f}")
print("-> B 가 더 높을 때만 아래 셀을 실행해 submission 을 갱신하세요.")

# %%
# --- 제출 파일 갱신 (방법 B 가 더 좋을 때만) ★ ---
p2_test_enc = p2_tokenizer(p2_X_test_text, truncation=True, padding=True,
                     max_length=p2_MAX_LENGTH, return_tensors='tf')
p2_test_ds = tf.data.Dataset.from_tensor_slices(dict(p2_test_enc)).batch(p2_BATCH_SIZE)
p2_test_pred_labels = np.argmax(p2_bert_model.predict(p2_test_ds).logits, axis=1)

if p2_label_map:
    p2_bert_out = pd.Series(p2_test_pred_labels).map(p2_label_map)
else:
    p2_bert_out = pd.Series(p2_test_pred_labels)
p2_bert_submit = pd.DataFrame({p2_PRED_COL: p2_bert_out})
p2_bert_submit.to_csv(p2_SUBMIT_CSV, index=False)
print(f"[갱신] {p2_SUBMIT_CSV} ({len(p2_bert_submit)}건)")

p2_bert_model.save_pretrained(f'{PHONE}_{p2_PROB_NO}_bert')
p2_tokenizer.save_pretrained(f'{PHONE}_{p2_PROB_NO}_bert')
print(f"BERT 모델 저장 완료: {PHONE}_{p2_PROB_NO}_bert/")

# %%
# --- 제출물 재검증 ★ (B 로 덮어썼으므로 다시) ---
try:
    check_submission(p2_SUBMIT_CSV, n_rows=len(p2_df_test),
                     required_cols=[p2_PRED_COL], label_col=p2_PRED_COL,
                     allowed_labels=p2_allowed)
except NameError:
    print("check_submission 미정의 — 세션 시작 셀을 먼저 실행하세요")

# 눈 검증 — B 결과도 직접 읽어 본다
p2_rng_b = np.random.RandomState(RANDOM_SEED)
for p2__i in p2_rng_b.choice(len(p2_df_test), min(10, len(p2_df_test)), replace=False):
    print(f"[예측: {p2_bert_submit[p2_PRED_COL].iloc[p2__i]}]  "
          f"{str(p2_df_test[p2_TEXT_COL].iloc[p2__i])[:60]}")
