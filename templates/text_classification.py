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
# ── 변수 이름 규칙: 왜 p1_ 이 붙어 있나 ──
# 시험은 주피터 커널 하나에서 여러 문제를 풉니다. 전역 변수는 커널 전체가 공유하므로,
# 접두사가 없으면 2번 문제의 X_train 이 1번 문제의 X_train 을 그대로 덮어씁니다.
# 1번 문제로 돌아와 셀 하나만 다시 돌리는 순간 조용히 틀린 결과가 나옵니다.
#
#   2번 문제에 쓸 때 (Colab: 편집 > 찾기 및 바꾸기, Ctrl+H)
#     p1_  ->  p2_        전체 바꾸기
#     p2_PROB_NO = 2      숫자도 함께 수정   ← 이 한 줄을 가장 자주 놓칩니다
#
# PHONE, RANDOM_SEED 는 세션 공용이라 접두사가 없습니다.

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
# 0. 문제 설정  ★ 이 문제에서 바꾸는 것은 p1_PROB_NO 한 줄뿐
# ============================================================
# GPU 확인과 PHONE 은 세션 시작 셀(00_session_start.py)에서 합니다. 여기서 다시 하지 않습니다.
# 단, 세션이 튕기면 GPU 연결이 끊기고 변수가 전부 사라지므로 그 셀부터 다시 실행해야 합니다.
p1_PROB_NO = 1                # ★ 문제 번호

# 세션 시작 셀을 실행했다면 PHONE 은 이미 정의돼 있습니다.
# 이 파일만 단독으로 연습할 때를 위한 기본값입니다.
try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

p1_SUBMIT_CSV = f'{PHONE}_{p1_PROB_NO}.csv'
p1_SUBMIT_MODEL = f'{PHONE}_{p1_PROB_NO}.pkl'      # sklearn 계열
p1_SUBMIT_KERAS = f'{PHONE}_{p1_PROB_NO}.keras'    # 딥러닝 (문제가 형식을 지정하면 그쪽을)
print(f"제출 파일: {p1_SUBMIT_CSV} / {p1_SUBMIT_MODEL}")

# ============================================================
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 ---
# p1_df_train = pd.read_csv('train.csv')
# p1_df_test  = pd.read_csv('test.csv')
# p1_TEXT_COL = 'text'            # ★ 텍스트 컬럼명
# p1_TARGET   = 'label'           # ★ 타겟 컬럼명
# p1_target_names = None          # ★ 클래스 이름 리스트 (있으면)

# --- 연습용: KLUE-YNAT 뉴스 토픽 (7클래스) ---
# pip install datasets
from datasets import load_dataset
p1_klue = load_dataset('klue', 'ynat')
p1_df_train = pd.DataFrame({'text': p1_klue['train']['title'], 'label': p1_klue['train']['label']})
p1_df_test = pd.DataFrame({'text': p1_klue['validation']['title'], 'label': p1_klue['validation']['label']})
p1_target_names = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
p1_TEXT_COL, p1_TARGET = 'text', 'label'

p1_NUM_CLASSES = p1_df_train[p1_TARGET].nunique()
p1_label_map = {i: n for i, n in enumerate(p1_target_names)} if p1_target_names else None

print(f"Train {p1_df_train.shape} / Test {p1_df_test.shape}")
print(f"클래스 {p1_NUM_CLASSES}개")
print(p1_df_train[p1_TARGET].value_counts().sort_index())
p1_df_train.head()

# ============================================================
# 2. EDA — 텍스트 길이가 p1_MAX_LENGTH 를 정한다
# ============================================================
# %%
p1_lengths = p1_df_train[p1_TEXT_COL].astype(str).apply(len)
print(p1_lengths.describe())
print(f"95 퍼센타일: {p1_lengths.quantile(0.95):.0f}자")
print(f"결측: {p1_df_train[p1_TEXT_COL].isnull().sum()}")

# ============================================================
# 3. 텍스트 정제
# ============================================================
# %%
# TF-IDF 용은 세게, BERT 용은 약하게(문장부호가 문맥 정보이므로)
def p1_clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)                 # HTML 태그
    text = re.sub(r'http\S+|www\.\S+', ' ', text)        # URL
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)     # 특수문자
    text = re.sub(r'\s+', ' ', text).strip()             # 다중 공백
    return text

p1_df_train[p1_TEXT_COL] = p1_df_train[p1_TEXT_COL].apply(p1_clean_text)
p1_df_test[p1_TEXT_COL] = p1_df_test[p1_TEXT_COL].apply(p1_clean_text)

# 정제 후 빈 문자열이 생겼는지 확인 (벡터화 단계에서 조용히 문제를 일으킴)
p1_empty_train = (p1_df_train[p1_TEXT_COL].str.strip() == '').sum()
print(f"정제 후 빈 텍스트: train {p1_empty_train}건")
if p1_empty_train:
    p1_df_train = p1_df_train[p1_df_train[p1_TEXT_COL].str.strip() != ''].reset_index(drop=True)

# --- Train → Train/Val 분할 (방법 A, B 가 같은 분할을 공유) ---
p1_df_tr, p1_df_val = train_test_split(
    p1_df_train, test_size=0.2, random_state=RANDOM_SEED, stratify=p1_df_train[p1_TARGET]
)
p1_df_tr = p1_df_tr.reset_index(drop=True)
p1_df_val = p1_df_val.reset_index(drop=True)
print(f"Train {len(p1_df_tr)} / Val {len(p1_df_val)} / Test {len(p1_df_test)}")

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
p1_USE_CHAR_NGRAM = True        # ★ 한국어면 True, 영어면 False

if p1_USE_CHAR_NGRAM:
    p1_tfidf = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=50000, min_df=2, sublinear_tf=True,
    )
else:
    p1_tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True,
    )

p1_X_train = p1_tfidf.fit_transform(p1_df_tr[p1_TEXT_COL])      # fit 은 train 에만
p1_X_val = p1_tfidf.transform(p1_df_val[p1_TEXT_COL])
p1_X_test = p1_tfidf.transform(p1_df_test[p1_TEXT_COL])
p1_y_train, p1_y_val = p1_df_tr[p1_TARGET], p1_df_val[p1_TARGET]
print(f"TF-IDF 행렬: Train {p1_X_train.shape} / Val {p1_X_val.shape}")

# %%
# 고차원 희소 행렬에서는 선형 모델이 강하다. RandomForest 는 느리고 성능도 밀려 제외.
p1_ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'LinearSVC': LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
    'MultinomialNB': MultinomialNB(alpha=0.1),
}

p1_results, p1_best_acc, p1_best_ml_name, p1_best_ml_model = {}, 0, '', None
for p1_name, p1_model in p1_ml_models.items():
    p1_model.fit(p1_X_train, p1_y_train)
    p1_y_pred = p1_model.predict(p1_X_val)
    p1_acc = accuracy_score(p1_y_val, p1_y_pred)
    p1_f1 = f1_score(p1_y_val, p1_y_pred, average='macro')
    p1_cv = cross_val_score(p1_model, p1_X_train, p1_y_train, cv=5, scoring='accuracy').mean()
    p1_results[p1_name] = {'Accuracy': p1_acc, 'F1_macro': p1_f1, 'CV_Acc': p1_cv}
    print(f"[{p1_name}] Acc={p1_acc:.4f} F1={p1_f1:.4f} CV={p1_cv:.4f}")
    if p1_acc > p1_best_acc:
        p1_best_acc, p1_best_ml_name, p1_best_ml_model = p1_acc, p1_name, p1_model

p1_results_df = pd.DataFrame(p1_results).T.sort_values('Accuracy', ascending=False)
print("\n[모델 비교]")
print(p1_results_df)
print(f"\n1위: {p1_best_ml_name}")

# %%
p1_y_pred_val = p1_best_ml_model.predict(p1_X_val)
p1_acc_final = accuracy_score(p1_y_val, p1_y_pred_val)
p1_f1_final = f1_score(p1_y_val, p1_y_pred_val, average='macro')
print(f"Accuracy {p1_acc_final:.4f} / F1(macro) {p1_f1_final:.4f}")
print(classification_report(p1_y_val, p1_y_pred_val, target_names=p1_target_names))

# Acc 는 높은데 F1 이 낮으면 다수 클래스만 맞히고 있다는 뜻
if p1_acc_final - p1_f1_final > 0.1:
    print("Accuracy >> F1 : 클래스 불균형 -> class_weight='balanced' 검토")

p1_grade = next((g for t, g in [(0.95, '매우 우수'), (0.90, '우수'),
                             (0.85, '양호'), (0.80, '보통')] if p1_acc_final >= t), '미흡')
print(f"-> {p1_grade}")

# %%
# --- 제출 파일 (방법 A) ★ ---
p1_y_pred_test = p1_best_ml_model.predict(p1_X_test)
p1_submit_df = p1_df_test.copy()
p1_submit_df['predicted'] = p1_y_pred_test
# ★ 문제가 label명('긍정'/'부정' 등)을 요구하면 반드시 매핑 — 숫자 그대로 내면 0점
if p1_label_map:
    p1_submit_df['predicted_name'] = p1_submit_df['predicted'].map(p1_label_map)

if p1_TARGET in p1_df_test.columns:
    print(f"[Test] Acc={accuracy_score(p1_df_test[p1_TARGET], p1_y_pred_test):.4f} "
          f"F1={f1_score(p1_df_test[p1_TARGET], p1_y_pred_test, average='macro'):.4f}")

p1_submit_df.to_csv(p1_SUBMIT_CSV, index=False)
print(f"[저장] {p1_SUBMIT_CSV} ({len(p1_submit_df)}건)")

import joblib, os
# 벡터라이저와 모델을 함께 담아야 나중에 그대로 예측할 수 있습니다
joblib.dump({'model': p1_best_ml_model, 'vectorizer': p1_tfidf}, p1_SUBMIT_MODEL)
print(f"모델 저장 완료: {p1_SUBMIT_MODEL}")

# %%
# --- 제출물 검증 ★ (check_submission 은 세션 시작 셀에서 정의) ---
# label 미복원(0점) 방지: 예측값이 train 타겟의 label 집합 안에 있는지까지 확인
p1_allowed = p1_df_train[p1_TARGET].unique()
try:
    check_submission(p1_SUBMIT_CSV, n_rows=len(p1_df_test),
                     label_col='predicted', allowed_labels=p1_allowed,
                     model_path=p1_SUBMIT_MODEL)
except NameError:
    p1__sub = pd.read_csv(p1_SUBMIT_CSV)
    assert len(p1__sub) == len(p1_df_test), "행 수 불일치!"
    assert not p1__sub['predicted'].isnull().any(), "예측 컬럼에 결측!"
    assert set(p1__sub['predicted']) <= set(p1_allowed), "label 미복원 의심 — 0점 위험!"
    print("최소 점검 통과 (전체 검증은 세션 시작 셀의 check_submission)")

# %%
# --- 눈 검증 ★ 원문과 예측을 나란히 출력해 직접 읽어 본다 ---
# 구조 검증이 통과해도 내용이 엉뚱할 수 있습니다.
# 무작위 10건을 읽고 예측이 상식적으로 맞는지 확인하세요.
# (전부 한 클래스로 쏠려 있거나, 명백한 문장을 반대로 찍고 있으면 제출 전에 원인부터)
p1_eyeball = p1_submit_df.sample(min(10, len(p1_submit_df)), random_state=RANDOM_SEED)
for p1__i, p1_row in p1_eyeball.iterrows():
    p1_pred_str = p1_row['predicted_name'] if 'predicted_name' in p1_row else p1_row['predicted']
    print(f"[예측: {p1_pred_str}]  {str(p1_row[p1_TEXT_COL])[:60]}")

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
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification

tf.random.set_seed(RANDOM_SEED)
# GPU 는 세션 시작 셀에서 확인합니다. 세션이 튕겼다면 그 셀부터 다시 실행하세요.

# ★ 데이터 언어에 맞춰 선택할 것. 한국어에 영어 모델을 쓰면 성능이 무너진다.
p1_BERT_MODEL_NAME = 'klue/bert-base'        # 한국어
# p1_BERT_MODEL_NAME = 'bert-base-uncased'   # 영어
p1_MAX_LENGTH = 128        # 위 EDA 의 95 퍼센타일을 참고. 늘리면 시간도 비례해 늘어남
p1_BATCH_SIZE = 32         # OOM 나면 16 → 8
p1_EPOCHS = 5              # EarlyStopping 이 알아서 멈춘다
p1_LEARNING_RATE = 2e-5    # BERT 표준. 1e-5 ~ 5e-5 를 벗어나면 학습이 망가진다

p1_X_train_text = p1_df_tr[p1_TEXT_COL].tolist()
p1_X_val_text = p1_df_val[p1_TEXT_COL].tolist()
p1_X_test_text = p1_df_test[p1_TEXT_COL].tolist()
p1_y_train_b = p1_df_tr[p1_TARGET].tolist()
p1_y_val_b = p1_df_val[p1_TARGET].tolist()

# %%
p1_tokenizer = BertTokenizerFast.from_pretrained(p1_BERT_MODEL_NAME)
p1_train_enc = p1_tokenizer(p1_X_train_text, truncation=True, padding=True,
                      max_length=p1_MAX_LENGTH, return_tensors='tf')
p1_val_enc = p1_tokenizer(p1_X_val_text, truncation=True, padding=True,
                    max_length=p1_MAX_LENGTH, return_tensors='tf')
print(f"토크나이징 완료: {p1_train_enc['input_ids'].shape}")

p1_train_ds = (tf.data.Dataset.from_tensor_slices((dict(p1_train_enc), p1_y_train_b))
            .shuffle(1000, seed=RANDOM_SEED).batch(p1_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
p1_val_ds = (tf.data.Dataset.from_tensor_slices((dict(p1_val_enc), p1_y_val_b))
          .batch(p1_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# %%
# from_pt=True : HF 의 한국어 모델은 PyTorch 가중치만 있는 경우가 많다
p1_bert_model = TFBertForSequenceClassification.from_pretrained(
    p1_BERT_MODEL_NAME, num_labels=p1_NUM_CLASSES, from_pt=True
)

# from_logits=True : 이 모델은 softmax 를 거치지 않은 logits 를 출력한다. 빠뜨리면 학습이 안 됨
p1_bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p1_LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

p1_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7),
]

p1_start = time.time()
p1_history = p1_bert_model.fit(p1_train_ds, validation_data=p1_val_ds, epochs=p1_EPOCHS,
                         callbacks=p1_callbacks, verbose=1)
print(f"학습 완료: {(time.time() - p1_start) / 60:.1f}분")

# %%
p1_fig, (p1_ax1, p1_ax2) = plt.subplots(1, 2, figsize=(12, 4))
p1_ax1.plot(p1_history.history['loss'], label='Train'); p1_ax1.plot(p1_history.history['val_loss'], label='Val')
p1_ax1.set_title('Loss'); p1_ax1.legend()
p1_ax2.plot(p1_history.history['accuracy'], label='Train'); p1_ax2.plot(p1_history.history['val_accuracy'], label='Val')
p1_ax2.set_title('Accuracy'); p1_ax2.legend()
plt.tight_layout(); plt.show()

# %%
# 예측 결과는 텐서가 아니라 객체 → .logits 에서 꺼낸다
p1_predictions = p1_bert_model.predict(p1_val_ds)
p1_pred_labels = np.argmax(p1_predictions.logits, axis=1)

p1_acc_bert = accuracy_score(p1_y_val_b, p1_pred_labels)
p1_f1_bert = f1_score(p1_y_val_b, p1_pred_labels, average='macro')
print(f"[BERT] Accuracy {p1_acc_bert:.4f} / F1(macro) {p1_f1_bert:.4f}")
print(classification_report(p1_y_val_b, p1_pred_labels, target_names=p1_target_names))

print(f"\n방법 A: {p1_acc_final:.4f}  vs  방법 B: {p1_acc_bert:.4f}")
print("-> B 가 더 높을 때만 아래 셀을 실행해 submission 을 갱신하세요.")

# %%
# --- 제출 파일 갱신 (방법 B 가 더 좋을 때만) ★ ---
p1_test_enc = p1_tokenizer(p1_X_test_text, truncation=True, padding=True,
                     max_length=p1_MAX_LENGTH, return_tensors='tf')
p1_test_ds = tf.data.Dataset.from_tensor_slices(dict(p1_test_enc)).batch(p1_BATCH_SIZE)
p1_test_pred_labels = np.argmax(p1_bert_model.predict(p1_test_ds).logits, axis=1)

p1_bert_submit = p1_df_test.copy()
p1_bert_submit['predicted'] = p1_test_pred_labels
if p1_label_map:
    p1_bert_submit['predicted_name'] = p1_bert_submit['predicted'].map(p1_label_map)

p1_bert_submit.to_csv(p1_SUBMIT_CSV, index=False)
print(f"[갱신] {p1_SUBMIT_CSV} ({len(p1_bert_submit)}건)")

p1_bert_model.save_pretrained(f'{PHONE}_{p1_PROB_NO}_bert')
p1_tokenizer.save_pretrained(f'{PHONE}_{p1_PROB_NO}_bert')
print(f"BERT 모델 저장 완료: {PHONE}_{p1_PROB_NO}_bert/")

# %%
# --- 제출물 재검증 ★ (B 로 덮어썼으므로 다시) ---
try:
    check_submission(p1_SUBMIT_CSV, n_rows=len(p1_df_test),
                     label_col='predicted',
                     allowed_labels=p1_df_train[p1_TARGET].unique())
except NameError:
    print("check_submission 미정의 — 세션 시작 셀을 먼저 실행하세요")

# 눈 검증 — B 결과도 직접 읽어 본다
p1_eyeball_b = p1_bert_submit.sample(min(10, len(p1_bert_submit)), random_state=RANDOM_SEED)
for p1__i, p1_row in p1_eyeball_b.iterrows():
    p1_pred_str = p1_row['predicted_name'] if 'predicted_name' in p1_row else p1_row['predicted']
    print(f"[예측: {p1_pred_str}]  {str(p1_row[p1_TEXT_COL])[:60]}")
