# %% [markdown]
# # Text 분류 템플릿 (TF-IDF 우선 · BERT 선택)
#
# 교재: docs/02-text.md
#
# 전략: 방법 A 를 먼저 돌려 submission.csv 를 확보한 뒤,
#       시간이 남고 점수가 부족할 때만 방법 B 로 갱신합니다.
#
#   방법 A  TF-IDF + ML       1~5분   CPU     실패 위험 거의 없음
#   방법 B  BERT Fine-tuning  20~40분 GPU     성능은 높지만 실패 위험 있음
#
# ★ 표시가 있는 곳만 문제에 맞게 고치면 됩니다.

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
# 0. 문제 설정  ★ 이 문제에서 바꾸는 것은 PROB_NO 한 줄뿐
# ============================================================
# GPU 확인과 PHONE 은 세션 시작 셀(00_session_start.py)에서 한 번만 합니다.
# 여기서 다시 확인하지 않습니다.
PROB_NO = 1                # ★ 문제 번호

# 세션 시작 셀을 실행했다면 PHONE 은 이미 정의돼 있습니다.
# 이 파일만 단독으로 연습할 때를 위한 기본값입니다.
try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

SUBMIT_CSV = f'{PHONE}_{PROB_NO}.csv'
SUBMIT_MODEL = f'{PHONE}_{PROB_NO}.pkl'      # sklearn 계열
SUBMIT_KERAS = f'{PHONE}_{PROB_NO}.keras'    # 딥러닝 (문제가 형식을 지정하면 그쪽을)
print(f"제출 파일: {SUBMIT_CSV} / {SUBMIT_MODEL}")

# ============================================================
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 ---
# df_train = pd.read_csv('train.csv')
# df_test  = pd.read_csv('test.csv')
# TEXT_COL = 'text'            # ★ 텍스트 컬럼명
# TARGET   = 'label'           # ★ 타겟 컬럼명
# target_names = None          # ★ 클래스 이름 리스트 (있으면)

# --- 연습용: KLUE-YNAT 뉴스 토픽 (7클래스) ---
# pip install datasets
from datasets import load_dataset
klue = load_dataset('klue', 'ynat')
df_train = pd.DataFrame({'text': klue['train']['title'], 'label': klue['train']['label']})
df_test = pd.DataFrame({'text': klue['validation']['title'], 'label': klue['validation']['label']})
target_names = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
TEXT_COL, TARGET = 'text', 'label'

NUM_CLASSES = df_train[TARGET].nunique()
label_map = {i: n for i, n in enumerate(target_names)} if target_names else None

print(f"Train {df_train.shape} / Test {df_test.shape}")
print(f"클래스 {NUM_CLASSES}개")
print(df_train[TARGET].value_counts().sort_index())
df_train.head()

# ============================================================
# 2. EDA — 텍스트 길이가 MAX_LENGTH 를 정한다
# ============================================================
# %%
lengths = df_train[TEXT_COL].astype(str).apply(len)
print(lengths.describe())
print(f"95 퍼센타일: {lengths.quantile(0.95):.0f}자")
print(f"결측: {df_train[TEXT_COL].isnull().sum()}")

# ============================================================
# 3. 텍스트 정제
# ============================================================
# %%
# TF-IDF 용은 세게, BERT 용은 약하게(문장부호가 문맥 정보이므로)
def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)                 # HTML 태그
    text = re.sub(r'http\S+|www\.\S+', ' ', text)        # URL
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)     # 특수문자
    text = re.sub(r'\s+', ' ', text).strip()             # 다중 공백
    return text

df_train[TEXT_COL] = df_train[TEXT_COL].apply(clean_text)
df_test[TEXT_COL] = df_test[TEXT_COL].apply(clean_text)

# 정제 후 빈 문자열이 생겼는지 확인 (벡터화 단계에서 조용히 문제를 일으킴)
empty_train = (df_train[TEXT_COL].str.strip() == '').sum()
print(f"정제 후 빈 텍스트: train {empty_train}건")
if empty_train:
    df_train = df_train[df_train[TEXT_COL].str.strip() != ''].reset_index(drop=True)

# --- Train → Train/Val 분할 (방법 A, B 가 같은 분할을 공유) ---
df_tr, df_val = train_test_split(
    df_train, test_size=0.2, random_state=RANDOM_SEED, stratify=df_train[TARGET]
)
df_tr = df_tr.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
print(f"Train {len(df_tr)} / Val {len(df_val)} / Test {len(df_test)}")

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
USE_CHAR_NGRAM = True        # ★ 한국어면 True, 영어면 False

if USE_CHAR_NGRAM:
    tfidf = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=50000, min_df=2, sublinear_tf=True,
    )
else:
    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True,
    )

X_train = tfidf.fit_transform(df_tr[TEXT_COL])      # fit 은 train 에만
X_val = tfidf.transform(df_val[TEXT_COL])
X_test = tfidf.transform(df_test[TEXT_COL])
y_train, y_val = df_tr[TARGET], df_val[TARGET]
print(f"TF-IDF 행렬: Train {X_train.shape} / Val {X_val.shape}")

# %%
# 고차원 희소 행렬에서는 선형 모델이 강하다. RandomForest 는 느리고 성능도 밀려 제외.
ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'LinearSVC': LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
    'MultinomialNB': MultinomialNB(alpha=0.1),
}

results, best_acc, best_ml_name, best_ml_model = {}, 0, '', None
for name, model in ml_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    results[name] = {'Accuracy': acc, 'F1_macro': f1, 'CV_Acc': cv}
    print(f"[{name}] Acc={acc:.4f} F1={f1:.4f} CV={cv:.4f}")
    if acc > best_acc:
        best_acc, best_ml_name, best_ml_model = acc, name, model

results_df = pd.DataFrame(results).T.sort_values('Accuracy', ascending=False)
print("\n[모델 비교]")
print(results_df)
print(f"\n1위: {best_ml_name}")

# %%
y_pred_val = best_ml_model.predict(X_val)
acc_final = accuracy_score(y_val, y_pred_val)
f1_final = f1_score(y_val, y_pred_val, average='macro')
print(f"Accuracy {acc_final:.4f} / F1(macro) {f1_final:.4f}")
print(classification_report(y_val, y_pred_val, target_names=target_names))

# Acc 는 높은데 F1 이 낮으면 다수 클래스만 맞히고 있다는 뜻
if acc_final - f1_final > 0.1:
    print("Accuracy >> F1 : 클래스 불균형 -> class_weight='balanced' 검토")

grade = next((g for t, g in [(0.95, '매우 우수'), (0.90, '우수'),
                             (0.85, '양호'), (0.80, '보통')] if acc_final >= t), '미흡')
print(f"-> {grade}")

# %%
# --- 제출 파일 (방법 A) ★ ---
y_pred_test = best_ml_model.predict(X_test)
submit_df = df_test.copy()
submit_df['predicted'] = y_pred_test
if label_map:
    submit_df['predicted_name'] = submit_df['predicted'].map(label_map)

if TARGET in df_test.columns:
    print(f"[Test] Acc={accuracy_score(df_test[TARGET], y_pred_test):.4f} "
          f"F1={f1_score(df_test[TARGET], y_pred_test, average='macro'):.4f}")

submit_df.to_csv(SUBMIT_CSV, index=False)
print(f"[저장] {SUBMIT_CSV} ({len(submit_df)}건)")

import joblib, os
# 벡터라이저와 모델을 함께 담아야 나중에 그대로 예측할 수 있습니다
joblib.dump({'model': best_ml_model, 'vectorizer': tfidf}, SUBMIT_MODEL)
print(f"모델 저장 완료: {SUBMIT_MODEL}")
print("현재 폴더:", [f for f in os.listdir('.') if f.startswith(PHONE)])

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
# GPU 는 세션 시작 셀에서 이미 확인했습니다. 여기서 다시 하지 않습니다.

# ★ 데이터 언어에 맞춰 선택할 것. 한국어에 영어 모델을 쓰면 성능이 무너진다.
BERT_MODEL_NAME = 'klue/bert-base'        # 한국어
# BERT_MODEL_NAME = 'bert-base-uncased'   # 영어
MAX_LENGTH = 128        # 위 EDA 의 95 퍼센타일을 참고. 늘리면 시간도 비례해 늘어남
BATCH_SIZE = 32         # OOM 나면 16 → 8
EPOCHS = 5              # EarlyStopping 이 알아서 멈춘다
LEARNING_RATE = 2e-5    # BERT 표준. 1e-5 ~ 5e-5 를 벗어나면 학습이 망가진다

X_train_text = df_tr[TEXT_COL].tolist()
X_val_text = df_val[TEXT_COL].tolist()
X_test_text = df_test[TEXT_COL].tolist()
y_train_b = df_tr[TARGET].tolist()
y_val_b = df_val[TARGET].tolist()

# %%
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
train_enc = tokenizer(X_train_text, truncation=True, padding=True,
                      max_length=MAX_LENGTH, return_tensors='tf')
val_enc = tokenizer(X_val_text, truncation=True, padding=True,
                    max_length=MAX_LENGTH, return_tensors='tf')
print(f"토크나이징 완료: {train_enc['input_ids'].shape}")

train_ds = (tf.data.Dataset.from_tensor_slices((dict(train_enc), y_train_b))
            .shuffle(1000, seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
val_ds = (tf.data.Dataset.from_tensor_slices((dict(val_enc), y_val_b))
          .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# %%
# from_pt=True : HF 의 한국어 모델은 PyTorch 가중치만 있는 경우가 많다
bert_model = TFBertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=NUM_CLASSES, from_pt=True
)

# from_logits=True : 이 모델은 softmax 를 거치지 않은 logits 를 출력한다. 빠뜨리면 학습이 안 됨
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7),
]

start = time.time()
history = bert_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                         callbacks=callbacks, verbose=1)
print(f"학습 완료: {(time.time() - start) / 60:.1f}분")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='Train'); ax1.plot(history.history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(history.history['accuracy'], label='Train'); ax2.plot(history.history['val_accuracy'], label='Val')
ax2.set_title('Accuracy'); ax2.legend()
plt.tight_layout(); plt.show()

# %%
# 예측 결과는 텐서가 아니라 객체 → .logits 에서 꺼낸다
predictions = bert_model.predict(val_ds)
pred_labels = np.argmax(predictions.logits, axis=1)

acc_bert = accuracy_score(y_val_b, pred_labels)
f1_bert = f1_score(y_val_b, pred_labels, average='macro')
print(f"[BERT] Accuracy {acc_bert:.4f} / F1(macro) {f1_bert:.4f}")
print(classification_report(y_val_b, pred_labels, target_names=target_names))

print(f"\n방법 A: {acc_final:.4f}  vs  방법 B: {acc_bert:.4f}")
print("-> B 가 더 높을 때만 아래 셀을 실행해 submission 을 갱신하세요.")

# %%
# --- 제출 파일 갱신 (방법 B 가 더 좋을 때만) ★ ---
test_enc = tokenizer(X_test_text, truncation=True, padding=True,
                     max_length=MAX_LENGTH, return_tensors='tf')
test_ds = tf.data.Dataset.from_tensor_slices(dict(test_enc)).batch(BATCH_SIZE)
test_pred_labels = np.argmax(bert_model.predict(test_ds).logits, axis=1)

bert_submit = df_test.copy()
bert_submit['predicted'] = test_pred_labels
if label_map:
    bert_submit['predicted_name'] = bert_submit['predicted'].map(label_map)

bert_submit.to_csv(SUBMIT_CSV, index=False)
print(f"[갱신] {SUBMIT_CSV} ({len(bert_submit)}건)")

bert_model.save_pretrained(f'{PHONE}_{PROB_NO}_bert')
tokenizer.save_pretrained(f'{PHONE}_{PROB_NO}_bert')
print(f"BERT 모델 저장 완료: {PHONE}_{PROB_NO}_bert/")
