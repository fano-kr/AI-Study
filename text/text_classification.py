# %% [markdown]
# # Text - 분류 (Classification) 통합 템플릿
# AICE Professional 시험 대비
#
# NUM_CLASSES = 2 → 이진분류 (감성분석) / NUM_CLASSES >= 3 → 다중분류 (카테고리)
# 코드 변경 없이 자동 대응
#
# 두 가지 접근법:
#   - 방법 A: TF-IDF + ML (빠름, 5~10분)
#   - 방법 B: BERT Fine-tuning (높은 성능, 20~40분)
#
# ── 데이터 흐름 ──
# df_train / df_test (로드)
#   ↓ 전처리 (clean_text)
# df_train → df_tr / df_val (분할)
#   ↓ TF-IDF 또는 BERT 토크나이징
# X_train, X_val, y_train, y_val (학습/검증)
# X_test (제출용)
#   ↓ 학습 → 검증 → 제출
# model.fit(X_train) → predict(X_val) → predict(X_test) → submission.csv

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. 데이터 로드
# ============================================================
# %%
# -------------------------------------------------------
# [시험 구조]
# df_train = pd.read_csv('train.csv')  → train + val 분리 (레이블 있음)
# df_test  = pd.read_csv('test.csv')   → 최종 예측용 (레이블 없음)
# -------------------------------------------------------
# df_train = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')

# ==============================================
# [데이터셋 선택] 아래 중 하나만 주석 해제
# ==============================================
from sklearn.model_selection import train_test_split

# --- (A) 영어: 20 Newsgroups (4클래스 다중분류) ---
# from sklearn.datasets import fetch_20newsgroups
# categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.politics.misc']
# newsgroups = fetch_20newsgroups(
#     subset='all', categories=categories,
#     remove=('headers', 'footers', 'quotes'), random_state=RANDOM_SEED
# )
# df_all = pd.DataFrame({'text': newsgroups.data, 'label': newsgroups.target})
# target_names = newsgroups.target_names
# df_train, df_test = train_test_split(
#     df_all, test_size=0.2, random_state=RANDOM_SEED, stratify=df_all['label']
# )
# df_train = df_train.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)
# TEXT_COL = 'text'
# TARGET = 'label'

# --- (B) 한국어: NSMC 네이버 영화 감성분석 (2클래스 이진분류) ---
# NSMC_TRAIN = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt'
# NSMC_TEST = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt'
# df_train = pd.read_csv(NSMC_TRAIN, sep='\t').dropna()
# df_test = pd.read_csv(NSMC_TEST, sep='\t').dropna()
# target_names = ['부정', '긍정']
# TEXT_COL = 'document'
# TARGET = 'label'

# --- (C) 한국어: KLUE-YNAT 뉴스 토픽분류 (7클래스 다중분류) ---
# pip install datasets
from datasets import load_dataset
klue = load_dataset('klue', 'ynat')
ynat_names = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
df_train = pd.DataFrame({'text': klue['train']['title'], 'label': klue['train']['label']})
df_test = pd.DataFrame({'text': klue['validation']['title'], 'label': klue['validation']['label']})
target_names = ynat_names
TEXT_COL = 'text'
TARGET = 'label'
NUM_CLASSES = df_train[TARGET].nunique()

print(f"Train 크기: {df_train.shape}")
print(f"Test  크기: {df_test.shape}")
print(f"클래스 수: {NUM_CLASSES} ({'이진분류' if NUM_CLASSES == 2 else '다중분류'})")
print(f"클래스 분포 (train):\n{df_train[TARGET].value_counts().sort_index()}")
print(f"클래스 이름: {target_names}")
df_train.head()

# ============================================================
# 2. EDA (탐색적 데이터 분석)
# ============================================================
# %%
print("=" * 50)
print("[결측치 - Train]")
print(df_train.isnull().sum())

print(f"\n[텍스트 길이 통계 - Train]")
df_train['text_len'] = df_train[TEXT_COL].astype(str).apply(len)
print(df_train['text_len'].describe())

print(f"\n[빈 텍스트 수 - Train]: {(df_train[TEXT_COL].astype(str).str.strip() == '').sum()}")

# ============================================================
# 3. 텍스트 전처리
# ============================================================
# %%
# -------------------------------------------------------
# 텍스트 정제 함수
# -------------------------------------------------------
# [주요 전처리 항목]
# HTML 태그 제거          → 웹 크롤링 데이터에 포함
# URL 제거                → 분류에 불필요
# 특수문자 제거            → 노이즈 제거 (한글/영문/숫자만 유지)
# 다중 공백 정리           → 토크나이저 입력 정규화
# 소문자 변환 (영어)       → 'Apple'과 'apple'을 같은 단어로 처리
#
# [선택적 전처리 - 필요 시 추가]
# 불용어(stopwords) 제거  → from nltk.corpus import stopwords
# 형태소 분석 (한국어)     → from konlpy.tag import Okt
# 어간 추출 (영어)         → from nltk.stem import PorterStemmer

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s가-힣a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

df_train[TEXT_COL] = df_train[TEXT_COL].apply(clean_text)
df_train = df_train[df_train[TEXT_COL].str.len() > 0].reset_index(drop=True)
df_test[TEXT_COL] = df_test[TEXT_COL].apply(clean_text)
df_test = df_test[df_test[TEXT_COL].str.len() > 0].reset_index(drop=True)
print(f"전처리 후 Train: {df_train.shape}, Test: {df_test.shape}")

# ============================================================
# 방법 A: TF-IDF + 머신러닝 (빠르고 안정적)
# ============================================================
# %% [markdown]
# ## 방법 A: TF-IDF + ML

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- Train → Train/Val 분할 ---
df_tr, df_val = train_test_split(
    df_train, test_size=0.2, random_state=RANDOM_SEED, stratify=df_train[TARGET]
)
df_tr = df_tr.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
print(f"Train: {len(df_tr)}, Val: {len(df_val)}, Test: {len(df_test)}")

# -------------------------------------------------------
# TF-IDF 벡터화
# -------------------------------------------------------
# max_features: 최대 단어 수 (메모리에 맞게 조절, 보통 5000~30000)
# ngram_range: (1,1)=단어, (1,2)=단어+바이그램, (1,3)=트라이그램까지
# min_df: 최소 등장 문서 수 (너무 희귀한 단어 제거)
# max_df: 최대 등장 비율 (너무 흔한 단어 제거, 0.95=95% 이상 문서에 등장하면 제거)
# sublinear_tf: TF에 로그 스케일 적용 (긴 문서의 영향 완화)
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
# fit은 train에만, transform은 val/test에도 적용
X_train = tfidf.fit_transform(df_tr[TEXT_COL])
X_val = tfidf.transform(df_val[TEXT_COL])
y_train = df_tr[TARGET]
y_val = df_val[TARGET]
print(f"TF-IDF 행렬: Train={X_train.shape}, Val={X_val.shape}")

# -------------------------------------------------------
# 모델 학습 및 비교
# -------------------------------------------------------
ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'LinearSVC': LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
    'MultinomialNB': MultinomialNB(alpha=0.1),
    #'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
}

print("\n[TF-IDF + ML 모델 비교]")
results = {}
best_ml_acc = 0
best_ml_name = ''
best_ml_model = None

for name, model in ml_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    results[name] = {'Accuracy': acc, 'F1_macro': f1, 'CV_Acc': cv}
    print(f"  [{name}] Accuracy={acc:.4f} | F1(macro)={f1:.4f} | CV={cv:.4f}")
    if acc > best_ml_acc:
        best_ml_acc = acc
        best_ml_name = name
        best_ml_model = model

results_df = pd.DataFrame(results).T.sort_values('Accuracy', ascending=False)
print(f"\n[모델 비교 결과]")
print(results_df)
print(f"\n1위 모델: {best_ml_name}")

# %%
# --- 최적 ML 모델 상세 결과 (Val) ---
y_pred_val = best_ml_model.predict(X_val)
acc_final = accuracy_score(y_val, y_pred_val)
f1_final = f1_score(y_val, y_pred_val, average='macro')

print(f"[최종 모델 성능 - Validation]")
print(f"Accuracy:   {acc_final:.4f}")
print(f"F1 (macro): {f1_final:.4f}")
print(f"\n[분류 보고서]")
print(classification_report(y_val, y_pred_val, target_names=target_names))

# %%
# -------------------------------------------------------
# 성능 평가 기준
# -------------------------------------------------------
# [Accuracy 기준]
#   0.95 이상 : 매우 우수 (Excellent)
#   0.90 ~ 0.95: 우수 (Good)
#   0.85 ~ 0.90: 양호 (Acceptable)
#   0.80 ~ 0.85: 보통 (Moderate)
#   0.80 미만 : 미흡 (Poor)
#
# [F1 Score (macro) 기준]
#   0.90 이상 : 매우 우수
#   0.80 ~ 0.90: 우수
#   0.70 ~ 0.80: 양호
#   0.60 ~ 0.70: 보통
#   0.60 미만 : 미흡
#
# [텍스트 분류 특이사항]
#   TF-IDF + LogisticRegression/LinearSVC가 텍스트에서는 매우 강력
#   BERT는 복잡한 문맥 이해가 필요한 경우에 유리

acc_grades = [(0.95, '매우 우수'), (0.90, '우수'), (0.85, '양호'), (0.80, '보통')]
acc_grade = '미흡'
for threshold, grade in acc_grades:
    if acc_final >= threshold:
        acc_grade = grade
        break

f1_grades = [(0.90, '매우 우수'), (0.80, '우수'), (0.70, '양호'), (0.60, '보통')]
f1_grade = '미흡'
for threshold, grade in f1_grades:
    if f1_final >= threshold:
        f1_grade = grade
        break

print(f"\n[성능 평가]")
print(f"Accuracy = {acc_final:.4f} -> {acc_grade}")
print(f"F1 Score = {f1_final:.4f} -> {f1_grade}")

# ============================================================
# 결과 CSV 파일 생성 (방법 A)
# ============================================================
# %%
# --- 모델 비교 결과 저장 ---
results_df.to_csv('text_model_comparison.csv')
print("[저장] text_model_comparison.csv")

# --- Validation 결과 CSV ---
val_result_df = df_val[[TEXT_COL, TARGET]].copy()
val_result_df['predicted'] = y_pred_val
val_result_df['correct'] = (y_val.values == y_pred_val).astype(int)
if target_names:
    label_map = {i: name for i, name in enumerate(target_names)}
    val_result_df['actual_name'] = val_result_df[TARGET].map(label_map)
    val_result_df['predicted_name'] = val_result_df['predicted'].map(label_map)

val_result_df.to_csv('text_classification_val_result.csv', index=False)
print(f"\n[저장] text_classification_val_result.csv ({len(val_result_df)}건)")
print(val_result_df.head(10))

# --- Test 예측 → 제출 파일 ---
X_test_tfidf = tfidf.transform(df_test[TEXT_COL])
y_pred_test = best_ml_model.predict(X_test_tfidf)
submit_df = df_test.copy()
submit_df['predicted'] = y_pred_test
if target_names:
    submit_df['predicted_name'] = submit_df['predicted'].map(label_map)
# 시험에서 test에 레이블이 있는 경우 정확도 확인
if TARGET in df_test.columns:
    test_acc = accuracy_score(df_test[TARGET], y_pred_test)
    test_f1 = f1_score(df_test[TARGET], y_pred_test, average='macro')
    submit_df['correct'] = (df_test[TARGET].values == y_pred_test).astype(int)
    print(f"\n[Test 성능] Accuracy={test_acc:.4f} | F1(macro)={test_f1:.4f}")

submit_df.to_csv('submission.csv', index=False)
print(f"[저장] submission.csv ({len(submit_df)}건)")
print(submit_df.head(10))

# --- 최종 성능 요약 저장 ---
summary_df = pd.DataFrame([{
    'method': 'TF-IDF + ML',
    'model': best_ml_name,
    'val_accuracy': acc_final,
    'val_f1_macro': f1_final,
    'accuracy_grade': acc_grade,
    'f1_grade': f1_grade,
}])
summary_df.to_csv('text_classification_summary.csv', index=False)
print(f"\n[저장] text_classification_summary.csv")
print(summary_df)

# ============================================================
# 모델 저장 (방법 A)
# ============================================================
# %%
import joblib
joblib.dump(best_ml_model, 'text_tfidf_model.pkl')
joblib.dump(tfidf, 'text_tfidf_vectorizer.pkl')
print("\nTF-IDF 모델 저장 완료: text_tfidf_model.pkl, text_tfidf_vectorizer.pkl")

# ============================================================
# 방법 B: BERT Fine-tuning (높은 성능)
# ============================================================
# %% [markdown]
# ## 방법 B: BERT Fine-tuning (TensorFlow)
# 시간이 충분하고 GPU가 있을 때 사용
# 방법 A로 성능이 충분하면 생략 가능

# %%
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification

tf.random.set_seed(RANDOM_SEED)

# -------------------------------------------------------
# BERT 하이퍼파라미터
# -------------------------------------------------------
# BERT_MODEL_NAME: 한국어='klue/bert-base' | 영어='bert-base-uncased'
# MAX_LENGTH: 토큰 최대 길이 (128이 일반적, 긴 텍스트는 256~512)
# BATCH_SIZE: GPU 메모리에 맞게 조절 (16GB VRAM → 32 정도)
# EPOCHS: 보통 3~5 (EarlyStopping이 알아서 조기 종료)
# LEARNING_RATE: 2e-5가 BERT 표준, 1e-5~5e-5 범위
# 영어 데이터 → 'bert-base-uncased' | 한국어 데이터 → 'klue/bert-base'
# BERT_MODEL_NAME = 'bert-base-uncased'
BERT_MODEL_NAME = 'klue/bert-base'
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5

# --- train/val/test 텍스트 준비 (방법 A와 동일 분할 사용) ---
X_train_text = df_tr[TEXT_COL].tolist()
X_val_text = df_val[TEXT_COL].tolist()
X_test_text = df_test[TEXT_COL].tolist()
y_train_b = df_tr[TARGET].tolist()
y_val_b = df_val[TARGET].tolist()
print(f"Train: {len(X_train_text)}, Val: {len(X_val_text)}, Test: {len(X_test_text)}")

# --- 토크나이저 ---
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

train_enc = tokenizer(
    X_train_text, truncation=True, padding=True,
    max_length=MAX_LENGTH, return_tensors='tf'
)
val_enc = tokenizer(
    X_val_text, truncation=True, padding=True,
    max_length=MAX_LENGTH, return_tensors='tf'
)
print(f"토크나이징 완료: Train={train_enc['input_ids'].shape}, Val={val_enc['input_ids'].shape}")

# --- TF Dataset ---
train_ds = tf.data.Dataset.from_tensor_slices((dict(train_enc), y_train_b))
train_ds = train_ds.shuffle(1000, seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((dict(val_enc), y_val_b))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# %%
# --- BERT 모델 (NUM_CLASSES로 이진/다중 자동 대응) ---
bert_model = TFBertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=NUM_CLASSES, from_pt=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7
    ),
]

start_time = time.time()
history = bert_model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks, verbose=1
)
print(f"학습 완료: {(time.time()-start_time)/60:.1f}분")

# %%
# --- 학습 곡선 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.legend(); ax1.set_title('Loss')

ax2.plot(history.history['accuracy'], label='Train Acc')
ax2.plot(history.history['val_accuracy'], label='Val Acc')
ax2.legend(); ax2.set_title('Accuracy')
plt.tight_layout()
plt.show()

# %%
# --- BERT 평가 (Validation) ---
predictions = bert_model.predict(val_ds)
pred_labels = np.argmax(predictions.logits, axis=1)

acc_bert = accuracy_score(y_val_b, pred_labels)
f1_bert = f1_score(y_val_b, pred_labels, average='macro')
print(f"[BERT - Validation] Accuracy={acc_bert:.4f} | F1(macro)={f1_bert:.4f}")
print(classification_report(y_val_b, pred_labels, target_names=target_names))

# ============================================================
# 결과 CSV 파일 생성 (방법 B)
# ============================================================
# %%
# --- Validation 결과 CSV ---
bert_val_df = pd.DataFrame({
    'text': X_val_text,
    'actual': y_val_b,
    'predicted': pred_labels,
    'correct': [1 if a == p else 0 for a, p in zip(y_val_b, pred_labels)],
})
if target_names:
    bert_val_df['actual_name'] = bert_val_df['actual'].map(label_map)
    bert_val_df['predicted_name'] = bert_val_df['predicted'].map(label_map)

bert_val_df.to_csv('text_bert_val_result.csv', index=False)
print(f"\n[저장] text_bert_val_result.csv ({len(bert_val_df)}건)")
print(bert_val_df.head(10))

# --- Test 예측 → 제출 파일 ---
test_enc_bert = tokenizer(
    X_test_text, truncation=True, padding=True,
    max_length=MAX_LENGTH, return_tensors='tf'
)
test_ds_bert = tf.data.Dataset.from_tensor_slices(dict(test_enc_bert))
test_ds_bert = test_ds_bert.batch(BATCH_SIZE)
test_preds = bert_model.predict(test_ds_bert)
test_pred_labels = np.argmax(test_preds.logits, axis=1)

bert_submit_df = df_test.copy()
bert_submit_df['predicted'] = test_pred_labels
if target_names:
    bert_submit_df['predicted_name'] = bert_submit_df['predicted'].map(label_map)
if TARGET in df_test.columns:
    test_acc_b = accuracy_score(df_test[TARGET], test_pred_labels)
    test_f1_b = f1_score(df_test[TARGET], test_pred_labels, average='macro')
    bert_submit_df['correct'] = (df_test[TARGET].values == test_pred_labels).astype(int)
    print(f"\n[BERT Test 성능] Accuracy={test_acc_b:.4f} | F1(macro)={test_f1_b:.4f}")

bert_submit_df.to_csv('submission_bert.csv', index=False)
print(f"[저장] submission_bert.csv ({len(bert_submit_df)}건)")

bert_summary_df = pd.DataFrame([{
    'method': 'BERT Fine-tuning',
    'model': BERT_MODEL_NAME,
    'val_accuracy': acc_bert,
    'val_f1_macro': f1_bert,
}])
bert_summary_df.to_csv('text_bert_summary.csv', index=False)
print(f"\n[저장] text_bert_summary.csv")
print(bert_summary_df)

# ============================================================
# 모델 저장 (방법 B)
# ============================================================
# %%
bert_model.save_pretrained('text_bert_model')
tokenizer.save_pretrained('text_bert_model')
print("BERT 모델 저장 완료: text_bert_model/")
