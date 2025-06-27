# %%
#%pip install transformers
#%pip install tensorflow
#%pip install pandas
#%pip install numpy
#%pip install matplotlib
#%pip install seaborn
#%pip install scikit-learn
#%pip install openpyxl -q

# %%
# 필수 라이브러리 임포트
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# 코드 실행 시작 시간 기록
start_time = time.time()
start_datetime = datetime.now()
print(f"코드 실행 시작: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# 재현 가능한 결과를 위한 시드 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# %%
print(transformers.__version__)
print(tf.__version__)

# %%
# pandas read_excel 함수 사용을 위한 openpyxl 설치

# %%
# 하이퍼파라미터 및 설정 값들을 변수로 정의
TRAIN_DATA_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/master/A_comment_train.xlsx'
TEST_DATA_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/master/A_comment_test.xlsx'
BERT_MODEL_NAME = 'klue/bert-base'
MAX_LENGTH = 128  # BERT 입력 시퀀스 최대 길이
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-6  # 학습률을 약간 높여서 더 안정적 학습
TEST_SIZE = 0.3

# 과적합 방지를 위한 추가 설정
# 모델 학습 시 정확도가 높아지지 않아서 추가
# 미 설정 시 훈련도가 빠르게 100%에 도달하고 검증 정확도가 96%에서 정체
DROPOUT_RATE = 0.1  # 드롭아웃 추가
WEIGHT_DECAY = 0.01  # 가중치 감쇠 추가

# 데이터 로드
try:
    comment_train = pd.read_excel(TRAIN_DATA_URL, engine='openpyxl')
    comment_test = pd.read_excel(TEST_DATA_URL, engine='openpyxl')
    print(f"훈련 데이터 크기: {comment_train.shape}")
    print(f"테스트 데이터 크기: {comment_test.shape}")
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    raise

# %%
comment_train.head()

# %%
comment_test.count()

# %%
# 데이터 전처리 및 정리
def preprocess_data(df):
    """데이터 전처리 함수"""
    # 불필요한 컬럼 제거
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # 결측값 확인 및 처리
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"결측값 {null_count}개 발견되었습니다:")
        print(df.isnull().sum())
        df = df.dropna()  # 결측값이 있는 행 제거
    
    # 텍스트 데이터 기본 정리 (공백 제거 등)
    df['data'] = df['data'].str.strip()
    
    # 빈 텍스트 제거
    before_len = len(df)
    df = df[df['data'].str.len() > 0]
    after_len = len(df)
    if before_len != after_len:
        print(f"빈 텍스트 {before_len - after_len}개가 제거되었습니다.")
    
    return df.reset_index(drop=True)

# 훈련 데이터만 사용 (실제로는 train/validation split을 할 예정)
comment = preprocess_data(comment_train.copy())
print(f"전처리 후 데이터 크기: {comment.shape}")
print(f"레이블 분포:\n{comment['label'].value_counts()}")

# %%
comment.head()

# %%
comment.isnull().sum()

# %%
# label 변환
#comment['label'] = comment['label'].replace(['부정', '긍정'],[0,1])
comment['label'] = comment['label'].apply(lambda x: 0 if x == '부정' else 1)

# %%
comment.tail()

# %%
comment.info()

# %%
# 데이터를 feature와 label로 분리
X = comment['data'].tolist()
y = comment['label'].tolist()

print(f"총 샘플 수: {len(X)}")
print(f"긍정 샘플: {sum(y)}개, 부정 샘플: {len(y) - sum(y)}개")

# %%
# 훈련/검증 데이터 분할 (stratify를 사용하여 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    stratify=y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED
)

print(f"훈련 데이터: {len(X_train)}개")
print(f"검증 데이터: {len(X_test)}개")
print(f"훈련 데이터 클래스 분포: 긍정 {sum(y_train)}개, 부정 {len(y_train) - sum(y_train)}개")
print(f"검증 데이터 클래스 분포: 긍정 {sum(y_test)}개, 부정 {len(y_test) - sum(y_test)}개")

# %%
len(X_train), len(X_test), len(y_train), len(y_test)

# %%
X_train[:2]

# %%
# BERT 모델 및 토크나이저 초기화
print(f"사용할 BERT 모델: {BERT_MODEL_NAME}")
print(f"최대 시퀀스 길이: {MAX_LENGTH}")

# %%
# BERT 토크나이저와 모델 컴포넌트 임포트 및 초기화
from transformers import AutoConfig, BertTokenizerFast, TFBertForSequenceClassification

try:
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    print(f"토크나이저 로드 완료. 어휘 크기: {tokenizer.vocab_size}")
except Exception as e:
    print(f"토크나이저 로드 중 오류 발생: {e}")
    raise

# %%
tokenizer.vocab_size

# %%
#tokenizer.vocab

# %%
# 텍스트를 BERT 입력 형태로 토크나이징
print("텍스트 토크나이징 중...")
train_encodings = tokenizer(
    X_train, 
    truncation=True, 
    padding=True, 
    max_length=MAX_LENGTH,
    return_tensors='tf'
)
test_encodings = tokenizer(
    X_test, 
    truncation=True, 
    padding=True, 
    max_length=MAX_LENGTH,
    return_tensors='tf'
)

print(f"훈련 데이터 토크나이징 완료. 형태: {train_encodings['input_ids'].shape}")
print(f"검증 데이터 토크나이징 완료. 형태: {test_encodings['input_ids'].shape}")

# %%
print(train_encodings['input_ids'][0])

# %%
print(test_encodings['input_ids'][0])

# %%
# TensorFlow 데이터셋 생성 및 최적화
print("TensorFlow 데이터셋 생성 중...")

# 훈련 데이터셋 생성 (셔플, 배치, 캐싱, 프리페치 적용)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
train_dataset = train_dataset.shuffle(1000, seed=RANDOM_SEED).batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

# 검증 데이터셋 생성 (배치, 캐싱, 프리페치 적용)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

print(f"데이터셋 생성 완료. 배치 크기: {BATCH_SIZE}")

# %%
# BERT 모델 설정 확인
config = AutoConfig.from_pretrained(BERT_MODEL_NAME)
print(f"BERT 모델 설정:")
print(f"- 숨겨진 크기: {config.hidden_size}")
print(f"- 어텐션 헤드 수: {config.num_attention_heads}")
print(f"- 숨겨진 레이어 수: {config.num_hidden_layers}")
print(f"- 어휘 크기: {config.vocab_size}")
print(f"- 최대 위치 임베딩: {config.max_position_embeddings}")
config

# %%
# BERT 분류 모델 생성 및 컴파일
print("BERT 분류 모델 초기화 중...")

# 사전 훈련된 BERT 모델을 분류 작업에 맞게 초기화 (2개 클래스: 긍정/부정)
try:
    model = TFBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, 
        num_labels=2,  # 긍정/부정 2개 클래스
        hidden_dropout_prob=DROPOUT_RATE,  # 드롭아웃 적용
        attention_probs_dropout_prob=DROPOUT_RATE,  # 어텐션 드롭아웃
        from_pt=True   # PyTorch 모델에서 TensorFlow로 변환
    )
    print("BERT 모델 로드 완료")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    raise

# 옵티마이저, 손실 함수, 메트릭 설정 (가중치 감쇠 추가)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY  # 가중치 감쇠로 과적합 방지
)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

# 모델 컴파일
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

print(f"모델 컴파일 완료:")
print(f"- 학습률: {LEARNING_RATE}")
print(f"- 가중치 감쇠: {WEIGHT_DECAY}")
print(f"- 드롭아웃 비율: {DROPOUT_RATE}")
print(f"- 손실 함수: SparseCategoricalCrossentropy")
print(f"- 메트릭: {metrics}")

# %%
# 모델 훈련
print("모델 훈련 시작...")
print(f"훈련 에포크: {EPOCHS}")
print(f"배치 크기: {BATCH_SIZE}")

# 콜백 설정 (조기 종료, 체크포인트 저장 등)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# 모델 훈련 실행
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

print("모델 훈련 완료!")

# %%
# 훈련 과정 시각화
plt.figure(figsize=(12, 4))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('모델 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('모델 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.legend()

plt.tight_layout()
plt.show()

# 최종 성능 출력
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\n최종 훈련 결과:")
print(f"- 훈련 정확도: {final_train_acc:.4f}")
print(f"- 검증 정확도: {final_val_acc:.4f}")
print(f"- 훈련 손실: {final_train_loss:.4f}")
print(f"- 검증 손실: {final_val_loss:.4f}")


# %%
model.summary()

# %%
# 실제 테스트 데이터로 모델 성능 평가
print("실제 테스트 데이터로 모델 평가 중...")

# %%
# 레이블이 있는 테스트 데이터 사용 (전처리 적용)
# comment_valid = preprocess_data(comment_test.copy())
comment_valid = comment_test.copy()
print(f"평가용 테스트 데이터 크기: {comment_valid.shape}")
print(f"테스트 데이터 레이블 분포:\n{comment_valid['label'].value_counts()}")
comment_valid.head()

# %%
comment_valid['data']

# %%
# 테스트 데이터 예측을 위한 전처리
print("테스트 데이터 토크나이징 및 예측 중...")

# 텍스트 데이터 추출
valid_texts = comment_valid['data'].tolist()
print(f"예측할 텍스트 수: {len(valid_texts)}")

# 토크나이징 (훈련 때와 동일한 설정 사용)
valid_encodings = tokenizer(
    valid_texts, 
    truncation=True, 
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='tf'
)

# TensorFlow 데이터셋으로 변환
valid_dataset = tf.data.Dataset.from_tensor_slices(dict(valid_encodings))
valid_dataset = valid_dataset.batch(BATCH_SIZE)

# 예측 실행
predictions = model.predict(valid_dataset)
print("예측 완료!")

# %%
predictions.logits

# %%
# 예측 결과 분석 및 평가
print("예측 결과 분석 중...")

# logits에서 클래스 예측값으로 변환
# logits : 모델이 출력하는 원시 점수 (확률화되기 전)
# argmax : 가장 큰 값의 인덱스를 반환
# axis=1 : 행 방향으로 최대값의 인덱스를 찾음
# 큰 값이 예측 레이블이 됨
predicted_labels = np.argmax(predictions.logits, axis=1)

# 예측 결과를 확률로 변환하기 위해 softmax 함수를 사용
# softmax : 확률로 변환
# axis=1 : 행 방향으로 최대값의 인덱스를 찾음
# numpy() : numpy 배열로 변환
prediction_probs = tf.nn.softmax(predictions.logits, axis=1).numpy()

print(predicted_labels[:10])
print(prediction_probs[:10])

# 실제 레이블 변환 (부정=0, 긍정=1)  
true_labels = comment_valid['label'].apply(lambda x: 0 if x == '부정' else 1).values

# 성능 지표 계산
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'정확도: {accuracy:.4f}')

# 상세한 분류 보고서 출력
print('\n분류 보고서:')
report = classification_report(true_labels, predicted_labels, target_names=['부정', '긍정'])
print(report)

# 혼동 행렬 시각화
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['부정', '긍정'], 
            yticklabels=['부정', '긍정'])
plt.title('혼동 행렬 (Confusion Matrix)')
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.show()

# 예측 결과를 DataFrame으로 저장
results_df = pd.DataFrame({
    '텍스트': comment_valid['data'],
    '실제 레이블': comment_valid['label'],
    '예측 레이블': ['부정' if label == 0 else '긍정' for label in predicted_labels],
    '부정 확률': prediction_probs[:, 0],
    '긍정 확률': prediction_probs[:, 1],
    '예측 신뢰도': np.max(prediction_probs, axis=1)
})

print("\n예측 결과 샘플:")
print(results_df[['텍스트', '실제 레이블', '예측 레이블', '예측 신뢰도']].head())

# %%
print(results_df.head())

# %%
# 잘못 예측된 샘플 분석
incorrect_predictions = results_df[results_df['실제 레이블'] != results_df['예측 레이블']].copy()
print(f"\n잘못 예측된 샘플 수: {len(incorrect_predictions)}")

if len(incorrect_predictions) > 0:
    print("\n잘못 예측된 샘플들:")
    print(incorrect_predictions[['텍스트', '실제 레이블', '예측 레이블', '예측 신뢰도']].head(10))
    
    # 신뢰도가 낮은 예측들 확인
    low_confidence = results_df[results_df['예측 신뢰도'] < 0.8].copy()
    print(f"\n신뢰도가 낮은 예측 수 (< 80%): {len(low_confidence)}")
    
    if len(low_confidence) > 0:
        print("신뢰도가 낮은 예측 샘플들:")
        print(low_confidence[['텍스트', '실제 레이블', '예측 레이블', '예측 신뢰도']].head())

print("\n" + "="*50)
print("모델 성능 요약:")
print(f"- 총 테스트 샘플: {len(results_df)}")
print(f"- 정확한 예측: {len(results_df) - len(incorrect_predictions)}")
print(f"- 잘못된 예측: {len(incorrect_predictions)}")
print(f"- 정확도: {accuracy:.4f}")
print(f"- 평균 예측 신뢰도: {results_df['예측 신뢰도'].mean():.4f}")
print("="*50)


# %%
# 모델 및 결과 저장
import os
from datetime import datetime

# 저장 디렉토리 생성
save_dir = "model_output"
os.makedirs(save_dir, exist_ok=True)

# 현재 시간을 포함한 파일명 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 저장 (SavedModel 형식과 H5 형식 모두 저장)
model_path = os.path.join(save_dir, f"sentiment_bert_model_{timestamp}")
h5_model_path = os.path.join(save_dir, f"sentiment_bert_model_{timestamp}.h5")

try:
    # SavedModel 형식 저장 (권장)
    model.save(model_path)
    print(f"SavedModel 형식으로 모델이 저장되었습니다: {model_path}")
    
    # H5 형식 저장 (가중치만 저장)
    model.save_weights(h5_model_path)
    print(f"H5 형식으로 모델 가중치가 저장되었습니다: {h5_model_path}")
    
    # 전체 모델을 H5 형식으로 저장 시도 (일부 모델에서는 제한이 있을 수 있음)
    h5_full_model_path = os.path.join(save_dir, f"sentiment_bert_full_model_{timestamp}.h5")
    try:
        model.save(h5_full_model_path, save_format='h5')
        print(f"H5 형식으로 전체 모델이 저장되었습니다: {h5_full_model_path}")
    except Exception as h5_e:
        print(f"H5 형식 전체 모델 저장 중 오류 (가중치만 저장됨): {h5_e}")
        
except Exception as e:
    print(f"모델 저장 중 오류 발생: {e}")

# 결과 CSV 저장
results_path = os.path.join(save_dir, f"prediction_results_{timestamp}.csv")
try:
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"예측 결과가 저장되었습니다: {results_path}")
except Exception as e:
    print(f"결과 저장 중 오류 발생: {e}")

# 모델 설정 정보 저장
config_info = {
    'model_name': BERT_MODEL_NAME,
    'max_length': MAX_LENGTH,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'test_size': TEST_SIZE,
    'final_accuracy': accuracy,
    'timestamp': timestamp
}

config_path = os.path.join(save_dir, f"model_config_{timestamp}.txt")
try:
    with open(config_path, 'w', encoding='utf-8') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
    print(f"모델 설정 정보가 저장되었습니다: {config_path}")
except Exception as e:
    print(f"설정 저장 중 오류 발생: {e}")

print(f"\n모든 결과가 '{save_dir}' 디렉토리에 저장되었습니다.")
results_df = pd.DataFrame({
    '텍스트': comment_test['data'],
    '실제 레이블': comment_test['label'],
    '예측 레이블': ['부정' if label == 0 else '긍정' for label in predicted_labels]
})

# 실제 레이블과 예측 레이블이 다른 경우만 필터링
incorrect_predictions = results_df[results_df['실제 레이블'] != results_df['예측 레이블']]

# 코드 실행 완료 시간 기록 및 총 소요 시간 계산
end_time = time.time()
end_datetime = datetime.now()
total_duration = end_time - start_time

print(f"\n코드 실행 완료: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"총 실행 소요 시간: {total_duration/60:.2f}분 ({total_duration:.2f}초)")

incorrect_predictions

# %%