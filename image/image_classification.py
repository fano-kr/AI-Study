# %% [markdown]
# # Image - 분류 (Classification) 통합 템플릿
#
# NUM_CLASSES = 2 → 이진분류 (Cats vs Dogs)
# NUM_CLASSES >= 3 → 다중분류 (꽃, 동물 등)
#
# 핵심: softmax + categorical_crossentropy로 통일하면
#       이진/다중 구분 없이 동일 코드 사용 가능
#
# 모델: MobileNetV2 Transfer Learning + Fine-tuning
#
# ── 데이터 흐름 ──
# data_dir (이미지 폴더, 클래스별 하위 폴더)
#   ↓ image_dataset_from_directory (validation_split)
# train_ds / val_ds (학습/검증)
#   ↓ 학습 → 검증
# model.fit(train_ds) → predict(val_pred_ds)
#   ↓ 별도 test 폴더가 있으면
# predict_test_directory(test_dir) → submission.csv

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# ============================================================
# 1. 하이퍼파라미터
# ============================================================
# %%
# -------------------------------------------------------
# [하이퍼파라미터 설명]
# INPUT_SHAPE    : 모델 입력 크기 (MobileNetV2 기본=224x224)
# BATCH_SIZE     : GPU 메모리에 맞게 조절 (16GB VRAM → 32~64)
# EPOCHS         : Phase 1(Feature Extraction) 학습 에폭
# LEARNING_RATE  : Phase 1 학습률 (1e-3이 일반적)
# FINE_TUNE_EPOCHS: Phase 2(Fine-tuning) 학습 에폭
# FINE_TUNE_LR   : Phase 2 학습률 (1e-5로 낮춰야 사전학습 가중치 보존)
# -------------------------------------------------------
INPUT_SHAPE = (224, 224, 3)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 1e-5

# ============================================================
# 2. 데이터 로드
# ============================================================
# %%
# === 시험 환경에 맞게 경로 수정 ===
#
# [방법 1] train/validation 분리된 경우:
#   DATA_DIR = './data'
#   TRAIN_DIR = os.path.join(DATA_DIR, 'train')
#   VAL_DIR = os.path.join(DATA_DIR, 'validation')
#
#   train_ds = tf.keras.utils.image_dataset_from_directory(
#       TRAIN_DIR, label_mode='categorical',
#       image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
#       seed=RANDOM_SEED, shuffle=True,
#   )
#   val_ds = tf.keras.utils.image_dataset_from_directory(
#       VAL_DIR, label_mode='categorical',
#       image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
#       shuffle=False,
#   )
#
# ==============================================
# [데이터셋 선택] 아래 중 하나만 주석 해제
# ==============================================
import pathlib

# --- (A) Flower Photos (5클래스 다중분류) ---
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)
# if (data_dir / 'flower_photos').exists():
#     data_dir = data_dir / 'flower_photos'
# USE_SPLIT = True

# --- (B) Cats and Dogs (2클래스 이진분류) ---
# Hugging Face에서 다운로드 후 폴더 구조로 저장
from datasets import load_dataset

label_names = ['cat', 'dog']
data_dir = pathlib.Path('./cats_vs_dogs')

if not data_dir.exists():
    print("데이터셋 다운로드 중 (Hugging Face)...")
    hf_ds = load_dataset('microsoft/cats_vs_dogs', split='train')
    for name in label_names:
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    print(f"이미지 저장 중 ({len(hf_ds)}장)...")
    for i, sample in enumerate(hf_ds):
        label_name = label_names[sample['labels']]
        sample['image'].convert('RGB').save(
            str(data_dir / label_name / f'{i:05d}.jpg'), 'JPEG'
        )
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}장 완료...")
    print(f"총 {i + 1}장 저장 완료")
else:
    print(f"기존 데이터 사용: {data_dir}")

USE_SPLIT = True

print(f"데이터 경로: {data_dir}")
print(f"하위 폴더: {sorted([d.name for d in data_dir.iterdir() if d.is_dir()])}")

# -------------------------------------------------------
# label_mode='categorical' → 이진/다중 모두 동일하게 처리
# USE_SPLIT=True  → 단일 폴더에서 validation_split 자동 분할
# USE_SPLIT=False → 이미 train/validation 분리된 경우
# -------------------------------------------------------
if USE_SPLIT:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode='categorical',
        validation_split=0.2, subset='training',
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        seed=RANDOM_SEED, shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode='categorical',
        validation_split=0.2, subset='validation',
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        seed=RANDOM_SEED, shuffle=True,
    )
else:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / 'train', label_mode='categorical',
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        seed=RANDOM_SEED, shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / 'validation', label_mode='categorical',
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        seed=RANDOM_SEED, shuffle=True,
    )

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# shuffle=True 이므로 file_paths 순서 ≠ 반복 순서
# → 예측 시에는 file_paths 기반 별도 데이터셋 사용
val_file_paths = list(val_ds.file_paths)
val_filenames = [os.path.basename(fp) for fp in val_file_paths]
val_parent_dirs = [os.path.basename(os.path.dirname(fp)) for fp in val_file_paths]

print(f"클래스({NUM_CLASSES}): {class_names}")
print(f"{'이진분류' if NUM_CLASSES == 2 else '다중분류'}")
print(f"Train 배치: {len(train_ds)}, Val 배치: {len(val_ds)}")

# --- 성능 최적화 ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# --- 예측용 정렬 데이터셋 (file_paths 순서 보장) ---
def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img

val_pred_ds = tf.data.Dataset.from_tensor_slices(val_file_paths)
val_pred_ds = val_pred_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
val_pred_ds = val_pred_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ============================================================
# 2-1. EDA (탐색적 데이터 분석)
# ============================================================
# %%
# 클래스별 이미지 수 확인
print("\n[클래스별 이미지 수]")
if USE_SPLIT:
    for cls in class_names:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            print(f"  {cls}: {len(list(cls_dir.glob('*')))}장")
else:
    for split_name in ['train', 'validation']:
        print(f"  [{split_name}]")
        for cls in class_names:
            cls_dir = data_dir / split_name / cls
            if cls_dir.exists():
                print(f"    {cls}: {len(list(cls_dir.glob('*')))}장")

# 샘플 시각화
plt.figure(figsize=(15, 6))
for images, labels in train_ds.take(1):
    for i in range(min(10, len(images))):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_idx = np.argmax(labels[i])
        plt.title(class_names[label_idx], fontsize=10)
        plt.axis("off")
plt.suptitle('Sample Images')
plt.tight_layout()
plt.show()

# ============================================================
# 3. 데이터 증강 (Data Augmentation)
# ============================================================
# %%
# -------------------------------------------------------
# [데이터 증강 설명]
# RandomFlip("horizontal") : 좌우 반전 (가장 기본적)
# RandomRotation(0.2)      : +-20% 회전 (=+-36도)
# RandomZoom(0.2)           : +-20% 확대/축소
# RandomContrast(0.2)       : 대비 변화 (조명 변화 대응)
#
# [선택적 증강 - 필요 시 추가]
# RandomBrightness(0.2)    : 밝기 변화
# RandomTranslation(0.1,0.1): 이동
# GaussianNoise(0.05)      : 노이즈 추가
# -------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# ============================================================
# 4. 모델 구축 (MobileNetV2 Transfer Learning)
# ============================================================
# %%
# -------------------------------------------------------
# [사전훈련 모델 선택]
# MobileNetV2     : 가볍고 빠름 (시험 추천, 시간 절약)
# ResNet50        : 더 깊고 강력함 (시간 여유 있을 때)
# EfficientNetB0  : 효율적 (최신, 성능/속도 균형)
# -------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=INPUT_SHAPE,
    weights='imagenet',
    include_top=False
)
base_model.trainable = False
print(f"Base model layers: {len(base_model.layers)}")

# --- 전체 모델 (이진/다중 동일 구조) ---
# softmax + NUM_CLASSES → 이진(2)이든 다중이든 동일
inputs = tf.keras.Input(shape=INPUT_SHAPE)
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# --- 컴파일 ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 5. Phase 1 - Feature Extraction 학습
# ============================================================
# %%
# -------------------------------------------------------
# Phase 1: base_model을 얼리고(frozen) 새 분류 헤드만 학습
# 빠르게 수렴, 보통 5~15 에폭이면 충분
# -------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', save_best_only=True,
        monitor='val_accuracy', mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7
    ),
]

print("Phase 1: Feature Extraction (base model frozen)")
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks, verbose=1
)

# ============================================================
# 6. Phase 2 - Fine-Tuning
# ============================================================
# %%
# -------------------------------------------------------
# Phase 2: base_model 하위 레이어를 풀어서 미세 조정
# 학습률을 매우 낮게(1e-5) 설정해야 사전학습 가중치 보존
# 마지막 30개 레이어만 풀기 → 너무 많이 풀면 과적합 위험
# -------------------------------------------------------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Fine-tuning: last {len(base_model.layers) - fine_tune_at} layers")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_ds, validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS, callbacks=callbacks, verbose=1
)

# ============================================================
# 7. 학습 결과 시각화
# ============================================================
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

all_acc = history.history['accuracy'] + history_ft.history['accuracy']
all_val_acc = history.history['val_accuracy'] + history_ft.history['val_accuracy']
all_loss = history.history['loss'] + history_ft.history['loss']
all_val_loss = history.history['val_loss'] + history_ft.history['val_loss']

ax1.plot(all_acc, label='Train')
ax1.plot(all_val_acc, label='Val')
ax1.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--')
ax1.legend(); ax1.set_title('Accuracy')

ax2.plot(all_loss, label='Train')
ax2.plot(all_val_loss, label='Val')
ax2.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--')
ax2.legend(); ax2.set_title('Loss')

plt.tight_layout()
plt.show()

# ============================================================
# 8. 최종 평가
# ============================================================
# %%
if os.path.exists('best_model.keras'):
    model = tf.keras.models.load_model('best_model.keras')

# val_pred_ds로 순서 보장된 예측 수행
all_preds = []
all_confs = []
for images in val_pred_ds:
    preds = model.predict(images, verbose=0)
    all_preds.extend(np.argmax(preds, axis=1))
    all_confs.extend([preds[i][np.argmax(preds[i])] for i in range(len(preds))])

# 실제 레이블은 디렉토리명(=클래스명)에서 추출
all_labels = [class_names.index(d) for d in val_parent_dirs]

acc_final = accuracy_score(all_labels, all_preds)
f1_final = f1_score(all_labels, all_preds, average='macro')

print(f"[최종 모델 성능]")
print(f"Accuracy:   {acc_final:.4f}")
print(f"F1 (macro): {f1_final:.4f}")
print(f"\n[분류 보고서]")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 혼동 행렬
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6 + NUM_CLASSES, 5 + NUM_CLASSES * 0.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

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
# [이미지 분류 특이사항]
#   Transfer Learning은 소량 데이터에서도 높은 성능 달성 가능
#   Phase 2(Fine-tuning)에서 성능이 크게 오를 수 있음
#   과적합 시: Dropout 증가, 데이터 증강 강화, EPOCHS 감소

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

# %%
# 예측 시각화
plt.figure(figsize=(16, 8))
for images, labels in val_ds.take(1):
    preds = model.predict(images, verbose=0)
    for i in range(min(12, len(images))):
        ax = plt.subplot(2, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_idx = np.argmax(labels[i])
        pred_idx = np.argmax(preds[i])
        conf = preds[i][pred_idx]
        color = 'green' if true_idx == pred_idx else 'red'
        plt.title(f'T:{class_names[true_idx]}\nP:{class_names[pred_idx]}\n{conf:.2f}',
                  color=color, fontsize=8)
        plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# 9. 결과 CSV 파일 생성
# ============================================================
# %%
# --- 디렉토리명 vs 예측명 일치 비율 ---
dir_pred_match = sum(d == p for d, p in zip(val_parent_dirs, [class_names[i] for i in all_preds]))
dir_pred_total = len(val_parent_dirs)
print(f"\n[디렉토리명 vs 예측명 일치 비율]")
print(f"  일치: {dir_pred_match} / {dir_pred_total} ({dir_pred_match/dir_pred_total:.4f})")
print(f"  불일치: {dir_pred_total - dir_pred_match}건")

# --- 이미지 파일명 + 예측 결과 컬럼 추가하여 저장 ---
result_df = pd.DataFrame({
    'directory': val_parent_dirs,
    'filename': val_filenames,
    'actual': all_labels,
    'predicted': all_preds,
    'confidence': all_confs,
    'correct': [1 if a == p else 0 for a, p in zip(all_labels, all_preds)],
    'actual_name': [class_names[i] for i in all_labels],
    'predicted_name': [class_names[i] for i in all_preds],
})

result_df = result_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
result_df.to_csv('image_classification_result.csv', index=False)
print(f"[저장] image_classification_result.csv ({len(result_df)}건)")
print(result_df.head(10))

# --- 최종 성능 요약 저장 ---
summary_df = pd.DataFrame([{
    'model': 'MobileNetV2 (Fine-tuned)',
    'accuracy': acc_final,
    'f1_macro': f1_final,
    'accuracy_grade': acc_grade,
    'f1_grade': f1_grade,
    'num_classes': NUM_CLASSES,
    'class_names': str(class_names),
}])
summary_df.to_csv('image_classification_summary.csv', index=False)
print(f"\n[저장] image_classification_summary.csv")
print(summary_df)

# ============================================================
# 10. 테스트 데이터 예측 및 제출 (Test → submission.csv)
# ============================================================
# %%
# -------------------------------------------------------
# 시험에서 별도 test 폴더가 제공되면 아래 코드 사용
# test 폴더: 레이블 없이 이미지만 있는 디렉토리
# -------------------------------------------------------
def predict_test_directory(model, test_dir, class_names,
                           image_size=IMAGE_SIZE, batch_size=32):
    """테스트 디렉토리 이미지 예측 -> DataFrame 반환"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    filenames = sorted([
        f for f in os.listdir(test_dir) if f.lower().endswith(extensions)
    ])

    images = []
    for fname in filenames:
        img = tf.keras.utils.load_img(
            os.path.join(test_dir, fname), target_size=image_size
        )
        images.append(tf.keras.utils.img_to_array(img))

    images = np.array(images)
    preds = model.predict(images, batch_size=batch_size, verbose=1)
    pred_indices = np.argmax(preds, axis=1)
    pred_labels = [class_names[i] for i in pred_indices]
    confidences = [preds[i][pred_indices[i]] for i in range(len(preds))]

    return pd.DataFrame({
        'filename': filenames,
        'predicted': pred_labels,
        'confidence': confidences
    })


# --- (A) 시험 환경: 별도 test 폴더가 있는 경우 ---
# TEST_DIR = './test_images'
# submission = predict_test_directory(model, TEST_DIR, class_names)
# submission.to_csv('submission.csv', index=False)
# print(f"[저장] submission.csv ({len(submission)}건)")

# --- (B) 별도 test 폴더가 없으면 val 결과를 submission으로 사용 ---
result_df.to_csv('submission.csv', index=False)
print(f"\n[저장] submission.csv ({len(result_df)}건)")

# %%
model.save('image_classification_model.keras')
print("\n모델 저장 완료: image_classification_model.keras")
