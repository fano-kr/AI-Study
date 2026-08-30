# %% [markdown]
# # Image 분류 템플릿 (MobileNetV2 2단계 전이학습)
#
# 교재: docs/03-image.md
#
# label_mode='categorical' + softmax + categorical_crossentropy 로 통일해
# 이진분류(2)든 다중분류(7)든 코드가 같습니다.
#
#   Phase 1  Feature Extraction  base 동결        lr = 1e-3
#   Phase 2  Fine-Tuning         상위 30층 해제   lr = 1e-5
#
# ★ 표시가 있는 곳만 문제에 맞게 고치면 됩니다.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"TensorFlow {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# ============================================================
# 1. 하이퍼파라미터
# ============================================================
# %%
INPUT_SHAPE = (224, 224, 3)     # MobileNetV2 기본 입력
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32                 # OOM 나면 16 → 8
EPOCHS = 30                     # EarlyStopping 이 알아서 멈춘다
LEARNING_RATE = 1e-3            # Phase 1
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 1e-5             # Phase 2 — 100배 낮춰야 사전학습 가중치가 보존된다
FINE_TUNE_LAYERS = 30           # 상위 몇 개 레이어를 풀 것인가

# ============================================================
# 2. 데이터 로드  ★
# ============================================================
# %%
# 폴더 구조가 곧 라벨입니다 (하위 폴더명이 알파벳 순으로 class 0,1,2...)
#   data_dir/
#     daisy/  img1.jpg ...
#     rose/   img7.jpg ...
import pathlib

# --- 시험 환경 ---
# data_dir = pathlib.Path('./train')      # ★ 실제 경로
# USE_SPLIT = True                        # 단일 폴더에서 8:2 자동 분할
# USE_SPLIT = False                       # 이미 train/ validation/ 으로 나뉘어 있으면

# --- 연습용: tf_flowers (5클래스) ---
_p = pathlib.Path(tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True,
))
# get_file 이 돌려주는 경로는 TF/Keras 버전마다 다르므로 클래스 폴더를 직접 찾는다
data_dir = _p if (_p / 'daisy').exists() else next(_p.rglob('flower_photos/daisy')).parent
USE_SPLIT = True

print(f"경로: {data_dir}")
print(f"하위 폴더: {sorted(d.name for d in data_dir.iterdir() if d.is_dir())}")

# %%
# label_mode='categorical' 이 이진/다중 통일의 핵심
common = dict(label_mode='categorical', image_size=IMAGE_SIZE,
              batch_size=BATCH_SIZE, seed=RANDOM_SEED, shuffle=True)

if USE_SPLIT:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='training', **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='validation', **common)
else:
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir / 'train', **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir / 'validation', **common)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"클래스 {NUM_CLASSES}개: {class_names}")

# shuffle=True 라 반복 순서 != file_paths 순서 → 예측용 데이터셋을 따로 만든다
val_file_paths = list(val_ds.file_paths)
val_filenames = [os.path.basename(p) for p in val_file_paths]
val_true_names = [os.path.basename(os.path.dirname(p)) for p in val_file_paths]
val_true_idx = [class_names.index(n) for n in val_true_names]

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return tf.image.resize(img, IMAGE_SIZE)


# 순서 보장 예측용 데이터셋 (이 차시에서 가장 중요한 함정 대응)
val_pred_ds = (tf.data.Dataset.from_tensor_slices(val_file_paths)
               .map(_load_image, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# %%
# --- EDA: 클래스별 장수와 샘플 확인 ---
if USE_SPLIT:
    for cls in class_names:
        print(f"  {cls}: {len(list((data_dir / cls).glob('*')))}장")

plt.figure(figsize=(15, 6))
for images, labels in train_ds.take(1):
    for i in range(min(10, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[np.argmax(labels[i])], fontsize=9)
        plt.axis('off')
plt.suptitle('Sample Images'); plt.tight_layout(); plt.show()

# ============================================================
# 3. 데이터 증강 — 모델 안에 넣으면 추론 시 자동으로 꺼진다
# ============================================================
# %%
# 현실에서 일어날 수 있는 변형만 넣을 것.
# 상하 반전은 대부분의 사진에서 비현실적이고, 숫자/문자 이미지는 좌우 반전도 빼야 한다.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# ============================================================
# 4. 모델 구축
# ============================================================
# %%
base_model = tf.keras.applications.MobileNetV2(
    input_shape=INPUT_SHAPE, weights='imagenet', include_top=False
)
base_model.trainable = False        # Phase 1: 동결
print(f"base model 레이어 수: {len(base_model.layers)}")

inputs = tf.keras.Input(shape=INPUT_SHAPE)
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(x)   # MobileNetV2 는 -1~1 입력
x = base_model(x, training=False)                         # 동결 시 BatchNorm 통계 고정
x = tf.keras.layers.GlobalAveragePooling2D()(x)           # Flatten 대신 (파라미터 폭증 방지)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# ============================================================
# 5. Phase 1 — Feature Extraction
# ============================================================
# %%
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True,
                                       monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
]

print('Phase 1: base 동결, 분류 헤드만 학습')
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, callbacks=callbacks, verbose=1)

# ============================================================
# 6. Phase 2 — Fine-Tuning
# ============================================================
# %%
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
print(f'Phase 2: 상위 {FINE_TUNE_LAYERS}개 레이어 해제')

# trainable 을 바꿨으면 반드시 재컴파일 (안 하면 변경이 반영되지 않는다)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

history_ft = model.fit(train_ds, validation_data=val_ds,
                       epochs=FINE_TUNE_EPOCHS, callbacks=callbacks, verbose=1)

# ============================================================
# 7. 학습 곡선
# ============================================================
# %%
acc = history.history['accuracy'] + history_ft.history['accuracy']
val_acc = history.history['val_accuracy'] + history_ft.history['val_accuracy']
loss = history.history['loss'] + history_ft.history['loss']
val_loss = history.history['val_loss'] + history_ft.history['val_loss']
split_at = len(history.history['accuracy'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(acc, label='Train'); ax1.plot(val_acc, label='Val')
ax1.axvline(split_at - 1, color='gray', ls='--', label='Fine-tune 시작')
ax1.set_title('Accuracy'); ax1.legend()
ax2.plot(loss, label='Train'); ax2.plot(val_loss, label='Val')
ax2.axvline(split_at - 1, color='gray', ls='--')
ax2.set_title('Loss'); ax2.legend()
plt.tight_layout(); plt.show()

# Train 만 오르고 Val 이 정체 → 과적합 (증강 강화, Dropout ↑, Phase2 에폭 ↓)
# Phase 2 에서 급락 → 학습률이 높음 (1e-5 확인)

# ============================================================
# 8. 최종 평가 — 순서 보장 데이터셋으로 예측
# ============================================================
# %%
probs = model.predict(val_pred_ds, verbose=1)
all_preds = np.argmax(probs, axis=1)
all_confs = probs.max(axis=1)
all_labels = np.array(val_true_idx)

acc_final = accuracy_score(all_labels, all_preds)
f1_final = f1_score(all_labels, all_preds, average='macro')
print(f"Accuracy {acc_final:.4f} / F1(macro) {f1_final:.4f}")
print(classification_report(all_labels, all_preds, target_names=class_names))

grade = next((g for t, g in [(0.95, '매우 우수'), (0.90, '우수'),
                             (0.85, '양호'), (0.80, '보통')] if acc_final >= t), '미흡')
print(f"-> {grade}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6 + NUM_CLASSES, 5 + NUM_CLASSES * 0.4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.show()

# %%
# 검증 결과 저장 (파일명과 예측이 1:1 대응)
val_result = pd.DataFrame({
    'filename': val_filenames,
    'actual_name': val_true_names,
    'predicted_name': [class_names[i] for i in all_preds],
    'confidence': all_confs,
    'correct': (all_labels == all_preds).astype(int),
})
val_result.to_csv('image_val_result.csv', index=False)
print(val_result.head(10))

# ============================================================
# 9. 테스트 예측 및 제출  ★
# ============================================================
# %%
def predict_test_directory(model, test_dir, class_names,
                           image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """라벨 없는 test 폴더의 이미지를 예측해 DataFrame 으로 반환."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    filenames = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(exts))
    images = np.array([
        tf.keras.utils.img_to_array(
            tf.keras.utils.load_img(os.path.join(test_dir, f), target_size=image_size))
        for f in filenames
    ])
    preds = model.predict(images, batch_size=batch_size, verbose=1)
    idx = np.argmax(preds, axis=1)
    return pd.DataFrame({
        'filename': filenames,
        'predicted': [class_names[i] for i in idx],
        'confidence': preds.max(axis=1),
    })


# --- 시험 환경: 별도 test 폴더가 있는 경우 ★ ---
# TEST_DIR = './test'
# submission = predict_test_directory(model, TEST_DIR, class_names)
# submission.to_csv('submission.csv', index=False)
# print(f"[저장] submission.csv ({len(submission)}건)")

# --- test 폴더가 없으면 검증 결과를 제출물로 ---
val_result.to_csv('submission.csv', index=False)
print(f"[저장] submission.csv ({len(val_result)}건)")

# %%
model.save('image_classification_model.keras')
print('모델 저장 완료')
