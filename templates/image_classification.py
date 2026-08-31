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
# GPU 는 세션 시작 셀(00_session_start.py)에서 확인합니다.
# 세션이 튕겼다면 그 셀부터 다시 실행하세요 (재연결 시 CPU 로 붙는 경우가 있습니다).

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
# 1. 하이퍼파라미터
# ============================================================
# %%
p1_INPUT_SHAPE = (224, 224, 3)     # MobileNetV2 기본 입력
p1_IMAGE_SIZE = (224, 224)
p1_BATCH_SIZE = 32                 # OOM 나면 16 → 8
p1_EPOCHS = 30                     # EarlyStopping 이 알아서 멈춘다
p1_LEARNING_RATE = 1e-3            # Phase 1
p1_FINE_TUNE_EPOCHS = 20
p1_FINE_TUNE_LR = 1e-5             # Phase 2 — 100배 낮춰야 사전학습 가중치가 보존된다
p1_FINE_TUNE_LAYERS = 30           # 상위 몇 개 레이어를 풀 것인가

# ============================================================
# 2. 데이터 로드  ★
# ============================================================
# %%
# 폴더 구조가 곧 라벨입니다 (하위 폴더명이 알파벳 순으로 class 0,1,2...)
#   p1_data_dir/
#     daisy/  img1.jpg ...
#     rose/   img7.jpg ...
import pathlib

# --- 시험 환경 ---
# p1_data_dir = pathlib.Path('./train')      # ★ 실제 경로
# p1_USE_SPLIT = True                        # 단일 폴더에서 8:2 자동 분할
# p1_USE_SPLIT = False                       # 이미 train/ validation/ 으로 나뉘어 있으면

# --- 연습용: tf_flowers (5클래스) ---
p1__p = pathlib.Path(tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True,
))
# get_file 이 돌려주는 경로는 TF/Keras 버전마다 다르므로 클래스 폴더를 직접 찾는다
p1_data_dir = p1__p if (p1__p / 'daisy').exists() else next(p1__p.rglob('flower_photos/daisy')).parent
p1_USE_SPLIT = True

print(f"경로: {p1_data_dir}")
print(f"하위 폴더: {sorted(d.name for d in p1_data_dir.iterdir() if d.is_dir())}")

# %%
# label_mode='categorical' 이 이진/다중 통일의 핵심
p1_common = dict(label_mode='categorical', image_size=p1_IMAGE_SIZE,
              batch_size=p1_BATCH_SIZE, seed=RANDOM_SEED, shuffle=True)

if p1_USE_SPLIT:
    p1_train_ds = tf.keras.utils.image_dataset_from_directory(
        p1_data_dir, validation_split=0.2, subset='training', **p1_common)
    p1_val_ds = tf.keras.utils.image_dataset_from_directory(
        p1_data_dir, validation_split=0.2, subset='validation', **p1_common)
else:
    p1_train_ds = tf.keras.utils.image_dataset_from_directory(p1_data_dir / 'train', **p1_common)
    p1_val_ds = tf.keras.utils.image_dataset_from_directory(p1_data_dir / 'validation', **p1_common)

p1_class_names = p1_train_ds.class_names
p1_NUM_CLASSES = len(p1_class_names)
print(f"클래스 {p1_NUM_CLASSES}개: {p1_class_names}")

# shuffle=True 라 반복 순서 != file_paths 순서 → 예측용 데이터셋을 따로 만든다
p1_val_file_paths = list(p1_val_ds.file_paths)
p1_val_filenames = [os.path.basename(p) for p in p1_val_file_paths]
p1_val_true_names = [os.path.basename(os.path.dirname(p)) for p in p1_val_file_paths]
p1_val_true_idx = [p1_class_names.index(n) for n in p1_val_true_names]

p1_AUTOTUNE = tf.data.AUTOTUNE
p1_train_ds = p1_train_ds.prefetch(p1_AUTOTUNE)
p1_val_ds = p1_val_ds.prefetch(p1_AUTOTUNE)


def p1__load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return tf.image.resize(img, p1_IMAGE_SIZE)


# 순서 보장 예측용 데이터셋 (이 차시에서 가장 중요한 함정 대응)
p1_val_pred_ds = (tf.data.Dataset.from_tensor_slices(p1_val_file_paths)
               .map(p1__load_image, num_parallel_calls=p1_AUTOTUNE)
               .batch(p1_BATCH_SIZE).prefetch(p1_AUTOTUNE))

# %%
# --- EDA: 클래스별 장수와 샘플 확인 ---
if p1_USE_SPLIT:
    for p1_cls in p1_class_names:
        print(f"  {p1_cls}: {len(list((p1_data_dir / p1_cls).glob('*')))}장")

plt.figure(figsize=(15, 6))
for p1_images, p1_labels in p1_train_ds.take(1):
    for p1_i in range(min(10, len(p1_images))):
        plt.subplot(2, 5, p1_i + 1)
        plt.imshow(p1_images[p1_i].numpy().astype('uint8'))
        plt.title(p1_class_names[np.argmax(p1_labels[p1_i])], fontsize=9)
        plt.axis('off')
plt.suptitle('Sample Images'); plt.tight_layout(); plt.show()

# ============================================================
# 3. 데이터 증강 — 모델 안에 넣으면 추론 시 자동으로 꺼진다
# ============================================================
# %%
# 현실에서 일어날 수 있는 변형만 넣을 것.
# 상하 반전은 대부분의 사진에서 비현실적이고, 숫자/문자 이미지는 좌우 반전도 빼야 한다.
p1_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# ============================================================
# 4. 모델 구축
# ============================================================
# %%
p1_base_model = tf.keras.applications.MobileNetV2(
    input_shape=p1_INPUT_SHAPE, weights='imagenet', include_top=False
)
p1_base_model.trainable = False        # Phase 1: 동결
print(f"base model 레이어 수: {len(p1_base_model.layers)}")

p1_inputs = tf.keras.Input(shape=p1_INPUT_SHAPE)
p1_x = p1_data_augmentation(p1_inputs)
p1_x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(p1_x)   # MobileNetV2 는 -1~1 입력
p1_x = p1_base_model(p1_x, training=False)                         # 동결 시 BatchNorm 통계 고정
p1_x = tf.keras.layers.GlobalAveragePooling2D()(p1_x)           # Flatten 대신 (파라미터 폭증 방지)
p1_x = tf.keras.layers.Dropout(0.3)(p1_x)
p1_x = tf.keras.layers.Dense(256, activation='relu')(p1_x)
p1_x = tf.keras.layers.Dropout(0.2)(p1_x)
p1_outputs = tf.keras.layers.Dense(p1_NUM_CLASSES, activation='softmax')(p1_x)

p1_model = tf.keras.Model(p1_inputs, p1_outputs)
p1_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p1_LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
p1_model.summary()

# ============================================================
# 5. Phase 1 — Feature Extraction
# ============================================================
# %%
p1_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # 학습 도중 세션이 튕겨도 여기까지의 최고 성능 가중치는 파일로 남는다
    tf.keras.callbacks.ModelCheckpoint(f'{PHONE}_{p1_PROB_NO}_ckpt.keras', save_best_only=True,
                                       monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
]

print('Phase 1: base 동결, 분류 헤드만 학습')
p1_history = p1_model.fit(p1_train_ds, validation_data=p1_val_ds,
                    epochs=p1_EPOCHS, callbacks=p1_callbacks, verbose=1)

# ============================================================
# 6. Phase 2 — Fine-Tuning
# ============================================================
# %%
p1_base_model.trainable = True
for p1_layer in p1_base_model.layers[:-p1_FINE_TUNE_LAYERS]:
    p1_layer.trainable = False
print(f'Phase 2: 상위 {p1_FINE_TUNE_LAYERS}개 레이어 해제')

# trainable 을 바꿨으면 반드시 재컴파일 (안 하면 변경이 반영되지 않는다)
p1_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p1_FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

p1_history_ft = p1_model.fit(p1_train_ds, validation_data=p1_val_ds,
                       epochs=p1_FINE_TUNE_EPOCHS, callbacks=p1_callbacks, verbose=1)

# ============================================================
# 7. 학습 곡선
# ============================================================
# %%
p1_acc = p1_history.history['accuracy'] + p1_history_ft.history['accuracy']
p1_val_acc = p1_history.history['val_accuracy'] + p1_history_ft.history['val_accuracy']
p1_loss = p1_history.history['loss'] + p1_history_ft.history['loss']
p1_val_loss = p1_history.history['val_loss'] + p1_history_ft.history['val_loss']
p1_split_at = len(p1_history.history['accuracy'])

p1_fig, (p1_ax1, p1_ax2) = plt.subplots(1, 2, figsize=(14, 5))
p1_ax1.plot(p1_acc, label='Train'); p1_ax1.plot(p1_val_acc, label='Val')
p1_ax1.axvline(p1_split_at - 1, color='gray', ls='--', label='Fine-tune 시작')
p1_ax1.set_title('Accuracy'); p1_ax1.legend()
p1_ax2.plot(p1_loss, label='Train'); p1_ax2.plot(p1_val_loss, label='Val')
p1_ax2.axvline(p1_split_at - 1, color='gray', ls='--')
p1_ax2.set_title('Loss'); p1_ax2.legend()
plt.tight_layout(); plt.show()

# Train 만 오르고 Val 이 정체 → 과적합 (증강 강화, Dropout ↑, Phase2 에폭 ↓)
# Phase 2 에서 급락 → 학습률이 높음 (1e-5 확인)

# ============================================================
# 8. 최종 평가 — 순서 보장 데이터셋으로 예측
# ============================================================
# %%
p1_probs = p1_model.predict(p1_val_pred_ds, verbose=1)
p1_all_preds = np.argmax(p1_probs, axis=1)
p1_all_confs = p1_probs.max(axis=1)
p1_all_labels = np.array(p1_val_true_idx)

p1_acc_final = accuracy_score(p1_all_labels, p1_all_preds)
p1_f1_final = f1_score(p1_all_labels, p1_all_preds, average='macro')
print(f"Accuracy {p1_acc_final:.4f} / F1(macro) {p1_f1_final:.4f}")
print(classification_report(p1_all_labels, p1_all_preds, target_names=p1_class_names))

p1_grade = next((g for t, g in [(0.95, '매우 우수'), (0.90, '우수'),
                             (0.85, '양호'), (0.80, '보통')] if p1_acc_final >= t), '미흡')
print(f"-> {p1_grade}")

p1_cm = confusion_matrix(p1_all_labels, p1_all_preds)
plt.figure(figsize=(6 + p1_NUM_CLASSES, 5 + p1_NUM_CLASSES * 0.4))
sns.heatmap(p1_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=p1_class_names, yticklabels=p1_class_names)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.show()

# %%
# 검증 결과 저장 (파일명과 예측이 1:1 대응)
p1_val_result = pd.DataFrame({
    'filename': p1_val_filenames,
    'actual_name': p1_val_true_names,
    'predicted_name': [p1_class_names[p1_i] for p1_i in p1_all_preds],
    'confidence': p1_all_confs,
    'correct': (p1_all_labels == p1_all_preds).astype(int),
})
p1_val_result.to_csv('image_val_result.csv', index=False)
print(p1_val_result.head(10))

# ============================================================
# 9. 테스트 예측 및 제출  ★
# ============================================================
# %%
def p1_predict_test_directory(p1_model, test_dir, p1_class_names,
                           image_size=p1_IMAGE_SIZE, batch_size=p1_BATCH_SIZE):
    """라벨 없는 test 폴더의 이미지를 예측해 DataFrame 으로 반환."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    filenames = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(exts))
    p1_images = np.array([
        tf.keras.utils.img_to_array(
            tf.keras.utils.load_img(os.path.join(test_dir, f), target_size=image_size))
        for f in filenames
    ])
    preds = p1_model.predict(p1_images, batch_size=batch_size, verbose=1)
    idx = np.argmax(preds, axis=1)
    return pd.DataFrame({
        'filename': filenames,
        'predicted': [p1_class_names[p1_i] for p1_i in idx],
        'confidence': preds.max(axis=1),
    })


# --- 시험 환경: 별도 test 폴더가 있는 경우 ★ ---
# TEST_DIR = './test'
# submission = p1_predict_test_directory(p1_model, TEST_DIR, p1_class_names)
# submission.to_csv(p1_SUBMIT_CSV, index=False)
# print(f"[저장] {p1_SUBMIT_CSV} ({len(submission)}건)")

# --- test 폴더가 없으면 검증 결과를 제출물로 ---
p1_val_result.to_csv(p1_SUBMIT_CSV, index=False)
print(f"[저장] {p1_SUBMIT_CSV} ({len(p1_val_result)}건)")

# %%
p1_model.save(p1_SUBMIT_KERAS)
print(f"모델 저장 완료: {p1_SUBMIT_KERAS}")

# 체크포인트(_ckpt)는 튕김 대비용 작업 파일입니다. 제출물과 구분해서 확인하세요.
_files = [f for f in os.listdir('.') if f.startswith(PHONE)]
print("제출물   :", [f for f in _files if '_ckpt' not in f])
print("작업 파일:", [f for f in _files if '_ckpt' in f])
