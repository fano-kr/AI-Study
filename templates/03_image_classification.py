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
# ── 변수 이름 규칙 ──
# 커널 하나에서 세 문제를 풀므로, 전역 변수가 서로 덮어쓰지 않도록
# 템플릿마다 시험 순서대로 접두사를 붙여 두었습니다.
#   1번 Tabular = p1_ · 2번 Text = p2_ · 3번 Image = p3_
# 실제 순서가 다르게 나오면 Ctrl+H 전체 바꾸기로 접두사를 맞추고 PROB_NO 도 함께 고치세요.
# 안 고치면 다른 문제의 제출물을 덮어씁니다.
# PHONE, RANDOM_SEED, check_submission 은 세션 공용이라 접두사가 없습니다.

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
# 0. 문제 설정
# ============================================================
# GPU 확인과 PHONE 은 세션 시작 셀(00_session_start.py)에서 합니다. 여기서 다시 하지 않습니다.
# 단, 세션이 튕기면 GPU 연결이 끊기고 변수가 전부 사라지므로 그 셀부터 다시 실행해야 합니다.
p3_PROB_NO = 3                # ★ 시험 순서: Image = 3번

# 세션 시작 셀을 실행했다면 PHONE 은 이미 정의돼 있습니다.
# 이 파일만 단독으로 연습할 때를 위한 기본값입니다.
try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

p3_SUBMIT_CSV = f'{PHONE}_{p3_PROB_NO}.csv'
p3_SUBMIT_MODEL = f'{PHONE}_{p3_PROB_NO}.pkl'      # sklearn 계열
p3_SUBMIT_KERAS = f'{PHONE}_{p3_PROB_NO}.keras'    # 딥러닝 (문제가 형식을 지정하면 그쪽을)
print(f"제출 파일: {p3_SUBMIT_CSV} / {p3_SUBMIT_MODEL}")

# ============================================================
# 1. 하이퍼파라미터
# ============================================================
# %%
p3_INPUT_SHAPE = (224, 224, 3)     # MobileNetV2 기본 입력
p3_IMAGE_SIZE = (224, 224)
p3_BATCH_SIZE = 32                 # OOM 나면 16 → 8
p3_EPOCHS = 30                     # EarlyStopping 이 알아서 멈춘다
p3_LEARNING_RATE = 1e-3            # Phase 1
p3_FINE_TUNE_EPOCHS = 20
p3_FINE_TUNE_LR = 1e-5             # Phase 2 — 100배 낮춰야 사전학습 가중치가 보존된다
p3_FINE_TUNE_LAYERS = 30           # 상위 몇 개 레이어를 풀 것인가

# ============================================================
# 2. 데이터 로드  ★
# ============================================================
# %%
# 폴더 구조가 곧 라벨입니다 (하위 폴더명이 알파벳 순으로 class 0,1,2...)
#   p3_data_dir/
#     daisy/  img1.jpg ...
#     rose/   img7.jpg ...
import pathlib

# --- 시험 환경 ---
# p3_data_dir = pathlib.Path('./train')      # ★ 실제 경로
# p3_USE_SPLIT = True                        # 단일 폴더에서 8:2 자동 분할
# p3_USE_SPLIT = False                       # 이미 train/ validation/ 으로 나뉘어 있으면

# --- 연습용: tf_flowers (5클래스) ---
p3__p = pathlib.Path(tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True,
))
# get_file 이 돌려주는 경로는 TF/Keras 버전마다 다르므로 클래스 폴더를 직접 찾는다
p3_data_dir = p3__p if (p3__p / 'daisy').exists() else next(p3__p.rglob('flower_photos/daisy')).parent
p3_USE_SPLIT = True

print(f"경로: {p3_data_dir}")
print(f"하위 폴더: {sorted(d.name for d in p3_data_dir.iterdir() if d.is_dir())}")

# %%
# label_mode='categorical' 이 이진/다중 통일의 핵심
p3_common = dict(label_mode='categorical', image_size=p3_IMAGE_SIZE,
              batch_size=p3_BATCH_SIZE, seed=RANDOM_SEED, shuffle=True)

if p3_USE_SPLIT:
    p3_train_ds = tf.keras.utils.image_dataset_from_directory(
        p3_data_dir, validation_split=0.2, subset='training', **p3_common)
    p3_val_ds = tf.keras.utils.image_dataset_from_directory(
        p3_data_dir, validation_split=0.2, subset='validation', **p3_common)
else:
    p3_train_ds = tf.keras.utils.image_dataset_from_directory(p3_data_dir / 'train', **p3_common)
    p3_val_ds = tf.keras.utils.image_dataset_from_directory(p3_data_dir / 'validation', **p3_common)

p3_class_names = p3_train_ds.class_names
p3_NUM_CLASSES = len(p3_class_names)
print(f"클래스 {p3_NUM_CLASSES}개: {p3_class_names}")

# shuffle=True 라 반복 순서 != file_paths 순서 → 예측용 데이터셋을 따로 만든다
p3_val_file_paths = list(p3_val_ds.file_paths)
p3_val_filenames = [os.path.basename(p) for p in p3_val_file_paths]
p3_val_true_names = [os.path.basename(os.path.dirname(p)) for p in p3_val_file_paths]
p3_val_true_idx = [p3_class_names.index(n) for n in p3_val_true_names]

p3_AUTOTUNE = tf.data.AUTOTUNE
p3_train_ds = p3_train_ds.prefetch(p3_AUTOTUNE)
p3_val_ds = p3_val_ds.prefetch(p3_AUTOTUNE)


def p3__load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return tf.image.resize(img, p3_IMAGE_SIZE)


# 순서 보장 예측용 데이터셋 (이 차시에서 가장 중요한 함정 대응)
p3_val_pred_ds = (tf.data.Dataset.from_tensor_slices(p3_val_file_paths)
               .map(p3__load_image, num_parallel_calls=p3_AUTOTUNE)
               .batch(p3_BATCH_SIZE).prefetch(p3_AUTOTUNE))

# %%
# --- EDA: 클래스별 장수와 샘플 확인 ---
if p3_USE_SPLIT:
    for p3_cls in p3_class_names:
        print(f"  {p3_cls}: {len(list((p3_data_dir / p3_cls).glob('*')))}장")

plt.figure(figsize=(15, 6))
for p3_images, p3_labels in p3_train_ds.take(1):
    for p3_i in range(min(10, len(p3_images))):
        plt.subplot(2, 5, p3_i + 1)
        plt.imshow(p3_images[p3_i].numpy().astype('uint8'))
        plt.title(p3_class_names[np.argmax(p3_labels[p3_i])], fontsize=9)
        plt.axis('off')
plt.suptitle('Sample Images'); plt.tight_layout(); plt.show()

# ============================================================
# 3. 데이터 증강 — 모델 안에 넣으면 추론 시 자동으로 꺼진다
# ============================================================
# %%
# 현실에서 일어날 수 있는 변형만 넣을 것.
# 상하 반전은 대부분의 사진에서 비현실적이고, 숫자/문자 이미지는 좌우 반전도 빼야 한다.
p3_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# ============================================================
# 4. 모델 구축
# ============================================================
# %%
p3_base_model = tf.keras.applications.MobileNetV2(
    input_shape=p3_INPUT_SHAPE, weights='imagenet', include_top=False
)
p3_base_model.trainable = False        # Phase 1: 동결
print(f"base model 레이어 수: {len(p3_base_model.layers)}")

p3_inputs = tf.keras.Input(shape=p3_INPUT_SHAPE)
p3_x = p3_data_augmentation(p3_inputs)
p3_x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(p3_x)   # MobileNetV2 는 -1~1 입력
p3_x = p3_base_model(p3_x, training=False)                         # 동결 시 BatchNorm 통계 고정
p3_x = tf.keras.layers.GlobalAveragePooling2D()(p3_x)           # Flatten 대신 (파라미터 폭증 방지)
p3_x = tf.keras.layers.Dropout(0.3)(p3_x)
p3_x = tf.keras.layers.Dense(256, activation='relu')(p3_x)
p3_x = tf.keras.layers.Dropout(0.2)(p3_x)
p3_outputs = tf.keras.layers.Dense(p3_NUM_CLASSES, activation='softmax')(p3_x)

p3_model = tf.keras.Model(p3_inputs, p3_outputs)
p3_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p3_LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
# p3_model.summary()   # 구조 확인이 필요할 때만

# ============================================================
# 5. Phase 1 — Feature Extraction
# ============================================================
# %%
p3_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # 학습 도중 세션이 튕겨도 여기까지의 최고 성능 가중치는 파일로 남는다
    tf.keras.callbacks.ModelCheckpoint(f'{PHONE}_{p3_PROB_NO}_ckpt.keras', save_best_only=True,
                                       monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
]

print('Phase 1: base 동결, 분류 헤드만 학습')
p3_history = p3_model.fit(p3_train_ds, validation_data=p3_val_ds,
                    epochs=p3_EPOCHS, callbacks=p3_callbacks, verbose=1)

# ============================================================
# 6. Phase 2 — Fine-Tuning
# ============================================================
# %%
p3_base_model.trainable = True
for p3_layer in p3_base_model.layers[:-p3_FINE_TUNE_LAYERS]:
    p3_layer.trainable = False
print(f'Phase 2: 상위 {p3_FINE_TUNE_LAYERS}개 레이어 해제')

# trainable 을 바꿨으면 반드시 재컴파일 (안 하면 변경이 반영되지 않는다)
p3_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p3_FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

p3_history_ft = p3_model.fit(p3_train_ds, validation_data=p3_val_ds,
                       epochs=p3_FINE_TUNE_EPOCHS, callbacks=p3_callbacks, verbose=1)

# ============================================================
# 7. 학습 곡선
# ============================================================
# %%
p3_acc = p3_history.history['accuracy'] + p3_history_ft.history['accuracy']
p3_val_acc = p3_history.history['val_accuracy'] + p3_history_ft.history['val_accuracy']
p3_loss = p3_history.history['loss'] + p3_history_ft.history['loss']
p3_val_loss = p3_history.history['val_loss'] + p3_history_ft.history['val_loss']
p3_split_at = len(p3_history.history['accuracy'])

p3_fig, (p3_ax1, p3_ax2) = plt.subplots(1, 2, figsize=(14, 5))
p3_ax1.plot(p3_acc, label='Train'); p3_ax1.plot(p3_val_acc, label='Val')
p3_ax1.axvline(p3_split_at - 1, color='gray', ls='--', label='Fine-tune 시작')
p3_ax1.set_title('Accuracy'); p3_ax1.legend()
p3_ax2.plot(p3_loss, label='Train'); p3_ax2.plot(p3_val_loss, label='Val')
p3_ax2.axvline(p3_split_at - 1, color='gray', ls='--')
p3_ax2.set_title('Loss'); p3_ax2.legend()
plt.tight_layout(); plt.show()

# Train 만 오르고 Val 이 정체 → 과적합 (증강 강화, Dropout ↑, Phase2 에폭 ↓)
# Phase 2 에서 급락 → 학습률이 높음 (1e-5 확인)

# ============================================================
# 8. 최종 평가 — 순서 보장 데이터셋으로 예측
# ============================================================
# %%
p3_probs = p3_model.predict(p3_val_pred_ds, verbose=1)
p3_all_preds = np.argmax(p3_probs, axis=1)
p3_all_confs = p3_probs.max(axis=1)
p3_all_labels = np.array(p3_val_true_idx)

p3_acc_final = accuracy_score(p3_all_labels, p3_all_preds)
p3_f1_final = f1_score(p3_all_labels, p3_all_preds, average='macro')
print(f"Accuracy {p3_acc_final:.4f} / F1(macro) {p3_f1_final:.4f}")
print(classification_report(p3_all_labels, p3_all_preds, target_names=p3_class_names))

p3_grade = next((g for t, g in [(0.95, '매우 우수'), (0.90, '우수'),
                             (0.85, '양호'), (0.80, '보통')] if p3_acc_final >= t), '미흡')
print(f"-> {p3_grade}")

p3_cm = confusion_matrix(p3_all_labels, p3_all_preds)
plt.figure(figsize=(6 + p3_NUM_CLASSES, 5 + p3_NUM_CLASSES * 0.4))
sns.heatmap(p3_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=p3_class_names, yticklabels=p3_class_names)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.show()

# %%
# 검증 결과 저장 (파일명과 예측이 1:1 대응)
p3_val_result = pd.DataFrame({
    'filename': p3_val_filenames,
    'actual_name': p3_val_true_names,
    # ★ argmax 결과(0/1/2)가 아니라 class_names 문자열로 — 숫자 그대로 내면 0점
    'predicted_name': [p3_class_names[p3_i] for p3_i in p3_all_preds],
    'confidence': p3_all_confs,
    'correct': (p3_all_labels == p3_all_preds).astype(int),
})
print(p3_val_result.head(10))

# ============================================================
# 9. 테스트 예측 및 제출  ★
# ============================================================
# %%
def p3_predict_test_directory(p3_model, test_dir, p3_class_names,
                           image_size=p3_IMAGE_SIZE, batch_size=p3_BATCH_SIZE):
    """라벨 없는 test 폴더의 이미지를 예측해 DataFrame 으로 반환."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    filenames = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(exts))
    p3_images = np.array([
        tf.keras.utils.img_to_array(
            tf.keras.utils.load_img(os.path.join(test_dir, f), target_size=image_size))
        for f in filenames
    ])
    preds = p3_model.predict(p3_images, batch_size=batch_size, verbose=1)
    idx = np.argmax(preds, axis=1)
    return pd.DataFrame({
        'filename': filenames,
        'predicted': [p3_class_names[p3_i] for p3_i in idx],
        'confidence': preds.max(axis=1),
    })


# --- 시험 환경: 별도 test 폴더가 있는 경우 ★ ---
# TEST_DIR = './test'
# submission = p3_predict_test_directory(p3_model, TEST_DIR, p3_class_names)
# submission.to_csv(p3_SUBMIT_CSV, index=False)
# print(f"[저장] {p3_SUBMIT_CSV} ({len(submission)}건)")

# --- test 폴더가 없으면 검증 결과를 제출물로 ---
p3_val_result.to_csv(p3_SUBMIT_CSV, index=False)
print(f"[저장] {p3_SUBMIT_CSV} ({len(p3_val_result)}건)")

# %%
p3_model.save(p3_SUBMIT_KERAS)
print(f"모델 저장 완료: {p3_SUBMIT_KERAS}")

# 체크포인트(_ckpt)는 튕김 대비용 작업 파일입니다. 제출물과 구분해서 확인하세요.
_files = [f for f in os.listdir('.') if f.startswith(PHONE)]
print("제출물   :", [f for f in _files if '_ckpt' not in f])
print("작업 파일:", [f for f in _files if '_ckpt' in f])

# ============================================================
# 10. 제출물 검증  ★ 문제를 끝낼 때마다
# ============================================================
# %%
# label 미복원(0점) 방지: 예측 컬럼 값이 class_names 문자열인지까지 확인.
# (별도 test 폴더로 제출했다면 n_rows 를 그 파일 수로 바꿀 것)
try:
    check_submission(p3_SUBMIT_CSV, n_rows=len(p3_val_result),
                     label_col='predicted_name', allowed_labels=p3_class_names,
                     model_path=p3_SUBMIT_KERAS)
except NameError:
    p3__sub = pd.read_csv(p3_SUBMIT_CSV)
    assert len(p3__sub) == len(p3_val_result), "행 수 불일치!"
    assert set(p3__sub['predicted_name']) <= set(p3_class_names), "label 미복원 의심 — 0점 위험!"
    assert os.path.getsize(p3_SUBMIT_KERAS) > 0, "모델 파일 없음!"
    print("최소 점검 통과 (전체 검증은 세션 시작 셀의 check_submission)")

# %%
# --- 눈 검증 ★ 이미지를 직접 띄워 예측과 대조한다 ---
# 초록 제목 = 실제와 일치, 빨강 = 불일치.
# 그림과 예측이 상식적으로 안 맞으면(꽃인데 전부 한 클래스 등) 제출 전에 원인부터.
p3_rng = np.random.RandomState(RANDOM_SEED)
p3_pick = p3_rng.choice(len(p3_val_file_paths),
                        size=min(12, len(p3_val_file_paths)), replace=False)
plt.figure(figsize=(15, 8))
for p3_j, p3_k in enumerate(p3_pick):
    plt.subplot(3, 4, p3_j + 1)
    plt.imshow(tf.keras.utils.load_img(p3_val_file_paths[p3_k]))
    p3_ok = p3_all_preds[p3_k] == p3_val_true_idx[p3_k]
    plt.title(f"P:{p3_class_names[p3_all_preds[p3_k]]} / T:{p3_val_true_names[p3_k]}",
              fontsize=9, color='green' if p3_ok else 'red')
    plt.axis('off')
plt.suptitle('Eyeball check: prediction vs actual')
plt.tight_layout(); plt.show()
