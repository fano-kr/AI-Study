# %% [markdown]
# # Image 로컬라이제이션 템플릿 (BBox 회귀 + 분류, MobileNetV2 이중 헤드)
#
# 교재: docs/03-image.md (로컬라이제이션 절)
#
# 3번 이미지 문제가 "분류"면 03_image_classification.py 를,
# "객체 위치(Bounding Box)"까지 요구하면 이 파일을 씁니다.
#
# 핵심 — 분류 템플릿에서 바뀌는 것은 셋뿐입니다.
#   1) 라벨: 폴더명 대신 XML 에서 좌표(x,y,w,h)와 클래스를 읽는다
#   2) 출력층: Dense(4, sigmoid) 좌표 + Dense(N, softmax) 클래스를 Concatenate
#   3) 손실: 앞 4개는 MSE, 뒤는 분류 CE — 커스텀 함수 하나
#
# 주의: 데이터 증강을 넣지 않습니다. 좌우 반전·회전은 이미지만 바꾸고
#       정답 좌표는 그대로 두므로 라벨이 통째로 틀려집니다.
#
# ── 변수 이름 규칙 ──
# 3번 문제 슬롯이므로 분류 템플릿과 같은 p3_ 접두사입니다.
# 같은 세션에서 분류와 로컬라이제이션을 둘 다 돌릴 일은 없습니다(문제는 하나).

# %%
import os
import xml.etree.ElementTree as et
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# 0. 문제 설정
# ============================================================
# GPU 확인과 PHONE 은 세션 시작 셀(00_session_start.py)에서 합니다.
p3_PROB_NO = 3                # ★ 시험 순서: Image = 3번

try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

p3_SUBMIT_CSV = f'{PHONE}_{p3_PROB_NO}.csv'
p3_SUBMIT_KERAS = f'{PHONE}_{p3_PROB_NO}.keras'        # 지문 ④의 확장자 그대로
p3_PRED_COL = 'pred'                                   # ★ 지문 ③의 클래스 컬럼명 그대로
print(f"제출 파일: {p3_SUBMIT_CSV} / {p3_SUBMIT_KERAS} / 결과 컬럼: {p3_PRED_COL}")

# ============================================================
# 1. 하이퍼파라미터
# ============================================================
# %%
p3_IMAGE_SIZE = (224, 224)
p3_INPUT_SHAPE = (224, 224, 3)
p3_BATCH_SIZE = 32
p3_EPOCHS = 30                # EarlyStopping 이 알아서 멈춘다
p3_LEARNING_RATE = 1e-3       # base 동결 상태
p3_LOC_WEIGHT = 5.0           # 손실에서 좌표 항 가중 — bbox 가 안 맞으면 올린다

# ============================================================
# 2. 데이터 준비 — XML 에서 좌표와 클래스를 읽는다  ★
# ============================================================
# %%
# 시험 데이터 구조 (Oxford Pet 형식):
#   data_dir/images/            *.jpg
#   data_dir/annotations/xmls/  *.xml   ← 이미지와 같은 이름
# 이미지 수 > XML 수인 경우가 보통 — 라벨이 있는 XML 을 기준으로 모은다.
p3_data_dir = './oxford_pet'                       # ★ 실제 경로
p3_image_dir = os.path.join(p3_data_dir, 'images')
p3_bbox_dir = os.path.join(p3_data_dir, 'annotations', 'xmls')

p3_bbox_files = sorted(glob(os.path.join(p3_bbox_dir, '*.xml')))
print(f"XML {len(p3_bbox_files)}개 (이 수가 학습 데이터 수)")


def p3_parse_xml(xml_path):
    """XML 하나 → (x, y, w, h) 0~1 정규화 + 클래스명.

    XML 은 실제 픽셀 좌표(xmin,ymin,xmax,ymax)로 저장돼 있다.
    모델이 회귀하기 좋게 이미지 크기로 나눠 0~1 의 중심점(x,y)+크기(w,h)로 바꾼다.
    (객체가 여러 개인 XML 은 첫 번째 object 만 쓴다 — 시험 데이터는 보통 1개)
    """
    t = et.parse(xml_path)
    W = float(t.find('./size/width').text)
    H = float(t.find('./size/height').text)
    xmin = float(t.find('./object/bndbox/xmin').text)
    ymin = float(t.find('./object/bndbox/ymin').text)
    xmax = float(t.find('./object/bndbox/xmax').text)
    ymax = float(t.find('./object/bndbox/ymax').text)
    name = t.find('./object/name').text
    x = (xmin + xmax) / 2 / W
    y = (ymin + ymax) / 2 / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return x, y, w, h, name


# XML 기준으로 DataFrame 구성 (짝이 되는 이미지가 있는 것만)
p3_rows = []
for p3_f in p3_bbox_files:
    p3_stem = os.path.splitext(os.path.basename(p3_f))[0]
    p3_img = os.path.join(p3_image_dir, p3_stem + '.jpg')
    if not os.path.exists(p3_img):
        continue
    p3_x, p3_y, p3_w, p3_h, p3_name = p3_parse_xml(p3_f)
    p3_rows.append((p3_img, os.path.basename(p3_img), p3_x, p3_y, p3_w, p3_h, p3_name))

p3_df = pd.DataFrame(p3_rows, columns=['path', 'filename', 'x', 'y', 'w', 'h', 'label'])
p3_class_names = sorted(p3_df['label'].unique())
p3_NUM_CLASSES = len(p3_class_names)
p3_df['label_idx'] = p3_df['label'].map({n: i for i, n in enumerate(p3_class_names)})

print(f"학습 {len(p3_df)}건 / 클래스 {p3_NUM_CLASSES}개: {p3_class_names}")
print(p3_df.head(3))

# ============================================================
# 3. tf.data 파이프라인 — y = [x, y, w, h, label_idx] 5개
# ============================================================
# %%
p3_df = p3_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
p3_n_val = max(1, int(len(p3_df) * 0.2))
p3_df_val, p3_df_tr = p3_df.iloc[:p3_n_val], p3_df.iloc[p3_n_val:]
print(f"Train {len(p3_df_tr)} / Val {len(p3_df_val)}")

p3_AUTOTUNE = tf.data.AUTOTUNE


def p3_load(path, target):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, p3_IMAGE_SIZE)
    return img, target


def p3_make_ds(d, shuffle):
    y = d[['x', 'y', 'w', 'h', 'label_idx']].values.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((d['path'].values, y))
    if shuffle:
        ds = ds.shuffle(1000, seed=RANDOM_SEED)
    return ds.map(p3_load, num_parallel_calls=p3_AUTOTUNE).batch(p3_BATCH_SIZE).prefetch(p3_AUTOTUNE)


p3_train_ds = p3_make_ds(p3_df_tr, shuffle=True)
p3_val_ds = p3_make_ds(p3_df_val, shuffle=False)   # 순서 보장 — 평가·제출에 그대로 사용

# ============================================================
# 4. 모델 — 출력층만 다르다: 좌표 4 + 클래스 N 을 Concatenate  ★
# ============================================================
# %%
p3_base = tf.keras.applications.MobileNetV2(
    input_shape=p3_INPUT_SHAPE, weights='imagenet', include_top=False)
p3_base.trainable = False

p3_inputs = tf.keras.Input(shape=p3_INPUT_SHAPE)
p3_x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(p3_inputs)
p3_x = p3_base(p3_x, training=False)
p3_x = tf.keras.layers.GlobalAveragePooling2D()(p3_x)
p3_x = tf.keras.layers.Dense(128, activation='relu')(p3_x)
p3_out_xywh = tf.keras.layers.Dense(4, activation='sigmoid', name='xywh')(p3_x)      # 좌표는 0~1 회귀
p3_out_cls = tf.keras.layers.Dense(p3_NUM_CLASSES, activation='softmax', name='cls')(p3_x)
p3_outputs = tf.keras.layers.Concatenate()([p3_out_xywh, p3_out_cls])

p3_model = tf.keras.Model(p3_inputs, p3_outputs)


def p3_loss_fn(y_true, y_pred):
    """앞 4개(좌표)는 MSE, 뒤(클래스)는 분류 CE — 둘을 합친다."""
    loc_loss = tf.reduce_mean(tf.square(y_true[:, :4] - y_pred[:, :4]))
    cls_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        tf.cast(y_true[:, 4], tf.int32), y_pred[:, 4:]))
    return p3_LOC_WEIGHT * loc_loss + cls_loss


p3_model.compile(optimizer=tf.keras.optimizers.Adam(p3_LEARNING_RATE), loss=p3_loss_fn)

# ============================================================
# 5. 학습
# ============================================================
# %%
p3_CKPT = f'{PHONE}_{p3_PROB_NO}_ckpt.weights.h5'   # 버전 무관 가중치 저장 (커스텀 loss → load_weights 로 복원)
p3_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # 학습 도중 세션이 튕겨도 최고 가중치는 파일로 남는다
    tf.keras.callbacks.ModelCheckpoint(p3_CKPT,
                                       save_best_only=True, save_weights_only=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
]

p3_history = p3_model.fit(p3_train_ds, validation_data=p3_val_ds,
                          epochs=p3_EPOCHS, callbacks=p3_callbacks, verbose=1)

# ★ best 가중치 보장 — restore_best_weights 에 의존하지 않고 체크포인트를 다시 읽는다
p3_model.load_weights(p3_CKPT)

# ============================================================
# 6. 평가 — 좌표는 IoU, 클래스는 정확도
# ============================================================
# %%
p3_pred = p3_model.predict(p3_val_ds, verbose=0)
p3_true = p3_df_val[['x', 'y', 'w', 'h', 'label_idx']].values.astype('float32')


def p3_iou(a, b):
    """xywh(0~1) 두 상자의 IoU. a, b: (N, 4)"""
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.clip(union, 1e-9, None)


p3_ious = p3_iou(p3_true[:, :4], p3_pred[:, :4])
p3_cls_pred = np.argmax(p3_pred[:, 4:], axis=1)
p3_cls_acc = (p3_cls_pred == p3_true[:, 4].astype(int)).mean()

print(f"mean IoU  : {p3_ious.mean():.4f}   (0.5 이상이면 대체로 양호)")
print(f"IoU ≥ 0.5 : {(p3_ious >= 0.5).mean():.2%}")
print(f"분류 정확도: {p3_cls_acc:.4f}")

# ============================================================
# 7. 눈 검증 ★ 이미지에 예측(파랑) vs 정답(초록) 상자를 겹쳐 본다
# ============================================================
# %%
def p3_draw_box(ax, xywh, W, H, color):
    x, y, w, h = xywh
    ax.add_patch(Rectangle(((x - w / 2) * W, (y - h / 2) * H), w * W, h * H,
                           fill=False, edgecolor=color, linewidth=2))


p3_rng = np.random.RandomState(RANDOM_SEED)
p3_pick = p3_rng.choice(len(p3_df_val), size=min(8, len(p3_df_val)), replace=False)
plt.figure(figsize=(16, 8))
for p3_j, p3_k in enumerate(p3_pick):
    ax = plt.subplot(2, 4, p3_j + 1)
    p3_img = plt.imread(p3_df_val['path'].iloc[p3_k])
    H, W = p3_img.shape[:2]
    ax.imshow(p3_img)
    p3_draw_box(ax, p3_true[p3_k, :4], W, H, 'lime')            # 정답
    p3_draw_box(ax, p3_pred[p3_k, :4], W, H, 'dodgerblue')      # 예측
    ax.set_title(f"P:{p3_class_names[p3_cls_pred[p3_k]]} IoU:{p3_ious[p3_k]:.2f}", fontsize=9)
    ax.axis('off')
plt.suptitle('Eyeball: green=actual, blue=predicted')
plt.tight_layout(); plt.show()

# ============================================================
# 8. 제출  ★ 좌표 형식(x,y,w,h vs xmin,ymin,xmax,ymax)은 문제 지문 그대로!
# ============================================================
# %%
# 좌표 컬럼명(x,y,w,h vs xmin~ymax)과 클래스 컬럼명 모두 지문 ③ 그대로
p3_submission = pd.DataFrame({
    'filename': p3_df_val['filename'].values,
    'x': p3_pred[:, 0], 'y': p3_pred[:, 1],
    'w': p3_pred[:, 2], 'h': p3_pred[:, 3],
    p3_PRED_COL: [p3_class_names[i] for i in p3_cls_pred],   # label 복원 — 숫자면 0점
})
# 문제가 xmin/ymin/xmax/ymax 픽셀 좌표를 요구하면 여기서 변환해서 내세요.

p3_submission.to_csv(p3_SUBMIT_CSV, index=False)
p3_model.save(p3_SUBMIT_KERAS)
print(f"[저장] {p3_SUBMIT_CSV} ({len(p3_submission)}건) / {p3_SUBMIT_KERAS}")

# %%
# --- 제출물 검증 (check_submission 은 세션 시작 셀에서 정의) ---
try:
    check_submission(p3_SUBMIT_CSV, n_rows=len(p3_df_val),
                     label_col=p3_PRED_COL, allowed_labels=p3_class_names,
                     model_path=p3_SUBMIT_KERAS)
except NameError:
    p3__sub = pd.read_csv(p3_SUBMIT_CSV)
    assert len(p3__sub) == len(p3_df_val), "행 수 불일치!"
    assert set(p3__sub[p3_PRED_COL]) <= set(p3_class_names), "label 미복원 — 0점 위험!"
    p3__xy = p3__sub[['x', 'y', 'w', 'h']].values
    assert ((p3__xy >= 0) & (p3__xy <= 1)).all(), "좌표가 0~1 을 벗어남!"
    print("최소 점검 통과 (전체 검증은 세션 시작 셀의 check_submission)")
