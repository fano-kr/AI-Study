# 03. Image — 2단계 전이학습

**소요 9분** · 이전: [02. Text](02-text.md) · 다음: [04. 시험 당일](04-exam-day.md)
**실습:** [notebooks/03_image_quickstart.ipynb](../notebooks/03_image_quickstart.ipynb)
**템플릿:** [templates/image_classification.py](../templates/image_classification.py)

---

이미지 분류는 세 도메인 중 **판단할 게 가장 적습니다.** 대신 레시피를 정확히 따라야 합니다.
CNN을 처음부터 학습시키는 선택지는 시험에 없습니다. 데이터도 시간도 부족하기 때문입니다.
답은 **전이학습(Transfer Learning)** 하나입니다.

## 1. 왜 전이학습인가

ImageNet 140만 장으로 이미 학습된 모델은 "선, 질감, 모서리, 눈, 바퀴" 같은 시각적 특징을 알고 있습니다.
꽃 5종을 구분하는 과제에서 우리가 새로 가르쳐야 할 것은 **그 특징들을 어떻게 조합해 꽃 이름을 붙이는가**뿐입니다.

그래서 학습을 두 단계로 나눕니다.

```
Phase 1 — Feature Extraction   base 동결, 새 분류기만 학습    lr = 1e-3   빠르게 수렴
Phase 2 — Fine-Tuning          base 상위 일부 해제, 미세 조정  lr = 1e-5   마지막 몇 %를 올림
```

**Phase 2에서 학습률을 100배 낮추는 것이 핵심입니다.**
높은 학습률로 base를 풀면 ImageNet에서 배운 가중치가 첫 배치에 파괴됩니다.
"미세 조정"이라는 이름이 붙은 이유가 그것입니다.

## 2. 데이터 로드

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',      # 이진/다중 통일 (중요)
    validation_split=0.2, subset='training',
    image_size=(224, 224),         # MobileNetV2 기본 입력
    batch_size=32, seed=RANDOM_SEED, shuffle=True,
)
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
```

폴더 구조가 곧 라벨입니다. 하위 폴더 이름이 알파벳 순으로 클래스 0, 1, 2… 가 됩니다.

```
data/
  daisy/     img1.jpg ...
  rose/      img7.jpg ...
  tulip/     img9.jpg ...
```

**`label_mode='categorical'`을 반드시 쓰세요.**
기본값 `'int'`나 `'binary'`를 쓰면 클래스 수가 바뀔 때마다 손실 함수와 출력층을 고쳐야 합니다.
`categorical` + `softmax` + `categorical_crossentropy`로 통일하면 **클래스가 2개든 7개든 코드가 같습니다.**
[00. 오리엔테이션](00-orientation.md)의 세 번째 습관이 여기서 가장 크게 값을 합니다.

성능을 위해 파이프라인을 예열합니다.

```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)   # GPU가 계산하는 동안 CPU가 다음 배치 준비
val_ds   = val_ds.prefetch(AUTOTUNE)
```

## 3. 데이터 증강 — 모델 안에 넣는다

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])
```

증강 레이어를 **모델의 일부로** 넣으면 학습 때만 동작하고 추론 때는 자동으로 꺼집니다.
데이터셋 파이프라인에서 따로 증강하는 방식보다 실수가 적습니다.

`RandomFlip`에서 `"horizontal"`만 쓰는 이유가 있습니다.
상하 반전은 대부분의 사진에서 물리적으로 말이 안 됩니다(뒤집힌 개는 데이터에 없습니다).
숫자·문자 이미지라면 좌우 반전조차 빼야 합니다. 6과 9가 뒤집히면 라벨이 틀려지니까요.
**증강은 "현실에서 일어날 수 있는 변형"만 넣는 것**이 원칙입니다.

## 4. 모델 구축

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), weights='imagenet', include_top=False
)
base_model.trainable = False                      # Phase 1: 동결

inputs  = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)   # 0~255 → -1~1
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
```

줄별로 짚을 것:

- **`Rescaling(1./127.5, offset=-1)`** — MobileNetV2는 −1~1 입력으로 학습됐습니다. 0~1로 넣으면 성능이 떨어집니다. 모델마다 요구 범위가 다르니 바꿀 때 확인하세요.
- **`GlobalAveragePooling2D`** — `Flatten` 대신 씁니다. `Flatten`은 파라미터가 폭증해 과적합을 부릅니다.
- **`base_model(x, training=False)`** — 동결 상태에서 BatchNorm이 통계를 갱신하지 않게 합니다. 이걸 빠뜨리면 학습이 불안정해집니다.
- **`Dense(NUM_CLASSES, softmax)`** — 이진이든 다중이든 동일. 클래스 수만 바뀝니다.

### 백본 선택

| 모델 | 특징 | 언제 |
|---|---|---|
| MobileNetV2 | 가볍고 빠름 | **시험 기본값** |
| EfficientNetB0 | 성능/속도 균형 | 시간 여유 있을 때 |
| ResNet50 | 무겁고 강력 | 정확도가 정말 부족할 때 |

시험에서는 **끝까지 도는 것**이 우선이므로 MobileNetV2로 시작하세요.

## 5. 두 단계 학습

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True,
                                       monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
]

# Phase 1
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

# Phase 2 — 상위 30개 레이어만 해제
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),   # 반드시 재컴파일
              loss='categorical_crossentropy', metrics=['accuracy'])

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
```

**`trainable`을 바꾼 뒤에는 반드시 재컴파일해야 합니다.** 안 하면 변경이 반영되지 않습니다.

**상위 30개만 푸는 이유** — 하위 레이어는 선·모서리 같은 범용 특징을 담고 있어 어떤 과제에나 유효합니다.
상위로 갈수록 ImageNet에 특화된 특징이라 우리 데이터에 맞게 조정할 가치가 있습니다.
전부 풀면 데이터가 적은 상황에서 과적합합니다.

`EarlyStopping(restore_best_weights=True)` 덕분에 `epochs`를 넉넉히 잡아도 안전합니다.
"몇 에폭이 적당할까"를 고민하지 말고 30을 넣고 콜백에 맡기세요.

## 6. 예측 순서 함정 — 이 차시에서 가장 중요한 함정

```python
val_ds = image_dataset_from_directory(..., shuffle=True)
preds  = model.predict(val_ds)
filenames = val_ds.file_paths        # ← preds 와 순서가 다르다!
```

`shuffle=True`인 데이터셋은 **반복할 때마다 순서가 바뀝니다.**
그런데 `file_paths`는 원래 정렬 순서 그대로입니다. 둘을 나란히 붙이면 **파일명과 예측이 어긋난 채 제출**됩니다.
검증 정확도는 정상이라 눈치채기 어렵고, 제출 점수만 랜덤 수준으로 나옵니다.

해결: 예측용 데이터셋을 순서 보장되게 따로 만듭니다.

```python
def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, IMAGE_SIZE)

val_pred_ds = (tf.data.Dataset.from_tensor_slices(val_ds.file_paths)
               .map(_load_image, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE).prefetch(AUTOTUNE))

preds = model.predict(val_pred_ds)    # file_paths 순서와 1:1 대응 보장
```

## 7. 제출

test 폴더는 보통 라벨 없이 이미지만 들어 있습니다.

```python
filenames = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg','.jpeg','.png')))
images = np.array([tf.keras.utils.img_to_array(
    tf.keras.utils.load_img(os.path.join(TEST_DIR, f), target_size=IMAGE_SIZE)
) for f in filenames])

preds = model.predict(images)
submission = pd.DataFrame({
    'filename':   filenames,
    'predicted':  [class_names[i] for i in np.argmax(preds, axis=1)],
    'confidence': preds.max(axis=1),
})
submission.to_csv('submission.csv', index=False)
```

`sorted()`로 파일명을 정렬해 두면 채점 기준 파일과 순서를 맞추기 쉽습니다.

## 8. 성능이 안 나올 때

| 증상 | 원인 | 조치 |
|---|---|---|
| train acc 높고 val acc 낮음 | 과적합 | 증강 강화, Dropout ↑, Phase 2 에폭 ↓ |
| 둘 다 낮음 | 과소적합 | Phase 2 레이어 더 풀기, 더 큰 백본 |
| Phase 2에서 급락 | 학습률이 높음 | `1e-5` 확인, 그래도 나쁘면 `1e-6` |
| 학습이 아예 안 됨 | 입력 정규화 불일치 | `Rescaling` 범위가 백본 요구와 맞는지 |

---

**정리** — 이미지는 레시피 도메인입니다. `categorical` 통일, 2단계 학습(1e-3 → 1e-5),
그리고 **shuffle된 데이터셋으로 예측하지 않기.** 이 셋이면 대부분 통과합니다.

다음: [04. 시험 당일 체크리스트](04-exam-day.md)
