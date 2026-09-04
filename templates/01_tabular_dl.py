# %% [markdown]
# # Tabular 딥러닝 템플릿 (문제가 "딥러닝으로 모델링"을 지시할 때)
#
# 교재: docs/01-tabular.md (딥러닝 절)
#
# AICE 는 정형 문제에서 딥러닝(DNN)을 **명시적으로 지시**하는 문항이 나옵니다.
# 보통 레이어 수·유닛 수·활성화까지 지문이 지정합니다 — 그때는 지문이 정답입니다.
# 이 템플릿의 구조는 기본값일 뿐, 지정이 있으면 그대로 바꾸세요.
#
# 공식 실습 정답 스타일을 따릅니다:
#   인코딩   pd.get_dummies (원-핫)  ← DNN 은 트리가 아니라 LabelEncoder 의 가짜 순서가 해롭다
#   스케일링 MinMaxScaler            ← 0~1 입력이 DNN 학습에 안정적
#   모델     Sequential + Dense(relu) 몇 층
#
# ── 변수 이름 규칙 ──
# 1번 문제 슬롯이므로 sklearn 템플릿과 같은 p1_ 접두사입니다.
# 같은 문제에서 sklearn 과 DNN 을 둘 다 요구하면 이 파일을 이어서 실행해도 됩니다
# (전처리 변수명이 겹치지 않게 이 파일은 p1_dl_ 로 시작합니다).

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# 0. 문제 설정
# ============================================================
p1_PROB_NO = 1                # ★ 시험 순서: Tabular = 1번

try:
    PHONE
except NameError:
    PHONE = '01012345678'  # ★ 본인 휴대폰 번호 (하이픈 없이)

p1_dl_SUBMIT_CSV = f'{PHONE}_{p1_PROB_NO}.csv'
p1_dl_SUBMIT_H5 = f'{PHONE}_{p1_PROB_NO}.h5'     # 공식 실습은 h5 — 문제 지정 형식 우선
print(f"제출 파일: {p1_dl_SUBMIT_CSV} / {p1_dl_SUBMIT_H5}")

# ============================================================
# 1. 데이터 로드  ★
# ============================================================
# %%
# --- 시험 환경 ---
# p1_dl_df_train = pd.read_csv('train.csv')
# p1_dl_df_test  = pd.read_csv('test.csv')
# p1_dl_TARGET   = 'target'

# --- 연습용: Titanic ---
import seaborn as sns
p1_dl_all = sns.load_dataset('titanic')
p1_dl_all = p1_dl_all.drop(columns=['alive', 'deck', 'class', 'embark_town',
                                    'who', 'adult_male', 'alone'])
p1_dl_TARGET = 'survived'
p1_dl_df_train, p1_dl_df_test = train_test_split(
    p1_dl_all, test_size=0.2, random_state=RANDOM_SEED, stratify=p1_dl_all[p1_dl_TARGET])
p1_dl_df_train = p1_dl_df_train.reset_index(drop=True)
p1_dl_df_test = p1_dl_df_test.reset_index(drop=True)

p1_dl_NUM_CLASSES = p1_dl_df_train[p1_dl_TARGET].nunique()
print(f"Train {p1_dl_df_train.shape} / Test {p1_dl_df_test.shape} / 클래스 {p1_dl_NUM_CLASSES}개")

# ============================================================
# 2. 전처리 — get_dummies + MinMaxScaler (공식 정답 스타일)
# ============================================================
# %%
p1_dl_X = p1_dl_df_train.drop(columns=[p1_dl_TARGET])
p1_dl_y = p1_dl_df_train[p1_dl_TARGET].copy()
p1_dl_has_label = p1_dl_TARGET in p1_dl_df_test.columns
p1_dl_X_test = p1_dl_df_test.drop(columns=[p1_dl_TARGET]) if p1_dl_has_label else p1_dl_df_test.copy()

# 결측: 수치형 중앙값, 범주형 최빈값 (train 기준으로 test 채움)
for p1_dl_c in p1_dl_X.columns:
    if p1_dl_X[p1_dl_c].dtype.kind in 'biufc':
        p1_dl_fill = p1_dl_X[p1_dl_c].median()
    else:
        p1_dl_fill = p1_dl_X[p1_dl_c].mode()[0]
    p1_dl_X[p1_dl_c] = p1_dl_X[p1_dl_c].fillna(p1_dl_fill)
    p1_dl_X_test[p1_dl_c] = p1_dl_X_test[p1_dl_c].fillna(p1_dl_fill)

# 원-핫 인코딩 — DNN 에는 LabelEncoder 의 가짜 순서(0<1<2)가 해로우므로 get_dummies
p1_dl_X = pd.get_dummies(p1_dl_X)
p1_dl_X_test = pd.get_dummies(p1_dl_X_test)
p1_dl_X_test = p1_dl_X_test.reindex(columns=p1_dl_X.columns, fill_value=0)  # 컬럼 맞추기

# MinMax 스케일링 — DNN 입력은 0~1 이 안정적
p1_dl_scaler = MinMaxScaler()
p1_dl_X_s = p1_dl_scaler.fit_transform(p1_dl_X)
p1_dl_X_test_s = p1_dl_scaler.transform(p1_dl_X_test)

p1_dl_X_tr, p1_dl_X_val, p1_dl_y_tr, p1_dl_y_val = train_test_split(
    p1_dl_X_s, p1_dl_y, test_size=0.2, random_state=RANDOM_SEED, stratify=p1_dl_y)

p1_dl_N_FEATURES = p1_dl_X_tr.shape[1]
print(f"피처 {p1_dl_N_FEATURES}개 (get_dummies 후) / Train {len(p1_dl_X_tr)} / Val {len(p1_dl_X_val)}")

# ============================================================
# 3. 모델 — 지문이 구조를 지정하면 그대로 따를 것  ★
# ============================================================
# %%
# 지문 예시: "input_shape=(38,), 히든 unit 4 → 3, 이진분류 출력층" → 그 숫자 그대로.
# 아래는 지정이 없을 때의 기본값입니다.
p1_dl_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(p1_dl_N_FEATURES,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
])

# 출력층·손실: 지문이 지정하는 형태가 최우선.
# 공식 실습은 이진 = Dense(1, sigmoid) + binary_crossentropy 를 씁니다.
# (습관 3의 softmax 통일은 "선택권이 있을 때"의 기본값 — 지시형 문항에선 지시가 정답)
if p1_dl_NUM_CLASSES == 2:
    p1_dl_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    p1_dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    p1_dl_model.add(tf.keras.layers.Dense(p1_dl_NUM_CLASSES, activation='softmax'))
    p1_dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ============================================================
# 4. 학습 — EarlyStopping + ModelCheckpoint
# ============================================================
# %%
p1_dl_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(f'{PHONE}_{p1_PROB_NO}_ckpt.h5',
                                       save_best_only=True, monitor='val_loss'),
]

p1_dl_history = p1_dl_model.fit(
    p1_dl_X_tr, p1_dl_y_tr,
    validation_data=(p1_dl_X_val, p1_dl_y_val),
    epochs=50, batch_size=32,
    callbacks=p1_dl_callbacks, verbose=1)

# %%
# 학습 곡선 — 지문이 "loss/accuracy 그래프를 그리시오"를 자주 요구합니다
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(p1_dl_history.history['loss'], label='loss')
plt.plot(p1_dl_history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss'); plt.xlabel('Epochs')
plt.subplot(1, 2, 2)
plt.plot(p1_dl_history.history['accuracy'], label='accuracy')
plt.plot(p1_dl_history.history['val_accuracy'], label='val_accuracy')
plt.legend(); plt.title('Accuracy'); plt.xlabel('Epochs')
plt.tight_layout(); plt.show()

# ============================================================
# 5. 평가 · 제출
# ============================================================
# %%
if p1_dl_NUM_CLASSES == 2:
    p1_dl_val_pred = (p1_dl_model.predict(p1_dl_X_val, verbose=0) > 0.5).astype(int).ravel()
    p1_dl_test_pred = (p1_dl_model.predict(p1_dl_X_test_s, verbose=0) > 0.5).astype(int).ravel()
else:
    p1_dl_val_pred = np.argmax(p1_dl_model.predict(p1_dl_X_val, verbose=0), axis=1)
    p1_dl_test_pred = np.argmax(p1_dl_model.predict(p1_dl_X_test_s, verbose=0), axis=1)

print(f"Val Accuracy {accuracy_score(p1_dl_y_val, p1_dl_val_pred):.4f} / "
      f"F1(macro) {f1_score(p1_dl_y_val, p1_dl_val_pred, average='macro'):.4f}")
print(classification_report(p1_dl_y_val, p1_dl_val_pred))

# ★ 타겟을 인코딩했다면 여기서 label 복원 — 숫자 그대로 내면 0점
p1_dl_submission = p1_dl_df_test.copy()
p1_dl_submission['predicted'] = p1_dl_test_pred
p1_dl_submission.to_csv(p1_dl_SUBMIT_CSV, index=False)
p1_dl_model.save(p1_dl_SUBMIT_H5)
print(f"[저장] {p1_dl_SUBMIT_CSV} ({len(p1_dl_submission)}건) / {p1_dl_SUBMIT_H5}")

# %%
# --- 제출물 검증 (check_submission 은 세션 시작 셀에서 정의) ---
try:
    check_submission(p1_dl_SUBMIT_CSV, n_rows=len(p1_dl_df_test),
                     label_col='predicted', allowed_labels=p1_dl_y.unique(),
                     model_path=p1_dl_SUBMIT_H5)
except NameError:
    import os
    p1_dl__sub = pd.read_csv(p1_dl_SUBMIT_CSV)
    assert len(p1_dl__sub) == len(p1_dl_df_test), "행 수 불일치!"
    assert set(p1_dl__sub['predicted']) <= set(p1_dl_y.unique()), "label 미복원 — 0점 위험!"
    assert os.path.getsize(p1_dl_SUBMIT_H5) > 0, "모델 파일 없음!"
    print("최소 점검 통과 (전체 검증은 세션 시작 셀의 check_submission)")
