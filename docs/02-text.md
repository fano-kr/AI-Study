# 02. Text — 두 갈래 길

**소요 10분** · 이전: [01. Tabular](01-tabular.md) · 다음: [03. Image](03-image.md)
**실습:** [notebooks/02_text_quickstart.ipynb](../notebooks/02_text_quickstart.ipynb)
**템플릿:** [templates/text_classification.py](../templates/text_classification.py)

---

텍스트 분류에는 길이 두 개 있습니다. 시험에서는 **어느 길로 갈지 먼저 정하는 것**이 실력입니다.

| | 방법 A — TF-IDF + ML | 방법 B — BERT Fine-tuning |
|---|---|---|
| 소요 시간 | 1~5분 | 20~40분 |
| 성능 | 짧은 문장·명확한 주제에서 매우 강함 | 문맥·뉘앙스가 중요할 때 우세 |
| 실패 위험 | 거의 없음 | 다운로드 실패, OOM, 시간 초과 |

*소요 시간은 Colab T4 기준입니다.*

## 1. 전략: A를 먼저, B는 보험

**항상 방법 A부터 돌리세요.** 5분 안에 점수가 나오고, `submission.csv`가 하나 확보됩니다.

이 순서가 중요한 이유는 시험이 시간 제한 경기이기 때문입니다.
BERT부터 시작했다가 30분 뒤에 OOM으로 죽으면 제출할 게 아무것도 없습니다.
반면 A를 먼저 끝내 두면, B가 실패해도 이미 제출물이 손에 있습니다.

```
A 실행 (5분) → submission.csv 확보 → 점수 확인
   ├─ 0.90 이상 → 끝. 다음 문제로.
   └─ 0.85 미만이고 시간 30분+ 남음 → B 시도 → 더 좋으면 덮어쓰기
```

TF-IDF + LinearSVC는 짧은 텍스트 분류에서 생각보다 훨씬 강합니다.
뉴스 제목 토픽 분류 같은 과제에서는 BERT와 차이가 크지 않은 경우도 흔합니다.

## 2. 공통 — 텍스트 정제

```python
def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)                  # HTML 태그
    text = re.sub(r'http\S+|www\.\S+', ' ', text)         # URL
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)      # 특수문자
    text = re.sub(r'\s+', ' ', text).strip()              # 다중 공백
    return text

df_train[TEXT_COL] = df_train[TEXT_COL].apply(clean_text)
df_test[TEXT_COL]  = df_test[TEXT_COL].apply(clean_text)
```

정제 강도는 방법에 따라 다르게 잡습니다.

- **TF-IDF** — 세게 해도 됩니다. 단어 등장 횟수만 보므로 문장부호가 사라져도 손해가 적습니다.
- **BERT** — 약하게 하세요. BERT는 문맥을 읽기 때문에 `?`, `!`, 어미 같은 신호가 정보입니다. HTML·URL 제거 정도면 충분합니다.

정제 후 **빈 문자열이 생겼는지** 확인하세요. 특수문자만으로 이루어진 행이 있으면 정제 후 빈칸이 되고, 벡터화 단계에서 조용히 문제를 일으킵니다.

## 3. 방법 A — TF-IDF + ML

```python
tfidf = TfidfVectorizer(
    max_features=10000,     # 상위 1만 단어만 (메모리/속도)
    ngram_range=(1, 2),     # 단어 + 2단어 조합. "not good" 같은 표현을 잡는다
    min_df=2,               # 문서 1개에만 나오는 단어 제거 (오타·고유명사 노이즈)
    max_df=0.95,            # 95% 이상 문서에 나오는 단어 제거 (변별력 없음)
    sublinear_tf=True,      # TF에 log 스케일. 긴 문서의 과대평가 완화
)

X_train = tfidf.fit_transform(df_tr[TEXT_COL])   # train: fit
X_val   = tfidf.transform(df_val[TEXT_COL])      # val:   transform
X_test  = tfidf.transform(df_test[TEXT_COL])     # test:  transform
```

`fit`은 train에만. Tabular에서 배운 원칙이 그대로입니다.
test까지 포함해 `fit`하면 test의 단어 분포가 모델에 새어 들어갑니다.

**`ngram_range=(1,2)`가 기본값인 이유** — 감성 분석에서 "좋지 않다"의 "않다"는 단어 하나로는 부정을 못 잡습니다.
바이그램을 넣으면 잡힙니다. 대신 피처 수가 몇 배로 늘어나므로 `max_features`로 상한을 겁니다.

### 한국어라면 반드시 확인할 것 — 문자 n-gram

`TfidfVectorizer`의 기본 토큰화는 **공백 기준**입니다. 영어에는 잘 맞지만 한국어에는 잘 맞지 않습니다.
조사와 어미가 단어에 붙기 때문입니다.

```
"영화가"  "영화는"  "영화를"  "영화도"   →  전부 다른 피처로 흩어진다
```

정석은 형태소 분석기(KoNLPy, Mecab)지만 **시험 환경에 설치돼 있으리라는 보장이 없습니다.**
설치 없이 쓸 수 있는 우회로가 문자 단위 n-gram입니다.

```python
tfidf = TfidfVectorizer(
    analyzer='char_wb',    # 단어 경계를 존중하는 문자 n-gram
    ngram_range=(2, 4),    # 2~4글자 조각
    max_features=50000,    # 문자 조합은 종류가 많으므로 상한을 올린다
    min_df=2, sublinear_tf=True,
)
```

한국어 데이터셋 두 개로 실측한 결과입니다(세 모델 중 최고 점수).

| 데이터 | 단어 (1,2)-gram | 문자 `char_wb` (2,4)-gram | 차이 |
|---|---|---|---|
| NSMC 감성분석 (2클래스, 3만 건) | 0.781 | **0.847** | +6.6%p |
| KLUE-YNAT 뉴스 토픽 (7클래스) | 0.670 | **0.785** | +11.6%p |

문장이 짧고 클래스가 많은 YNAT에서 차이가 더 벌어집니다.
조사·어미가 붙어도 `재미없`, `최악이` 같은 조각은 그대로 잡히기 때문입니다.

대가는 피처가 5배로 늘어 벡터화가 3배쯤 느려지는 것인데, 몇 초 수준이라 시험에서는 거의 항상 이득입니다.

> **기억할 것** — 한국어 텍스트에서 TF-IDF 점수가 기대보다 낮으면,
> 모델을 바꾸기 전에 `analyzer='char_wb'`를 먼저 시도하세요.
> 실습 노트북에서 두 방식을 나란히 돌려 확인합니다.

### 모델 후보

```python
ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'LinearSVC':          LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
    'MultinomialNB':      MultinomialNB(alpha=0.1),
}
```

이 셋이면 충분합니다. 고차원 희소 행렬(TF-IDF의 출력)에서 선형 모델이 강하기 때문입니다.

RandomForest는 **일부러 뺐습니다.** 1만 차원 희소 행렬에서 트리를 키우면 느린 데다 성능도 선형 모델에 밀립니다.
Tabular에서는 주력이던 모델이 여기서는 후보에도 못 든다는 점이, 도메인마다 모델 선택이 달라진다는 걸 보여 줍니다.

> `LinearSVC`는 `predict_proba`가 없습니다. AUC가 필요하면 `LogisticRegression`을 쓰거나
> `CalibratedClassifierCV`로 감싸세요. 시험에서 AUC를 요구하지 않으면 신경 쓸 필요 없습니다.

## 4. 방법 B — BERT Fine-tuning

```python
BERT_MODEL_NAME = 'klue/bert-base'      # 한국어
# BERT_MODEL_NAME = 'bert-base-uncased' # 영어
MAX_LENGTH    = 128    # 토큰 최대 길이
BATCH_SIZE    = 32     # GPU 메모리에 맞춰 조절 (OOM 나면 16 → 8)
EPOCHS        = 5      # EarlyStopping이 알아서 멈춘다
LEARNING_RATE = 2e-5   # BERT 표준. 1e-5 ~ 5e-5 범위를 벗어나지 말 것
```

**모델 선택이 첫 번째 갈림길입니다.** 한국어 데이터에 `bert-base-uncased`를 쓰면
토크나이저가 한글을 거의 `[UNK]`로 처리해 성능이 무너집니다. 데이터 언어를 먼저 확인하세요.

**`LEARNING_RATE = 2e-5`는 협상 대상이 아닙니다.**
사전학습된 가중치를 "미세하게" 조정하는 게 fine-tuning의 전부입니다.
Tabular에서 쓰던 `1e-3` 같은 값을 넣으면 사전학습 지식이 첫 배치에서 파괴되고, 정확도가 랜덤 수준에 머뭅니다.

```python
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
train_enc = tokenizer(X_train_text, truncation=True, padding=True,
                      max_length=MAX_LENGTH, return_tensors='tf')

model = TFBertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=NUM_CLASSES, from_pt=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
```

세 가지 함정:

- **`from_logits=True`** — `TFBertForSequenceClassification`은 softmax를 거치지 않은 logits를 뱉습니다. 이걸 빠뜨리면 손실이 이상하게 계산되고 학습이 안 됩니다.
- **`from_pt=True`** — Hugging Face의 한국어 모델 다수가 PyTorch 가중치만 올라와 있습니다. TF에서 쓰려면 이 옵션이 필요합니다.
- **예측은 `.logits`에서** — `model.predict(ds)`의 결과는 텐서가 아니라 객체입니다. `np.argmax(preds.logits, axis=1)`로 꺼내세요.

### MAX_LENGTH 정하는 법

감으로 128을 넣지 말고 데이터를 보고 정합니다.

```python
lengths = df_train[TEXT_COL].astype(str).apply(len)
print(lengths.describe())
print(f"95 퍼센타일: {lengths.quantile(0.95):.0f}")
```

95% 지점을 덮는 값이면 충분합니다. 길이를 2배로 늘리면 학습 시간도 대략 2배가 되므로,
"긴 문서 5%를 살리려고 전체 시간을 2배 쓸 것인가"의 트레이드오프입니다. 시험에서는 보통 자르는 쪽이 이깁니다.

### OOM이 나면

`ResourceExhaustedError`를 만났을 때 순서대로 시도합니다.

1. `BATCH_SIZE` 32 → 16 → 8
2. `MAX_LENGTH` 128 → 64
3. 그래도 안 되면 방법 A로 후퇴 (미련 갖지 말 것)

## 5. 지표와 제출

분류이므로 Tabular와 같습니다. `Accuracy`와 `F1(macro)`를 함께 봅니다.

텍스트는 클래스 불균형이 흔합니다(뉴스 카테고리 중 '정치'만 많다든가).
`Accuracy`가 높은데 `F1(macro)`가 낮으면 **다수 클래스만 맞히고 있다**는 뜻이므로,
`class_weight='balanced'`를 넣거나 소수 클래스 데이터를 확인해야 합니다.

```python
submit_df = df_test.copy()
submit_df['predicted'] = y_pred_test
submit_df['predicted_name'] = submit_df['predicted'].map(label_map)   # 라벨명 요구 시
submit_df.to_csv('submission.csv', index=False)
```

---

**정리** — 텍스트는 A로 안전하게 점수를 확보한 뒤 B로 올리는 도메인입니다.
BERT에서는 `2e-5`, `from_logits=True`, `from_pt=True` 세 개만 정확히 기억하면 나머지는 템플릿이 처리합니다.

다음: [03. Image — 2단계 전이학습](03-image.md)
