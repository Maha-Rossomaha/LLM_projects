# Введение в каскад reranking

## 1. Зачем нужен reranking после retrieval

Первичный retrieval (поиск кандидатов) обычно строится на **быстрых, но приближённых** методах:

* **BM25 / инвертированный индекс** (sparse retrieval),
* **bi-encoder** с векторным ANN (dense retrieval).

Эти методы позволяют эффективно отобрать **top-K кандидатов** из миллионов документов. Но их качество ограничено:

* bi-encoder кодирует query и документ независимо, что теряет часть взаимодействий,
* sparse retrieval плохо работает с синонимами и семантикой.

Поэтому вводится **reranking** — второй этап, где отобранные кандидаты ранжируются более точными, но дорогими моделями.

**Ключевая идея:** *retrieval даёт широкое покрытие, reranking повышает точность на малом числе кандидатов.*

---

## 2. Общая схема каскада

### 1. Retrieval (дешёвый слой)

* BM25, SPLADE или bi-encoder (Sentence-BERT, E5, BGE).
* Возвращает top-K (например, K=1000).

### 2. Reranking (средний слой)

* **Late interaction (ColBERT)**: учитывает токеновые взаимодействия (query × doc токены).
* Более точный, но тяжелее, чем bi-encoder.
* Сужает кандидатов до top-M (например, M=100).

### 3. Финальный reranker (дорогой слой)

* **Cross-encoder**: объединяет query и документ в одну последовательность, полное взаимодействие токенов.
* Очень точный, но дорогой (O(L²) self-attention).
* Применяется только к top-N (например, N=20).

### 4. Downstream использование

* В поиске: выдаётся top-N пользователю.
* В рекомендациях: top-N → post-ranking (персонализация, бизнес-правила).
* В RAG: top-N передаётся в LLM как контекст.

**Схема:**

```
Corpus → Retrieval (BM25 / bi-encoder, top-K) → Reranker (ColBERT, top-M) → Cross-encoder (top-N) → LLM / UI
```

---

## 3. Latency vs качество

Почему нельзя сразу кросс-энкодером по всему корпусу:

* Пусть корпус содержит \$N = 10^7\$ документов.
* Cross-encoder оценивает query–doc за \$t\_{pair} = 5\$ мс на GPU.
* Полный перебор: \$N·t\_{pair} = 10^7·5\$ мс = **\~14 часов на один запрос**.

Даже если распараллелить, это не масштабируется. Поэтому:

* Retrieval снижает кандидатов до \$K ≈ 10^3\$ за **<100 мс**.
* Reranker обрабатывает \$M ≈ 100\$ за **\~200 мс**.
* Cross-encoder оценивает \$N ≈ 20\$ за **\~100 мс**.

**Баланс:** поиск работает в SLA (например, <1 сек), а качество близко к максимально возможному.

---

## 4. Типовые сценарии применения

1. **Поиск (Search)**

   * Web search, корпоративные базы знаний.
   * Ключ: высокая точность и устойчивость к синонимам.

2. **Рекомендации (RecSys)**

   * Item retrieval: сначала быстрые фильтры (категория, популярность), потом reranking по персонализации.
   * Латентные фичи + пользовательские сигналы.

3. **Retrieval-Augmented Generation (RAG)**

   * Вопрос → Retriever (top-K) → Reranker (top-M) → Cross-encoder (top-N) → LLM.
   * Ключ: уменьшение шума и нерелевантных документов в контексте.

---

## 5. Примеры

### 1. Retrieval с bi-encoder

```python
from sentence_transformers import SentenceTransformer, util

# Bi-encoder (независимое кодирование)
bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

corpus = [
    "The Eiffel Tower is located in Paris",
    "The capital of France is Paris",
    "Python is a programming language",
    "Transformers are models for NLP"
]

corpus_emb = bi_encoder.encode(corpus, convert_to_tensor=True)
query = "Where is the Eiffel Tower?"
query_emb = bi_encoder.encode(query, convert_to_tensor=True)

# Быстрый поиск по косинусной близости
hits = util.semantic_search(query_emb, corpus_emb, top_k=3)
print(hits)
```

### 2. Reranking с cross-encoder

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Берём кандидатов из bi-encoder (top-3)
candidates = [(query, corpus[idx]) for idx in [0, 1, 2]]

# Cross-encoder даёт более точные оценки
scores = cross_encoder.predict(candidates)
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

for (q, doc), score in reranked:
    print(f"{score:.3f} | {doc}")
```
