# Bi-encoder как базовый слой

## 1. Принцип работы

**Bi-encoder** — это архитектура, в которой **запрос (query)** и **документ (doc)** кодируются **независимо**:

$$
\text{query\_embedding} = f(q), \quad \text{doc\_embedding} = f(d)
$$

После кодирования, их **сходство** вычисляется с помощью косинусной меры:

$$
\text{score}(q, d) = \cos(f(q), f(d)) = \frac{f(q)^T \cdot f(d)}{\|f(q)\| \cdot \|f(d)\|}
$$

Это позволяет **предварительно индексировать** эмбеддинги документов, а затем сравнивать их с query-эмбеддингом без повторного прогонки модели.

---

## 2. Сильные стороны

* **Быстро**: только одно прямое прохождение модели на сторону запроса.
* **Масштабируемо**: документы кодируются заранее → поддерживается миллионы записей.
* **Просто реализуется**: можно использовать Sentence-BERT, E5, BGE и др.
* **Совместимо с ANN**: после кодирования можно использовать Faiss, ScaNN и другие библиотеки.

---

## 3. Ограничения

* Нет взаимодействия между токенами query и document.

  * Например, model не знает, что слово "Python" в query и "Python" в doc совпадают по позиции и смыслу.
* Иногда пропускает тонкие соответствия (синонимы, контекстные намёки).
* Качество ниже, чем у cross-encoder, особенно на коротких и шумных запросах.

---

## 4. Практика: top-K retrieval на bi-encoder

### Вопрос: где ставить cut-off K?

Обычно:

* $K = 1000$ — если потом будет reranking.
* $K = 10 \dots 100$ — если отдаём напрямую пользователю (например, в простых системах).
* Выбор зависит от:

  * downstream latency budget;
  * потребности в recall vs precision.

---

## 5. Пример

```python
from sentence_transformers import SentenceTransformer, util

# Загружаем bi-encoder модель
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Документы
corpus = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "Berlin is the capital of Germany.",
    "Python is a programming language."
]

# Запрос
query = "Where is the Eiffel Tower located?"

# Кодируем всё
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# Semantic search
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]

# Вывод результатов
for hit in hits:
    print(f"Score: {hit['score']:.4f} | {corpus[hit['corpus_id']]}")
```