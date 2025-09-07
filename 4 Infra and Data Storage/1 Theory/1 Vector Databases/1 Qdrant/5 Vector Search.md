# Vector Search

## 1. Типы векторов

Qdrant поддерживает два основных типа векторов:

* **Dense (плотные)** — стандартные эмбеддинги из нейросетевых моделей (Sentence-BERT, OpenAI Embeddings и т.д.). Используются для семантического поиска, рекомендаций, кластеризации.
* **Sparse (разреженные)** — вектора, основанные на bag-of-words или нейросетевых sparse-моделях (BM25, SPLADE и др.). Хорошо работают для поиска редких терминов и в гибридных сценариях.

### Named vectors

* В одной коллекции можно хранить несколько векторных пространств.
* Каждый вектор имеет **имя**.
* Один поинт может содержать несколько векторов разных типов, например:

  * `title_vector` (dense, 384 размерность),
  * `body_vector` (dense, 768 размерность),
  * `bm25_vector` (sparse).

Пример конфигурации коллекции:

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="docs",
    vectors_config={
        "title_vector": rest.VectorParams(size=384, distance=rest.Distance.COSINE),
        "body_vector": rest.VectorParams(size=768, distance=rest.Distance.COSINE),
    }
)
```

---

## 2. Нормализация косинуса

* При выборе метрики **cosine** Qdrant **автоматически нормализует** векторы и вычисляет скалярное произведение.
* Внутри движка это реализовано как dot product на нормализованных векторах.
* Пользователю не нужно заранее нормализовать эмбеддинги.

---

## 3. Поиск (search)

В Python-клиенте поиск выполняется методом `search`. Основные поля запроса:

* `vector` — вектор или `NamedVector`, по которому идёт поиск.
* `filter` — условия на payload (фильтрация).
* `limit` — сколько результатов вернуть.
* `offset` — смещение для пагинации.
* `with_payload` — возвращать ли payload в ответе.
* `with_vectors` — возвращать ли вектора.
* `score_threshold` — минимальный скор для включения в результат.

Пример запроса:

```python
search_result = client.search(
    collection_name="docs",
    query_vector=("body_vector", [0.1, 0.2, 0.3, ...]),
    query_filter=rest.Filter(
        must=[rest.FieldCondition(
            key="lang", match=rest.MatchValue(value="ru")
        )]
    ),
    limit=5,
    with_payload=True,
    score_threshold=0.7
)

for r in search_result:
    print(r.id, r.score, r.payload)
```

---

## 4. Query API

* Вместо `search` можно использовать `query_points` (новый API).
* Поля во многом совпадают, но `query_points` поддерживает **batch-запросы**.

Пример batch-запроса:

```python
result = client.query_points(
    collection_name="docs",
    query_points=[
        rest.Query(
            vector=rest.NamedVector("title_vector", [0.5, 0.2, 0.1]),
            filter=rest.Filter(
                must=[rest.FieldCondition(key="lang", match=rest.MatchValue(value="en"))]
            ),
            limit=3
        ),
        rest.Query(
            vector=rest.NamedVector("body_vector", [0.9, 0.1, 0.2]),
            limit=2
        )
    ]
)
```
