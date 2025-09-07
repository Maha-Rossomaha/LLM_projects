# Qdrant Intro

## 1. Введение

Qdrant — это современная **векторная база данных** (Vector DB), оптимизированная для поиска ближайших соседей (ANN — Approximate Nearest Neighbor) и семантического поиска. Она позволяет хранить и индексировать векторы эмбеддингов, связывая их с дополнительными метаданными (payload), и выполнять быстрый поиск по большим коллекциям.

Qdrant используется в задачах:

* семантический поиск,
* рекомендательные системы,
* RAG (retrieval‑augmented generation),
* дедупликация,
* кластеризация и поиск аномалий.

---

## 2. Модель данных

### Points

* Основная единица хранения в Qdrant — **point**.
* Point имеет:

  * `id` (числовой или строковый идентификатор),
  * один или несколько **векторов** (dense или sparse),
  * **payload** — произвольные метаданные (JSON‑подобная структура).

### Векторы

* **Dense** (плотные) — стандартные эмбеддинги моделей (например, Sentence‑BERT, OpenAI Embeddings).
* **Sparse** (разреженные) — векторы на основе inverted index (BM25, SPLADE и т.п.).

### Payload

* Позволяет фильтровать результаты поиска по категориям, датам, числовым диапазонам.
* Примеры: язык документа, теги, пользовательские атрибуты.

---

## 3. Коллекции

* В Qdrant все данные хранятся в **коллекциях** (аналог таблиц в SQL).
* В коллекции настраиваются:

  * размерность вектора,
  * используемая метрика (cosine, dot, euclidean),
  * схема шардирования и репликации.

Пример создания коллекции на Python:

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="docs",
    vectors_config={
        "text_vector": rest.VectorParams(
            size=768,
            distance=rest.Distance.COSINE
        )
    }
)
```

---

## 4. Метрики расстояния

Qdrant поддерживает несколько метрик для поиска ближайших соседей:

* **Cosine similarity** — $1 - \cos(\theta)$, хорошо подходит для нормализованных эмбеддингов.
* **Euclidean (L2)** — стандартное евклидово расстояние.
* **Dot product** — скалярное произведение; удобно для моделей без нормализации.

Выбор метрики зависит от модели эмбеддингов.

---

## 5. Named Vectors

* В Qdrant одна коллекция может содержать несколько векторных пространств.
* У каждого вектора есть имя (`named vectors`).
* Это позволяет хранить разные типы представлений (например, `title_vector` и `body_vector`) и искать по ним независимо.

Пример:

```python
client.recreate_collection(
    collection_name="articles",
    vectors_config={
        "title_vector": rest.VectorParams(size=384, distance=rest.Distance.COSINE),
        "body_vector": rest.VectorParams(size=768, distance=rest.Distance.COSINE),
    }
)
```

---

## 6. Sparse и Dense

Qdrant умеет работать и с dense‑, и с sparse‑векторами:

* **Dense**: хорошо подходят для семантического поиска, RAG, рекомендаций.
* **Sparse**: классические inverted index (BM25, SPLADE), полезны для точного поиска по редким термам.
* **Hybrid search**: можно комбинировать dense и sparse векторы в одном запросе.

Пример гибридного поиска:

```python
search_result = client.search_batch(
    collection_name="articles",
    requests=[
        rest.SearchRequest(
            vector=rest.NamedVector("body_vector", [0.1, 0.2, 0.3, ...]),
            filter=rest.Filter(
                must=[rest.FieldCondition(key="lang", match=rest.MatchValue(value="ru"))]
            ),
            limit=5
        )
    ]
)
```

---

## 7. Где Qdrant силён

* **Гибридный поиск**: объединение dense и sparse сигналов.
* **Метаданные**: мощная система фильтров по payload.
* **Named vectors**: хранение нескольких векторных пространств в одной коллекции.
* **Простота запуска**: бинарник, Docker, Kubernetes.
* **Скорость**: высокое качество HNSW‑индекса.
* **Поддержка streaming ingestion**: можно писать данные онлайн.

## 8. Где Qdrant слаб

* **Нет полного SQL**: payload не заменяет полноценную СУБД.
* **Тяжёлые join‑операции невозможны**.
* **Очень большие кластеры** (сотни миллиардов векторов) лучше масштабировать в Milvus или Vespa.
* **Нет встроенного reranker’а**: нужно подключать внешние модели.

---

## 9. Когда брать Qdrant, а когда нет

### Брать Qdrant, если:

* Нужен **семантический поиск** по документам.
* Требуется **гибридный поиск** (dense + sparse).
* Важна **фильтрация по метаданным**.
* Нужно просто и быстро поднять **продакшн‑ready сервис**.

### Лучше взять другое решение:

* **Postgres + pgvector**, если данных мало (до \~10M векторов) и важны транзакции.
* **Elasticsearch / OpenSearch**, если основа — полнотекстовый поиск, а dense нужен как дополнение.
* **Milvus**, если объём векторов — **сотни миллиардов** и нужен масштаб.
* **FAISS**, если нужен офлайн‑анализ без продакшн‑сервиса.
