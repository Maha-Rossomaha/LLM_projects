# Hybrid and Sparse

## 1. Зачем нужен гибридный поиск

Чисто dense-поиск (по эмбеддингам) хорошо улавливает семантику, но может плохо работать с:

* редкими терминами,
* точными совпадениями (например, артикулы, коды, даты),
* узкоспециализированными запросами.

Sparse-поиск (BM25, SPLADE и др.) наоборот, учитывает частоту терминов и точные совпадения, но не понимает семантики.

**Гибридный поиск** = комбинация dense + sparse, позволяющая брать лучшее из обоих миров.

---

## 2. Sparse-вектора в Qdrant

* Хранятся в коллекциях наряду с dense.
* Представляют собой **разреженный словарь признаков**: `{index: weight}`.
* В Python-клиенте используется класс `SparseVector`.

Пример вставки sparse-вектора:

```python
from qdrant_client.http import models as rest

point = rest.PointStruct(
    id=1,
    vector={
        "dense_vector": [0.1, 0.2, 0.3, ...],
        "sparse_vector": rest.SparseVector(
            indices=[5, 10, 42],
            values=[0.3, 0.7, 1.2]
        )
    },
    payload={"lang": "en"}
)

client.upsert(collection_name="docs", points=[point])
```

---

## 3. Конфигурация коллекции для dense+sparse

Пример создания коллекции:

```python
client.recreate_collection(
    collection_name="docs",
    vectors_config={
        "dense_vector": rest.VectorParams(size=768, distance=rest.Distance.COSINE),
        "sparse_vector": rest.SparseVectorParams(),
    }
)
```

---

## 4. Поиск по гибридным векторам

Можно искать одновременно по dense и sparse вектору и комбинировать скоры.

Пример:

```python
query = rest.Query(
    vector={
        "dense_vector": [0.5, 0.1, 0.3],
        "sparse_vector": rest.SparseVector(
            indices=[7, 13],
            values=[0.8, 1.0]
        )
    },
    limit=5
)

results = client.query_points(collection_name="docs", query_points=[query])
```

---

## 5. Рецепты гибридного ранжирования

### Взвешенная сумма

```python
score = w_dense * dense_score + w_sparse * sparse_score
```

* `w_dense` и `w_sparse` подбираются экспериментально.
* Обычно `w_dense` чуть выше (например, 0.7/0.3).

### Reciprocal Rank Fusion (RRF)

* Берём ранги документов из обоих поисков.
* Итоговый скор = $\frac{1}{(k + rank)}$, затем суммируем по системам.
* Хорошо работает при разных шкалах скорингов.

### Нормализация скоров

* Перед суммированием dense и sparse скоры приводят к единой шкале (min-max, z-score).

---

## 6. Когда использовать гибрид

* Запросы могут содержать редкие термины (имена, артикулы).
* Нужно балансировать точность (precision) и полноту (recall).
* Есть риск, что dense пропустит важный результат.
* В системах рекомендаций и RAG для повышения релевантности.
