# Metadata Filtering 

Metadata filtering — это техника ограничения поиска по векторному индексу с учётом дополнительных атрибутов объектов (язык, дата, категория, user‑id и т.п.). Используется в векторных БД (Qdrant, Milvus, Weaviate, Pinecone) и retrieval‑системах.

---

## 1. Идея
- У объекта кроме эмбеддинга $x \in \mathbb{R}^d$ есть метаданные $m$ (например, {"lang":"ru", "date":"2025-01-01"}).
- Запрос $q$ ищет ближайшие соседи по эмбеддингам **с учётом ограничений на $m$**.
- Варианты:
  - **Post-filter:** сначала ANN‑поиск по всем, потом фильтрация по метаданным.
  - **Pre-filter:** сразу ограничиваем кандидатов только нужными объектами.

---

## 2. Архитектура
1. **Index Layer (ANN):** IVF, HNSW, ScaNN и т.п. для поиска по эмбеддингам.
2. **Metadata Store:** хранение атрибутов (обычно key-value или колоночное хранилище).
3. **Query Executor:** комбинирует ANN‑результаты и условия WHERE.

---

## 3. Подходы
### 3.1. Post-filter
- ANN возвращает top‑K кандидатов.
- Применяем фильтр (например, только lang="ru").
- Если мало кандидатов — увеличиваем K и повторяем.
- **Плюсы:** простота, работает с любыми индексами.
- **Минусы:** может быть неэффективно при редких фильтрах (надо сильно завышать K).

### 3.2. Pre-filter
- Фильтр применяется на этапе ANN‑поиска.
- Векторная БД знает, какие объекты в каких сегментах/шардах удовлетворяют условию.
- **Плюсы:** быстрее, меньше кандидатов.
- **Минусы:** сложнее реализовать (нужно хранить inverted index по метаданным).

### 3.3. Hybrid
- Комбинация: быстрый pre-filter по грубому признаку (например, дата>2024), затем ANN и post-filter для уточнения.

---

## 4. Параметры и SLA
- **selectivity фильтра:** чем реже условие, тем важнее pre-filter.
- **K (размер кандидатов):** при post-filter нужно завышать K.
- **Latency budget:** фильтрация не должна ломать p95 SLA.
- **Consistency:** метаданные и эмбеддинги должны обновляться синхронно.

---

## 5. Сложность и ресурсы
- Post-filter: $O(K)$ проверок метаданных.
- Pre-filter: $O(\log N)$ для поиска по inverted‑метаданным + $O(cost_{ANN})$.
- Память: хранение метаданных + индексов по ним (B‑деревья, inverted lists, bitmap‑индексы).

---

## 6. Достоинства
- Более гибкий поиск (условия WHERE как в SQL).
- Поддержка многоарендности (tenant‑id фильтры).
- Подходит для production retrieval (e‑commerce, поиск по документам, персонализация).

---

## 7. Недостатки
- Post-filter может требовать огромный K.
- Pre-filter усложняет реализацию и синхронизацию.
- Tail‑latency при сложных фильтрах.

---

## 8. Практические советы
- Для «узких» фильтров (например, user_id) — использовать pre-filter.
- Для «широких» фильтров (например, lang in {"en","ru"}) — достаточно post-filter.
- Следить за балансом recall/latency.
- Использовать vector DB с нативной поддержкой фильтров (Qdrant, Milvus, Weaviate).
- Для Elastic/OpenSearch — комбинировать dense_vector с обычными полями.

---

## 9. Примеры

### 9.1. Post-filter с FAISS
```python
import faiss
import numpy as np

# Допустим, у каждого вектора есть метаданные lang
langs = ["en", "ru", "en", "de", "ru"]

# Строим простой индекс
xb = np.random.randn(5, 128).astype('float32')
index = faiss.IndexFlatL2(128)
index.add(xb)

# Запрос
xq = np.random.randn(1, 128).astype('float32')
D, I = index.search(xq, 5)

# Post-filter (lang=="ru")
results = [(i, D[0][j]) for j, i in enumerate(I[0]) if langs[i] == "ru"]
print(results)
```

### 9.2. Pre-filter в Qdrant
```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

client = QdrantClient("localhost", port=6333)

# Создание коллекции с фильтруемыми полями
client.recreate_collection(
    collection_name="docs",
    vectors_config=rest.VectorParams(
        size=128, 
        distance="Cosine"
    ),
)

# Загрузка векторов с метаданными
client.upsert(
    collection_name="docs",
    points=[
        rest.PointStruct(
            id=1, 
            vector=np.random.rand(128).tolist(), 
            payload={"lang":"ru"}
        ),
        rest.PointStruct(
            id=2, 
            vector=np.random.rand(128).tolist(), 
            payload={"lang":"en"}
        ),
    ]
)

# Поиск только по lang="ru"
hits = client.search(
    collection_name="docs",
    query_vector=np.random.rand(128).tolist(),
    limit=5,
    query_filter=rest.Filter(
        must=[
            rest.FieldCondition(
                key="lang", 
                match=rest.MatchValue(value="ru")
            )
        ]
    )
)
print(hits)
```

---

## 10. Чеклист тюнинга
- Определить частоту фильтра (узкий/широкий).
- Решить: post‑filter, pre‑filter или hybrid.
- Тюнинговать K для post‑filter.
- Настроить индексы по метаданным для pre‑filter.
- Тестировать p95 latency на реальных нагрузках.

