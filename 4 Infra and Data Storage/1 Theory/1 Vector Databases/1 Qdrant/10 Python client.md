# Python client

## 1. Создание и настройка клиента

Подключение к локальному или удалённому серверу Qdrant:

```python
from qdrant_client import QdrantClient

# Локальный инстанс (Docker, binary)
client = QdrantClient(host="localhost", port=6333)

# Удалённый сервер с API-ключом
client = QdrantClient(
    url="https://qdrant.example.com",
    api_key="YOUR_API_KEY"
)
```

---

## 2. CRUD поинтов

### Upsert (создание/обновление)

```python
from qdrant_client.http import models as rest

client.upsert(
    collection_name="docs",
    points=[
        rest.PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3],
            payload={"lang": "ru", "views": 10}
        )
    ]
)
```

### Чтение по id

```python
points = client.retrieve(collection_name="docs", ids=[1], with_payload=True)
```

### Удаление

```python
client.delete(collection_name="docs", points_selector=rest.PointIdsList(points=[1]))
```

---

## 3. Батчи

Можно добавлять/обновлять поинты пачками:

```python
batch = [
    rest.PointStruct(id=i, vector=[0.1*i, 0.2*i, 0.3*i], payload={"idx": i})
    for i in range(1000)
]
client.upsert(collection_name="docs", points=batch)
```

---

## 4. Upsert vs Update

- **Upsert**: если `id` существует — заменяет вектор и payload.
- **Update payload**: можно менять только payload без изменения вектора.

```python
client.set_payload(
    collection_name="docs",
    payload={"verified": True},
    points=[1]
)
```

---

## 5. Поиск (Search)

```python
results = client.search(
    collection_name="docs",
    query_vector=[0.1, 0.2, 0.3],
    limit=5,
    with_payload=True
)
for r in results:
    print(r.id, r.score, r.payload)
```

---

## 6. Query API (batch-запросы)

```python
from qdrant_client.http.models import Query

queries = [
    Query(vector=[0.1, 0.2, 0.3], limit=3),
    Query(vector=[0.5, 0.6, 0.7], limit=2)
]

results = client.query_points(collection_name="docs", query_points=queries)
```

---

## 7. Фильтры

```python
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

query_filter = Filter(
    must=[FieldCondition(key="lang", match=MatchValue(value="ru"))]
)

results = client.search(
    collection_name="docs",
    query_vector=[0.1, 0.2, 0.3],
    query_filter=query_filter,
    limit=5
)
```

---

## 8. Пагинация (offset/limit)

```python
results = client.search(
    collection_name="docs",
    query_vector=[0.1, 0.2, 0.3],
    limit=5,
    offset=5
)
```

---

## 9. Recommend

Функция **recommend** ищет похожие точки на указанные `positive` и непохожие на `negative`.

```python
results = client.recommend(
    collection_name="docs",
    positive=[1, 2],
    negative=[3],
    limit=5
)
```

---

## 10. Scroll

Используется для обхода всей коллекции.

```python
points, next_page = client.scroll(
    collection_name="docs",
    limit=100,
    with_payload=True
)
```

---

## 11. Ошибки и ретраи

### Обработка ошибок

```python
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    client.search(collection_name="nonexistent", query_vector=[0.1, 0.2], limit=1)
except UnexpectedResponse as e:
    print("Ошибка:", e)
```

### Ретраи

Можно использовать внешние библиотеки (`tenacity`) для повторных попыток:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def robust_search():
    return client.search(collection_name="docs", query_vector=[0.1, 0.2, 0.3], limit=5)
```

---
