# Collections, points and payload

## 1. Коллекции

**Коллекция** — это логическая группа данных в Qdrant (аналог таблицы в SQL). Она определяет:

* размерность векторов,
* используемую метрику (cosine, dot, euclidean),
* схему хранения (named vectors, payload-индиксы, оптимизация).

Пример создания коллекции:

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="articles",
    vectors_config={
        "title_vector": rest.VectorParams(size=384, distance=rest.Distance.COSINE),
        "body_vector": rest.VectorParams(size=768, distance=rest.Distance.COSINE),
    }
)
```

Особенности:

* Можно хранить несколько **named vectors** в одной коллекции.
* Метрика фиксируется при создании и применяется ко всем вкладам вектора.
* Payload может быть любым JSON-подобным объектом.

---

## 2. Поинты (Points)

**Point** — минимальная единица хранения в Qdrant. Каждый поинт содержит:

* `id`: уникальный идентификатор (числовой или строковый),
* один или несколько векторов (в зависимости от конфигурации коллекции),
* `payload`: произвольные метаданные.

Пример добавления поинтов:

```python
client.upsert(
    collection_name="articles",
    points=[
        rest.PointStruct(
            id=1,
            vector={
                "title_vector": [0.1, 0.2, 0.3, ...],
                "body_vector": [0.4, 0.5, 0.6, ...]
            },
            payload={
                "lang": "ru",
                "tags": ["llm", "retrieval"],
                "published": "2025-09-07"
            }
        )
    ]
)
```

---

## 3. Upsert и overwrite

* `upsert` добавляет новые поинты или обновляет существующие по `id`.
* Если `id` уже есть, старые данные будут **перезаписаны** (overwrite).
* Это касается и векторов, и payload.

Пример обновления:

```python
client.upsert(
    collection_name="articles",
    points=[
        rest.PointStruct(
            id=1,
            vector={"title_vector": [0.9, 0.8, 0.7]},
            payload={"lang": "en"}
        )
    ]
)
```

---

## 4. Частичные обновления payload

Qdrant поддерживает **частичное обновление payload** без перезаписи всего поинта.

Пример:

```python
client.set_payload(
    collection_name="articles",
    payload={"views": 100},
    points=[1]  # id поинта
)
```

Удаление ключа из payload:

```python
client.delete_payload(
    collection_name="articles",
    keys=["views"],
    points=[1]
)
```

---

## 5. Типы payload

Поддерживаемые типы:

* **Строки** — ключевые слова, тексты.
* **Числа** — целые и вещественные.
* **Булевы значения**.
* **Списки** — массивы значений (например, теги).
* **Даты и временные метки**.
* **Гео-данные** — координаты широты и долготы.

---

## 6. Индексация payload-полей

Для ускорения фильтрации можно создавать индексы на payload-поля.

Пример индекса по строковому полю:

```python
client.create_payload_index(
    collection_name="articles",
    field_name="lang",
    field_schema=rest.PayloadSchemaType.KEYWORD
)
```

Пример индекса по числовому полю:

```python
client.create_payload_index(
    collection_name="articles",
    field_name="views",
    field_schema=rest.PayloadSchemaType.INTEGER
)
```

Пример индекса по гео-полю:

```python
client.create_payload_index(
    collection_name="articles",
    field_name="location",
    field_schema=rest.PayloadSchemaType.GEO
)
```