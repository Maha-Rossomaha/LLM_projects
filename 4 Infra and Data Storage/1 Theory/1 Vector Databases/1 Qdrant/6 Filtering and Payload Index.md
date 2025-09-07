# Filtering and Payload Index

## 1. Общая идея

Фильтрация в Qdrant позволяет **сужать результаты векторного поиска** по метаданным (payload). Это особенно важно для RAG и рекомендательных систем, где необходимо учитывать язык документа, категорию, даты или геолокацию.

---

## 2. Язык условий (Filters)

Фильтры в Qdrant описываются в терминах:

* **must** — условия, которые должны выполняться (логическое AND),
* **should** — условия, которые желательно выполнять (логическое OR, повышают скор),
* **must\_not** — условия-исключения (NOT).

Пример фильтрации:

```python
from qdrant_client.http import models as rest

query_filter = rest.Filter(
    must=[
        rest.FieldCondition(
            key="lang", match=rest.MatchValue(value="ru")
        )
    ],
    must_not=[
        rest.FieldCondition(
            key="blocked", match=rest.MatchValue(value=True)
        )
    ],
    should=[
        rest.FieldCondition(
            key="priority", match=rest.MatchValue(value="high")
        )
    ]
)
```

---

## 3. Операторы фильтрации

* **match** — точное совпадение (строки, булевы, числа).
* **range** — числовые диапазоны (`gt`, `lt`, `gte`, `lte`).
* **geo\_bounding\_box / geo\_radius** — гео-фильтрация по координатам.
* **is\_empty** — проверка наличия/отсутствия значения.

Пример фильтра с числовым диапазоном:

```python
rest.FieldCondition(
    key="views",
    range=rest.Range(gte=100, lt=1000)
)
```

Пример фильтра с гео-радиусом:

```python
rest.FieldCondition(
    key="location",
    geo_radius=rest.GeoRadius(
        center=rest.GeoPoint(lon=37.62, lat=55.75),
        radius=5000  # в метрах
    )
)
```

---

## 4. Индексация payload-полей

Чтобы фильтрация была быстрой, нужно создавать индексы:

* **Строковые поля** — `KEYWORD`.
* **Числовые поля** — `INTEGER` или `FLOAT`.
* **Гео-поля** — `GEO`.
* **Булевы** — автоматически индексируются как ключевые слова.

Пример индексации:

```python
client.create_payload_index(
    collection_name="docs",
    field_name="lang",
    field_schema=rest.PayloadSchemaType.KEYWORD
)
```

Создание числового индекса:

```python
client.create_payload_index(
    collection_name="docs",
    field_name="views",
    field_schema=rest.PayloadSchemaType.INTEGER
)
```

---

## 5. Влияние индексации на производительность

* Без индексов фильтрация идёт линейным перебором → медленно.
* С индексами фильтрация масштабируется и даёт быстрый отклик.
* Индексация полезна для часто используемых фильтров (язык, теги, категории).
* Минус: создание индекса занимает время и потребляет память.

---

## 6. Паттерны использования

### «Сначала вектор, потом фильтр»

* Сценарий: ищем топ-N ближайших векторов, а потом фильтруем по payload.
* Минус: можно потерять релевантные результаты, которые были отфильтрованы.

### «Фильтр внутри поиска» (рекомендуется)

* Сценарий: фильтрация применяется прямо на этапе поиска.
* Плюс: поиск выполняется только среди кандидатов, удовлетворяющих фильтру.
* Это быстрее и точнее.

Пример:

```python
results = client.search(
    collection_name="docs",
    query_vector=("body_vector", [0.1, 0.2, 0.3]),
    query_filter=query_filter,
    limit=5,
    with_payload=True
)
```