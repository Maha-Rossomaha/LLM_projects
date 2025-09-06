# Индексация, Bulk и Reindex в OpenSearch

## 1. Индексация (Indexing)

### 1.1. Основные операции

* `index` — вставка нового документа или замена по `id`.
* `create` — вставка документа, если `id` ещё не существует.
* `update` — частичное обновление документа.
* `delete` — удаление по `id`.

### 1.2. Версионирование документов

* OpenSearch хранит `version`, `seq_no` и `primary_term`.
* Для оптимистичной блокировки использовать `if_seq_no` + `if_primary_term`.

Пример:

```json
POST products/_update/1?if_seq_no=7&if_primary_term=1
{
  "doc": {"price": 600}
}
```

### 1.3. Авто-генерация ID

* Если не передавать `_id`, OpenSearch создаст UUID.
* Лучше задавать `_id` для детерминированного апдейта/удаления.

---

## 2. Bulk API

### 2.1. Зачем нужен Bulk

* Массовая загрузка документов в одном HTTP-запросе.
* Существенно быстрее по сравнению с поштучной индексацией.

### 2.2. Формат запроса

* Каждая операция занимает **две строки**: метаданные + данные.

Пример:

```json
POST _bulk
{ "index": { "_index": "products", "_id": 1 }}
{ "name": "Смартфон", "brand": "Samsung", "price": 500 }
{ "update": { "_index": "products", "_id": 1 }}
{ "doc": {"price": 600}}
{ "delete": { "_index": "products", "_id": 2 }}
```

### 2.3. Лучшие практики

* Размер батча: 5–15 MB или 1000–5000 документов.
* Параллельная загрузка несколькими потоками.
* Backoff и retry при ошибках (`429 Too Many Requests`).
* DLQ (dead-letter queue) для проблемных документов.

### 2.4. Python-пример

```python
from opensearchpy import OpenSearch, helpers

client = OpenSearch(hosts=[{
    "host": "localhost",
    "port": 9200
}])

actions = [
    {
        "_op_type": "index",
        "_index": "products",
        "_id": i,
        "_source": {
            "name": f"Товар {i}",
            "price": i*10
        }
    }
    for i in range(1, 1001)
]

helpers.bulk(client, actions)
```

---

## 3. Reindex API

### 3.1. Зачем нужен Reindex

* Копирование данных из одного индекса в другой.
* Используется при:

  * смене мэппинга,
  * изменении анализа текста,
  * миграциях данных.

### 3.2. Пример

```json
POST _reindex
{
  "source": {"index": "products_v1"},
  "dest": {"index": "products_v2"}
}
```

### 3.3. С фильтрацией и скриптом

```json
POST _reindex
{
  "source": {
    "index": "products_v1",
    "query": {"range": {"price": {"gte": 100}}}
  },
  "dest": {"index": "products_v2"},
  "script": {
    "source": "ctx._source.discount = ctx._source.price * 0.1"
  }
}
```

### 3.4. Best practices

* Делать reindex в **новый индекс**, затем переключать алиас.
* Для больших объёмов — `slices` (параллельная обработка).
* Проверять `conflicts` и использовать `retry_on_conflict`.
* Reindex блокирует ресурсы — запускать вне пиковых часов.

---

## 4. Index Settings (настройки индекса)

### 4.1. number\_of\_shards и number\_of\_replicas

* **number\_of\_shards** — число первичных шардов (разбиений индекса). По умолчанию 1.

  * Используется для горизонтального масштабирования.
  * Изменить нельзя после создания индекса.
  * Не стоит задавать слишком много шардов для маленьких индексов (over-sharding).

* **number\_of\_replicas** — число реплик для каждого шарда (резервные копии). По умолчанию 1.

  * Обеспечивает отказоустойчивость и параллельный поиск.
  * Можно изменять динамически.

```json
PUT my_index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2
  },
  "mappings": {
    "properties": {
      "name": {"type": "text"}
    }
  }
}
```

### 4.2. refresh\_interval

* Интервал, с которым сегменты становятся доступными для поиска. По умолчанию — 1s.
* При массовой индексации можно временно установить `-1`, чтобы отключить авто-обновление.
* После завершения загрузки — вернуть значение обратно.

```json
PUT my_index/_settings
{
  "index": {
    "refresh_interval": "-1"
  }
}
```

### 4.3. max\_result\_window

* Максимальное значение `from + size` для пагинации.
* По умолчанию 10 000. Увеличение — может сильно нагрузить память.
* Для глубокой пагинации лучше использовать scroll или search\_after.

```json
PUT my_index/_settings
{
  "index.max_result_window": 50000
}
```

### 4.4. Другие важные настройки

* `index.mapping.total_fields.limit` — максимум полей в индексе (по умолчанию \~1000).
* `index.query.default_field` — поле по умолчанию для полнотекстового поиска.
* `index.routing.allocation.require._name` — указание, на каких узлах хранить индекс.

---

## 5. Edge cases и подводные камни

* **Mapping explosion**: нельзя бездумно индексировать JSON с тысячами полей.
* **Ограничение полей**: `index.mapping.total_fields.limit` (по умолчанию \~1000).
* **Reindex не изменяет mapping**: нужно сначала создать новый индекс с нужным мэппингом.
* **Bulk не атомарен**: часть операций может выполниться, часть нет.
* **Version conflicts**: при reindex/update может быть конфликт версий.

---

## 6. Troubleshooting

* Ошибка `document_missing_exception` — документ с `_id` не найден.
* Ошибка `illegal_argument_exception` при bulk — неправильный формат запроса.
* Высокая нагрузка при bulk → увеличить `refresh_interval` или временно отключить реплики.

Пример:

```json
PUT products/_settings
{
  "index": {
    "refresh_interval": "-1",
    "number_of_replicas": 0
  }
}
```

После загрузки вернуть настройки.
