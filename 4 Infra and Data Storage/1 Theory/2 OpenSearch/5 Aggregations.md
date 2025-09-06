# Aggregations

## 1. Общая структура

```json
GET /sales/_search
{
  "size": 0,
  "aggs": {
    "my_agg": {
      "terms": {"field": "product"}
    }
  }
}
```

* `size: 0` отключает возврат документов — только агрегации.
* `aggs` может быть вложенным.

---

## 2. Buckets: группирующие агрегации

### 2.1. `terms`

* Группировка по значениям поля.
* По умолчанию — top 10 по убыванию doc\_count.

```json
"terms": {
  "field": "category.keyword",
  "size": 5,
  "shard_size": 100
}
```

* `shard_size` влияет на **точность**: чем больше — тем меньше потерь из-за локальных top-N.

### 2.2. `range`

```json
"range": {
  "field": "price",
  "ranges": [
    {"to": 100},
    {"from": 100, "to": 500},
    {"from": 500}
  ]
}
```

### 2.3. `date_histogram`

```json
"date_histogram": {
  "field": "timestamp",
  "calendar_interval": "day"
}
```

> Альтернатива: `fixed_interval` (например, `60m`, `1d`, `30s`).

### 2.4. `filters`

* Несколько независимых фильтров в одной агрегации.

```json
"filters": {
  "filters": {
    "cheap": {"range": {"price": {"lt": 100}}},
    "mid": {"range": {"price": {"gte": 100, "lt": 500}}}
  }
}
```

### 2.5. composite
- Итеративная агрегация для очень больших наборов.
- Позволяет пагинацию (`after_key`) и сбор миллионов bucket'ов без потери.

---

## 3. Metrics: агрегаты внутри bucket'ов

### 3.1. `avg`, `sum`, `min`, `max`

```json
"avg": {
  "field": "price"
}
```

### 3.2. `cardinality`

* Подсчёт уникальных значений.
* Использует HyperLogLog++, **может быть неточно**.

```json
"cardinality": {
  "field": "user_id",
  "precision_threshold": 1000
}
```

* `precision_threshold`: максимум 40 000, выше → точнее, но медленнее и дороже по памяти.

### 3.3. `percentiles`

```json
"percentiles": {
  "field": "duration",
  "percents": [50, 90, 99]
}
```

* Алгоритм t-digest / HDRHistogram (настраивается).

### 3.4. stats
Возвращает сразу `min`, `max`, `avg`, `sum`, `count`.
```json
"stats": {
  "field": "price"
}
```

### 3.5. extended_stats
Добавляет `дисперсию`, `std_dev`, `skewness` и `kurtosis`.
```json
"extended_stats": {
  "field": "price"
}
```

---

## 4. Вложенные агрегации

* Можно вложить одну агрегацию внутрь другой:

```json
"aggs": {
  "by_category": {
    "terms": {"field": "category"},
    "aggs": {
      "avg_price": {
        "avg": {"field": "price"}
      }
    }
  }
}
```

* Пример: top категорий и средняя цена в каждой.

---

## 5. Pipeline Aggregations

Агрегации **над результатами других агрегаций** (post-processing).

### 5.1. `derivative`

```json
"derivative": {
  "buckets_path": "sales_over_time"
}
```

### 5.2. `moving_avg` (устарел) → `moving_fn` (новый способ)

```json
"moving_fn": {
  "buckets_path": "avg_sales",
  "window": 5,
  "script": "MovingFunctions.unweightedAvg(values)"
}
```

### 5.3. `bucket_script`

```json
"bucket_script": {
  "buckets_path": {
    "revenue": "total_revenue",
    "orders": "order_count"
  },
  "script": "params.revenue / params.orders"
}
```

* Пишется на painless. Использует значения других агрегаций.


### 5.4. `bucket_selector`

- Фильтрует bucket'ы по условию
  
```json
"bucket_selector": {
  "buckets_path": {"totalSales": "sales"},
  "script": "params.totalSales > 100"
}
```

---

## 6. Специальные агрегации

### 6.1. nested и reverse_nested

Используются для агрегаций по вложенным объектам. `nested` погружается внутрь массива объектов и строит агрегацию только по ним, а `reverse_nested` позволяет вернуться обратно на уровень родителя, чтобы, например, сгруппировать документы по полю родителя, удовлетворяющему условиям вложенной агрегации.

```json
"nested": {
  "path": "comments",                  // указываем путь до вложенного массива
  "aggs": {
    "top_authors": {
      "terms": {"field": "comments.author.keyword", "size": 5}, // топ-5 авторов комментариев
      "aggs": {
        "avg_length": {"avg": {"field": "comments.length"}} // средняя длина комментария каждого автора
      }
    }
  }
}
```

```json
"reverse_nested": {                     // возвращаемся на уровень родителя
  "aggs": {
    "posts_by_category": {
      "terms": {"field": "category.keyword", "size": 3} // топ-3 категории постов, где встречаются такие комментарии
    }
  }
}
```

### 6.2. significant_terms

Выделяет термины, которые статистически значимо отличаются от корпуса.

```json
"significant_terms": {
    "field": "tags"
}
```

### 6.3. sampler

Агрегация только по случайной подвыборке документов.

```json
"sampler": {
    "shard_size": 1000
}
```

---

## 7. Edge cases и подводные камни

### 7.1. `cardinality` неточен при малом `precision_threshold`

* Тестировать на своих данных.
* Важен при отчётах по уникальным пользователям.

### 7.2. `terms` обрезает редкие значения

* По умолчанию `size: 10`, а `shard_size` мал → теряются важные данные.
* Увеличить `shard_size` до 2–3x `size` или использовать `composite`.

### 7.3. Порядок агрегаций

* Pipeline нельзя ставить вне buckets.
* `bucket_script` работает только внутри `aggs`, а не `root`.

### 7.4. Nested ограничения
* При агрегации по `nested` не видны поля родителя без `reverse_nested`.

---

## 8. Практика оптимизации агрегаций
* Использовать `size: 0`, если нужны только агрегации.
* Следить за `shard_size` и Cardinality.
* Для больших датасетов использовать `composite` вместо `terms`.
* Для временных рядов — `date_histogram` + `moving_fn`.
* Для вложенных — `nested` + `reverse_nested`.
* Для фильтрации результатов — `bucket_selector`.
* Проверять производительность через `_profile`.