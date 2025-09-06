# Релевантность и скоринг в OpenSearch


## 1. Основы скоринга

- Все запросы в разделе `query` возвращают `_score` — численную оценку релевантности.
- Чем выше `_score`, тем выше документ в результатах.
- Скоринг зависит от:
  - типа запроса (full-text vs term-level),
  - частоты термина (tf),
  - распространённости термина в корпусе (idf),
  - длины поля (dl),
  - boost'а и функции.

---

## 2. BM25: формула и параметры

**Официальная формула:**

$$
score = idf(t) \cdot \frac{tf(t) \cdot (k_1 + 1)}{tf(t) + k_1 \cdot (1 - b + b \cdot \frac{dl}{avgdl})}
$$

Где:

- `tf(t)` — сколько раз термин `t` встречается в документе,
- `idf(t)` — логарифмическая функция редкости термина,
- `dl` — длина документа (в токенах),
- `avgdl` — средняя длина документа в индексе,
- `k_1` — параметр насыщения, обычно 1.2–2.0,
- `b` — параметр нормализации по длине, от 0 до 1 (обычно 0.75).

**Пояснение:**

- При `b=0` длина документа не влияет.
- При `b=1` документ с длиной выше средней получает меньший скор.
- Чем выше `k_1`, тем слабее влияние tf.

**Изменение параметров:**

```json
PUT my_index
{
  "settings": {
    "similarity": {
      "default": {
        "type": "BM25",
        "k1": 1.5,
        "b": 0.6
      }
    }
  }
}
```

---
## 3. Другие модели сходства (similarity)
Помимо BM25, поддерживаются:
- **classic TF-IDF** — устаревшая модель, использует tf * idf и координатный коэффициент. Подходит для обратной совместимости.
- **boolean** — бинарная модель: если терм найден, скор фиксированный. Используется для фильтрации.
- **DFR (Divergence From Randomness)** — семейство моделей, основанных на вероятностной дивергенции. Более гибкая, чем BM25.
- **IB (Information-Based)** — сходна с DFR, применяет информационные меры.
- **LMDirichlet** — языковая модель с Дирихле-гладжингом. Хороша для длинных документов.
- **LMJelinekMercer** — языковая модель с параметром $\lambda$ (линейное сглаживание). Хороша для коротких текстов.

Пример настройки:
```json
PUT my_index
{
  "settings": {
    "similarity": {
      "default": {
        "type": "DFR"
      }
    }
  }
}
```

---

## 4. Custom similarity
Можно писать собственные формулы через `scripted_similarity` или плагины.

```json
"similarity": {
  "custom_sim": {
    "type": "scripted",
    "script": {
      "source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1)/(doc.freq+1)); return query.boost * tf * idf;"
    }
  }
}
```

---

## 5. Boost: повышение веса полей и условий

Boost используется для задания приоритета:

### 5.1. Boost поля

```json
"multi_match": {
  "query": "смартфон",
  "fields": ["title^3", "description"]
}
```

Поле `title` вносит вклад в `_score` в 3 раза сильнее.

### 5.2. Boost условия

```json
"bool": {
  "should": [
    {"match": {"name": {"query": "iphone", "boost": 2}}},
    {"match": {"description": "iphone"}}
  ]
}
```

---

## 6. dis\_max и tie\_breaker

`dis_max` (disjunction max) — альтернатива `bool->should`, выбирает **лучшее совпадение**, а не складывает все `_score`:

`tie_breaker` — насколько сильны вторые и следующие совпадения:
  - `0.0` → учитывается только лучший скор.
  - `1.0` → суммируются все, как в `should`.

```json
"dis_max": {
  "queries": [
    {"match": {"title": "смартфон"}},
    {"match": {"description": "смартфон"}}
  ],
  "tie_breaker": 0.3
}
```

## 7. function\_score: расширенный контроль

Позволяет модифицировать `_score` по формулам, decay-функциям, полям и скриптам.

```json
"function_score": {
  "query": {"match": {"name": "смартфон"}},
  "field_value_factor": {
    "field": "popularity",
    "modifier": "log1p",
    "missing": 0
  },
  "boost_mode": "multiply"
}
```

### 7.1. Доступные функции

- `field_value_factor` — числовое поле влияет на скор.
- `random_score` — случайность (например, для ротации).
- `decay` — уменьшение влияния по расстоянию:

```json
"gauss": {
  "release_date": {
    "origin": "2023-01-01",
    "scale": "30d"
  }
}
```

- `script_score` — использовать произвольный `painless` скрипт:

```json
"script_score": {
  "script": {
    "source": "Math.log(2 + doc['popularity'].value)"
  }
}
```

### 7.2. boost\_mode 

`boost_mode`: как функция влияет на базовый `_score`:
- `multiply` (по умолчанию) — умножает на функцию.
- `sum` — добавляет.
- `replace` — заменяет базовый _score.
- `avg`, `max`, `min` — усреднение или выбор экстремальных значений.

### 7.3. score\_mode 
`score_mode`: если несколько функций — как они комбинируются.
- `multiply` — перемножаются.
- `sum` — складываются.
- `avg` — усредняются.
- `first` — берётся первая.
- `max `/ `min `— берутся крайние.
---

## 8. Nested scoring
Для вложенных (`nested`) документов:
- Каждый вложенный объект индексируется как скрытый документ.
- `_score` считается отдельно для каждого вложенного объекта.
- Итоговый `_score` родительского документа агрегирует вложенные (по умолчанию `avg`).
- Это означает, что если хотя бы один вложенный объект релевантен, родительский документ может получить высокий `_score`.
- С помощью параметра `score_mode` можно управлять, как именно комбинируются скоры вложенных документов: например, брать сумму (`sum`) всех, среднее (`avg`), минимум или максимум.
- Если выбрать `none`, то `_score` вложенных не учитывается, и родительский документ получает `_score=0` или фиксированный при других условиях.

```json
// Пример маппинга: массив comments — nested-объекты
PUT my_index
{
  "mappings": {
    "properties": {
      "title":   { "type": "text" },
      "comments": {
        "type": "nested",
        "properties": {
          "text":       { "type": "text" },
          "author":     { "type": "keyword" },
          "created_at": { "type": "date" }
        }
      }
    }
  }
}
```

```json
// Поиск документов, где есть релевантные вложенные комментарии.
//  - match по comments.text даёт _score на КАЖДЫЙ вложенный комментарий;
//  - filter ограничивает по дате (за последние 90 дней);
//  - score_mode = "sum" суммирует скоры по всем попавшим комментариям,
//    усиливая документы с несколькими хорошими комментариями;
//  - inner_hits возвращает топ-3 лучших вложенных комментария с подсветкой.
POST my_index/_search
{
  "query": {
    "nested": {
      "path": "comments",                        // путь до вложенного массива
      "query": {
        "bool": {
          "must": [
            {
              "match": {                         // полнотекстовый матч по тексту комментария
                "comments.text": {
                  "query": "смартфон с хорошей камерой",
                  "operator": "and"              // требуем совпадение всех токенов
                }
              }
            }
          ],
          "filter": [
            {
              "range": {                         // только свежие комментарии за 90 дней
                "comments.created_at": { "gte": "now-90d" }
              }
            }
          ]
        }
      },
      "score_mode": "sum",                       // как агрегировать скоры: sum/avg/min/max/none
      "inner_hits": {                            // вернуть сами вложенные, чтобы показать пользователю
        "name": "top_comments",
        "size": 3,
        "sort": [{ "comments.created_at": "desc" }],
        "highlight": { "fields": { "comments.text": {} } }
      }
    }
  },
  "size": 10,
  "track_scores": true                           // гарантируем, что вернётся _score (если сортируете по полю)
}
```

Доступные `score_mode` для nested: `avg`, `sum`, `min`, `max`, `none`. Кроме того, можно использовать фильтры внутри nested-запросов для более точного управления.

---

## 9. Гибридный поиск: BM25 + векторный

Можно объединять **векторный поиск (ANN)** с классическим текстовым BM25 в одном запросе через `should`:

```json
"bool": {
  "should": [
    {"match": {"text": "смартфон"}},
    {
      "knn": {
        "embedding": {
          "vector": [0.123, 0.234, ...],
          "k": 10
        }
      }
    }
  ]
}
```

Или использовать гибридный ранжировщик:

```json
"rank": {
  "rrf": {
    "window_size": 10
  }
}
```

- Поддерживается в **OpenSearch 2.6+**.
- Поддерживает **reciprocal rank fusion (RRF)**.

---

## 10. Отладка скоринга

### 10.1. explain API

Использовать параметр `explain=true`, чтобы увидеть расчёт `_score`:

```bash
GET my_index/_search?explain=true
```

```python
from opensearchpy import OpenSearch

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

response = client.search(
    index="my_index",
    body={
        "query": {"match": {"text": "смартфон"}},
        "explain": True
    }
)

for hit in response["hits"]["hits"]:
    print(hit["_id"], hit["_score"])
    print(hit["_explanation"])
```

### 10.2. profile API

Используется для анализа затрат на подзапросы.

```bash
GET my_index/_search?profile=true
```

---
## 11. Edge cases

- Если использовать `filter` или `constant_score`, то `_score=1.0`.
- При сортировке по полю можно отключить вычисление `_score` (`"track_scores": false`).
- В агрегациях `_score` не используется.


