# Mappings and Analyzers

## 1. Мэппинг (Mapping)

### 1.1. Что такое мэппинг

- **Mapping** — схема индекса, определяющая типы полей и правила их обработки.
- Аналог схемы таблицы в SQL, но гибче.
- Контролирует хранение, анализ текста, сортировки и агрегации.

### 1.2. Основные режимы

- **Dynamic mapping**: OpenSearch сам определяет типы по первым документам.
  - Удобно для прототипирования.
  - Опасно в проде (неявные типы, лишние поля).
- **Explicit mapping**: явное задание типов.
  - Надёжно, предсказуемо.

### 1.3. Типы полей

1. **Строки**
   - `text` — для полнотекстового поиска. Анализируется.
   - `keyword` — для точных значений, сортировок, агрегаций.
   - Ошибка: использовать `text` в фильтрах или агрегациях (дорого и неточно).
2. **Числовые**
   - `integer`, `long`, `short`, `byte`, `double`, `float`, `half_float`, `scaled_float`.
   - Выбор зависит от диапазона и точности.
3. **Date/Time**
   - `date`, `date_nanos`.
   - Поддержка форматов (`yyyy-MM-dd`, `epoch_millis`).
4. **Логические**
   - `boolean`.
5. **Структуры**
   - `object` — вложенный JSON.
   - `nested` — специальные объекты для поиска внутри массивов.
6. **Специальные**
   - `geo_point`, `geo_shape` — геоданные.
   - `ip` — IP-адреса.
   - `dense_vector` — эмбеддинги.
   - `join` — связи родитель-дочерний документ.

### 1.4. doc_values и fielddata
- **doc_values** — это колоночное хранение значений полей на диске:
  - Используются для агрегаций, сортировки, скриптов.
  - Включены по умолчанию для большинства типов (`keyword`, `numeric`, `date`). 
  - Позволяют экономить оперативную память (heap), т.к. данные считываются с диска при необходимости.
- **fielddata** — используется только для `text`-полей, если по ним всё же нужно делать агрегации или сортировки:
  - По умолчанию отключён.
  - При включении данные загружаются в heap-память.
  - Использование `fielddata` может быть очень ресурсоёмким и вызывать `OutOfMemoryError`.  
  
**Пример включения fielddata** (не рекомендуется без крайней необходимости):
```json
PUT my_index/_mapping
{
  "properties": {
    "description": {
      "type": "text",
      "fielddata": true
    }
  }
}
```
**Рекомендации**:
- Используйте `keyword` для полей, по которым требуется сортировка или агрегация.
- Не включайте `fielddata`, если можно переопределить поле в `keyword` (или как `multi_field`). 
- Убедитесь, что `doc_values` не отключены (опция `doc_values: false`).

### 1.5. Пример мэппинга

```json
PUT products
{
  "mappings": {
    "properties": {
      "name": { "type": "text", "analyzer": "standard" },
      "brand": { "type": "keyword" },
      "price": { "type": "float" },
      "release_date": { "type": "date" },
      "in_stock": { "type": "boolean" },
      "location": { "type": "geo_point" }
    }
  }
}
```

---

## 2. Анализаторы (Analyzers)

### 2.1. Что такое анализатор

- **Analyzer** = tokenizer + фильтры токенов + нормализаторы.
- Используется при **индексации** и при **поиске**.

### 2.2. Стандартные анализаторы

- `standard` — дефолт, делит по пробелам, приводит к нижнему регистру.
- `simple` — делит по не-буквенным символам.
- `whitespace` — делит только по пробелам.
- `stop` — удаляет стоп-слова.
- `keyword` — не разбивает строку, индексирует как есть.
- `language` — для языков (например, `russian`).

### 2.3. Кастомные анализаторы

Пример:

```json
PUT my_index
{
  "settings": {
    "analysis": {
      "analyzer": {
        "custom_ru": {
          "tokenizer": "standard",
          "filter": ["lowercase", "russian_stop", "russian_stemmer"]
        }
      },
      "filter": {
        "russian_stop": {"type": "stop", "stopwords": "_russian_"},
        "russian_stemmer": {"type": "stemmer", "language": "russian"}
      }
    }
  }
}
```

### 2.4. Search analyzer vs index analyzer

- **index\_analyzer** — как текст обрабатывается при записи.
- **search\_analyzer** — как текст обрабатывается при поиске.
- Пример: при индексации использовать стемминг, при поиске — синонимы.

---

## 3. Edge cases и подводные камни

- `text` нельзя эффективно сортировать и агрегировать — используйте `keyword`.
- Поля `keyword` длиннее 256 символов нужно ограничивать через `ignore_above`.
- Массивы индексируются как список значений, но для сложных вложенных структур нужен `nested`.
- `date` требует указания формата при загрузке нестандартных значений.
- `dense_vector` поддерживает ограничение по размерности (например, до 4096).

---

## 4. Примеры запросов в Python

```python
from opensearchpy import OpenSearch

client = OpenSearch(hosts=[{
    "host": "localhost",
    "port": 9200
}])

# Создание индекса с мэппингом
client.indices.create(
    index="products",
    body={
        "mappings": {
            "properties": {
                "name": {
                    "type": "text", 
                    "analyzer": "standard"
                },
                "brand": {"type": "keyword"},
                "price": {"type": "float"}
            }
        }
    }
)

# Добавление документа
client.index(
    index="products",
    id=1,
    body={
        "name": "Смартфон", 
        "brand": "Samsung", 
        "price": 500.0
    }
)
```

---

## 5. Troubleshooting

- Ошибка `mapper_parsing_exception`: неверный тип данных (например, строка вместо числа).
- Ошибка `illegal_argument_exception` при агрегации: поле `text` без `fielddata=true`.
- Производительность падает: проверить количество полей (`mapping explosion`) и статус `doc_values` / `fielddata`.
