# Жизненный цикл индекса (Index State Management, ISM)

OpenSearch поддерживает **автоматическое управление жизненным циклом индексов** через механизм **ISM (Index State Management)**. Это аналог ILM в Elasticsearch. Он позволяет:

* переносить индексы между "горячими", "тёплыми" и "холодными" узлами,
* запускать rollover, force\_merge, snapshot и другие действия,
* уменьшать нагрузку и стоимость хранения без потери данных.

---

## 1. Основные понятия

* **Политика ISM** — JSON-объект, описывающий состояния индекса и действия в каждом из них.
* **Состояние (state)** — логическая стадия жизни индекса (hot, warm, cold, delete).
* **Переход (transition)** — условие перехода между состояниями (по времени, размеру и др.).
* **Действие (action)** — что делать с индексом в этом состоянии (rollover, snapshot, allocation и т.п.).

---

## 2. Hot-Warm-Cold архитектура

* **Hot** — индексы для записи и активного поиска (SSD, много ресурсов).
* **Warm** — старые, редко обновляемые индексы (дешевле узлы).
* **Cold** — только чтение, архивные данные (медленные диски).

> ISM позволяет автоматически перемещать индексы между слоями.

---

## 3. Пример политики ISM

```json
PUT _plugins/_ism/policies/my_policy
{
  "policy": {
    "description": "Hot-Warm-Cold-Delete policy",
    "default_state": "hot",
    "states": [
      {
        "name": "hot",
        "actions": [
          {"rollover": {"min_index_age": "1d", "min_size": "10gb"}}
        ],
        "transitions": [
          {"state_name": "warm", "conditions": {"min_index_age": "3d"}}
        ]
      },
      {
        "name": "warm",
        "actions": [
          {"replica_count": {"number_of_replicas": 0}},
          {"force_merge": {"max_num_segments": 1}},
          {"allocation": {"require": {"data": "warm"}}}
        ],
        "transitions": [
          {"state_name": "cold", "conditions": {"min_index_age": "7d"}}
        ]
      },
      {
        "name": "cold",
        "actions": [
          {"snapshot": {"repository": "my_backup_repo", "snapshot": "{{index}}-{{date}}"}},
          {"allocation": {"require": {"data": "cold"}}},
          {"read_only": {}}
        ],
        "transitions": [
          {"state_name": "delete", "conditions": {"min_index_age": "30d"}}
        ]
      },
      {
        "name": "delete",
        "actions": [
          {"delete": {}}
        ],
        "transitions": []
      }
    ]
  }
}
```

---

## 4. Подключение политики к шаблону

Чтобы новые индексы получали политику:

```json
PUT _index_template/logs_template
{
  "index_patterns": ["logs-*"] ,
  "template": {
    "settings": {
      "plugins.index_state_management.policy_id": "my_policy",
      "number_of_shards": 1
    }
  }
}
```

---

## 5. Основные действия в ISM

### 5.1. `rollover`

* Создаёт новый индекс, если старый достиг возраста/размера.

```json
"rollover": {"min_index_age": "1d", "min_size": "10gb"}
```

### 5.2. `snapshot`

* Делает snapshot индекса в указанный репозиторий.

```json
"snapshot": {"repository": "s3_backup", "snapshot": "{{index}}-snap"}
```

### 5.3. `force_merge`

* Удаляет старые сегменты, уменьшает размер.

```json
"force_merge": {"max_num_segments": 1}
```

### 5.4. `allocation`

* Переносит индекс на определённые узлы по тегу:

```json
"allocation": {"require": {"data": "cold"}}
```

> У узлов в `opensearch.yml` должно быть прописано: `node.attr.data: cold`

---

## 6. Команды и статус

### Проверить политику:

```bash
GET _plugins/_ism/policies/my_policy
```

### Проверить статус индекса:

```bash
GET my_index/_plugins/_ism/explain/
```

### Применить политику вручную:

```bash
POST _plugins/_ism/add/
{
  "index": "logs-2023-09",
  "policy_id": "my_policy"
}
```

---

## 7. Чеклист для продакшена

- Политика охватывает весь жизненный цикл (hot → warm → cold → delete).
- Используется rollover, чтобы не накапливать огромные индексы.
- Настроены force\_merge и replica\_count для warm фаз.
- Используется snapshot перед удалением cold-данных.
- allocation переносит индексы на нужные узлы.
- Все узлы имеют нужные атрибуты `node.attr.data`.
- Настроены шаблоны, подключающие ISM-политику.
- Проверен статус через `/_plugins/_ism/explain`.
