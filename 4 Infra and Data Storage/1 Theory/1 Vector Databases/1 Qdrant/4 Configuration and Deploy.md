# Configuration and Deploy

## 1. Конфигурационные файлы

Qdrant поддерживает конфигурацию через файл **`config.yaml`** (по умолчанию) и альтернативные форматы (`TOML`, `JSON`, переменные окружения).

Пример `config.yaml`:

```yaml
service:
  host: 0.0.0.0
  grpc_port: 6334
  http_port: 6333

storage:
  path: ./storage
  snapshots_path: ./snapshots

cluster:
  enabled: false

log_level: INFO
```

Основные разделы:

* **service** — порты и сетевые настройки.
* **storage** — путь к данным и снапшотам.
* **cluster** — параметры кластера (включение/отключение).
* **log\_level** — уровень логирования.

Альтернативные способы:

* `QDRANT__SERVICE__HTTP_PORT=8080` (через переменные окружения),
* `config.json` или `config.toml` с аналогичной структурой.

---

## 2. Основные параметры

* `service.host` — IP-адрес для биндинга.
* `service.http_port` — порт REST API (по умолчанию 6333).
* `service.grpc_port` — порт gRPC API (по умолчанию 6334).
* `storage.path` — директория хранения сегментов и индексов.
* `storage.snapshots_path` — директория для снапшотов.
* `cluster.enabled` — включение кластерного режима.
* `log_level` — уровень логов: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`.

---

## 3. Docker-запуск

Базовый запуск Qdrant через Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

* `-p 6333:6333` — REST API.
* `-p 6334:6334` — gRPC API.
* `-v ./qdrant_storage:/qdrant/storage` — монтирование локальной папки для сохранения данных.

Запуск с кастомным конфигом:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/config.yaml:/qdrant/config/config.yaml \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

---

## 4. Где хранится дата

* По умолчанию данные хранятся в директории **`/qdrant/storage`** (в Docker-контейнере).
* Для персистентности стоит монтировать локальную папку (`-v ./qdrant_storage:/qdrant/storage`).
* Снапшоты и бэкапы сохраняются в `/qdrant/snapshots`.
