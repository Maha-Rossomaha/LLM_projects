# Snapshot & Restore

Снапшоты (резервные копии) — ключевая часть отказоустойчивости и миграций. OpenSearch позволяет сохранять и восстанавливать:

* отдельные индексы,
* весь кластер (включая шаблоны, алиасы, настройки).

Поддерживаются различные хранилища: локальный диск, S3, HDFS, Azure, Google Cloud и др.

---

## 1. Репозитории

Перед использованием необходимо зарегистрировать репозиторий.

### 1.1. Репозиторий на локальном диске (fs)

```bash
PUT _snapshot/my_fs_repo
{
  "type": "fs",
  "settings": {
    "location": "/mnt/backups",
    "compress": true
  }
}
```

> Директория `/mnt/backups` должна быть указана в `path.repo` в `opensearch.yml`.

```yaml
path.repo: ["/mnt/backups"]
```

### 1.2. Репозиторий в S3

```bash
PUT _snapshot/my_s3_repo
{
  "type": "s3",
  "settings": {
    "bucket": "my-opensearch-backups",
    "region": "us-east-1"
  }
}
```

> Требует плагина `repository-s3` и настроенных AWS credentials.

---

## 2. Создание снапшота

```bash
PUT _snapshot/my_fs_repo/snapshot_2023_09_06
{
  "indices": "logs-*,metrics-*",
  "ignore_unavailable": true,
  "include_global_state": true
}
```

* `indices` — список или шаблон индексов.
* `include_global_state` — сохранять алиасы, шаблоны, роли.

### Async статус:

```bash
GET _snapshot/my_fs_repo/snapshot_2023_09_06
```

---

## 3. Настройка snapshot-политик (automated backups)

```bash
PUT _plugins/_snapshot_policy/policy_daily
{
  "enabled": true,
  "schedule": "0 30 2 * * ?",  
  "name": "daily-snap-{now/d}",
  "repository": "my_fs_repo",
  "config": {
    "indices": ["*"],
    "ignore_unavailable": true,
    "include_global_state": true
  },
  "retention": {
    "expire_after": "30d",
    "min_count": 3,
    "max_count": 50
  }
}
```

> Поддерживается с OpenSearch 2.6+. Использует Quartz cron-синтаксис.

---

## 4. Восстановление из снапшота

### Восстановить конкретный индекс

```bash
POST _snapshot/my_fs_repo/snapshot_2023_09_06/_restore
{
  "indices": "logs-2023-08",
  "rename_pattern": "logs-(.+)",
  "rename_replacement": "logs-restored-$1",
  "include_aliases": false
}
```

* По умолчанию индекс будет перезаписан. Используй `rename_*` для избежания конфликтов.

### Восстановить всё

```bash
POST _snapshot/my_fs_repo/snapshot_2023_09_06/_restore
```