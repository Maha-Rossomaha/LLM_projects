# Volumes, Backups and Migration

## Зачем нужны тома

Файловая система контейнера эфемерна: при пересборке/перезапуске изменения исчезают. **Том** даёт постоянное хранилище: настройки, данные БД, индексы поиска, артефакты ETL.

---

## Два типа: bind‑mount и named volume

### Bind‑mount (`/host/path:/container/path`)

**Bind-mount** — это способ примонтировать в контейнер **реальную директорию или файл с хоста**.\
То есть контейнер видит не копию, а **прямой доступ** к этим данным.

```
docker run -v /host/path:/container/path image
```

или

```
docker run --mount type=bind,source=/host/path,target=/container/path image
```

### Ключевые свойства

- **Источник** (`/host/path`) — путь на хосте.
- **Цель** (`/container/path`) — куда в контейнере примонтировать.
- Изменения **двусторонние**: меняешь в контейнере → видно на хосте и наоборот.
- Используется, когда нужно:
  - работать с локальным кодом без пересборки образа;
  - сохранять данные вне контейнера (например, логи, конфиги);
  - передавать сокеты, устройства, сертификаты и т.д.

### Named volume (`volname:/container/path`)

\
**Named volume** — это постоянное хранилище данных, которым управляет Docker сам.\
В отличие от bind-mount, путь на хосте не указывается вручную, вместо этого задаётся **имя тома**.

```
docker volume create mydata   # создать том
docker run -v mydata:/app/data image
```

или 

```
docker run --mount type=volume,source=mydata,target=/app/data image
```

### Ключевые свойства

- **Источник** (`mydata`) — имя тома (не путь).
- **Цель** (`/app/data`) — путь внутри контейнера.
- Данные сохраняются в специальной директории Docker (обычно `/var/lib/docker/volumes/mydata/_data`).
- Изменения сохраняются между перезапусками контейнера.
- Можно подключать один и тот же том к разным контейнерам.

**Коротко:**

- Dev: чаще **bind** (код/конфиги), плюс **named** для БД.
- Prod: преимущественно **named volumes** для состояния; bind — только там, где нужно строгое соответствие пути (например, внешние ключи/сертификаты с `:ro`).

---

## Layout данных (как раскладывать)

Рекомендуемая схема путей в контейнере:

- `/app` — код приложения (часто read‑only).
- `/config` — конфигурация (часто read‑only, можно bind‑mount).
- `/data` — рабочие данные приложения (named volume, read‑write).
- `/var/lib/<svc>` — данные сервисов (Postgres, Qdrant, OpenSearch).
- `/var/log/<svc>` — логи (можно в stdout/stderr без отдельного тома).
- `tmpfs` для временных файлов/кеша: `--tmpfs /tmp:rw,size=256m`.

**Права:** запускай процесс **под непривилегированным UID/GID**, делай предварительный `chown` в образе или настрой `user:` в compose.

---

## Мини‑пример: Python пишет в том

`app/write_data.py`:

```python
from pathlib import Path
p = Path("/data/counter.txt")
p.write_text(str(int(p.read_text())+1) if p.exists() else "1")
print("counter =", p.read_text())
```

Запуск с **named volume**:

```
docker volume create mydata
docker run --rm -v mydata:/data python:3.11-slim \
  python - <<'PY'
from pathlib import Path
p = Path('/data/counter.txt')
p.write_text(str(int(p.read_text())+1) if p.exists() else '1')
print('counter =', p.read_text())
PY
```

---

## Бэкапы: стратегии

### 1) Специфичные для СУБД/движка (лучше всего)

- **Postgres:** `pg_dump`/`pg_basebackup` + WAL, или логические дампы.
- **OpenSearch/Elasticsearch:** snapshot repository (S3/NFS).
- **Qdrant:** snapshots API.

Плюс: согласованность формата. Минус: настраивать отдельно под каждый движок.

### 2) «Снимок» тома tar‑ом (универсально)

Подходит для файловых данных/индексов (лучше останавливать или переводить в read‑only на время бэкапа).

**Резервная копия named volume → tar:**

```
BACKUP=backup_$(date +%F).tar.gz
VOL=appdata

docker run --rm -v $VOL:/data:ro -v $(pwd):/backup \
  busybox sh -c "cd /data && tar -czf /backup/$BACKUP ."
```

**Восстановление tar → named volume:**

```
BACKUP=backup_2025-09-29.tar.gz
VOL=appdata

docker run --rm -v $VOL:/data -v $(pwd):/backup \
  busybox sh -c "cd /data && tar -xzf /backup/$BACKUP"
```

**Bind‑mount** бэкапится обычным `tar` на хосте:

```
# на хосте
sudo tar -czf appdata.tar.gz -C /srv/appdata .
```

**Совет:** проверяй бэкап восстановлением в тестовый том перед тем, как считать его валидным.

---

## Миграции (перенос между серверами)

### Named volume → архив → перенос

1. На исходном хосте — сделать архив (см. выше) и перенести файл.
2. На целевом хосте — создать том и развернуть архив.

```
docker volume create appdata
scp backup_2025-09-29.tar.gz user@dest:/tmp/
docker run --rm -v appdata:/data -v /tmp:/backup \
  busybox sh -c "cd /data && tar -xzf /backup/backup_2025-09-29.tar.gz"
```

### Bind‑mount

Скопируй директорию на целевой хост, сохрани UID/GID/права: `rsync -aHAX` или `tar --xattrs --same-owner`.

---

## Права и безопасность

- Запускай контейнеры под фиксированным `UID:GID`, совпадающим с владельцем каталога на хосте (для bind‑mount).
- Для SELinux/Podman: пометки `:z`/`:Z` на bind‑mount.
- Делай каталог и том **минимально необходимыми правами**; секреты не клади в тома (используй менеджеры секретов).
- Для критичных данных думай об **шифровании на хосте** (LUKS/FS‑уровень) и шифровании бэкапов.

---

## Частые ошибки

1. Монтировать весь проект `.:/` и случайно перекрыть системные пути контейнера.
2. Запускать несколько писателей в один и тот же том → гонки/повреждение (особенно БД/индексы).
3. Отсутствие `.dockerignore` → случайные файлы инвалидируют кеш.
4. На проде держать данные на bind‑mount из непредназначенного для этого каталога (`/home/...`) без бэкапов.
5. Делать «живые» файловые бэкапы БД без согласованного снапшота.

---

## Шпаргалка CLI

```
# список и детали томов
docker volume ls
docker volume inspect <name>

# создать/удалить
docker volume create <name>
docker volume rm <name>

# очистить неиспользуемые
docker volume prune

# временный контейнер для операций с томом
docker run --rm -v <vol>:/data busybox ls -la /data
```

---

## Что выбрать в случае ETL + Search

- **ETL (Airflow):** результаты и артефакты кладем в **named volumes** (например, `/var/lib/airflow-data`). Логи — в stdout или отдельный bind‑mount, если нужно читать с хоста.
- **Search‑сервис (GPU, индексы):** индексы/кэши — отдельный **named volume** (легко бэкапить/мигрировать); большие статические модели можно держать на bind‑mount (read‑only) или в отдельном volume‑кэше.
- Разделяем тома по назначению: `etl_out`, `search_index`, `pgdata` — проще бэкапить и переносить по частям.

