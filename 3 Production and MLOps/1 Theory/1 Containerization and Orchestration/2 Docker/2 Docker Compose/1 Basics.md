# Docker Compose 

Короткий, но полный конспект: что такое `docker-compose.yml`, как описывать мультисервисные окружения (Postgres, Redis, MinIO, Airflow, API), пробрасывать порты, работать с `override`‑файлами, `.env` и профилями.

---

## Что такое `docker-compose.yml`
`docker-compose.yml` — декларативное описание **набора сервисов** (контейнеров), их образов/билда, сетей, томов, переменных и зависимостей.
Одной командой запускает весь стек: `docker compose up`.

- **Dockerfile** описывает *как собрать образ одного сервиса*.
- **Compose** описывает *как вместе запустить несколько сервисов* и связать их.

> Рекомендуемая версия синтаксиса: `version: "3.9"` (актуальна для Docker Compose V2).

---

## Базовый `docker-compose.yml` (API + Postgres + Redis + MinIO + Airflow)

```yaml
version: "3.9"

x-env: &default-env
  env_file:
    - .env

networks:
  backend: { }
  public: { }

volumes:
  pgdata: { }
  minio_data: { }
  airflow_data: { }

services:
  api:
    build: ./api
    <<: *default-env
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis:6379/0
      - MINIO_ENDPOINT=http://minio:9000
    depends_on: [postgres, redis, minio]
    networks: [backend, public]
    ports: ["8000:8000"] # dev: наружу

  postgres:
    image: postgres:15
    <<: *default-env
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks: [backend]

  redis:
    image: redis:7-alpine
    networks: [backend]

  minio:
    image: minio/minio:latest
    command: server /data --console-address :9001
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    networks: [backend, public]
    ports: ["9000:9000", "9001:9001"]

  airflow:
    build: ./airflow
    <<: *default-env
    # для простоты — монолитный контейнер webserver+scheduler (для prod обычно отдельные)
    volumes:
      - airflow_data:/opt/airflow
    networks: [backend, public]
    ports: ["8080:8080"]
    depends_on: [postgres, redis]
```

**Пояснения:**
- Все сервисы в **одной user‑defined сети** видят друг друга по имени (`postgres`, `redis`, `minio`, `api`).
- Порты публикуем только у того, что нужно снаружи (в dev это `api`, `minio`, `airflow`).
- Данные БД/объектного хранилища кладём в **named volumes**.

---

## Проброс портов наружу (`ports:`)
- Формат: `"HOST:CONTAINER"` — например, `"8000:8000"`.
- Чтобы слушать только на localhost хоста: `"127.0.0.1:8000:8000"`.
- Внутри контейнера сервис должен слушать `0.0.0.0`, иначе паблиш не сработает.

---

## `.env` и переменные окружения

Compose **автоматически** подхватывает файл `.env` из каталога с `docker-compose.yml`.
Пример `.env`:
```
POSTGRES_USER=app
POSTGRES_PASSWORD=app
POSTGRES_DB=app
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio_secret
```

Использование в `docker-compose.yml`:
- подстановка: `${VAR}` или `${VAR:-default}`;
- секция `environment:` и/или `env_file:`;
- можно иметь несколько `env_file`.

> Переменные из окружения *шела* перекрывают значения в `.env`.

---

## Override‑файлы: `docker-compose.override.yml`

`docker compose` по умолчанию читает **оба** файла, если есть:
- `docker-compose.yml` — базовая конфигурация (ближе к prod, без лишних портов/маунтов);
- `docker-compose.override.yml` — локальные изменения для **dev/test** (маунты кода, дополнительные порты, дебаг‑флаги).

Пример `docker-compose.override.yml` для разработки:
```yaml
version: "3.9"
services:
  api:
    volumes:
      - ./api:/app:rw   # live‑код
    environment:
      - DEBUG=1
    ports: ["8000:8000"]  # оставим наружу

  airflow:
    environment:
      - AIRFLOW__CORE__LOAD_EXAMPLES=true
    ports: ["8080:8080"]
```

Явное указание файлов:
```
docker compose -f docker-compose.yml -f docker-compose.override.yml up
```
Порядок важен: последующие файлы **переопределяют** предыдущие ключи.

---

## Профили (`profiles:`)

**Профили** позволяют включать/выключать группы сервисов без правок файлов.

В базовом `docker-compose.yml`:
```yaml
services:
  minio:
    image: minio/minio:latest
    profiles: [storage]
    ...

  airflow:
    build: ./airflow
    profiles: [etl]
    ...
```

Запуск с профилями:
```
# только API+DB+Redis (без MinIO/Airflow)
docker compose up

# включить MinIO
docker compose --profile storage up

# включить Airflow
docker compose --profile etl up

# сразу несколько
docker compose --profile storage --profile etl up
```

> Профили удобно использовать для «тяжёлых» сервисов (MinIO, Airflow, GPU‑сервисы): в dev они выключены по умолчанию.

---

## Dev/Test стенды: рекомендуемый подход

- **`docker-compose.yml`** — максимально близок к prod (без bind‑маунтов кода, минимум открытых портов, только нужные тома/сети).
- **`docker-compose.override.yml`** — локальный dev: bind‑маунты исходников, включённые порты, DEBUG‑флаги, удобные инструменты.
- **`profiles:`** — включение тяжёлых/опциональных сервисов по требованию.
- **`.env`** — секретов не кладём; только безобидные переменные. Настоящие секреты — через менеджеры секретов/пер‑сервис ENV вне VCS.

Примеры команд:
```
# базовый dev (API+DB+Redis)
docker compose up -d --build

# с профилем storage (MinIO)
docker compose --profile storage up -d

# тесты API с изолированным env
docker compose -f docker-compose.yml -f docker-compose.override.yml \
  run --rm api pytest -q
```

---

## Нюансы и частые ошибки
- `depends_on` **не ждёт готовности** сервиса, только запускает раньше. Для готовности используй `healthcheck` и скрипты ожидания.
- Старайся разделять **сети**: `public` для API/прокси, `backend` для внутренних (DB/кеш/очереди).
- Данные БД/хранилищ держи в **named volumes**; bind‑маунты кода — только в dev.
- Следи, чтобы сервис внутри контейнера слушал `0.0.0.0`, иначе публикация порта не работает.
- Для больших проектов разбивай на несколько файлов и подключай через `-f`.

Мини‑`healthcheck` (пример):
```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)"]
      interval: 10s
      timeout: 2s
      retries: 3
```

---

## Шпаргалка команд
```
# запустить/остановить
docker compose up -d
docker compose down

# пересобрать
docker compose build --no-cache

# логи и статус
docker compose logs -f api
docker compose ps

# выполнить команду внутри сервиса
docker compose exec api bash

# с профилями
docker compose --profile etl up -d
```

