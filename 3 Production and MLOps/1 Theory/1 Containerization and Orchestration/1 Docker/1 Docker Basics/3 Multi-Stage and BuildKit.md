# Multi-stage and BuildKit&#x20;

## Multi‑stage builds (многостейдж‑сборки)

**Идея:** разделяем стадии «builder» (где ставим компиляторы/SDK и собираем артефакты) и «runtime» (минимальная среда запуска). В финальный образ копируем только артефакты.

### Базовый паттерн (Python)

```Dockerfile
# syntax=docker/dockerfile:1.7

########## STAGE 1: builder ##########
FROM python:3.11-bookworm AS builder
WORKDIR /src
# 1) фиксируем зависимости отдельно — для кеша
COPY requirements.txt ./
# BuildKit кеш для pip (ускоряет повторные билды)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --wheel-dir=/wheels -r requirements.txt

# 2) копируем исходники и собираем приложение (если нужны артефакты)
COPY . .
# пример: сборка C-расширений/минимизация/линтеры — всё оставляем тут

########## STAGE 2: runtime ##########
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
# Копируем только нужное из builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY app/ ./app/
ENTRYPOINT ["python", "-m", "app"]
```

**Разбор:**

- FROM python:3.11-bookworm AS builder

  WORKDIR /src

**Плюсы:**

- маленький финальный образ (без компиляторов/заголовков);

- ускорение повторных сборок (кеш слоёв `requirements.txt` → `pip wheel`);

- чистые границы: всё «шумное» (build‑time зависимости) остаётся в builder.

**Советы:**

- давай имена стадиям и копируй по именам: `FROM ... AS builder`, `COPY --from=builder ...`;
- можно иметь несколько специализированных стадий (например, `test`, `lint`) и выбирать `--target`.

---

## Layer cache (кеш слоёв) и invalidation

- Docker кеширует слой, если **инструкция и её контент** идентичны прошлому билду.
- Порядок важен: всё, что меняется реже (библиотеки/базовый слой) — выше, а то, что чаще (исходники) — ниже.
- `.dockerignore` критичен: исключай всё лишнее, чтобы не инвалидировать `COPY . .`.

**Классический паттерн Python:**

```Dockerfile
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY . .
```

Пока `requirements.txt` не меняется, слой с зависимостями кешируется, и повторная сборка быстрая.

---

## BuildKit: что это и как включить

BuildKit — новый движок сборки Docker: параллелизм, `RUN --mount`, секреты, внешние кеши, кросс‑платформенность, аттестации.

**Включение:**

- Разово в команде: `DOCKER_BUILDKIT=1 docker build ...`
- Глобально в конфиге Docker: `{ "features": { "buildkit": true } }`
- Рекомендуемая шапка Dockerfile для новых фич:

```Dockerfile
# syntax=docker/dockerfile:1.7
```

> Версия синтаксиса влияет на доступные расширения (`--mount`, секреты, `--chmod`, и т.д.).

---

## `RUN --mount=type=cache` (горячее кеширование в рантайме сборки)

Позволяет монтировать каталоги как кеш между билдами **без** попадания содержимого в слой образа.

### pip cache

```Dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt
```

### apt cache (осторожно, только под BuildKit)

```Dockerfile
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*
```

**Плюсы:** ускорение без «утяжеления» слоёв; **минусы:** требуются BuildKit и аккуратность с правами/платформами.

---

## `--mount=type=secret` (секреты в момент сборки)

Хранить секреты в ARG/ENV — плохо: они попадут в историю образа. BuildKit позволяет «прокинуть» секрет как временный файл:

```Dockerfile
# syntax=docker/dockerfile:1.7
RUN --mount=type=secret,id=pip_token \
    pip config set global.index-url "https://__token__:@my-private/simple" && \
    pip config set global.extra-index-url "https://pypi.org/simple"
```

Сборка:

```
docker build \
  --secret id=pip_token,env=PIP_TOKEN \
  -t myimg:dev .
```

Содержимое секрета не попадёт в слои.

---

## buildx: кросс‑платформенная сборка и удалённые кеши

`buildx` — надстройка CLI, использующая BuildKit builders. Умеет:

- собирать под несколько платформ `--platform linux/amd64,linux/arm64`;
- пушить образы и кеш в реестр;
- использовать локальные/удалённые кеши (`--cache-from/--cache-to`).

### Быстрый старт

```bash
# создать и активировать builder (один раз)
docker buildx create --use --name xbuilder

# multi-arch сборка и пуш
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t registry.example.com/myapp:1.0 \
  --push .
```

### Кеш в реестре

```bash
docker buildx build \
  --cache-to type=registry,ref=registry.example.com/myapp:buildcache,mode=max \
  --cache-from type=registry,ref=registry.example.com/myapp:buildcache \
  -t registry.example.com/myapp:1.0 --push .
```

Это ускоряет сборки в CI: следующий билд подтянет кеш из реестра.

---

## Reproducible builds (воспроизводимые сборки)

**Задача:** одинаковый Dockerfile при одинаковом контенте всегда даёт **бит‑в‑бит** одинаковый образ.

### Практики

1. **Пиновать версии** базовых образов и зависимостей:
   - базовый образ: `python:3.11.9-slim-bookworm` (с конкретным тегом),
   - `requirements.txt` с фиксированными версиями (`==`), лучше с `--require-hashes`.
2. **Исключить нефинализированные артефакты**:
   - `ENV PYTHONDONTWRITEBYTECODE=1` чтобы не плодить разновременные `.pyc`,
   - или использовать hash‑based pyc (PEP 552) при необходимости.
3. **Избегать времени/локали**:
   - задавать `TZ`, `LANG`, `LC_ALL` стабильно,
   - не включать в финальный слой файлы с «текущей датой/временем».
4. **Одинаковый порядок шагов** и стабильные фоновые источники (apt/pip‑зеркала).
5. **Зависимости из lock‑файлов** (pip‑tools/Poetry/PDM) с хешами.
6. **Сборка колес в builder‑стадии** и перенос готовых артефактов (исключить повторные пересборки).
7. **Строгие COPY**: копируй только нужные файлы (`COPY app/ ./app/`, а не `COPY . .`).

### Мини‑шаблон

```Dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11.9-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8

FROM base AS builder
WORKDIR /src
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --wheel-dir=/wheels -r requirements.txt --require-hashes
COPY app/ ./app/

FROM base AS runtime
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY app/ ./app/
ENTRYPOINT ["python", "-m", "app"]
```

---

## Частые ошибки и анти‑паттерны

1. **Один жирный образ без стадий** → огромный размер, медленные доставки. Делай builder/runtime.
2. `\*\* без \*\*` → случайно утащил кэш/секреты, сломал кеш; используй явные пути.
3. **Неиспользование BuildKit** → нет `--mount=cache`, нет секретов, медленнее.
4. **Секреты в ENV/ARG** → утечка в историю образа. Используй `--mount=type=secret`.
5. **Сброс кеша из‑за порядка шагов** → перенос `COPY requirements.txt` выше.
6. **Нестабильные зеркала/пакеты** → фиксируй репозитории, пингуй версии, применяй lock‑файлы.

---

## Шпаргалка команд

```bash
# Включить BuildKit разово
DOCKER_BUILDKIT=1 docker build -t myimg:dev .

# buildx: создать builder и использовать
docker buildx create --use --name xbuilder

# multi-arch + пуш + удалённый кеш
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --cache-to type=registry,ref=reg.example.com/my:cache,mode=max \
  --cache-from type=registry,ref=reg.example.com/my:cache \
  -t reg.example.com/my:1.0 --push .

# таргет стадии
docker build --target builder -t myimg:builder .
```

---

## Мини‑чек‑лист

- Включён BuildKit; указан `# syntax=docker/dockerfile:1.7`.
- Стадии: `builder` → `runtime`; строгие `COPY`.
- Кеш: `RUN --mount=type=cache` для pip/apt.
- Секреты: `--mount=type=secret` при необходимости.
- Reproducible: фиксированные версии, локали, без случайных временных артефактов.
- buildx: multi‑arch и удалённые кеши для CI.

