# Python Service

## Выбор базового образа: `python:*‑slim`

- Используй официальные образы `python:3.11‑slim` (или с точной версией: `python:3.11.9‑slim‑bookworm`).
- Плюсы: меньше слоёв и размер, быстрее доставки. Устанавливай build‑time пакеты только в builder‑стадии.
- Рекомендуемые переменные окружения:

```Dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8
```

---

## `venv` внутри контейнера

- Изолируй рантайм‑зависимости в виртуальном окружении, например в `/opt/venv`.
- В рантайме используй только этот venv (без system‑site‑packages).

```Dockerfile
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"
```

---

## Фиксация зависимостей (pinned deps) и reproducible builds

- Фиксируй версии в `requirements.txt` (лучше через lock‑файл: pip‑tools/Poetry/PDM).
- По возможности используй хеши: `pip install --require-hashes -r requirements.txt`.
- Не клади секреты/токены в ARG/ENV — они попадают в историю образа.

Пример фрагмента `requirements.txt`:

```
uvicorn==0.30.0
fastapi==0.115.0
```

---

## Wheels и многостейдж‑сборка

- В builder‑стадии собирай **колёса** (`*.whl`) и устанавливай их в venv.
- В runtime‑стадию копируй только venv и код приложения.

```Dockerfile
# syntax=docker/dockerfile:1.7

########## builder ##########
FROM python:3.11-bookworm AS builder
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /src

# 1) зависимости отдельно для кеша
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --wheel-dir /wheels -r requirements.txt

# 2) код
COPY app/ ./app/
RUN pip install --no-cache-dir /wheels/*

########## runtime ##########
FROM python:3.11-slim AS runtime
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8

# безопасность: свой пользователь
RUN adduser --disabled-password --gecos "" appuser && mkdir -p /app
WORKDIR /app

# переносим только venv и код
COPY --from=builder $VENV_PATH $VENV_PATH
COPY app/ ./app/

USER appuser
EXPOSE 8000
# HEALTHCHECK ниже; ENTRYPOINT в конце
```

---

## Кеширование слоёв и BuildKit

- Порядок важен: `COPY requirements.txt` → установка deps → `COPY app/`.
- Включай BuildKit, чтобы использовать горячий кеш: `RUN --mount=type=cache` для pip/apt.
- Не забывай `.dockerignore` (чтобы не инвалидавать кеш из‑за лишних файлов).

`.dockerignore`:

```
__pycache__/
*.pyc
.git
.env
.DS_Store
*.log
```

---

## Минимальный HTTP‑сервис и healthcheck

Простой сервер и health‑проверка на чистом Python.

`app/server.py`:

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import signal, threading, sys

shutdown_event = threading.Event()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Hello")

    def log_message(self, fmt, *args):
        # тише в контейнере
        return

def _handle_signal(sig, frame):
    shutdown_event.set()

for s in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
    signal.signal(s, _handle_signal)

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # ждём сигнал завершения
        shutdown_event.wait()
        server.shutdown()
        thread.join(timeout=2)
        sys.exit(0)
    except Exception as e:
        print("fatal:", e)
        sys.exit(1)
```

`app/healthcheck.py`:

```python
import sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
```

Добавим `HEALTHCHECK` и `ENTRYPOINT` в runtime‑стадию:

```Dockerfile
HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
  CMD python /app/app/healthcheck.py

ENTRYPOINT ["python", "/app/app/server.py"]
# Либо запускай с встроенным init: docker run --init ...
```

---

## Graceful shutdown (корректное завершение)

- Запускай контейнер с инитом: `docker run --init ...` (или используй `tini` в ENTRYPOINT), чтобы:
  - корректно форвардить SIGTERM/SIGINT приложению,
  - reaper‑ить зомби‑процессы.
- Лови сигналы в Python и закрывай соединения/пулы (см. `server.py`).
- При необходимости укажи `STOPSIGNAL SIGTERM` и настрой `--stop-timeout`.

---

## Практические мелочи

- Логи в stdout/stderr (пусть оркестратор собирает их); без буферизации (`PYTHONUNBUFFERED=1`).
- Не ставь компиляторы в runtime‑образ.
- Запускай под непривилегированным пользователем (`USER appuser`).
- Храни конфигурацию в ENV/файлах, а не в образе (12‑factor).

---

## Сборка и запуск

```bash
# сборка
DOCKER_BUILDKIT=1 docker build -t mypy:0.1 .

# запуск (c init для сигналов)
docker run --rm --init -p 8000:8000 mypy:0.1

# проверка здоровья
docker inspect --format='{{json .State.Health}}' $(docker ps -q | head -n1) | jq
```

---

## Короткий чек‑лист

- `python:*‑slim` + точный тег версии.
- `venv` в `/opt/venv`, `PATH` указывает на него.
- Зависимости зафиксированы (lock/pins), при желании `--require-hashes`.
- В builder собраны wheels; в runtime только venv + код.
- Порядок слоёв и `.dockerignore` для кеша; BuildKit `--mount=cache`.
- `HEALTHCHECK` вызывает Python‑скрипт, есть `/health`.
- Graceful shutdown: сигналы + `--init` (или `tini`).

