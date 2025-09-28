# Dockerfile и сборка образа

## Минимальная структура проекта

```
.
├── app.py
├── Dockerfile
└── .dockerignore
```

### `app.py`

```python
from pathlib import Path
import platform, os

print("Hello from container!")
print("Python:", platform.python_version())
print("Hostname:", platform.node())
print("ENV FOO:", os.getenv("FOO", "<not set>"))

p = Path("/tmp/hello.txt")
p.write_text("data inside container layer")
print("Wrote:", p)
```

### `.dockerignore`

```
__pycache__/
*.pyc
.git
.env
.DS_Store
```

---

## Dockerfile (базовый)

```Dockerfile
# Базовый образ с интерпретатором
FROM python:3.11-slim

# Рабочая директория внутри образа
WORKDIR /app

# (Если есть зависимости: сначала requirements для лучшего кеша)
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY app.py ./

# Команда по умолчанию (PID 1 — python)
CMD ["python", "app.py"]
```

---

## Сборка и запуск

1. **Сборка образа** в текущей директории (где лежит Dockerfile):

```
docker build -t hello:0.1 .
```

2. **Запуск контейнера** из образа:

```
docker run --rm hello:0.1
```

3. **Передача переменной окружения** и переопределение команды:

```
docker run --rm -e FOO=bar hello:0.1
# или так (заменить CMD):
docker run --rm hello:0.1 python -c "print('override CMD')"
```

4. **С примонтированным томом** (для сохранения данных между перезапусками):

```
docker run --rm -v mydata:/data hello:0.1 python - <<'PY'
from pathlib import Path
p = Path('/data/counter.txt')
p.write_text(str(int(p.read_text())+1) if p.exists() else '1')
print('counter =', p.read_text())
PY
```

---

## ENTRYPOINT vs CMD (и формы записи)

* `ENTRYPOINT` — неизменяемая «основная команда» образа (обычно бинарь/интерпретатор).
* `CMD` — аргументы по умолчанию для `ENTRYPOINT` **или** команда по умолчанию, если `ENTRYPOINT` не задан.
* В `docker run <image> <args...>` — переданные `<args...>` **заменяют** `CMD`, но **не** `ENTRYPOINT`.
* Переопределить `ENTRYPOINT` можно флагом `--entrypoint`.

**Формы записи:**

* **Exec‑форма (рекомендуется):** JSON‑массив, без шелла. Пример: `CMD ["python", "app.py"]`. Сигналы попадают прямо в процесс.
* **Shell‑форма:** строка, выполняется через `/bin/sh -c`. Пример: `CMD python app.py`. В контейнере PID 1 — `sh`, что мешает корректной доставке сигналов и создает риск «зомби»-процессов.

**Комбинирование:**

```Dockerfile
# ENTRYPOINT — «что запускать», CMD — «с чем»
ENTRYPOINT ["python"]
CMD ["app.py"]
```

* `docker run img` → запустит `python app.py`.
* `docker run img -m site` → запустит `python -m site` (CMD заменён на `-m site`).
* `docker run --entrypoint bash img -lc 'echo hi'` → ENTRYPOINT заменён полностью.

**Когда что использовать**

* Фиксируй `ENTRYPOINT`, если образ предназначен для конкретного приложения/демона и должен всегда стартовать «именно им».
* Оставляй только `CMD`, если образ — универсальная среда (dev‑утилиты и т.п.).

---

## Сигналы, PID 1 и tini (зачем это нужно)

### Почему это важно

* В Linux процесс с PID 1 — «инит». У него **особая семантика сигналов**: многие сигналы с «поведением по умолчанию» игнорируются, кроме `SIGKILL`/`SIGSTOP`. Если твое приложение — PID 1 и **не обрабатывает сигналы**, `docker stop` (который шлёт `SIGTERM`) может не завершить его корректно.
* PID 1 обязан **reap‑ить зомби‑процессы** (дочерние завершённые процессы). Если этого не делать, они копятся.

### Что делает Docker при остановке

* `docker stop` отправляет `SIGTERM`, ждёт `T=10s` и, если процесс не завершился, шлёт `SIGKILL`.
* Таймаут на остановку можно задать: `docker stop -t 30 <ctr>` или при запуске: `docker run --stop-timeout=30 ...`.
* Можно указать желаемый сигнал остановки в Dockerfile: `STOPSIGNAL SIGTERM` (по умолчанию и так `SIGTERM`).

### Решения

1. **Exec‑форма CMD/ENTRYPOINT.** Избегай shell‑формы, чтобы сигналы шли прямо приложению.
2. **Обработка сигналов в коде.** Лови `SIGTERM`/`SIGINT` и завершайся аккуратно (закрыть соединения, сохранить состояние).
3. **Лёгкий init‑процесс (****`tini`****).** Он становится PID 1, форвардит сигналы дочерним процессам и reaper‑ит зомби.

#### Вариант А: встроенный init Docker

```
docker run --init --rm your-image
```

`--init` включает встроенный «tiny‑init», достаточно для большинства случаев.

#### Вариант Б: tini в образе

```Dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY app.py ./
# -g: пересылать сигналы всей группе; -- : разделитель аргументов
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["python", "app.py"]
```

---

## Пример: корректное завершение на Python

```python
# graceful.py — корректная обработка SIGTERM/SIGINT
import signal, sys, time

shutdown = False

def handle(sig, frame):
    global shutdown
    print(f"got signal: {sig}")
    shutdown = True

for s in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
    signal.signal(s, handle)

print("service started")
try:
    while not shutdown:
        # здесь — полезная работа
        time.sleep(0.5)
    print("shutting down: flushing state ...")
    # закрыть соединения/файлы/пулы
    time.sleep(1)
    print("bye")
    sys.exit(0)
except Exception as e:
    print("fatal error:", e)
    sys.exit(1)
```

**Dockerfile:**

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY graceful.py ./
ENTRYPOINT ["python", "graceful.py"]
# либо с tini:
# RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*
# ENTRYPOINT ["/usr/bin/tini", "-g", "--", "python", "graceful.py"]
```

**Проверка:**

```
docker build -t graceful:demo .
# Остановить с задержкой 3 секунды, увидишь логи корректного завершения
CID=$(docker run -d graceful:demo)
sleep 1
docker stop -t 3 "$CID"
```

---

## STOPSIGNAL и тесты сигналов

* В Dockerfile можно задать желаемый «мягкий» сигнал:

```Dockerfile
STOPSIGNAL SIGTERM
```

* Отправить произвольный сигнал в контейнер:

```
docker kill --signal=SIGINT <container>
```

* Установить таймаут на остановку по умолчанию при запуске:

```
docker run --stop-timeout=20 your-image
```

---

## Частые ошибки (и быстрые фиксы)

1. **Shell‑форма CMD/ENTRYPOINT** (`CMD python app.py`) → сигналы уходят в `/bin/sh`, приложение их не видит. **Исправление:** exec‑форма JSON.
2. **Отсутствует обработка SIGTERM** → долгий стоп и `SIGKILL`, потеря состояния. **Исправление:** обработчики сигналов в коде.
3. **Зомби‑процессы** при спауне дочерних → рост PID, утечки. **Исправление:** `--init` или `tini` в ENTRYPOINT.
4. **Неправильно объединены ENTRYPOINT и CMD** → неудобно переопределять поведение. **Правило:** ENTRYPOINT — «что», CMD — «с чем».
5. **Длинные цепочки shell‑скриптов** в качестве ENTRYPOINT → сложно дебажить и форвардить сигналы. **Фикс:** минимальный бинарь/интерпретатор + явные аргументы.

---

## Паттерны использования

* **Одно приложение:**

```Dockerfile
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "python", "server.py"]
CMD ["--port", "8080"]
```

`docker run img` → `python server.py --port 8080`,
`docker run img --port 9090` → порт меняется, ENTRYPOINT остаётся тем же.

* **Утилитарный образ:**

```Dockerfile
# без ENTRYPOINT
CMD ["python", "-V"]
```

`docker run img` → покажет версию; `docker run img python -m http.server` → поведёт себя как утилита.

---

## Проверка слоёв и размеров

* Список образов: `docker image ls`
* История слоёв: `docker history hello:0.1`
* Инспект (манифест/конфиг): `docker image inspect hello:0.1`

**Почему важен порядок команд?** Если ты сначала копируешь `requirements.txt` и делаешь `pip install`, этот слой кешируется и не пересобирается, пока `requirements.txt` не изменится. Это ускоряет билды.

---

## Чек‑лист

* Правильная форма `ENTRYPOINT`/`CMD` (exec‑форма JSON).
* Либо `--init`, либо `tini` в ENTRYPOINT — для сигналов и reaper‑а.
* Обработчики `SIGTERM`/`SIGINT` в коде, тест `docker stop`.
* `STOPSIGNAL` при необходимости, разумный `--stop-timeout`.
* `.dockerignore`, порядок слоёв для кеша.
* Сборка: `docker build -t hello:0.1 .`
* Запуск: `docker run --rm hello:0.1`
