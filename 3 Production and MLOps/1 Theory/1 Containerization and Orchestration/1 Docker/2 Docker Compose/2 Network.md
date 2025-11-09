# Сеть Docker

## 1. Ключевые идеи&#x20;

- Каждый контейнер живёт в своём **network namespace**.
- Режим **bridge** — дефолт: Docker делает виртуальный мост (обычно `docker0`) и NAT для выхода в интернет.
- **user‑defined bridge** сети дают: имена сервисов (встроенный DNS), изоляцию и управляемость. Используй **их**, а не default `bridge`.

---

## 2. Минимальная сеть (user‑defined bridge)

Создаём сеть и запускаем два контейнера в ней:

```bash
docker network create --driver bridge appnet

docker run -d --name db --network appnet postgres:15
# условный API (пример):
docker run -d --name api --network appnet -p 8000:8000 myapi:dev
```

Теперь `api` и `db` видят друг друга по именам: `db:5432`, `api:8000`.

Проверка DNS внутри `api`:

```bash
docker exec -it api sh -lc 'cat /etc/resolv.conf; getent hosts db'
# в resolv.conf ожидаем nameserver 127.0.0.11
```

---

## 3. Встроенный DNS Docker

- В **user‑defined** сетях Docker поднимает резолвер на `127.0.0.11`.
- Он резолвит **имена контейнеров** и их **алиасы** в IP внутри сети.
- В Compose **имена сервисов** = DNS‑имена.

Пример DNS‑проверки на Python:

```python
import socket
print(socket.gethostbyname("db"))  # -> IP контейнера db
```

Алиасы в Compose:

```yaml
networks:
  backnet: { }
services:
  api:
    image: my/api
    networks:
      backnet:
        aliases: [api.internal]
  db:
    image: postgres:15
    networks: [backnet]
```

Теперь доступны `db` и `api.internal` по именам.

---

## 4. Порты `EXPOSE` vs `ports:`

### Что делает `EXPOSE`

- `EXPOSE 8000` в Dockerfile — **метаданные**: «этот контейнер слушает 8000 внутри». Само по себе это **не публикует** порт наружу.
- Полезно как документация и чтобы сработал `docker run -P` (заглавная `-P`): он опубликует **все** `EXPOSE`‑порты на **случайные** порты хоста.

### `ports:` и `-p` — реальная публикация

- Публикация делается **при запуске**: NAT‑правила пробрасывают трафик с хоста в контейнер.

- Форматы (CLI):

  - `-p 8000:8000` — слушать на **всех интерфейсах** хоста (0.0.0.0:8000 → контейнер:8000).
  - `-p 127.0.0.1:8000:8000` — слушать **только** на localhost хоста.
  - `-p 8000` — опубликовать контейнерный порт **8000** на **случайный** порт хоста (узнать через `docker port <ctr> 8000`).
  - Диапазон: `-p 8000-8005:8000-8005`.
  - UDP: `-p 8125:8125/udp`.

- В Compose (short‑form) эквиваленты:

```yaml
services:
  api:
    image: my/api
    ports:
      - "8000:8000"               # все интерфейсы
      - "127.0.0.1:8001:8000"     # localhost хоста → контейнер:8000
      - "8125:8125/udp"           # UDP‑порт
```

> Внутри контейнера приложение **обязательно** должно слушать `0.0.0.0:PORT` (или `[::]` для IPv6). Если оно слушает только `127.0.0.1`, публикация через `-p/ports:` работать не будет.

### `expose:` в Compose

- `expose:` — открыть порт **только для других контейнеров** в той же сети (без публикации на хост):

```yaml
services:
  db:
    image: postgres:15
    expose: ["5432"]  # доступен как db:5432 внутри сети, но не с хоста
```

### Диагностика публикации порта

```bash
# С хоста:
curl -sS http://127.0.0.1:8000/health
ss -lntp | grep :8000           # кто слушает порт на хосте

docker port api 8000            # узнать сопоставление портов

docker exec -it api sh -lc 'python - <<PY
import socket as s; sock=s.socket(); sock.bind(("0.0.0.0",8000)); print("can bind: ok")
PY'  # (пример проверки бинда)
```

---

## 5. Общение сервис‑к‑сервису (Compose)

```yaml
version: "3.9"
services:
  api:
    build: ./api
    environment:
      - DATABASE_URL=postgresql://app:app@db:5432/app
    depends_on: [db]
    networks: [backnet]
    ports: ["8000:8000"]

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: app
    networks: [backnet]

networks: { backnet: { } }
```

Здесь `api` общается с `db` по адресу `db:5432` благодаря встроенному DNS.

> Помни: `depends_on` **не ждёт готовности** сервиса. Для готовности используй `healthcheck`.

Мини‑healthcheck:

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=1)"]
      interval: 10s
      timeout: 2s
      retries: 3
```

---

## 6. Базовая диагностика (cheatsheet)

**На хосте:**

```bash
docker network ls                    # сети
docker network inspect appnet        # кто подключён, подсеть, DNS
```

**В контейнере:**

```bash
docker exec -it api sh -lc 'ip addr; echo; ip route; echo; cat /etc/resolv.conf'
```

**Проверка DNS/сокета Python‑скриптом:**

```python
# netcheck.py
import socket, sys
host = sys.argv[1] if len(sys.argv)>1 else 'db'
port = int(sys.argv[2]) if len(sys.argv)>2 else 5432
print('resolve', host, '->', socket.gethostbyname(host))
s = socket.socket(); s.settimeout(2)
s.connect((host, port)); print('tcp ok'); s.close()
```

Запуск:

```bash
docker exec -it api python /app/netcheck.py db 5432
```

**Конфликт портов на хосте:**

```bash
ss -lntp | grep :8000    # занял ли порт другой процесс
```
