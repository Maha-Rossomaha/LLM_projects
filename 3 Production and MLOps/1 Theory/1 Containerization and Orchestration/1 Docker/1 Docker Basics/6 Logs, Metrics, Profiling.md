# Logs, Metrics, Profiling

---

## 1. Логи контейнеров

### Принцип

- Пиши **в stdout/stderr**. Не создавай файлов логов внутри контейнера: ими хуже управлять.
- Пусть оркестратор (Docker/Compose/k8s) собирает логи и отправляет их в централизованное хранилище.

### Драйверы логов Docker

- `json-file` (по умолчанию): прост, локальный JSON‑лог на хосте. Обязательно включай **ротацию (**`max-size`, `max-file`).
- `local`: компактнее, чем `json-file` (бинарный формат), тоже локально.
- `journald` / `syslog` — пишут в системные демоны хоста (`systemd-journald`, `syslog`).
- `fluentd` / `gelf` / `awslogs` / `splunk` / `etwlogs` (Windows) — отправка логов во внешние системы логирования.
- `none` : отключает сбор логов Docker (используй только если у тебя s`idec`ar‑агент читает прямо из приложения).

**Ротация для ****\`\`****\*\*\*\*:**

```bash
docker run \
  --log-driver json-file \
  --log-opt max-size=10m --log-opt max-file=5 \
  myimg
```

Compose:

```yaml
services:
  api:
    image: my/api
    logging:
      driver: json-file
      options: { max-size: "10m", max-file: "5" }
```

### Логирование в приложении (Python)

Минимальная настройка структурированного логгера в stdout:

```python
import logging, json, sys
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        })

h = logging.StreamHandler(sys.stdout)
h.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[h])
logging.getLogger("app").info("service started")
```

---

## 2. Метрики Docker и события

Быстрая диагностика:

```bash
# загрузка CPU/MEM/IO/NET
docker stats --no-stream

# события жизненного цикла контейнеров
docker events --filter 'container=api' --since 1h

# сетевые пространства, IP и т.п.
docker inspect api | jq '.[0].NetworkSettings'
```

Ресурсные лимиты (пример Compose):

```yaml
services:
  api:
    image: my/api
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1g
```

---

## 3. Метрики приложения → Prometheus

### Быстрый способ (Python `prometheus_client`)

Экспонируем `/metrics` и инкрементим счётчики:

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import random, time

REQS = Counter('app_requests_total', 'Total requests')
LAT = Histogram('app_request_seconds', 'Request latency')
STATE = Gauge('app_state', 'Arbitrary state')

if __name__ == '__main__':
    start_http_server(9100)  # /metrics на 9100
    while True:
        with LAT.time():
            time.sleep(random.random()/5)
        REQS.inc()
        STATE.set(random.randint(0, 5))
```

Запусти контейнер с пробросом порта `9100`, а Prometheus настроь на скрейп этого эндпойнта.

### Docker‑уровень метрик

- **node‑exporter** — метрики хоста; **cAdvisor** — метрики контейнеров.
- Их можно добавить отдельными сервисами в Compose и настроить Prometheus на их скрейп.

---

## 4. OpenTelemetry (OTel): трейсы/метрики/логи

Идея: приложение шлёт данные в **OTLP** (gRPC/HTTP) → **otel‑collector** → далее в хранилища (Prometheus для метрик, Jaeger/Tempo для трейсов, система логов).

### Мини‑инструментирование Python

```python
# базовая инициализация трассировки и метрик через OTLP
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

resource = Resource.create({"service.name": "api", "service.version": "1.0"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)))
trace.set_tracer_provider(provider)

tr = trace.get_tracer("api")
with tr.start_as_current_span("startup"):
    pass
```

### Фрагмент Compose с otel‑collector

```yaml
services:
  api:
    image: my/api
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
    networks: [obs]

  otel-collector:
    image: otel/opentelemetry-collector:latest
    command: ["--config=/etc/otel/config.yaml"]
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml:ro
    networks: [obs]
networks: { obs: {} }
```

Пример `otel-config.yaml` (минимум):

```yaml
receivers:
  otlp:
    protocols: { grpc: {}, http: {} }
exporters:
  logging: {}
service:
  pipelines:
    traces: { receivers: [otlp], exporters: [logging] }
```

(в реале вместо `logging` подключи Jaeger/Tempo/Prometheus/Логи.)

---

## 5. Профилирование Python в контейнерах

### Быстрый встроенный профилировщик (cProfile)

Запуск с профилированием участка кода:

```python
import cProfile, pstats, io
pr = cProfile.Profile(); pr.enable()
# ... ваш код ...
pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(20)
print(s.getvalue())
```

### Сэмплирующие профилировщики (низкий overhead)

- **py-spy** (внешний, без модификации кода): можно прикрепиться к PID процесса внутри контейнера.

Пример: профиль контейнера `api` 30 секунд с flamegraph `out.svg`:

```bash
PID=$(docker inspect -f '{{.State.Pid}}' api)
py-spy record -p $PID -o out.svg -d 30
```

Если запускаешь из контейнера‑утилиты: `--pid=container:api` и `--cap-add SYS_PTRACE`.

### Советы

- Профилируй **на репрезентативной нагрузке**.
- В проде используй сэмплирующие методы, чтобы не тормозить сервис.
