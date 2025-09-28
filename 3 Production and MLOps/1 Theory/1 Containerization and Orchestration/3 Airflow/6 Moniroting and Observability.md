# Moniroting and Observability

## 1. Что такое Observability в контексте Airflow

- **Логи** — подробное выполнение каждой задачи (stdout/stderr) локально или в удалённом хранилище.
- **Метрики** — агрегированные числа во времени: длительность, количество успешных/проваленных задач, очередь пулов, heartbeat шедулера.
- **События и алерты** — письма/уведомления при падениях, ретраях, пропуске SLA.
- **Трассировка** — обычно не требуется; для тяжёлых задач можно логировать шаги внутри PythonOperator.

Ключевая идея: **метрики — для трендов и алертов, логи — для детального разбора**.

---

## 2. Метрики: что смотреть в первую очередь

### 2.1 Уровень DAG

- **DAG runtime** — длительность запуска DAG (от начала окна до завершения всех задач).
- **Success/fail/queued count** — количество успешных/проваленных/ожидающих задач на запуск.
- **Backfill debt** — сколько «долга» по запускам накопилось.

### 2.2 Уровень задачи (Task)

- **Task duration** — длительность выполнения.
- **Retries** — число повторов; всплески = проблемы со стабильностью.
- **Queue wait** — время ожидания слота (пулы/очереди/ресурсы).

### 2.3 Откуда брать метрики

- **Web UI** (Grid/Gantt/Duration) — быстрый взгляд.
- **Metadata DB** — можно считать агрегаты SQL‑запросами (см. ниже).
- **StatsD/Prometheus** — системные метрики планировщика и задач.

#### Пример: агрегаты из metadata DB (Postgres) — PythonOperator

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task

@task
def dag_runtime_avg(dag_id: str) -> float:
    sql = """
    SELECT AVG(EXTRACT(EPOCH FROM (end_date - start_date))) AS avg_sec
    FROM dag_run
    WHERE dag_id = %s AND state = 'success' AND end_date IS NOT NULL
      AND start_date > NOW() - INTERVAL '7 days'
    """
    hook = PostgresHook(postgres_conn_id="airflow_db")  # подключение к мета‑БД
    with hook.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (dag_id,))
        row = cur.fetchone()
        return float(row[0] or 0.0)
```

> Такой таск можно запускать раз в день и писать результат в отдельную таблицу/лог.

---

## 3. Web UI: где смотреть статус и логи

- **DAGs list** — общее состояние, последние запуски, быстрые действия (play/pause, trigger).
- **Grid view** — клетчатая доска по запускам: клик по клетке → контекст задач, **лог** конкретного инстанса.
- **Graph view** — граф зависимостей; полезно видеть «узкие места».
- **Gantt view** — перекрытия задач во времени (удобно искать блокировки пулов/очередей).
- **Task duration** — распределение длительности по инстансам; хорошо ловит регрессии.
- **Code** — версия DAG‑файла, отрендеренный шаблон.

### Удалённые логи (S3/MinIO/GCS)

В проде включай remote logging, чтобы логи были доступны независимо от контейнеров/нод:

```env
AIRFLOW__LOGGING__REMOTE_LOGGING=True
AIRFLOW__LOGGING__REMOTE_LOG_CONN_ID=aws_default   # или gcs_default
AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER=s3://my-bucket/airflow-logs
```

> Дай воркерам доступ по Connection (ключ/секрет), не храни креды в коде.

---

## 4. Prometheus + Grafana: системные метрики

### 4.1 Вариант без плагинов: StatsD → statsd\_exporter → Prometheus

1. Поднимаем **statsd\_exporter** (конвертит UDP‑метрики StatsD в `/metrics`).
2. Включаем в Airflow **StatsD‑метрики**.
3. Prometheus скрейпит statsd\_exporter, Grafana строит дашборды.

**docker‑compose (фрагмент):**

```yaml
services:
  statsd-exporter:
    image: prom/statsd-exporter:latest
    command: ["--statsd.listen-udp=:9125", "--web.listen-address=:9102"]
    ports: ["9102:9102", "9125:9125/udp"]

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
```

**Env Airflow (в webserver/scheduler/worker):**

```env
AIRFLOW__METRICS__STATSD_ON=True
AIRFLOW__METRICS__STATSD_HOST=statsd-exporter
AIRFLOW__METRICS__STATSD_PORT=9125
AIRFLOW__METRICS__STATSD_PREFIX=airflow
```

**prometheus.yml (фрагмент):**

```yaml
scrape_configs:
  - job_name: 'airflow-statsd'
    static_configs:
      - targets: ['statsd-exporter:9102']
```

### 4.2 Что визуализировать в Grafana

- **Успехи/падения задач**: success/fail по dag\_id/task\_id.
- **Длительность задач**: среднее/перцентили за 1д/7д; топ‑N долгих задач.
- **Очереди/пулы**: занятость пулов, время ожидания слота.
- **Scheduler health**: очереди к запуску, heartbeat, длина backlog.

> Имена метрик зависят от экспорта, но базовая идея — видеть тренды и пороги. Если нет Prometheus, считай метрики из мета‑БД и шли в любую TSDB.

---

## 5. Алерты: email, Slack, Telegram

### 5.1 Email

Включи SMTP и настрой уведомления по умолчанию:

```env
AIRFLOW__SMTP__SMTP_HOST=smtp.example.org
AIRFLOW__SMTP__SMTP_USER=ops
AIRFLOW__SMTP__SMTP_PASSWORD=secret
AIRFLOW__SMTP__SMTP_MAIL_FROM=airflow@example.org
```

В DAG:

```python
default_args = {
    "email": ["ops@example.org"],
    "email_on_failure": True,
    "email_on_retry": False,
}
```

### 5.2 Slack (через Webhook)

Удобно завести **on\_failure\_callback**:

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

def notify_slack_failure(context):
    msg = (
        f"DAG: {context['dag'].dag_id}\n"
        f"Task: {context['task_instance'].task_id}\n"
        f"Run: {context['run_id']}\n"
        f"Log: {context['task_instance'].log_url}"
    )
    return SlackWebhookOperator(
        task_id="slack_alert",
        http_conn_id="slack_webhook",  # Connection со Slack webhook URL
        message=msg,
        username="airflow",
    ).execute(context=context)

with DAG(..., default_args={"on_failure_callback": notify_slack_failure}):
    ...
```

### 5.3 Telegram

Через провайдер `telegram` (бот‑токен/чат в Connections/Variables):

```python
from airflow.providers.telegram.operators.telegram import TelegramOperator

def notify_telegram_failure(context):
    text = (
        f"❌ {context['dag'].dag_id} / {context['task_instance'].task_id}\n"
        f"Run: {context['run_id']}\nLog: {context['task_instance'].log_url}"
    )
    TelegramOperator(
        task_id="tg_alert",
        telegram_conn_id="telegram_default",  # хранит token/chat_id
        text=text,
    ).execute(context=context)
```

> Все секреты и webhook URL держи в Connections/Variables/Secret backend, не в коде.

---

## 6. SLA miss callback (автоматические оповещения)

SLA — дедлайн на **время выполнения** задачи. При нарушении Airflow вызывает коллбек и пишет событие SLA miss.

```python
from airflow.utils.email import send_email

def on_sla_miss(dag, task_list, blocking_task_list, slas, blocking_tis):
    body = f"SLA missed in {dag.dag_id}: tasks={task_list}"
    send_email(to=["ops@example.org"], subject="[Airflow] SLA missed", html_content=body)

with DAG(
    dag_id="with_sla_alerts",
    start_date=datetime(2025, 9, 1),
    schedule="@hourly",
    catchup=False,
    sla_miss_callback=on_sla_miss,
):
    ...
```

На уровне задач:

```python
from datetime import timedelta
from airflow.operators.empty import EmptyOperator

EmptyOperator(task_id="heavy", sla=timedelta(minutes=30))
```

> Не путай SLA с дедлайном старта. Для дедлайна старта используй сенсоры подготовки входных данных + `max_active_runs=1`.
