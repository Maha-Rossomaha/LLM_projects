# Scaling and production

## 1. Executors: Local, Celery, Kubernetes

### 1.1 LocalExecutor

**Где применять:** одиночная машина/VM, dev/stage, небольшой прод.

**Идея:** Scheduler кладёт задачи в локальную очередь, воркеры — это процессы на той же машине.

**Плюсы:** минимум инфраструктуры; просто отлаживать.

**Минусы:** ограничен ресурсами одной ноды; нет устойчивости к падению хоста.

**Мини‑настройка (docker‑compose):**

```yaml
services:
  postgres:
    image: postgres:15
  airflow-webserver:
    image: apache/airflow:2
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
    volumes:
      - ./dags:/opt/airflow/dags
```

---

### 1.2 CeleryExecutor

**Где применять:** горизонтально масштабируемый прод на VMs/контейнерах.

**Идея:** Scheduler ставит задачи в брокер (Redis/RabbitMQ); пул воркеров Celery забирает и исполняет.

**Плюсы:** гибко масштабируется числом воркеров; изоляция задач по очередям.

**Минусы:** нужен брокер; операционные издержки (мониторинг/тюнинг Celery).

**Мини‑стек (compose, фрагмент):**

```yaml
services:
  redis:
    image: redis:7
  airflow-scheduler:
    image: apache/airflow:2
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
  airflow-worker:
    image: apache/airflow:2
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
```

**Очереди и приоритеты:** указывай `queue="high_io"` в операторах и заводи воркеров с `--queues high_io,default`.

---

### 1.3 KubernetesExecutor

**Где применять:** облака и крупный прод, когда нужно изолировать каждую задачу.

**Идея:** каждая задача — отдельный Pod; ресурсы и окружение задаются на уровне PodSpec.

**Плюсы:** эластичность; точные ресурсы (CPU/RAM/GPU) на таск; изоляция зависимостей.

**Минусы:** нужен кластер k8s; настройка ролей/подключений/сетей.

**Конфигурация (в общих чертах):**

```env
AIRFLOW__CORE__EXECUTOR=KubernetesExecutor
AIRFLOW__KUBERNETES__NAMESPACE=airflow
AIRFLOW__KUBERNETES__IN_CLUSTER=True
```

**Задачи в k8s (пример):**

```python
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

kpo = KubernetesPodOperator(
    task_id="heavy_train",
    name="heavy-train",
    image="python:3.11-slim",
    cmds=["python", "-c"],
    arguments=["print('train...')"],
    resources={"request_memory": "2Gi", "request_cpu": "1", "limit_memory": "4Gi", "limit_cpu": "2"},
)
```

> Для Celery/Kubernetes имей **shared storage** для логов/артефактов (S3/NFS/GCS) и понятную стратегию доставки DAG’ов (git‑sync, образ с DAG внутри, или бакет).

---

## 2. Параллелизм, retrieс, SLA

### 2.1 Крутилки параллелизма (QoS)

**Глобально (airflow\.cfg):**

- `parallelism` — максимум одновременно исполняемых task instances во всём инстансе.

**На уровне DAG:**

```python
with DAG(
    dag_id="example_qos",
    start_date=datetime(2025, 9, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,      # одновременно активных запусков DAG
    max_active_tasks=8,     # лимит тасков от этого DAG
    default_args={"retries": 2},
):
    ...
```

- `max_active_runs` — одновременно активных запусков DAG.
- `max_active_tasks` — одновременно исполняемых задач данного DAG.

**На уровне задачи:**

- `task_concurrency` — сколько инстансов этой **конкретной задачи** можно одновременно (полезно для idempotent‑тасков).
- `pool` — общий лимитер ресурса (см. ниже).
- `queue` — в Celery: на какую очередь отправить.

**Pools (пулы):** ограничивают число одновременно исполняемых задач, использующих общий ресурс (например, внешний API или БД).

```python
@task(pool="pg_io", priority_weight=5)
def upsert_batch(...):
    ...
```

В UI создай pool `pg_io` со слотов, скажем, 5.

---

### 2.2 Retries и backoff

```python
from datetime import timedelta

default_args = {
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=60),
    "retry_jitter": True,  # сгладить «стампид»
}
```

Рекомендация: ретраи должны быть **идемпотентны**; если не так — добавляй защиту (уникальные ключи/замки/версионирование).

---

### 2.3 SLA (срок выполнения)

SLA — «обещание» уложиться в время выполнения задачи **после** начала интервала. Если просрочено, создаётся событие «SLA miss» и вызывается коллбек/алерт.

**На уровне задачи:**

```python
from datetime import timedelta
from airflow.operators.empty import EmptyOperator

task = EmptyOperator(
    task_id="heavy_step",
    sla=timedelta(minutes=30),  # если не уложился — SLA miss
)
```

**Глобальный коллбек на DAG:**

```python
from airflow.utils.email import send_email

def on_sla_miss(dag, task_list, blocking_task_list, slas, blocking_tis):
    send_email(to=["ops@example.org"], subject="SLA missed", html_content=str(slas))

with DAG(
    dag_id="with_sla",
    start_date=datetime(2025, 9, 1),
    schedule="@hourly",
    catchup=False,
    sla_miss_callback=on_sla_miss,
):
    ...
```

> SLA — это не «дедлайн старта», а «время до завершения». Для «дедлайна старта» используй сенсоры/валидаторы входных данных и `max_active_runs=1`.

---

## 3. Секреты и конфигурации: Connections, Variables, Vault

### 3.1 Connections (подключения)

Используются hooks/operators через `conn_id`.

**Заведение через переменные окружения (удобно в CI):**

```bash
export AIRFLOW_CONN_PG_DEFAULT='postgresql+psycopg2://user:pass@host:5432/db'
```

**Использование в коде:**

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook

hook = PostgresHook(postgres_conn_id="pg_default")
with hook.get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
```

**Jinja‑доступ:** `{{ conn.pg_default.login }}`, `{{ conn.pg_default.password }}` (для templated‑полей).

---

### 3.2 Variables (переменные)

Храни «нетайные» конфиги (фич‑флаги, пути, лимиты).

**Окружение:**

```bash
export AIRFLOW_VAR_WB_DATASET_BUCKET='my-bucket'
```

**Python/Jinja:**

```python
from airflow.models import Variable
bucket = Variable.get("WB_DATASET_BUCKET", default_var="dev-bucket")
# Jinja: {{ var.value.WB_DATASET_BUCKET }}
```

---

### 3.3 Secret Backends (Vault/SM/KMS)

Для секрета в продакшне подключай бэкенд, чтобы не хранить их в БД Airflow.

**Пример (HashiCorp Vault):**

```env
AIRFLOW__SECRETS__BACKENDS=airflow.providers.hashicorp.secrets.vault.VaultSecrets
AIRFLOW__SECRETS__BACKEND_KWARGS={"connections_path": "airflow/connections", "variables_path": "airflow/variables", "mount_point": "kv"}
```

Теперь `conn_id`/`Variable` будут искаться в Vault. Аналоги есть для AWS Secrets Manager и GCP Secret Manager.

> Секреты **не клади** в XCom/логи/репозиторий. Для временных токенов используй TTL в самом бэкенде.

---

## 4. CI/CD: lint, unit tests, деплой DAG’ов

### 4.1 Структура репо (пример)

```
repo/
  dags/
    etl_*.py
    ml_*.py
  plugins/
  tests/
    test_dag_imports.py
    test_ml_pipeline.py
  docker/
    Dockerfile
  .pre-commit-config.yaml
```

### 4.2 Линтеры и статанализ

- **ruff/flake8** — стиль/ошибки Python.
- **black** — авто‑форматирование.
- **mypy** — типы (особенно для TaskFlow‑функций).
- **pre-commit** — запуск локально и в CI.

`.pre-commit-config.yaml` (фрагмент):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
```

### 4.3 Unit/интеграционные тесты

**Быстрый импорт‑тест DAG’ов:**

```python
from airflow.models import DagBag

def test_no_import_errors():
    dag_bag = DagBag(dag_folder="dags", include_examples=False)
    assert len(dag_bag.import_errors) == 0, dag_bag.import_errors
```

**Тест одной задачи (tasks test):**

```bash
airflow tasks test etl_local_taskflow extract 2025-09-26
```

**pytest‑airflow (по желанию):** даёт фикстуры и helpers для контекста Airflow.

### 4.4 Проверки в CI (GitHub Actions, фрагмент)

```yaml
name: ci
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install apache-airflow pytest ruff black
      - run: ruff dags plugins
      - run: black --check dags plugins
      - run: pytest -q
```

### 4.5 Стратегии деплоя DAG’ов

1. **git‑sync**: webserver/scheduler подхватывают DAG’и из git‑репо (sidecar). → Просто пуш в main.
2. **Образ с DAG’ами**: собираем Docker‑образ с кодом и выкатываем (Helm/compose).
3. **Бакет**: складываем DAG’и в S3/GCS, монтируем в Airflow как read‑only.

**Dockerfile (фрагмент, вариант «образ с DAG’ами»):**

```dockerfile
FROM apache/airflow:2
COPY dags/ /opt/airflow/dags/
COPY plugins/ /opt/airflow/plugins/
```

**Helm (идея):** включить gitSync или указать `dags.persistence`/`dags.gitSync` (в зависимости от чарта).

> В проде обязательно: **read‑only DAGs**, понятный roll‑out (blue/green или canary), и откат на предыдущий commit/образ.
