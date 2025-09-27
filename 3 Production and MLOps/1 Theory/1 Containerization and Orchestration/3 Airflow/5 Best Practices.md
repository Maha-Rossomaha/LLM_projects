# Best Practices

## 1. Организация репозитория DAG‑ов

### 1.1 Базовая структура

```
repo/
  dags/                    # только декларации DAG/Task (минимум логики)
    etl_*.py
    ml_*.py
  libs/                    # бизнес‑логика, чистые функции (не импортируют airflow)
    __init__.py
    etl/
      transform.py
      validate.py
    ml/
      features.py
      train.py
  plugins/                 # хуки/операторы/сенсоры
    __init__.py
    hooks/
    operators/
  configs/                 # yaml/json c описанием источников/таблиц
    datasets.yml
  tests/
    test_dag_imports.py
    test_libs_*.py
  docker/
    Dockerfile
  pyproject.toml / setup.cfg
  .pre-commit-config.yaml
```

**Идея:** DAG‑файлы тонкие. Вся тяжёлая логика — в `libs/` (тестируемая как обычный Python).

### 1.2 Импорты и зависимости

- В `libs/` никаких импортов `airflow`. Это упрощает юнит‑тесты и повторное использование.
- В DAG используем **тонкие адаптеры**: `PythonOperator`/TaskFlow только «оборачивают» функции из `libs/`.
- Зависимости фиксируем в образе/poetry, версии библиотек — pinned, чтобы пайплайны были воспроизводимы.

### 1.3 Конфигурации

- Переменные среды для окружений: dev/stage/prod.
- Airflow Variables/Connections — только для значений, которые меняются между окружениями.
- Бизнес‑конфиги (список таблиц, схемы) — в `configs/*.yml`, читаются синхронно при парсинге DAG.

---

## 2. Dynamic DAGs и Task Groups (вместо SubDAGs)

### 2.1 Почему не SubDAGs

SubDAGs исторически создавали отдельные DAG’и внутри DAG. Сейчас это анти‑паттерн: сложнее трекинг, проблемы с параллелизмом и планировщиком. Вместо них — **TaskGroup** и **Dynamic Task Mapping**.

### 2.2 TaskGroup: группировка этапов

```python
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task
from datetime import datetime

with DAG(
    dag_id="etl_with_groups",
    start_date=datetime(2025, 9, 1),
    schedule="@daily",
    catchup=False,
):
    @task
    def extract_table(tbl: str) -> str:
        # возвращаем путь сырых данных
        return f"/opt/airflow/data/raw/{tbl}_{{ ds_nodash }}.jsonl"

    @task
    def transform_table(path: str) -> str:
        # нормализация и путь на выходе
        return path.replace("/raw/", "/normalized/").replace(".jsonl", ".parquet")

    @task
    def load_table(path: str) -> int:
        # запись в БД/S3
        return 1

    tables = ["orders", "items", "users"]

    with TaskGroup(group_id="orders_flow") as orders_flow:
        load_table(transform_table(extract_table("orders")))

    with TaskGroup(group_id="items_flow") as items_flow:
        load_table(transform_table(extract_table("items")))

    with TaskGroup(group_id="users_flow") as users_flow:
        load_table(transform_table(extract_table("users")))

    orders_flow >> items_flow >> users_flow
```

Граф читаемый в UI, каждая группа — мини‑пайплайн.

### 2.3 Dynamic Task Mapping: fan‑out

```python
from airflow.decorators import task

@task
def list_sources() -> list[str]:
    return ["orders", "items", "users"]

@task
def extract(tbl: str) -> str:
    return f"/opt/airflow/data/raw/{tbl}_{{ ds_nodash }}.jsonl"

@task
def transform(path: str) -> str:
    return path.replace("/raw/", "/normalized/").replace(".jsonl", ".parquet")

@task
def load(path: str) -> int:
    return 1

# fan-out по списку таблиц и fan-in в load
paths = transform.expand(path=extract.expand(tbl=list_sources()))
_ = load.expand(path=paths)
```

Вместо ручного цикла создаются параллельные инстансы задач.

### 2.4 «Фабрика DAG’ов» по конфигам

```python
import yaml
from airflow import DAG
from datetime import datetime
from airflow.decorators import task

with open("/opt/airflow/configs/datasets.yml", "r") as f:
    cfg = yaml.safe_load(f)

for ds_cfg in cfg["datasets"]:
    dag_id = f"etl_{ds_cfg['name']}"
    with DAG(
        dag_id=dag_id,
        start_date=datetime(2025, 9, 1),
        schedule=ds_cfg.get("schedule", "@daily"),
        catchup=False,
        tags=["etl", ds_cfg["name"]],
    ) as dag:
        @task
        def extract() -> str:
            return f"/opt/airflow/data/raw/{ds_cfg['name']}_{{ ds_nodash }}.jsonl"
        @task
        def transform(path: str) -> str:
            return path.replace("/raw/", "/normalized/").replace(".jsonl", ".parquet")
        @task
        def load(path: str) -> int:
            return 1
        load(transform(extract()))
    globals()[dag_id] = dag
```

Один шаблон порождает N DAG’ов по конфигурации.

---

## 3. Версионирование пайплайнов и артефактов

### 3.1 Версии DAG

- **Не переименовывать** `dag_id` без нужды. Для крупных несовместимых изменений добавляй суффикс: `etl_orders_v2`.
- Теги `tags=["v2"]`, `params={"dag_version": 2}` — фиксируй версию в рантайме.
- В логи/метаданные прокидывай git sha: окружение `GIT_SHA` и `tags=[GIT_SHA[:7]]`.

### 3.2 Версии артефактов

- Храни с явной датой/версией: `s3://bucket/models/model_{{ ds_nodash }}.pkl`, алиас `production/model.pkl`.
- Для данных — партишены `date={{ ds_nodash }}`, `run={{ run_id }}`; атомарный `tmp → final`.

### 3.3 Схемы и миграции

- Изменения структуры БД — через миграции (Alembic/DB‑скрипты) отдельным DAG или шагом до `load`.
- Совместимость: временно писать в два столбца/таблицы, затем переключение чтения.
- Протоколировать версии схем в Variables или отдельной таблице `schema_versions`.

---

## 4. Типовые ошибки и как их предотвратить

### 4.1 Планирование и расписание

- Отсутствует `start_date` или неверная TZ → некорректные окна. Правильно: `start_date` в прошлом, TZ‑aware через `pendulum`.
- Забыт `catchup=False` для онлайн‑пайплайнов → взрыв исторических запусков.
- Неверный CRON. Проверь пресетами или сервисами проверки cron.

### 4.2 Граф и зависимости

- Скрытые зависимости через внешние файлы. Делай зависимости явными в графе, данные передавай через S3/БД.
- Слишком длинные цепочки. Делай группы и параллелизм, устанавливай лимиты (`max_active_tasks`, `pools`).

### 4.3 XCom и артефакты

- Передача больших объектов через XCom → раздутая мета‑БД. Передавай только **пути/ключи**.
- Секреты в XCom/логах. Храни их в Connections/Secret Backends.

### 4.4 Идемпотентность

- Отсутствие партишенов и UPSERT. Всегда пиши в `date={{ ds_nodash }}` и используй `MERGE/ON CONFLICT`.
- Неповторяемые ретраи. Сделай операции безопасными или защищайся уникальными ключами/замками.

### 4.5 Инфраструктура

- Весь прод на LocalExecutor. Для продакшна — Celery/Kubernetes.
- Один общий pool для всего. Делай отдельные пулы под внешние ресурсы (БД, API).
- Нет мониторинга и SLA. Включи метрики, алерты, `sla_miss_callback`.

### 4.6 Качество кода и тесты

- Логика внутри DAG‑файлов. Выноси в `libs/` и тестируй как обычный Python.
- Нет smoke‑теста импортов DAG. Добавь `DagBag`‑тест.
- Нет линтеров/форматтера. Подключи ruff/black, pre‑commit.

---

## 5. Когда лучше Prefect/Dagster вместо Airflow

**Prefect** подойдёт, если:

- нужен очень **pythonic** опыт разработки, минимум «DSL», локально как обычные функции;
- важна **динамика графа** и гибкие ретраи на уровне кода;
- хочется управлять оркестрацией через лёгкий агент и облачный UI;
- нужен простой локальный dev‑опыт без тяжёлой инсталляции.

**Dagster** подойдёт, если:

- ценишь концепцию **data assets** и декларативные зависимости между ними;
- нужны строгие **IO‑контракты** и типизация артефактов;
- важна единая **catalog/lineage** и интеграция с dbt как с asset‑графом;
- хочется first‑class **sensors/partitioning** на уровне сущностей данных.

**Airflow** остаётся лучшим выбором, когда:

- нужен зрелый, проверенный батч‑оркестратор с огромной экосистемой провайдеров;
- сильна опора на **операторы/хуки** к DWH/облакам/SaaS;
- основная нагрузка — **ETL/ML batch**, а не миллисекундные стримы;
- есть требования к строгому контролю раскатки/аудиту в enterprise.

**Не для Airflow**: чистый **event‑driven** стриминг с миллисекундами SLA, оркестрация внутри k8s только манифестами — логичнее Argo/Flink/Kafka Streams.
