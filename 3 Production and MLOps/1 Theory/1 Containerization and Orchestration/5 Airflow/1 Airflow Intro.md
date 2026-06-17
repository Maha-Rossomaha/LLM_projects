# Airflow Intro

Airflow — это оркестратор: он планирует и запускает задачи по зависимостям в виде DAG (ориентированный ациклический граф), логирует результаты, ретраит упавшие шаги и даёт UI для наблюдения.&#x20;

---

## 1. Что такое оркестратор и зачем он нужен в MLOps

**Оркестратор** — системный «дирижёр» батч‑процессов: ETL, обучение моделей, batch‑inference, отчёты, регламенты. Он **не исполняет** бизнес‑логику сам, а **управляет** её запуском и очередностью.

### Задачи в MLOps

- **ETL/ELT:** регулярная загрузка сырых данных → чистка → сохранение витрин.
- **Обучение модели по расписанию:** детект новых данных → препроцессинг → train → validate → регистрировать артефакты.
- **Batch inference:** раз в N часов предсказывать для накопившегося пула объектов.
- **Переиндексация/пересборка индексов:** для поиска/RAG.
- **Регламентные процедуры:** архивы, проверки качества, прогрев кэшей.

### Что оркестратор НЕ делает

- Не заменяет **стриминговые фреймворки** (Kafka/Flink/Spark Streaming) — но может триггерить их джобы.
- Не является **очередью сообщений** (RabbitMQ/Redis Streams) — он планирует и координирует.
- Не хранит ваши данные как DWH.

---

## 2. Архитектура Airflow (высокоуровнево)

- **DAG‑файлы** (Python) — описание графов задач. Хранятся на общей FS/образе.
- **Webserver** — UI: просмотр DAG’ов, графов, логов, запуск руками.
- **Scheduler** — читает DAG‑файлы, рассчитывает, какие таски пора ставить в очередь.
- **Executor** — стратегия исполнения: Local/Celery/Kubernetes (см. ниже).
- **Workers** — исполняют задачи (процессы/под‑контейнеры).
- **Metadata DB** — БД состояния (PostgreSQL/MySQL): статусы DAG/Task, расписания, XCom‑метаданные.
- **Triggerer** — фоновые, асинхронные триггеры (например, для сенсоров).

> Инвариант: **DAG’и — код на Python**; **состояние — в БД**; **исполнение — через Executor/Workers**.

---

## 3. Базовые сущности Airflow

- **DAG** — граф из задач + расписание (schedule), без циклов.
- **Task** — узел графа; создаётся с помощью **операторов**.
- **Operator** — «адаптер» способа исполнения (Python, Bash, SQL, Docker и т.д.).
- **Sensor** — специальный оператор ожидания события (файл появился, таблица готова и т.п.).
- **Hook / Connection** — унифицированные подключения к внешним системам (S3, Postgres, GCS…).
- **XCom** — обмен маленькими артефактами данных между задачами (метаданные, пути файлов).
- **TaskGroup** — логическая группировка задач в графе (удобнее SubDAG’ов).

---

## 4. Операторы: что это и какие бывают

Оператор определяет **как** исполняется задача.

### Часто используемые

- **PythonOperator** — запуск произвольной Python‑функции.
- **BashOperator** — команда shell (скрипт).
- **BranchPythonOperator** — ветвление по условию (выбор следующей ветки).
- **ShortCircuitOperator** — остановка ветки, если условие не выполнено.
- **Dummy/EmptyOperator** — «пустая» заглушка (маркер, разделитель).
- **TriggerDagRunOperator** — запуск другого DAG.

### Сенсоры (Sensors)

- **FileSensor / S3KeySensor / GCSObjectExistenceSensor** — ждать появления файла/объекта.
- **ExternalTaskSensor** — ждать завершения конкретной задачи в другом DAG.

### Другие популярные

- **DockerOperator** — запуск контейнера (нужен провайдер Docker).
- **KubernetesPodOperator** — запуск пода в k8s.
- **SQL/Transfer операторы** — выполнение SQL, перенос данных между хранилищами.

> На практике в ML/ETL достаточно сочетания: **PythonOperator + сенсоры + Branch/Empty**; для инфраструктурных задач — Docker/Kubernetes операторы.

---

## 5. Установка через Docker (локальная среда)

Ниже — минимальные шаги. 

1. Установить Docker Desktop/Engine и Docker Compose.
2. Подготовить каталог проекта: *./airflow/*, а внутри подпапки: *dags/*, *logs/*, *plugins/*.
3. Создать docker‑compose с сервисами: webserver, scheduler, triggerer, postgres, (по желанию) flower/redis для CeleryExecutor.
4. Инициализировать мета‑БД (командой инициализации) и поднять стек.
5. Открыть UI в браузере и авторизоваться.

> Примечания

- Для локалки достаточно **LocalExecutor** (или SequentialExecutor для самого простого стенда). Для параллельной работы и продакшна — **CeleryExecutor** или **KubernetesExecutor**.
- Хранить DAG’и в томе/volume, чтобы правки в файлах сразу подхватывались UI и Scheduler.

*(Если хочешь автоматизировать подготовку окружения, можно написать маленький Python‑скрипт, который создаёт структуру каталогов и базовый DAG — пример ниже.)*

---

## 6. Пример

Ниже — минимальный пример DAG, который:

1. печатает «hello» в Python;
2. выводит дату через bash;
3. связывает задачи зависимостью `>>`.

```python
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def say_hello() -> None:
    print("Hello, Airflow!")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="example_hello_dag",
    description="Базовый DAG: Python + Bash",
    default_args=default_args,
    start_date=datetime(2025, 9, 1),  # фиксируй start_date в прошлом
    schedule_interval="@daily",      # предустановленный пресет (@hourly, @daily, @weekly, @monthly)
    catchup=False,                    # не догонять «историю» при первом старте
    tags=["demo", "basics"],
) as dag:

    t1 = PythonOperator(
        task_id="say_hello",
        python_callable=say_hello,
    )

    t2 = BashOperator(
        task_id="print_date",
        bash_command="date",
    )

    # Зависимость: t1 → t2
    t1 >> t2
```

### Вариант с TaskFlow API

TaskFlow — это «сахар» над PythonOperator, позволяющий писать задачи как функции с декоратором.

```python
from __future__ import annotations

from datetime import datetime
from airflow import DAG
from airflow.decorators import task

with DAG(
    dag_id="example_taskflow_dag",
    start_date=datetime(2025, 9, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    @task
    def extract() -> list[int]:
        return [1, 2, 3]

    @task
    def transform(xs: list[int]) -> list[int]:
        return [x * 10 for x in xs]

    @task
    def load(xs: list[int]) -> None:
        print("Loaded:", xs)

    load(transform(extract()))
```

---

## 7. Расписания: cron, timedelta, пресеты, catchup

В Airflow расписание можно задать тремя основными способами: **пресетом** (`"@daily"`, `"@hourly"` и т.д.), **CRON‑выражением** (`"30 3 * * *"`) или \*\*объектом \*\*`` (интервал). В версиях Airflow ≥ 2.4 дополнительно есть **Timetables** (настраиваемые календарные расписания и датасеты), но для базовой практики достаточно трёх способов ниже.

### Пресеты

- `@once` — один запуск.
- `@hourly`, `@daily`, `@weekly`, `@monthly`, `@yearly` (`@annually`) — стандартные периоды.
- `@daily` эквивалентен CRON `0 0 * * *` (запуск в конце дневного интервала).

### CRON

Формат: `минуты часы день_месяца месяц день_недели`.

- Ежедневно в 03:30 — `"30 3 * * *"`.
- По будням в 20:05 — `"5 20 * * 1-5"`.
- Раз в 2 часа — `"0 */2 * * *"`.

### `timedelta`

Интервал‑основанное расписание (просто периодичность без CRON‑календаря):

```python
from datetime import timedelta
# эквивалент: запуск каждые 15 минут
schedule_interval = timedelta(minutes=15)
```

### Важные флаги и понятия

- `start_date` — момент, с которого DAG становится валидным для планировщика. Рекомендуется ставить в прошлом (и TZ‑aware), иначе первый запуск может не создаться.
- `end_date` — опциональная «линия завершения» расписания.
- `catchup` — по умолчанию `True`. Если включено, Airflow «догонит» все пропущенные интервалы от `start_date` до текущего времени. Для большинства прикладных DAG лучше ставить `catchup=False`.
- `schedule` / `schedule_interval` — в новых версиях предпочтительно `schedule`, но `schedule_interval` всё ещё поддерживается. `schedule=None` → без расписания (ручные/внешние триггеры).
- **Data Interval** — запуск планируется в **конце** окна данных. Пример: при `@daily` запуск за интервал `2025‑09‑01` произойдёт около `2025‑09‑02 00:00`.
- **Часовой пояс** — по умолчанию UTC. Задавай TZ глобально (airflow\.cfg) или делай `start_date` TZ‑aware (через `pendulum`).

### Примеры

**1. Пресет **`@daily`** + локальная таймзона:**

```python
import pendulum
with DAG(
    dag_id="example_daily",
    start_date=pendulum.datetime(2025, 9, 1, tz="Europe/Moscow"),
    schedule="@daily",          # или schedule_interval="@daily"
    catchup=False,
):
    ...
```

**2. CRON‑расписание:**

```python
with DAG(
    dag_id="example_cron",
    start_date=pendulum.datetime(2025, 9, 1, tz="Europe/Moscow"),
    schedule="30 3 * * *",      # ежедневно в 03:30
    catchup=True,                # при необходимости догнать прошлые окна
):
    ...
```

**3 Интервал через `timedelta`:**

```python
from datetime import timedelta
with DAG(
    dag_id="example_every_15min",
    start_date=pendulum.datetime(2025, 9, 1, tz="Europe/Moscow"),
    schedule_interval=timedelta(minutes=15),
    catchup=False,
):
    ...
```

> Примечание про Timetables: для сложных сценариев (например, запуска по готовности **Dataset** или «последняя пятница месяца») используются кастомные Timetable‑классы. Это расширение удобно подключать позже, когда базовые типы расписаний освоены.

---

## 8. Зависимости задач: операторы `>>` и `<<`

Есть три способа задать зависимости:

1. **Операторы сдвига:** `a >> b` (a перед b) и `a << b` (a после b).
2. **Методы:** `a.set_downstream(b)`, `b.set_upstream(a)`.
3. **Компоновка через выражения:** `(a1, a2) >> b >> (c1, c2)`.

### Пример

```python
from airflow.operators.empty import EmptyOperator

start = EmptyOperator(task_id="start")
A = EmptyOperator(task_id="A")
B = EmptyOperator(task_id="B")
C = EmptyOperator(task_id="C")
end = EmptyOperator(task_id="end")

start >> (A, B)   # параллельно
(A, B) >> C
C >> end
```

> Рекомендации: избегай «скрытых» зависимостей внутри операторов (например, через глобальные файлы), старайся, чтобы граф был самодокументируем.

---

## 9. Мини‑практика: создать проект и первый DAG (скрипт‑помощник)

Ниже — Python‑скрипт, который создаёт структуру каталогов Airflow‑проекта и кладёт базовый DAG в *dags/*.

```python
import os
from pathlib import Path

tree = {
    "airflow": {
        "dags": {},
        "logs": {},
        "plugins": {},
    }
}

BASE = Path.cwd()

def ensure_tree(base: Path, spec: dict) -> None:
    for name, children in spec.items():
        p = base / name
        p.mkdir(parents=True, exist_ok=True)
        if isinstance(children, dict):
            ensure_tree(p, children)

ensure_tree(BASE, tree)

DAG_CODE = """from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def hello():
    print("Hello from generated DAG!")

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}
with DAG(
    dag_id="generated_hello_dag",
    start_date=datetime(2025, 9, 1),
    schedule_interval="@daily",
    catchup=False,
):
    PythonOperator(task_id="hello", python_callable=hello)
"""

(dag_path := BASE / "airflow" / "dags" / "generated_hello_dag.py").write_text(DAG_CODE, encoding="utf-8")
print(f"Created {dag_path}")
```

> Дальше подключи каталог *./airflow* к контейнеру webserver/scheduler как volume — и DAG появится в UI.
