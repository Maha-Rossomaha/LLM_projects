# ETL‑паттерн

**Цель:** собрать прикладной ETL‑пайплайн под данные: Extract → Transform → Load, обмен артефактами между задачами через XCom, и зафиксировать практические правила (идемпотентность, схемы, backfill).

---

## 0. Картина целиком

- **Extract:** забираем JSON из внешнего источника → сохраняем «как есть» в объектное хранилище или на диск (JSONL/Parquet) в слое *Raw*.
- **Transform:** чистим/нормализуем в *Normalized* (pandas для небольших объёмов; PySpark — для больших).
- **Load:** грузим в витрину/БД (Postgres), плюс складываем артефакты в S3/MinIO.
- **XCom:** передаём **пути/метаданные**, а не тяжёлые DataFrame’ы.

> Ключ: каждая стадия **идемпотентна** (повторный запуск даёт тот же результат для того же окна данных).

---

## 1. Контракты и разметка слоёв

**Слои хранения:**

- `raw/` — данные без изменений (сигнатура источника сохранена).
- `normalized/` — очищенные и унифицированные таблицы.
- `indexdoc/` — документы для поиска (если нужно).

**Схемы:** описываем структуры в Pydantic (валидация на Transform):

```python
from pydantic import BaseModel, Field
from typing import Optional

class ProductRaw(BaseModel):
    id: int
    name: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[int] = None

class ProductNorm(BaseModel):
    id: int
    name: str = Field(default="")
    brand: str = Field(default="")
    price: float = 0.0
```

---

## 2. Мини‑DAG (TaskFlow API): Extract → Transform → Load

Этот пример работает локально (без внешних провайдеров): сохранение на диск в подкаталог проекта. Для S3/DB см. следующие разделы.

```python
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
from airflow import DAG
from airflow.decorators import task

BASE = Path("/opt/airflow/data")  # смонтируй volume
RAW_DIR = BASE / "raw"
NORM_DIR = BASE / "normalized"

with DAG(
    dag_id="etl_local_taskflow",
    start_date=datetime(2025, 9, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["etl", "demo"],
) as dag:

    @task
    def extract(exec_date: str) -> str:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        # имитация внешнего API
        payload = [
            {"id": 1, "name": "Наушники", "brand": "X", "price": 12990},
            {"id": 2, "name": None, "brand": "Y", "price": None},
        ]
        out = RAW_DIR / f"items_{exec_date}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in payload:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(out)  # путь уходит в XCom

    @task
    def transform(raw_path: str) -> str:
        NORM_DIR.mkdir(parents=True, exist_ok=True)
        # читаем jsonl → pandas
        df = pd.read_json(raw_path, lines=True)
        # очистка/нормализация
        df["name"] = df["name"].fillna("").astype(str).str.strip()
        df["brand"] = df["brand"].fillna("").astype(str)
        df["price"] = df["price"].fillna(0).astype(float) / 100 if df["price"].dtype != float else df["price"]
        # сохраняем parquet
        norm_path = NORM_DIR / (Path(raw_path).stem.replace("items_", "items_norm_") + ".parquet")
        df.to_parquet(norm_path, index=False)
        return str(norm_path)

    @task
    def load(norm_path: str) -> None:
        # здесь могла бы быть загрузка в Postgres; для демо просто считаем строки
        df = pd.read_parquet(norm_path)
        print(f"Loaded {len(df)} rows from {norm_path}")

    # Пайплайн
    ds = "{{ ds_nodash }}"  # 20250926
    load(transform(extract(ds)))
```

> Здесь XCom автоматически передаёт **строки‑пути** между задачами. Это дёшево и безопасно.

---

## 3. Интеграция с БД (пример для Postgres)

### `PostgresOperator` (DDL/DML)

```python
from airflow.providers.postgres.operators.postgres import PostgresOperator

create_table = PostgresOperator(
    task_id="create_table",
    postgres_conn_id="pg_default",
    sql="""
    CREATE TABLE IF NOT EXISTS public.products (
        id BIGINT PRIMARY KEY,
        name TEXT,
        brand TEXT,
        price DOUBLE PRECISION,
        dt DATE DEFAULT CURRENT_DATE
    );
    """,
)

upsert = PostgresOperator(
    task_id="upsert_data",
    postgres_conn_id="pg_default",
    sql="""
    INSERT INTO public.products (id, name, brand, price)
    SELECT id, name, brand, price
    FROM staging_products
    ON CONFLICT (id) DO UPDATE SET
        name = EXCLUDED.name,
        brand = EXCLUDED.brand,
        price = EXCLUDED.price;
    """,
)
```

> Часто делают так: `transform` создаёт временный `staging_*` в БД → отдельной задачей выполняется `MERGE/UPSERT` в основную таблицу.

---

## 4. XCom: как обмениваться артефактами

**XCom (Cross-Communication)** — механизм обмена маленькими данными между задачами одного запуска DAG. По сути это **ключ–значение**, которое Airflow сохраняет в **metadata DB** для пары `(dag_id, run_id, task_id, key)`.

### Что реально происходит

- Каждая задача может **положить** (`push`) значение в XCom и **забрать** (`pull`) из XCom, зная `task_id` (и опционально `key`).
- В **TaskFlow API** возврат значения из функции `@task` **автоматически** кладётся в XCom под ключом `return_value`.
- Значения сериализуются (JSON/пикл в зависимости от backend) и хранятся в мета‑БД. Поэтому **нельзя** класть большие объекты.

### Когда XCom уместен

- Пути к файлам/объектам (`"/path/..."`, `"s3://bucket/key"`).
- Небольшие счетчики, статусы, агрегаты (например, `{"rows": 1234}`).
- Лёгкие конфиги (пара строк), флаги ветвления.

### Когда НЕ уместен

- Большие DataFrame/массивы/бинарники → складывай в S3/MinIO/БД и передавай **указатель** (путь/ключ).
- Секреты/пароли → используй **Connections / Variables / Secret backends** (XCom виден в UI!).

---

### Способы работы с XCom

#### 1. TaskFlow API (автоматически через `return`)

```python
from airflow.decorators import task

@task
def extract() -> str:
    path = "/opt/airflow/data/raw/items_{{ ds_nodash }}.jsonl"
    return path  # попадёт в XCom как key='return_value'

@task
def transform(raw_path: str) -> str:
    norm_path = raw_path.replace("/raw/", "/normalized/").replace(".jsonl", ".parquet")
    return norm_path

@task
def load(norm_path: str) -> int:
    # читаем parquet и пишем в БД
    return 1

load(transform(extract()))
```

> Здесь между задачами передаются **строки** (пути). За кулисами Airflow создаёт `XComArg`, который подтягивает нужное значение при выполнении.

#### 2. Классический способ: `xcom_push` / `xcom_pull`

```python
from airflow.operators.python import PythonOperator

# Пушим с произвольным ключом

def _extract(ti):  # ti = TaskInstance
    ti.xcom_push(key="raw_path", value="/opt/airflow/data/raw/items_{{ ds_nodash }}.jsonl")

extract = PythonOperator(task_id="extract", python_callable=_extract)

# Пуллим из другой задачи

def _transform(ti):
    raw_path = ti.xcom_pull(task_ids="extract", key="raw_path")
    norm_path = raw_path.replace("/raw/", "/normalized/").replace(".jsonl", ".parquet")
    ti.xcom_push(key="norm_path", value=norm_path)

transform = PythonOperator(task_id="transform", python_callable=_transform)
```

#### 3. Шаблоны Jinja в операторах

В templated‑полях можно брать XCom прямо из Jinja:

```python
from airflow.operators.bash import BashOperator

bash = BashOperator(
    task_id="echo_norm",
    bash_command="echo {{ ti.xcom_pull(task_ids='transform', key='norm_path') }}",
)
```

#### 4. Dynamic Task Mapping (fan‑out/fan‑in)

Списки в XCom позволяют «размножать» задачи по элементам:

```python
from airflow.decorators import task

@task
def list_files() -> list[str]:
    return ["a.jsonl", "b.jsonl", "c.jsonl"]  # список пойдёт в XCom

@task
def normalize_one(fname: str) -> str:
    return fname.replace(".jsonl", ".parquet")

# expand создаст по задаче на каждый элемент списка
normalize_one.expand(fname=list_files())
```

---

### Область видимости и жизненный цикл

- XCom привязан к **конкретному запуску** DAG (`run_id`). Другой запуск не увидит эти значения.
- Ключ по умолчанию — `return_value`; свои ключи называй явно (`"raw_path"`, `"norm_path"`, `"rows"`).
- Ретеншн: XCom‑записи копятся в БД. Чисти их через `airflow db cleanup` (или соответствующие утилиты/параметры).

### Сериализация и лимиты

- По умолчанию backend сериализует в JSON (с Airflow‑кодером) и/или хранит бинарь как base64‑пикл.
- Ограничения БД и накладные расходы → **держи XCom маленьким** (десятки–сотни байт/кило‑байт, но не мегабайты).
- Для сложных кейсов можно подключить **кастомный XCom backend** (например, сохранять большие объекты в S3, а в БД класть только ссылку).

---

### Паттерны и анти‑паттерны

**Паттерны**

- Передавай **указатели** (пути/ключи), а не сами данные.
- Используй TaskFlow `return`/аргументы — чище и короче, чем ручные `xcom_push/pull`.
- Явно называй ключи, если пушишь несколько значений.

**Анти‑паттерны**

- Пихать DataFrame/большие JSON прямо в XCom.
- Хранить токены/пароли → XCom виден в UI.
- Полагаться на XCom как на долговременное хранилище.

---

## 5. Идемпотентность, backfill, окна данных

- **Идемпотентность:**

  - Пиши в партишены по дате: `.../date={{ ds_nodash }}/` (или столбец `dt` в БД).
  - Для БД используй `UPSERT`/`MERGE` по бизнес‑ключу (`PRIMARY KEY`/`UNIQUE`) + `ON CONFLICT DO UPDATE`.
  - Вариант с чисткой партиции: `DELETE FROM tbl WHERE dt = {{ ds }}` → затем вставка из staging.
  - Записывай во временную директорию и делай атомарное переименование: `.../tmp/run={{ run_id }}` → `.../date={{ ds_nodash }}` (защита от «полупрожаренных» партиций).
  - Добавь технические поля: `ingestion_ts`, `run_id`, `source_hash` (для аудита и дедупликации).

- **Backfill (догон):**

  - При `catchup=True` планировщик создаёт запуски для всех прошедших окон. Код должен уметь **повторно** обработать то же окно без дублей и побочных эффектов.
  - Ограничь конкуренцию по окнам: `max_active_runs=1` (или защита на уровне БД — `PRIMARY KEY`/`UNIQUE`).
  - Политика опоздавших событий (late data): либо «не принимаем» (только текущее окно), либо «скользящее переобновление» N прошлых окон (параметр `lookback_days`).
  - Для backfill’а тяжёлых DAG включай лимиты параллелизма и ретраи, либо запускай окна батчами.

- **Выборка по окну данных (data interval):**

  - В Airflow 2.4+ ориентируйся на макросы `{{ data_interval_start }}` и `{{ data_interval_end }}` (а не на legacy `execution_date`).
  - Окно интерпретируется как $[data\_interval\_start, data\_interval\_end)\$ — включая левую границу и исключая правую.
  - Пример SQL:
    ```sql
    SELECT *
    FROM events
    WHERE created_at >= '{{ data_interval_start }}'
      AND created_at <  '{{ data_interval_end }}';
    ```
  - Пример путей/партишенов:
    ```
    s3://bucket/dataset=date={{ ds_nodash }}/run={{ run_id }}/part-*.parquet
    ```
  - Учитывай таймзону: задавай `start_date` с `pendulum` и одной TZ (например, `Europe/Moscow`), иначе окна считаются в UTC.
