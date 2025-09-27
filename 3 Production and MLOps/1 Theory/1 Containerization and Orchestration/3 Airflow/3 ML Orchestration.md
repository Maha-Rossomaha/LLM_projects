# Оркестрация ML

## 0. Подготовка окружения и договорённости

Каталоги (монтируем в контейнеры как volume):

```
/opt/airflow/data/raw/
/opt/airflow/data/normalized/
/opt/airflow/data/predict/{in,out}/
/opt/airflow/models/{staging,production}/
```

Файлы реестра/метрик:

```
/opt/airflow/models/registry.json   # {"best_score": 0.842, "model_path": "/opt/.../production/model.pkl"}
```

Соединения (Connections) — по мере необходимости: `pg_default`, `aws_default`.

---

## 1. DAG обучения: prep → train → validate → save (с ветвлением)

### Идея

- **prep**: собрать обучающий датасет за окно данных, вернуть путь.
- **train**: обучить модель, сохранить в staging, вернуть путь модели.
- **validate**: посчитать метрику на hold‑out, вернуть `score`.
- **branch**: если `score` лучше текущего лучшего из `registry.json` → `save_model`, иначе `skip_save`.
- **save\_model**: атомарно обновить *production*‑модель.

### Полный пример (TaskFlow + BranchPythonOperator)

```python
from datetime import datetime, timedelta
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.operators.branch import BranchPythonOperator

BASE = Path("/opt/airflow")
DATA = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(parents=True, exist_ok=True)
(MODELS / "staging").mkdir(exist_ok=True)
(MODELS / "production").mkdir(exist_ok=True)
REGISTRY = MODELS / "registry.json"

DEFAULT_ARGS = {"retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="ml_train_branching",
    start_date=datetime(2025, 9, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ml", "train"],
) as dag:

    @task
    def prep(exec_date: str) -> str:
        """Готовим датасет за окно (демо: синтетика или чтение parquet)."""
        # Для демо — синтетика. На практике: читать из normalized/items_{{ ds_nodash }}.parquet
        df = pd.DataFrame({
            "x1": [0.1, 0.5, 0.7, 1.2, 0.3, 0.9],
            "x2": [1.0, 0.8, 0.2, 0.1, 1.1, 0.4],
            "y":  [0,   0,   1,   1,   0,   1],
        })
        out = DATA / "normalized" / f"train_{exec_date}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return str(out)

    @task
    def train(train_path: str) -> str:
        df = pd.read_parquet(train_path)
        X = df[["x1", "x2"]].values
        y = df["y"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=13)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        # Сохраняем модель в staging
        model_path = MODELS / "staging" / f"model_{{ ds_nodash }}.pkl"
        joblib.dump({"model": clf, "valid_X": Xte, "valid_y": yte}, model_path)
        return str(model_path)

    @task
    def validate(model_path: str) -> float:
        bundle = joblib.load(model_path)
        clf = bundle["model"]
        Xte, yte = bundle["valid_X"], bundle["valid_y"]
        proba = clf.predict_proba(Xte)[:, 1]
        score = float(roc_auc_score(yte, proba))
        return score

    def _branch(**context) -> str:
        """Сравниваем со старым best в registry.json, решаем — сохранять ли модель."""
        ti = context["ti"]
        score = ti.xcom_pull(task_ids="validate")  # key='return_value'
        best = -1.0
        if REGISTRY.exists():
            try:
                best = float(json.loads(REGISTRY.read_text()).get("best_score", -1.0))
            except Exception:
                best = -1.0
        return "save_model" if score > best else "skip_save"

    branch = BranchPythonOperator(
        task_id="branch_on_metric",
        python_callable=_branch,
        provide_context=True,
    )

    @task
    def save_model(model_path: str, score: float) -> str:
        """Атомарно обновляем production‑модель и registry.json"""
        prod_path = MODELS / "production" / "model.pkl"
        tmp_path = MODELS / "production" / f"model_{datetime.now().timestamp()}.tmp"
        # копируем staging → tmp → rename
        bytes_data = Path(model_path).read_bytes()
        tmp_path.write_bytes(bytes_data)
        tmp_path.replace(prod_path)
        REGISTRY.write_text(json.dumps({
            "best_score": score,
            "model_path": str(prod_path),
            "updated_at": datetime.utcnow().isoformat(),
        }))
        return str(prod_path)

    skip_save = EmptyOperator(task_id="skip_save")

    # Связки
    ds = "{{ ds_nodash }}"
    t_prep = prep(ds)
    t_train = train(t_prep)
    t_valid = validate(t_train)
    t_branch = branch
    t_prep >> t_train >> t_valid >> t_branch

    # Ветки
    t_save = save_model(t_train, t_valid)
    t_branch >> t_save
    t_branch >> skip_save
```

**Ключевые моменты:**

- Метрика (`score`) и путь к модели — **через XCom**: вернуть из `@task` и принять как аргумент.
- `BranchPythonOperator` возвращает **task\_id ветки** (строку с идентификатором задачи) или список `task_id`, которые нужно исполнить.
- Обновление production‑артефакта — **атомарным переименованием** (tmp → final) + обновление `registry.json`.

---

## 2. Batch inference: данные → предсказание → результаты (с mapping)

### Идея

- Получить список входных файлов/партиций за окно (`list_inputs`).
- Прочитать production‑модель (`read_production_model`).
- Прогнать предсказания **параллельно** на каждом файле (`predict_one.expand(...)`).
- Смерджить вывод в единый результат (`merge_results`) → БД.

### Полный пример

```python
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from airflow import DAG
from airflow.decorators import task

BASE = Path("/opt/airflow")
DATA = BASE / "data"
MODELS = BASE / "models"

with DAG(
    dag_id="ml_batch_inference",
    start_date=datetime(2025, 9, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "inference"],
) as dag:

    @task
    def list_inputs(exec_date: str) -> list[str]:
        # На практике: найти все parquet за дату. Демка — создаём три CSV.
        IN = DATA / "predict" / "in"
        IN.mkdir(parents=True, exist_ok=True)
        files = []
        for i in range(3):
            p = IN / f"batch_{exec_date}_{i}.csv"
            pd.DataFrame({"x1": [0.2, 0.7], "x2": [0.9, 0.1]}).to_csv(p, index=False)
            files.append(str(p))
        return files

    @task
    def read_production_model() -> str:
        prod = MODELS / "production" / "model.pkl"
        if not prod.exists():
            raise FileNotFoundError("Нет production‑модели — обучи её или зафиксируй путь")
        return str(prod)

    @task
    def predict_one(model_path: str, csv_path: str) -> str:
        bundle = joblib.load(model_path)
        clf = bundle["model"]
        df = pd.read_csv(csv_path)
        proba = clf.predict_proba(df[["x1", "x2"]].values)[:, 1]
        out = DATA / "predict" / "out" / (Path(csv_path).stem + "_preds.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"proba": proba}).to_parquet(out, index=False)
        return str(out)

    @task
    def merge_results(paths: list[str]) -> str:
        dfs = [pd.read_parquet(p) for p in paths]
        out = DATA / "predict" / "out" / f"merged_{{ ds_nodash }}.parquet"
        pd.concat(dfs, ignore_index=True).to_parquet(out, index=False)
        return str(out)

    ds = "{{ ds_nodash }}"
    files = list_inputs(ds)
    model = read_production_model()
    preds = predict_one.partial(model_path=model).expand(csv_path=files)
    merged = merge_results(preds)
```

**Ключевые моменты:**

- `predict_one.partial(...).expand(csv_path=files)` — **dynamic task mapping**, создаёт по задаче на файл.
- Модель считывается один раз, путь прокидывается всем задачам через XComArg (`partial`).
- Итог собирается в `merge_results`.

---

## 3. «Обучать только при новых данных» (ветвление по условию)

Варианты:

1. **Sensor** → ждать, пока появятся файлы/таблица (например, `FileSensor`/`S3KeySensor`).
2. **BranchPythonOperator** → в рантайме проверять новизну и выбирать ветку.

### Пример с BranchPythonOperator

```python
from airflow.operators.branch import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from pathlib import Path

DATA = Path("/opt/airflow/data/normalized")

# функция ветвления

def _has_new_data(ds_nodash: str) -> str:
    expected = DATA / f"train_{ds_nodash}.parquet"
    return "run_training" if expected.exists() else "no_new_data"

branch = BranchPythonOperator(
    task_id="check_new_data",
    python_callable=lambda **kw: _has_new_data(kw["ds_nodash"]),
    provide_context=True,
)

run_training = EmptyOperator(task_id="run_training")
no_new_data = EmptyOperator(task_id="no_new_data")

branch >> run_training
branch >> no_new_data
```

На практике лучше объединить это с этапом `prep`: если `prep` вернул пустой датасет/0 файлов — ветка `no_new_data`.

---

## 4. Практические советы

- **Идемпотентность:** хранить модели с версией (`model_{{ ds_nodash }}.pkl`), производственный alias — `production/model.pkl`.
- **Атомарность:** писать во временный файл/каталог, затем `rename`.
- **Reproducibility:** фиксировать версии библиотек, seed, сохранять `training_config.json` рядом с моделью.
- **Метрики:** хранить не только `best_score`, но и историю в отдельной таблице/файле для наблюдения.
- **Ресурсы:** тяжёлое обучение/инференс лучше выносить в `KubernetesPodOperator`/`DockerOperator`.
- **Секреты:** никакого хардкода, только Connections/Variables/Secret backends.