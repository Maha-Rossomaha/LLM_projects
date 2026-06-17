# Model Registry

## 1. Что такое Model Registry

**Model Registry** — централизованное хранилище для управления версиями ML-моделей на протяжении жизненного цикла: от обучения до деплоя и вывода из эксплуатации.

**Проблема без registry:** модель лежит в S3/директории, версия — в имени файла (`model_v3_final_2.pkl`), нет связи с кодом/данными/метриками, нельзя откатить.

**Что даёт Model Registry:**
* версионирование моделей (SemVer или автоинкремент),
* связь с артефактами (код, данные, конфигурация обучения),
* хранение метрик (accuracy, latency, размер),
* управление стадиями (Staging → Production → Archived),
* promotion/demotion с approval flow,
* интеграция с CI/CD для автоматического деплоя.

## 2. MLflow Model Registry

### 2.1 Модель

**MLflow** — open-source платформа для ML lifecycle. Model Registry — компонент MLflow для управления моделями.

**Ключевые концепты:**
* **Registered Model** — именованная сущность (например, `my-retriever`).
* **Model Version** — конкретная версия зарегистрированной модели (1, 2, 3...).
* **Stage** — стадия версии: `None` → `Staging` → `Production` → `Archived`.
* **Transition** — перемещение между стадиями (API или UI).

### 2.2 Регистрация модели

```python
import mlflow

with mlflow.start_run() as run:
    # обучение модели
    model = train_model(...)
    mlflow.log_metric("mrr", 0.87)
    mlflow.log_param("learning_rate", 1e-4)

    # регистрация
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        registered_model_name="my-retriever"  # создаёт или добавляет версию
    )
```

### 2.3 Promotion модели

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Найти версию в Staging
versions = client.get_latest_versions("my-retriever", stages=["Staging"])
version = versions[0].version

# Промоут в Production
client.transition_model_version_stage(
    name="my-retriever",
    version=version,
    stage="Production",
    archive_existing_versions=True   # архивировать предыдущую Production
)
```

### 2.4 Загрузка модели для inference

```python
# Загрузка конкретной стадии
model = mlflow.pyfunc.load_model("models:/my-retriever/Production")

# Загрузка конкретной версии
model = mlflow.pyfunc.load_model("models:/my-retriever/3")

# Inference
result = model.predict(query)
```

### 2.5 Approval flow

MLflow поддерживает ручной approval через UI или API:
1. Разработчик регистрирует новую версию → `None`.
2. Request transition → `Staging` (с комментарием о метриках).
3. Reviewer approves → `Staging`.
4. После валидации → `Production`.

## 3. SageMaker Model Registry

### 3.1 Модель

**AWS SageMaker Model Registry** — управляемый сервис AWS для каталогизации моделей.

**Ключевые концепты:**
* **Model Group** — группа версий одной модели.
* **Model Version** — версия с артефактами (S3 URI образа) и метаданными.
* **Model Package** — упакованная модель для деплоя.
* **Status** — `PendingManualApproval` → `Approved` → `Rejected`.
* **Stage** — `Staging` → `Production`.

### 3.2 Регистрация модели

```python
import sagemaker
from sagemaker.model import Model

# Создание модели
model = Model(
    image_uri="<account>.dkr.ecr.<region>.amazonaws.com/my-model:1.0",
    model_data="s3://my-bucket/models/model.tar.gz",
    role=role
)

# Регистрация в Model Registry
model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    model_package_group_name="my-retriever",
    approval_status="PendingManualApproval",
    model_metrics={
        "mrr": {"value": 0.87, "standard_deviation": 0.02}
    }
)
```

### 3.3 Promotion rules

* **Manual approval**: человек во в консоли AWS или через API.
* **Auto-approve**: при достижении метрик (через Lambda/Step Functions).
* **CI/CD интеграция**: CodePipeline → deploy при `Approved`.

### 3.4 SemVer

SageMaker поддерживает SemVer для версий моделей:
* `1.0.0` → `1.1.0` (minor — совместимое улучшение)
* `1.1.0` → `2.0.0` (major — несовместимое изменение)
* Promotion rules можно привязать к major/minor/patch.

## 4. Сравнение MLflow vs SageMaker

| | MLflow | SageMaker Model Registry |
|---|---|---|
| **Деплой** | Self-hosted или managed | AWS managed |
| **Интеграция** | Открытый API, любые платформы | Только AWS экосистема |
| **Стоимость** | Бесплатно (open-source) | Платно за хранение и запросы |
| **Stages** | None → Staging → Production → Archived | PendingApproval → Approved → Rejected |
| **Approval** | UI / REST API | Консоль / API / Step Functions |
| **Метрики** | `mlflow.log_metric()` | `model_metrics` при регистрации |

## 5. Интеграция Model Registry с CI/CD

### 5.1 Типовой flow

```
1. Тренировка (CI)
   ├── mlflow run → log model + metrics
   └── register model (version N, stage=None)

2. Промоут в Staging (CD)
   ├── transition to Staging
   └── deploy to staging cluster (ArgoCD/Helm)

3. Валидация
   ├── smoke tests
   ├── load tests
   └── метрики (latency, error rate)

4. Промоут в Production
   ├── manual approval
   ├── transition to Production
   └── deploy to production cluster
```

### 5.2 GitLab CI пример

```yaml
deploy-staging:
  stage: deploy-staging
  script:
    - python promote_model.py --stage Staging
    - |
      MODEL_URI=$(mlflow models get-uri --stage Staging my-retriever)
      helm upgrade --install my-app ./charts/my-app \
        -f configs/staging.yaml \
        --set model.uri=$MODEL_URI
  only:
    - main

deploy-prod:
  stage: deploy-prod
  script:
    - python promote_model.py --stage Production
    - |
      MODEL_URI=$(mlflow models get-uri --stage Production my-retriever)
      helm upgrade --install my-app ./charts/my-app \
        -f configs/prod.yaml \
        --set model.uri=$MODEL_URI
  when: manual
  only:
    - main
```

## 6. Rollback

```python
# Откат на предыдущую Production-версию
client = MlflowClient()
versions = client.get_latest_versions("my-retriever", stages=["Archived"])
prev_version = versions[0].version

client.transition_model_version_stage(
    name="my-retriever",
    version=prev_version,
    stage="Production",
    archive_existing_versions=True
)

# Helm rollback до предыдущей ревизии с правильным model.uri
```

## 7. Чек-лист

1. **Именование модели**: осмысленное имя (`my-retriever`, `query-classifier`).
2. **Версионирование**: SemVer или автоинкремент.
3. **Метрики**: каждая версия залогирована с метриками (MRR, accuracy, latency).
4. **Stages**: Staging (тестирование) → Production (бой) → Archived (история).
5. **Approval flow**: ручное подтверждение перед Production.
6. **Rollback**: предыдущая версия доступна в Archived.
7. **Связь с артефактами**: код (git commit hash), данные (dataset version), конфиг обучения.

## 8. Типичные ошибки

1. **Регистрировать модель без метрик**: нельзя сравнить версии, promotion «на глаз».
2. **Не архивировать старую Production**: несколько версий в Production → неопределённость.
3. **Model URI жёстко в коде**: должен приходить из Model Registry через CI.
4. **Забыть про rollback**: нет процедуры отката, прод сломается на невалидированной модели.
5. **Staging без реального трафика**: метрики на синтетических данных ≠ прод.