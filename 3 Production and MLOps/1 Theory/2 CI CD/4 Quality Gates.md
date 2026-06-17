# Quality Gates in CI/CD

## 1. Что такое Quality Gates

**Quality Gates** — автоматические проверки в CI/CD pipeline, которые блокируют продвижение артефакта дальше, если он не соответствует критериям качества.

**Проблема без quality gates:** модель прошла unit-тесты, но latency выросла в 2 раза или MRR упал на 5% — деплой в прод, инцидент.

**Принцип:** каждый этап pipeline имеет «ворота» (gates) — условия, без выполнения которых артефакт не идёт дальше.

```
PR → Unit tests (gate) → Integration tests (gate) → Staging (gate) → Manual approval → Production
```

## 2. Уровни Quality Gates

### 2.1 Unit-тесты (код)

Быстрые тесты отдельных функций/модулей. Без внешних зависимостей.

```python
# test_retriever.py
def test_tokenize():
    tokenizer = get_tokenizer()
    tokens = tokenizer("Hello world")
    assert len(tokens) == 2
    
def test_embedding_shape():
    model = load_model("test_fixture")
    emb = model.encode("test query")
    assert emb.shape == (768,)
```

**Инструменты:** `pytest`, `unittest`.

**Gate:** все тесты проходят, coverage ≥ 80%.

### 2.2 Интеграционные тесты (сервис)

Тесты взаимодействия с реальными зависимостями (БД, API, модель).

```python
def test_retrieval_pipeline():
    # Поднимаем тестовый контейнер с OpenSearch
    retriever = RetrievalService(endpoint="localhost:9200")
    results = retriever.search("machine learning")
    assert len(results) > 0
    assert results[0].score > 0.5
```

**Инструменты:** `docker-compose` для поднятия зависимостей в CI.

**Gate:** все интеграционные тесты проходят.

### 2.3 End-to-End тесты

Полный сценарий — от запроса пользователя до ответа.

```python
def test_full_rag_pipeline():
    response = client.post("/query", json={"query": "What is Kubernetes?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] is not None
    assert len(data["sources"]) > 0
    assert data["latency_ms"] < 1000
```

**Gate:** e2e тесты проходят, latency ≤ SLA.

### 2.4 Метрики модели (offline evaluation)

Сравнение новой модели с baseline на отложенной выборке.

```python
def test_model_regression():
    new_model = load_model("staging")
    baseline = load_model("production")
    
    # Оцениваем на тестовом датасете
    new_mrr = evaluate_mrr(new_model, test_queries, test_qrels)
    baseline_mrr = evaluate_mrr(baseline, test_queries, test_qrels)
    
    assert new_mrr >= baseline_mrr * 0.98  # не хуже чем -2%
```

**Gate:** MRR / NDCG / Recall ≥ baseline (порог зависит от критичности).

### 2.5 Нагрузочные тесты (performance)

Измерение latency и throughput под нагрузкой.

```bash
# Запуск нагрузочного теста
locust -f locustfile.py --headless \
  --users 50 --spawn-rate 5 --run-time 60s \
  --host https://staging.example.com
```

```python
# locustfile.py
from locust import HttpUser, task

class SearchUser(HttpUser):
    @task
    def search(self):
        self.client.post("/query", json={"query": "test query"})
```

**Gate:** p95 latency ≤ 500ms, p99 ≤ 1000ms, error rate < 1%.

### 2.6 Continuous evaluation (дрифт и качество)

Регулярная проверка модели в staging/prod на реальных данных.

```python
from evidently.metrics import RegressionPerformanceMetric
from evidently.report import Report

# Сравнение с предыдущим периодом
report = Report(metrics=[
    RegressionPerformanceMetric(),
])
report.run(reference_data=baseline_df, current_data=current_df)

# Gate: PSI < 0.2, accuracy drift < 5%
assert report.datasets.drift.share_drifted_features < 0.2
```

**Инструменты:** Evidently AI, DeepChecks, NannyML, WhyLogs.

## 3. Типовой pipeline с Quality Gates

### 3.1 GitLab CI

```yaml
stages:
  - unit-test
  - integration-test
  - model-eval
  - perf-test
  - staging-deploy
  - e2e-test
  - prod-deploy

unit-test:
  stage: unit-test
  script:
    - pytest tests/unit/ --junitxml=unit-report.xml --cov=src
  artifacts:
    reports:
      junit: unit-report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

integration-test:
  stage: integration-test
  services:
    - name: opensearch:2.11
  script:
    - docker-compose up -d
    - pytest tests/integration/ --junitxml=integration-report.xml
    
model-eval:
  stage: model-eval
  script:
    - python -m evaluation.compare_models
      --new-model staging
      --baseline production
      --test-dataset s3://datasets/eval-v3.jsonl
      --min-mrr 0.85
      --max-mrr-drop 0.02
    # Если порог не пройден — exit 1

perf-test:
  stage: perf-test
  script:
    - kubectl wait --for=condition=available deployment/my-app -n staging --timeout=120s
    - locust -f locustfile.py --headless --users 50 --run-time 120s
    - python -m evaluation.check_perf --p95-latency 500 --p99-latency 1000

e2e-test:
  stage: e2e-test
  script:
    - pytest tests/e2e/ --base-url https://staging.example.com --junitxml=e2e-report.xml

staging-deploy:
  stage: staging-deploy
  needs: [unit-test, integration-test, model-eval]
  script:
    - helm upgrade --install my-app ./charts/my-app -f configs/staging.yaml

prod-deploy:
  stage: prod-deploy
  needs: [staging-deploy, perf-test, e2e-test]
  script:
    - helm upgrade --install my-app ./charts/my-app -f configs/prod.yaml
  when: manual
```

## 4. Schema checks (контракты API)

Проверка, что API модели соответствует контракту (OpenAPI / Pydantic).

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    filters: dict | None = None
    top_k: int = Field(default=10, ge=1, le=100)

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    model_version: str

# В тестах
def test_api_schema():
    response = client.post("/query", json={"query": "test"})
    QueryResponse(**response.json())  # падает если не соответствует
```

**Инструменты:** Pydantic, OpenAPI/Swagger, Schemathesis.

**Gate:** все эндпоинты проходят schema validation.

## 5. Метрики quality gates (какие и где)

| Этап | Что проверяем | Инструмент | Порог |
|------|-------------|-----------|-------|
| Unit-test | Coverage | pytest-cov | ≥ 80% |
| Integration | Функциональность | pytest + docker | 100% pass |
| Model eval | MRR / NDCG / Recall | Кастомный скрипт | ≥ baseline × 0.98 |
| Perf test | p95/p99 latency | Locust / Gatling | p95 ≤ 500ms |
| E2E | Полный сценарий | pytest | 100% pass |
| Schema | API-контракт | Pydantic / OpenAPI | 100% pass |
| Continuous eval | Drift, accuracy | Evidently AI | PSI < 0.2 |

## 6. Автоматический откат

Если gate не пройден на staging — деплой в prod блокируется. Если проблема обнаружена post-deploy (continuous evaluation):

```yaml
post-deploy-monitor:
  stage: monitor
  script:
    - |
      for i in $(seq 1 12); do
        ERROR_RATE=$(curl -s https://my-app.example.com/metrics | grep error_rate)
        if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
          echo "Error rate too high: $ERROR_RATE, rolling back..."
          helm rollback my-app
          exit 1
        fi
        sleep 5
      done
  when: always
```

## 7. Чек-лист

1. **Unit-тесты**: покрытие кода, быстрые (<1 мин).
2. **Интеграционные тесты**: реальные зависимости через docker-compose.
3. **Model eval**: метрики качества против baseline на отложенной выборке.
4. **Performance**: latency/throughput под нагрузкой, пороги по SLA.
5. **E2E**: полный сценарий от запроса до ответа.
6. **Schema checks**: API-контракт через Pydantic/OpenAPI.
7. **Continuous evaluation**: дрифт и деградация в staging/prod.
8. **Автоматический откат**: если post-deploy метрики выходят за пределы.

## 8. Типичные ошибки

1. **Только unit-тесты без model eval**: код работает, но модель отвечает хуже — не заметим.
2. **Performance тесты без реалистичной нагрузки**: 1 user ≠ 50 concurrent users.
3. **Model eval на тренировочных данных**: переобучение не видно. Тестовый датасет должен быть изолирован.
4. **E2E тесты без таймаута**: висят бесконечно, блокируют pipeline.
5. **Отсутствие baseline**: не с чем сравнить новую модель.
6. **Staging = prod данные**: тесты на реальных пользователях до валидации — риск.