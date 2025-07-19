# Production & MLOps для LLM, Поиска и Рекомендательных Систем

## 1. Контейнеризация и оркестрация

**Необходимые навыки:**
- Docker: сборка production-образов для LLM-сервисов (с GPU, CUDA, Python env).
- docker-compose: dev/test окружения.
- Kubernetes:
  - Deployment моделей (retriever, reranker, API).
  - Helm charts, autoscaling (в т.ч. GPU).
  - Resource limits, rollout, blue-green/canary деплой.

**Что изучить:**
- dockerfile best practices
- helm, kubectl, GPU scheduling
- K8s secrets, configMaps
- [Docker best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Kubernetes basics (официальный гайд)](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [Helm Getting Started](https://helm.sh/docs/intro/quickstart/)
- [GPU scheduling в K8s](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

---

## 2. CI/CD пайплайны

**Необходимые навыки:**
- GitHub Actions / GitLab CI / Jenkins / Argo:
  - автоматическая сборка docker-образов
  - прогон тестов и валидации моделей
  - деплой в облако или локальную инфраструктуру
- Проверка схем, метрик, bias (в CI)

**Что изучить:**
- CI для моделей и inference-сервисов
- rollback-стратегии, multi-env CD
- [GitHub Actions для ML](https://mlops.community/github-actions-for-mlops/)
- [CI/CD for Machine Learning](https://mlops.community/mlops-cicd/)
- [Argo Workflows — Kubernetes-native pipelines](https://argoproj.github.io/argo-workflows/)

---

## 3. Облачные платформы и inference

**Необходимые навыки:**
- AWS SageMaker: train job, endpoint, multi-model endpoint.
- Bedrock: вызов моделей (Claude, Titan).
- GCP Vertex AI, Azure ML (желательно).
- FastAPI/gRPC inference endpoints.
- text-generation-inference / vLLM / Triton Inference Server.

**Что изучить:**
- SageMaker SDK, vLLM API
- Деплой кастомных моделей с автоскейлом
- [Deploy models with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [vLLM — fast inference for LLMs](https://github.com/vllm-project/vllm)
- [text-generation-inference (Hugging Face)](https://github.com/huggingface/text-generation-inference)
- [Triton Inference Server](https://github.com/triton-inference-server/server)

---

## 4. Monitoring, observability, logging

**🔧 Необходимые навыки:**
- Метрики: latency, throughput, GPU util, model quality (CTR, Gini, NDCG).
- Drift detection по embedding, token dist, user segments.
- Prometheus + Grafana: мониторинг.
- OpenTelemetry / Jaeger: трассировка.

**📚 Что изучить:**
- Метрики inference latency
- Интеграция FastAPI ↔ Prometheus
- [Prometheus + FastAPI integration](https://github.com/stephenhillier/prometheus-fastapi-instrumentator)
- [Grafana Dashboards для ML](https://grafana.com/grafana/dashboards/)
- [OpenTelemetry для Python](https://opentelemetry.io/docs/instrumentation/python/)

---

## 5. Логирование и трассировка

**Необходимые навыки:**
- Структурированные логи: structlog, loguru, logger.exception.
- TraceID, RequestID через все сервисы.
- Хранение логов: ELK, Loki, Cloud Logging.

**Что изучить:**
- Прокси логов из FastAPI
- Trace context propagation
- [Structlog documentation](https://www.structlog.org/en/stable/)
- [Loguru: Python logging made (stupidly) simple](https://github.com/Delgan/loguru)
- [OpenTelemetry Logging](https://opentelemetry.io/docs/specs/otel/logs/overview/)

---

## 6. Автоматизация валидации и тестирования

**Необходимые навыки:**
- Юнит/интеграционные тесты: pytest + FastAPI.
- Проверка метрик моделей.
- Контроль схем (Pydantic).
- Robustness тесты: переформулировки промптов.

**Что изучить:**
- Test coverage для LLM-инференса
- Pydantic schema enforcement
- [Testing ML Systems](https://madewithml.com/courses/mlops/testing/)
- [Pytest + FastAPI examples](https://fastapi.tiangolo.com/tutorial/testing/)
- [Deepchecks: ML validation framework](https://github.com/deepchecks/deepchecks)

---

## 7. A/B-тестирование моделей

**Необходимые навыки:**
- Параллельный деплой двух моделей (routing по user_id).
- Метрики: uplift, CTR, latency.
- Анализ логов, визуализация.

**Что изучить:**
- AB-инфраструктура FastAPI + логгинг
- A/B evaluation pipeline
- [AB Testing for ML Models — MLOps Guide](https://mlops.community/ab-testing/)
- [Feature toggles / split traffic](https://launchdarkly.com/blog/feature-flags-ab-testing/)
- [MLOps A/B testing with FastAPI](https://towardsdatascience.com/a-b-testing-in-machine-learning-d3b8e8f4de3c)

---

## 8. Batch / streaming inference

**Необходимые навыки:**
- Batch inference: cron, Airflow, CSV → embeddings → save.
- Streaming: Kafka/Redis → модель → хранилище.
- Умение переключаться между batch / online пайплайнами.

**Что изучить:**
- Airflow DAGs
- Streaming inference server
- [Batch inference with Airflow](https://blog.roboflow.com/batch-inference-airflow/)
- [Streaming ML with Kafka](https://developer.confluent.io/learn/kafka-streams/)
- [Redis Streams + LLM](https://redis.com/blog/real-time-stream-processing-llm/)

---

## 9. Feature Store и управление артефактами

**Необходимые навыки:**
- Feature Store (Feast, dbt, custom): хранение признаков.
- MLflow / DVC / S3: модели, эмбеддинги, скрипты.

**Что изучить:**
- Версионирование артефактов
- MLflow Registry API
- [Feast — open-source Feature Store](https://feast.dev/)
- [MLflow Tracking and Registry](https://mlflow.org/docs/latest/tracking.html)
- [DVC for ML pipelines](https://dvc.org/doc/start)

---

## 10. Безопасность и приватность

**Необходимые навыки:**
- Ограничение доступа к LLM endpoints.
- Обфускация PII, маскирование.
- Rate limiting, abuse protection.
- Аудит, compliance (SOC2, ISO, GDPR-ready).

**Что изучить:**
- OpenAPI tokens + access scopes
- Privacy filters для текстов/логов
- [OWASP ML Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [Ethical AI checklist (Google)](https://pair-code.github.io/ethicalml/)
- [GDPR for ML Systems](https://gdpr.eu/)
  
---

## 11. Частые задачи и навыки

| Задача | Навыки |
|-------|--------|
| Выкатить retriever | Docker, CI/CD, latency тесты |
| Проверить деградацию | Тесты метрик, drift |
| Внедрить quantized reranker | Quantization, latency tests |
| Настроить мониторинг | Prometheus, middlewares |
| A/B-тест retrievers | Routing, uplift анализ |
| Поднять vLLM | CUDA, API gateway |
| Обновить эмбеддинги | Batch infer, cron, Airflow |
| Обработка 10M запросов | Async inference, batching |

---
