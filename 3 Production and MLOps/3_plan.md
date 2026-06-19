# План компетенций: Production & MLOps для LLM‑сервисов (Search / RecSys)

> **Фокус:** эксплуатация и сопровождение LLM‑моделей (retriever, reranker, генератор) в продакшне: контейнеры, CI/CD, низколатентный inference, мониторинг, безопасность, сетевые основы, back‑end dev.  
> Алгоритмы retrieval и хранилища вынесены в отдельные планы.

---

## I. Контейнеризация и оркестрация

### Mini‑Networks (сетевые основы)  
📄 [Networks Intro](1%20Theory/1%20Containerization%20and%20Orchestration/1%20Mini-Networks/0%20Networks%20Intro.md)  
📄 [TCP](1%20Theory/1%20Containerization%20and%20Orchestration/1%20Mini-Networks/1%20TCP.md)  
📄 [UDP](1%20Theory/1%20Containerization%20and%20Orchestration/1%20Mini-Networks/1%20UDP.md)  
📄 [DNS](1%20Theory/1%20Containerization%20and%20Orchestration/1%20Mini-Networks/3%20DNS.md)  
📄 [HTTP](1%20Theory/1%20Containerization%20and%20Orchestration/1%20Mini-Networks/4%20HTTP.md)  
📄 [TLS](1%20Theory/1%20Containerization%20and%20Orchestration/1%20Mini-Networks/5%20TLS.md)

### Docker  
📄 [Docker Cheatsheet](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/0%20Docker%20Cheatsheet.md)  
📄 [Containers and Images](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/1%20Docker%20Basics/1%20Containers%20and%20Images.md)  
📄 [Dockerfile building image](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/1%20Docker%20Basics/2%20Dockerfile%20building%20image.md)  
📄 [Multi-Stage and BuildKit](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/1%20Docker%20Basics/3%20Multi-Stage%20and%20BuildKit.md)  
📄 [Python Service](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/1%20Docker%20Basics/4%20Python%20Service.md)  
📄 [Volumes, Backups and Migration](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/1%20Docker%20Basics/5%20Volumes,%20Backups%20and%20Migration.md)  
📄 [Logs, Metrics, Profiling](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/1%20Docker%20Basics/6%20Logs,%20Metrics,%20Profiling.md)

### Docker Compose  
📄 [Compose Basics](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/2%20Docker%20Compose/1%20Basics.md)  
📄 [Compose Network](1%20Theory/1%20Containerization%20and%20Orchestration/2%20Docker/2%20Docker%20Compose/2%20Network.md)  

### Kubernetes  
📄 [Kubernetes Intro](1%20Theory/1%20Containerization%20and%20Orchestration/3%20Kubernetes/1%20Kubernetes%20Intro.md)  
📄 [Pod, Deployment and Lifecycle](1%20Theory/1%20Containerization%20and%20Orchestration/3%20Kubernetes/2%20Pod,%20Deployment%20and%20Lifecycle.md)  
📄 [Service, Network and Configuration](1%20Theory/1%20Containerization%20and%20Orchestration/3%20Kubernetes/3%20Service,%20Network%20and%20Configuration.md)  
📄 [Scheduling, Affinity and Taints](1%20Theory/1%20Containerization%20and%20Orchestration/3%20Kubernetes/4%20Scheduling,%20Affinity%20and%20Taints.md)  

### Helm  
📄 [Helm Basics](1%20Theory/1%20Containerization%20and%20Orchestration/4%20Helm/1%20Helm%20Basics.md)  
📄 [Templating and Advanced](1%20Theory/1%20Containerization%20and%20Orchestration/4%20Helm/2%20Templating%20and%20Advanced.md)  

### Airflow
📄 [Airflow Intro](1%20Theory/1%20Containerization%20and%20Orchestration/5%20Airflow/1%20Airflow%20Intro.md)  
📄 [ETL and XCom](1%20Theory/1%20Containerization%20and%20Orchestration/5%20Airflow/2%20ETL%20and%20XCom.md)  
📄 [ML Orchestration](1%20Theory/1%20Containerization%20and%20Orchestration/5%20Airflow/3%20ML%20Orchestration.md)  
📄 [Scaling and Production](1%20Theory/1%20Containerization%20and%20Orchestration/5%20Airflow/4%20Scaling%20and%20Production.md)  
📄 [Best Practices](1%20Theory/1%20Containerization%20and%20Orchestration/5%20Airflow/5%20Best%20Practices.md)  
📄 [Monitoring and Observability](1%20Theory/1%20Containerization%20and%20Orchestration/5%20Airflow/6%20Moniroting%20and%20Observability.md)

---

## II. CI/CD

📄 [CI/CD Intro](1%20Theory/2%20CI%20CD/0%20CI%20CD%20Intro.md)  
📄 [CI Platforms](1%20Theory/2%20CI%20CD/1%20CI%20Platforms.md)  
📄 [CD Configs](1%20Theory/2%20CI%20CD/2%20CD%20Configs.md)  
📄 [Model Registry](1%20Theory/2%20CI%20CD/3%20Model%20Registry.md)  
📄 [Quality Gates](1%20Theory/2%20CI%20CD/4%20Quality%20Gates.md)  
📄 [Security Scanning](1%20Theory/2%20CI%20CD/5%20Security%20Scanning.md)

---

## III. Низко‑латентный inference и оптимизация

### Inference движки

📄 [vLLM](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/1%20Inference%20Engines/1%20vLLM.md)  
📄 [Text Generation Inference](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/1%20Inference%20Engines/2%20Text%20Generation%20Inference.md)  
📄 [Triton Inference Server](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/1%20Inference%20Engines/3%20Triton%20Inference%20Server.md)  
📄 [DeepSpeed-MII](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/1%20Inference%20Engines/4%20DeepSpeed-MII.md)  

### Inference движки

📄 [KV-cache](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/2%20Optimization/1%20KV-cache.md)  
📄 [Continuous Batching](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/2%20Optimization/2%20Continuous%20Batching.md)  
📄 [Speculative Decoding](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/2%20Optimization/3%20Speculative%20Decoding.md)  

### Autoscale

📄 [FastAPI Batch Endpoints](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/3%20Autoscale/1%20FastAPI%20Batch%20Endpoints.md)  
📄 [Concurrency Gate](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/3%20Autoscale/2%20Concurrency%20Gate.md)  
📄 [Dynamic Batch Size](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/3%20Autoscale/3%20Dynamic%20Batch%20Size.md)  

### Cost-control

📄 [Spot Instances](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/4%20Cost-control/1%20Spot%20Instances.md)  
📄 [Heterogeneous Fleet](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/4%20Cost-control/2%20Heterogeneous%20Fleet.md)  
📄 [Scale to Zero](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/4%20Cost-control/3%20Scale%20to%20Zero.md)  

### Compilers

📄 [Torch Compile](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/5%20Compilers/1%20Torch%20Compile.md)  
📄 [TensorRT](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/5%20Compilers/2%20TensorRT.md)  
📄 [ONNX Runtime](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/5%20Compilers/3%20ONNX%20Runtime.md)  
📄 [OpenVINO](1%20Theory/3%20Low-Latency%20Inference%20and%20Optimization/5%20Compilers/4%20OpenVINO.md)  

---

## IV. Облачные и on‑prem платформы
> 📝 Конспектов пока нет. Планируемые темы:
> - AWS SageMaker / Bedrock, GCP Vertex AI, Azure ML.
> - KServe / KFServing, HF Inference Endpoints.

---

## V. Back‑end разработка и архитектура сервисов
> 📝 Конспектов пока нет. Планируемые темы:
> - **Серверы:** uvicorn, gunicorn, FastAPI, Starlette. ASGI vs WSGI.
> - **Асинхронность:** `async`/`await`, event loop, `asyncio.gather()`, `httpx.AsyncClient`, таймауты, retry.
> - **Обработка ошибок:** санация, логирование трейса, проброс безопасных сообщений.
> - **JSON-RPC:** структура запроса/ответа/ошибки.
> - **Stateless vs Stateful:** когда что выбирать, StatefulSet, sticky sessions.
> - **Транспорты:** stdio, SSE, Streamable HTTP, WebSocket, gRPC — сравнение, плюсы/минусы.

### OpenAI API Format
📄 [OpenAI API Format](1%20Theory/5%20Backend%20and%20API/1%20OpenAI%20API%20Format.md)
>— Completions API (`/v1/completions`) vs Chat Completions API (`/v1/chat/completions`), структура запроса и ответа, streaming через SSE, function calling, различия между реализациями (OpenAI, vLLM, TGI, Anthropic).

---

## VI. Сетевые основы и Service Mesh (Istio)
> 📝 Конспектов пока нет. Планируемые темы:

### Сетевые основы
> - **HTTP:** редиректы (301/302/307/308), Reverse Proxy, Forward Proxy.
> - **DNS:** A, CNAME, TTL, резолвинг в Kubernetes (CoreDNS, headless services).

### mTLS
> - Цепочка сертификатов, рукопожатие, ротация, интеграция с Istio sidecar.

### Istio
> 📌 Вынесен из раздела Kubernetes, так как относится к сетевой инфраструктуре.
> - **Control plane (istiod):** Pilot (traffic), Citadel (security), Galley (config) — роль каждого компонента.
> - **Data plane (Envoy):** Sidecar injection, перехват трафика, фильтры, метрики.
> - **Traffic management:** VirtualService (routing, retries, timeouts, fault injection, mirroring), DestinationRule (subset, load balancing, connection pool, outlier detection), Gateway (ingress).
> - **Security:** PeerAuthentication (PERMISSIVE / STRICT), AuthorizationPolicy (allow/deny, JWT, RBAC), RequestAuthentication.
> - **Observability:** встроенные метрики Envoy, интеграция с Prometheus, Kiali, Jaeger.
> - **Multi-cluster:** ServiceEntry, multicluster deployment models, federation.

---

## VII. Мониторинг и Observability
> 📝 Конспектов пока нет. Планируемые темы:
> - **Три столпа:** логи vs метрики vs трейсы — что, зачем, инструменты.
> - **Fluent Bit:** pipeline (Input → Parser → Filter → Buffer → Output), DaemonSet в Kubernetes.
> - **OpenTelemetry:** context propagation (traceparent), стек Prometheus + Grafana + Jaeger + Sentry.

---

## VIII. Логирование и аудит
> 📝 Конспектов пока нет. Планируемые темы:
> - Structured logs (structlog/loguru, JSON), TraceID/SpanID, stdout → Fluent Bit.
> - Log storage: Loki, Elastic Stack.
> - Аудит вызовов, санация ошибок, централизованное логирование.

---

## IX. Валидация и тестирование
> 📝 Конспектов пока нет. Планируемые темы:
> - pytest (unit, integration), нагрузочное (Gatling/Locust), robustness (prompt mutations).
> - Schema checks (Pydantic, OpenAPI), continuous evaluation (DeepChecks/Evidently ML).

---

## X. A/B‑тестирование и rollout
> 📝 Конспектов пока нет. Планируемые темы:
> - Traffic splitting: header‑hash, user‑id bucketing.
> - Метрики uplift, sequential testing, CUPED, Bayesian bandits.
> - Shadow deploy, champion/challenger, feature‑flags.
> - **Deploy‑паттерны:** Blue‑Green, Canary, Shadow, Feature‑flags.

---

## XI. Batch и streaming inference
> 📝 Конспектов пока нет. Планируемые темы:
> - Batch: Airflow/Prefect — CSV/Parquet → model → S3/Table.
> - Streaming: Kafka/Redis Streams → inference worker → sink.
> - Hot‑swap: переключение real‑time ↔ batch.

---

## XII. Feature Store и артефакты
> 📝 Конспектов пока нет. Планируемые темы:
> - Feature Store: Feast, Hopsworks — online ↔ offline parity.
> - Артефакты: MLflow, DVC, Weights & Biases — versioning моделей, эмбеддингов, данных.
> - Data lineage & governance: tags, model cards, datasheets.

---

## XIII. Безопасность и приватность
> 📝 Конспектов пока нет. Планируемые темы:
> - **Auth & AuthZ:** OAuth2, service tokens, scope‑based ACL, JWO-токены, Origin header validation.
> - **Rate‑limiting / WAF:** abuse‑detection, IP‑reputation.
> - **Prompt safety:** prompt‑injection filters, jailbreak detection, output sandboxing.
> - **PII masking & encryption:** at‑rest (AES-GCM), in‑transit (TLS).
> - **Vault:** хранение секретов, dynamic credentials, интеграция с Kubernetes.
> - **Container security:** readOnlyRootFilesystem, runAsNonRoot, seccomp.
> - **Compliance:** SOC‑2, ISO‑27001, GDPR, HIPAA.

---

## XIV. FinOps и cost‑monitoring
> 📝 Конспектов пока нет. Планируемые темы:
> - Метрики стоимости: GPU‑hours, $/M tokens, utilization heatmaps.
> - Budget alerts, rightsizing, chargeback & showback.

---

## XV. Чеклист типовых задач

| Сценарий | Ключевые шаги / навыки |
|---|---|
| Вкат retriever на GPU | Docker → CI/CD → Helm chart → HPA |
| Снизить p95 latency на 30% | KV‑кэш, FlashAttention‑2, quantization, batching |
| Drift‑мониторинг | embed sampling → PSI/KL → Prometheus alert |
| A/B двух версий | traffic split → Metrics store → uplift analysis |
| Re‑index без даунтайма | batch infer → shadow index → alias switch |
| Anti‑abuse | rate‑limit middleware → JWT scope check |
| Деплой MCP‑сервера | Docker → CI-контракт → ConfigMap → Deployment → Service → Istio AuthorizationPolicy |
| Диагностика ProgressDeadlineExceeded | `kubectl describe deploy` → probes → ресурсы → логи пода |
| Откат Helm-релиза | `helm history` → `helm rollback <release> <revision>` |
| Настройка канареечного деплоя | VirtualService → DestinationRule → мониторинг → повышение веса |
| Подключение Fluent Bit | DaemonSet → ConfigMap → Output в Elasticsearch |
| Настройка mTLS | PeerAuthentication STRICT → DestinationRule ISTIO_MUTUAL (см. раздел VI) |
| Создание белых списков | AuthorizationPolicy → тест доступа (см. раздел VI) |
| Выбор транспорта для сервиса | Анализ требований → SSE vs Streamable HTTP vs stdio |

---

## XVI. Ресурсы
- **Dockerfile best practices** — docs.docker.com
- **Helm quickstart** — helm.sh/docs
- **Terraform Cloud** — developer.hashicorp.com
- **Kubernetes docs** — kubernetes.io/docs
- **Istio docs** — istio.io/latest/docs
- **OpenTelemetry Python** — opentelemetry.io
- **Fluent Bit manual** — docs.fluentbit.io
- **FAISS docs** — github.com/facebookresearch/faiss
- **vLLM** — github.com/vllm‑project/vllm
- **Prometheus FastAPI Instrumentator** — GitHub
- **Evidently AI drift monitor** — docs.evidentlyai.com
- **Feast Feature Store** — feast.dev
- **MLflow Tracking** — mlflow.org