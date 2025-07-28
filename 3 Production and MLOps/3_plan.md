# План компетенций: Production & MLOps для LLM‑сервисов (Search / RecSys)

> **Фокус:** эксплуатация и сопровождение LLM‑моделей (retriever, reranker, генератор) в продакшне: контейнеры, CI/CD, низколатентный inference, мониторинг, безопасность. Алгоритмы retrieval и хранилища вынесены в отдельные планы.

---

## I. Контейнеризация и оркестрация
- **Docker:** многостейджевые Dockerfile, CUDA‑базовые образы, тонкие runtime‑образы (distroless, slim).
- **docker‑compose:** dev / test окружения, GPU‑override.
- **Kubernetes + Helm:** GPU scheduling, resource limits, auto‑scaling (HPA / KEDA), blue‑green / canary rollout, secrets + configMaps.

## II. CI/CD и Infrastructure‑as‑Code
- **CI платформы:** GitHub Actions / GitLab CI / Jenkins / Argo — build → test → scan → push → deploy.
- **Model Registry:** MLflow Registry, SageMaker Model Registry — semver, promotion rules, rollback.
- **IaC:** Terraform / Pulumi для VPC, GPU nodes, LB; policy as code, drift detection.
- **Quality gates:** unit + integration + e2e тесты; метрики (MRR, latency) в pipeline.
- **Deploy‑паттерны:** blue‑green, canary, shadow, feature‑flags.

## III. Низко‑латентный inference и оптимизация
- **Inference движки:** vLLM, text‑generation‑inference, Triton, DeepSpeed‑MII.
- **Оптимизации модели:** KV‑кэш, FlashAttention‑2, quantization (INT8/4, GPTQ), continuous batching, speculative decoding.
- **Скалирование:** Tensor / Pipeline MP (DeepSpeed, Megatron), multi‑GPU NCCL, FP8 pilot.
- **Автоскейл нагрузок:** FastAPI batch endpoints, concurrency gate, dynamic batch‑size.
- **Cost‑control:** GPU spot, heterogeneous fleet, scale‑to‑zero.  
- **Компиляторы и runtime-оптимизаторы:** torch.compile, TensorRT, ONNX Runtime, OpenVINO

## IV. Облачные и on‑prem платформы
- **AWS SageMaker / Bedrock, GCP Vertex AI, Azure ML:** endpoints, multi‑model endpoints, traffic‑splitting.
- **KServe / KFServing:** serverless модели в кластере.
- **HF Inference Endpoints** для хостинга retriever/LLM.

## V. Мониторинг и observability
- **Сервисные метрики:** p50/p95 latency, throughput, GPU util, memory, error rate, токены / сек.
- **Качественные метрики:** MRR, nDCG, drift (PSI/KL), hallucination‑rate.
- **Стек:** Prometheus + Grafana, OpenTelemetry traces (Jaeger), Sentry alerts.
- **Алёртинг:** SLO burn‑rate, anomaly detection on metrics.

## VI. Логирование и трассировка
- **Structured logs:** structlog / loguru, JSON формат, TraceID / SpanID.
- **Log storage:** Loki, Elastic Stack, Cloud Logging.
- **Tracing:** context propagation (FastAPI middleware, gRPC interceptors).

## VII. Валидация и тестирование
- **Тесты:** pytest unit, integration, load (Gatling / Locust), robustness (prompt mutations).
- **Schema checks:** Pydantic, OpenAPI contract.
- **Continuous evaluation:** DeepChecks / Evidently ML; regression guardrails.

## VIII. A/B‑тестирование и rollout
- Traffic splitting: header‑hash, user‑id bucketing, dynamic routing.
- Метрики uplift, sequential testing, CUPED, Bayesian bandits.
- Shadow deploy, champion/challenger, feature‑flags.

## IX. Batch и streaming inference
- **Batch:** Airflow / Prefect — CSV/Parquet → model → S3/Table.
- **Streaming:** Kafka / Redis Streams → inference worker → sink.
- **Hot‑swap:** переключение real‑time ↔ batch без достоев.

## X. Feature Store и артефакты
- **Feature Store:** Feast, Hopsworks, dbt‑metrics — online ↔ offline parity.
- **Артефакты:** MLflow, DVC, Weights & Biases — versioning моделей, эмбеддингов, данных.
- **Data lineage & governance:** tags, model cards, datasheets.

## XI. Безопасность и приватность
- **Auth & AuthZ:** OAuth2, service tokens, scope‑based ACL.
- **Rate‑limiting / WAF:** abuse‑detection, IP‑reputation, captcha‑flows.
- **Prompt safety:** prompt‑injection filters, jailbreak detection, output sandboxing.
- **PII masking & encryption:** at‑rest (EBS/KMS) & in‑transit (TLS), field‑level crypto.
- **Compliance:** SOC‑2, ISO‑27001, GDPR (DSAR, RTBF), HIPAA.

## XII. FinOps и cost‑monitoring
- **Метрики стоимости:** GPU‑hours, $/M tokens, utilization heatmaps.
- **Budget alerts:** dynamic thresholds, forecast anomalies.
- **Rightsizing:** GPU bin‑packing, spot diversity, idle‑scale‑down.
- **Chargeback & showback:** per‑team tagging, cost‑reports (FinOps dashboards).

## XIII. Чеклист типовых задач
| Сценарий | Ключевые шаги / навыки |
|---|---|
| Вкат retriever на GPU | Docker → CI/CD → Helm chart → HPA |
| Снизить p95 latency на 30 % | KV‑кэш, FlashAttention‑2, quantization, batching |
| Drift‑мониторинг | embed sampling → PSI/KL → Prometheus alert |
| A/B двух версий | traffic split → Metrics store → uplift analysis |
| Re‑index без даунтайма | batch infer → shadow index → alias switch |
| Anti‑abuse | rate‑limit middleware → JWT scope check |

## XIV. Ресурсы
- **Dockerfile best practices** — docs.docker.com.
- **Helm quickstart** — helm.sh/docs.
- **Terraform Cloud** — developer.hashicorp.com.
- **vLLM** — github.com/vllm‑project/vllm.
- **Triton Inference Server** — NVIDIA repo.
- **DeepSpeed‑MII** — deepspeed.ai.
- **Prometheus FastAPI Instrumentator** — GitHub.
- **OpenTelemetry Python** — opentelemetry.io.
- **Evidently AI drift monitor** — docs.evidentlyai.com.
- **LaunchDarkly Feature Flags** — launchdarkly.com.
- **Feast Feature Store** — feast.dev.
- **MLflow Tracking** — mlflow.org.
- **FinOps Foundation Guides** — finops.org.

---

**Краткое резюме:** файл охватывает полный MLOps‑цикл LLM‑сервиса: контейнеризация, IaC + CI/CD, опт‑драйверы latency, мониторинг, A/B, FinOps и безопасность. Дубликаты с другими планами устранены; недостающие темы (IaC, Model Registry, FinOps, prompt safety) добавлены.

