# План изучения System Design

Цель раздела — не запомнить список технологий, а научиться:

- выделять требования и ограничения;
- строить корректную domain model;
- разделять данные, процессы и компоненты;
- проектировать хранение, состояния и жизненный цикл;
- объяснять consistency, retries, idempotency и failure recovery;
- проходить ML System Design интервью от начала до конца;
- аргументировать альтернативы и компромиссы.

---

# Часть 1. Core Theory

## Блок A. Архитектурное мышление

1. System Design, Domain Modeling и Data Architecture
2. Functional и Non-functional Requirements
3. Data Flow, Control Flow и Processing Pipeline
4. Компоненты, границы ответственности и зависимости
5. Modular Monolith, Services и границы декомпозиции

## Блок B. Domain Model и данные

6. Entity, Value Object, Artifact и Aggregate
7. Identity и виды идентификаторов
8. Business Keys, External IDs и Composite Keys
9. Checksum, content addressing и дедупликация
10. Provenance, Lineage и Audit Trail
11. Status Models и State Machines
12. Operational Status против Business Lifecycle
13. Versioning, immutable revisions и active pointers

## Блок C. Хранение и публикация

14. Storage Abstraction и Dependency Inversion
15. Storage Keys, Namespaces и Physical Layout
16. Metadata Database против Blob/Object Storage
17. Atomic Writes, Staging и Publication
18. Manifests, Readiness и Consistency Gates
19. Local Filesystem, HDFS и Object Storage
20. Repository и Artifact Service Patterns

## Блок D. Надёжные процессы

21. Idempotency и Idempotency Keys
22. Retries, Timeouts и Exponential Backoff
23. Partial Failures и Failure Recovery
24. SAGA: orchestration и choreography
25. Compensation и Business Rollback
26. Transactional Outbox
27. Inbox Pattern и Deduplication
28. At-least-once Delivery
29. Dead Letter Queues
30. Reconciliation и Repair Jobs
31. Optimistic Locking и конкурентные изменения

## Блок E. Production System Design

32. Queues, Workers и Backpressure
33. Horizontal Scaling и Partitioning
34. Replication и Consistency Models
35. Caching и Invalidation
36. Rate Limiting и Load Shedding
37. Circuit Breakers и Graceful Degradation
38. Logs, Metrics, Traces и Correlation IDs
39. SLI, SLO и Error Budgets
40. Security, Access Control и Data Isolation
41. Data Retention, Tombstones и Physical Deletion

---

# Часть 2. Applied Architectures

## 1. Versioned Document Ingestion Platform

1. Требования и верхнеуровневый data flow
2. IngestionPackage и граница поступления
3. SourceFile и embedded attachments
4. Canonical document model
5. Logical Document и Document Revision
6. Artifact scopes и provenance
7. Storage abstraction и artifact layout
8. Staging и atomic publication
9. Manifest и READY gate
10. Status model и lifecycle
11. Orchestrator, repositories и builders
12. Idempotency, retry и recovery
13. Serving только опубликованных данных

## 2. Artifact and Dataset Platform

- immutable datasets;
- lineage;
- snapshots;
- metadata catalog;
- retention;
- reproducibility;
- publication aliases.

## 3. Search and Indexing Platform

- lexical и dense indexes;
- index versioning;
- dual build;
- alias switch;
- freshness;
- partial reindexing;
- rollback.

## 4. Knowledge Platform and LLM Wiki

- source of truth против compiled knowledge;
- entity pages и topic pages;
- claims, evidence и citations;
- incremental compilation;
- invalidation и stale state;
- versioned wiki snapshots;
- human review;
- hybrid Wiki + RAG serving.

## 5. Model Serving Platform

- online inference;
- model registry;
- rollout;
- autoscaling;
- batching;
- fallback;
- monitoring.

## 6. Training and Evaluation Platform

- dataset snapshots;
- experiment tracking;
- distributed jobs;
- model artifacts;
- reproducibility;
- quality gates;
- promotion workflow.

---

# Часть 3. ML System Design Interviews

## Общий interview framework

Для каждой задачи нужно уметь последовательно пройти:

1. Уточнение задачи и пользователей
2. Functional requirements
3. Non-functional requirements
4. Оценка масштаба
5. Product и ML metrics
6. Данные, labels и feedback loops
7. High-level architecture
8. Offline training pipeline
9. Online inference pipeline
10. Candidate generation и ranking, если применимо
11. Storage и feature/data freshness
12. Reliability и failure modes
13. Monitoring и retraining
14. Security и privacy
15. Bottlenecks, alternatives и trade-offs

## Полные задачи

1. Design a Recommendation System
2. Design a Search System
3. Design a RAG Platform
4. Design a Feature Store
5. Design a Model Serving Platform
6. Design an ML Training Platform
7. Design an Experimentation Platform
8. Design a Document Knowledge Platform
9. Design a Content Moderation System
10. Design a Personalization Platform

Каждая задача должна содержать:

- постановку;
- уточняющие вопросы;
- пример диалога с интервьюером;
- расчёты масштаба;
- архитектурную схему;
- data и model pipelines;
- failure scenarios;
- альтернативы;
- финальный walkthrough на 35–45 минут.

---

# Часть 4. Case Studies

Case studies публикуются только в анонимизированном виде.

Планируемые синтетические кейсы:

1. Versioned Document Processing Platform
2. Knowledge Base Assistant
3. External API Gateway for Analytical Systems
4. Search Index Rebuild without Downtime
5. Reliable Background Processing Pipeline

---

# Формат каждого теоретического конспекта

1. Интуиция
2. Определения
3. Основная модель
4. Пошаговый пример
5. Альтернативы и компромиссы
6. Частые ошибки и заблуждения
7. Вопросы на собеседованиях
8. Практические задачи
9. Краткое резюме

# Формат обучения

```text
лекция
→ вопросы и задача
→ самостоятельный ответ
→ разбор ошибок
→ уточнение конспекта
→ следующая лекция
```

Ошибки из практики должны превращаться в отдельные объяснения и разделы «Частые заблуждения», а не сохраняться как персональный диалог.
