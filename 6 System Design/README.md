# System Design

Этот раздел разделяет четыре разных типа материала, которые часто ошибочно смешиваются в одном «плане по system design»:

1. **Core Theory** — универсальные архитектурные понятия и паттерны.
2. **Applied Architectures** — связные архитектуры типовых платформ.
3. **ML System Design Interviews** — полные прохождения задач в формате собеседования.
4. **Case Studies** — анонимизированные практические разборы.

## Зачем такое разделение

Список терминов вроде «Kafka, HDFS, ANN, retries, SAGA, feature store» сам по себе не учит проектировать системы. Для уверенного system design нужны три разных навыка:

- знать строительные блоки и ограничения;
- понимать, как соединять их в работающую архитектуру;
- уметь последовательно вести интервью: от требований до компромиссов и отказов.

Поэтому теория, прикладная архитектура и интервью-задача не должны быть одним и тем же конспектом.

## Структура

```text
6 System Design/
├── 1 Core Theory/
├── 2 Applied Architectures/
├── 3 ML System Design Interviews/
└── 4 Case Studies/
```

### 1 Core Theory

Здесь находятся темы, которые применимы не только к ML:

- system design, domain modeling и data architecture;
- identity и идентификаторы;
- status models и state machines;
- storage abstraction;
- транзакции и consistency;
- idempotency, retries и timeouts;
- SAGA и compensation;
- Outbox, Inbox и дедупликация;
- failure recovery и reconciliation;
- масштабирование, наблюдаемость и безопасность.

### 2 Applied Architectures

Здесь теория собирается в целые системы:

- versioned document ingestion platform;
- artifact и dataset platform;
- search и indexing platform;
- knowledge platform и LLM Wiki;
- model serving platform;
- training и evaluation platform.

### 3 ML System Design Interviews

Каждая тема здесь должна быть не перечнем технологий, а полным прохождением интервью:

```text
requirements
→ scale estimation
→ metrics
→ data and labels
→ training pipeline
→ online serving
→ storage
→ reliability
→ monitoring
→ trade-offs
```

Примеры будущих задач:

- Design a Recommendation System;
- Design a Search System;
- Design a RAG Platform;
- Design a Feature Store;
- Design a Model Serving Platform;
- Design an ML Training Platform;
- Design an Experimentation Platform;
- Design a Document Knowledge Platform.

### 4 Case Studies

Case study показывает, как решения проявляются в реальном коде и эксплуатации. Публичные case studies должны быть синтетическими и не раскрывать внутренние названия, API, роли, идентификаторы, инфраструктуру или бизнес-процессы конкретной организации.

## Правило анонимизации

Публичный конспект должен сохранять архитектурную идею, но не рабочий контекст.

Используем нейтральные сущности:

```text
External Registry
ExternalEntityRef
Business Entity
Document Processing Platform
Logical Document
Document Revision
ENTITY-123
```

Не публикуем:

- названия внутренних систем и организаций;
- реальные URL, API paths и роли;
- production storage layout;
- реальные идентификаторы;
- внутренние статусы, если они могут раскрывать процесс;
- фрагменты рабочего кода без отдельной очистки.

Универсальные термины вроде `IngestionPackage`, `SourceFile`, `ParsedDocument`, `ArtifactStorage`, `ReportRevision`, `Manifest` допустимы, если пример полностью синтетический.

## Как читать раздел

Рекомендуемый путь:

```text
Core Theory
→ Applied Architecture
→ Interview Walkthrough
→ Case Study
```

Например:

```text
Identity and Identifiers
→ Versioned Document Ingestion Platform
→ Design a Document Knowledge Platform
→ Synthetic Document Platform Case Study
```

Так одна и та же идея рассматривается на четырёх уровнях: определение, архитектура, интервью и практика.
