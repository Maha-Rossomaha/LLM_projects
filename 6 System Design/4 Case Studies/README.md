# Case Studies

Case studies связывают теорию с практической архитектурой, но не должны раскрывать контекст конкретной организации.

## Формат case study

1. Синтетическая постановка
2. Functional и non-functional requirements
3. Domain model
4. Data flow и control flow
5. High-level architecture
6. Storage и state model
7. Failure scenarios
8. Alternatives и trade-offs
9. Evolution path
10. Interview questions

## Правила анонимизации

Перед публикацией нужно удалить или заменить:

- названия компаний, подразделений и внутренних систем;
- реальные API endpoints;
- production hostnames;
- внутренние роли и permissions;
- реальные entity IDs;
- точный storage layout;
- уникальные названия сервисов и классов;
- детали, по которым можно восстановить бизнес-процесс.

Нейтральные замены:

```text
Internal System     → External Registry
Company Model ID    → ENTITY-123
Internal Report     → Analytical Document
Production Service  → Document Processing Service
Internal Relation   → ExternalEntityRef
```

## Что можно сохранять

Можно сохранять общие архитектурные идеи:

- immutable revisions;
- staging и atomic publication;
- manifests;
- idempotency;
- status models;
- storage abstraction;
- SAGA;
- Outbox/Inbox;
- provenance и lineage;
- knowledge compilation.

## План синтетических кейсов

1. Versioned Document Processing Platform
2. Knowledge Base Assistant
3. External API Gateway
4. Search Index Rebuild without Downtime
5. Reliable Background Processing Pipeline

## Главное правило

Case study должен позволять изучить решение, но не позволять определить, где и для кого оно было реализовано.
