# ML System Design Interviews

Этот раздел предназначен не для хранения разрозненных терминов, а для полного прохождения интервью-задач.

## Общий сценарий ответа

```text
1. Clarify the problem
2. Define functional requirements
3. Define non-functional requirements
4. Estimate scale
5. Define product and ML metrics
6. Design data collection and labels
7. Draw high-level architecture
8. Design offline training
9. Design online inference
10. Design storage and freshness
11. Handle failures and degradation
12. Design monitoring and retraining
13. Discuss security and privacy
14. Identify bottlenecks and trade-offs
```

## Что должно быть в каждом полном разборе

- исходная формулировка задачи;
- уточняющие вопросы кандидата;
- предположения и оценки масштаба;
- product metrics и ML metrics;
- data generation и feedback loops;
- baseline и развитие модели;
- training pipeline;
- online serving;
- storage и caching;
- freshness и consistency;
- scaling;
- failure modes;
- observability;
- privacy и abuse cases;
- альтернативы;
- финальный walkthrough на 35–45 минут.

## План задач

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

## Как теория используется на интервью

Core Theory не пересказывается целиком. Она применяется по необходимости.

Пример:

```text
Проблема повторной обработки событий
→ Idempotency + Inbox

Несколько сервисов без общей транзакции
→ SAGA + Outbox

Новая версия индекса
→ Immutable build + validation + alias switch

Состояние background workflow
→ State Machine + retries + reconciliation
```

## Принцип хорошего ответа

На собеседовании важен не максимально сложный дизайн, а последовательность рассуждения:

```text
requirement
→ design decision
→ guarantee
→ cost
→ alternative
```

Например:

> Для первой версии достаточно modular monolith и background workers. Отдельные сервисы появятся, если parsing и indexing потребуют независимого масштабирования. Это уменьшает начальную сложность, но сохраняет модульные границы.

Такой ответ сильнее, чем необоснованный список Kafka, Kubernetes и нескольких БД.
