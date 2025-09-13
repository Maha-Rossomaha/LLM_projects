

6. Фичи и сигналы для rerank

Что внутри:

Dense similarity (dot/cosine).

Lexical (BM25).

Позиция (позиционные признаки).

Свежесть документа (temporal decay).

User signals (CTR, dwell-time) — только как сигнал, без ML.

Комбинация сигналов (score fusion, нормализация).

7. Метрики и оценка reranking

Что внутри:

Offline: Recall@K, nDCG, MRR, Precision@K.

Online: CTR, dwell-time, conversions.

Как оценивать вклад каждого слоя каскада.

Bootstrap доверительные интервалы, variance reduction.

Edge cases: падение качества при агрессивном уменьшении K.

8. Оптимизация latency и ресурсоёмкости

Что внутри:

Деление latency budget по слоям (пример: 50% retrieval, 30% rerank1, 20% rerank2).

Adaptive $K/M$: уменьшение числа кандидатов при высокой нагрузке.

Tail-latency mitigation: hedged queries, early exit.

Кэширование результатов rerank.

Практика: как балансировать между SLA и качеством.