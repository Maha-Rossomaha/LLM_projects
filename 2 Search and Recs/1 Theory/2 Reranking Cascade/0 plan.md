Оптимизация latency и ресурсоёмкости

Что внутри:

Деление latency budget по слоям (пример: 50% retrieval, 30% rerank1, 20% rerank2).

Adaptive $K/M$: уменьшение числа кандидатов при высокой нагрузке.

Tail-latency mitigation: hedged queries, early exit.

Кэширование результатов rerank.

Практика: как балансировать между SLA и качеством.