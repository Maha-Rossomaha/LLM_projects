# BAAI General Embedding (BGE)

## Общее описание

**BGE (BAAI General Embedding)** — это семейство embedding-моделей, разработанных институтом BAAI (Beijing Academy of Artificial Intelligence) с акцентом на **универсальность**, **высокое качество retrieval** и **интеграцию с LLM**. Модель особенно популярна в задачах reranking и используется как альтернатива GTE, E5 и другим universal embedding-моделям.

Основная цель — генерация **dense embeddings**, пригодных для широкого спектра downstream-задач без дополнительного обучения.

---

## Архитектура и особенности

* Основана на **BERT/RoBERTa-like encoder**
* Предобучена на **мультизадачном датасете**: semantic search, paraphrasing, QA matching, NLI
* Поддерживает **instruction-style prompting**: запросы задаются как "Represent this sentence for retrieval: ..."
* Варианты моделей: `bge-small`, `bge-base`, `bge-large` (есть версии для query/document отдельно)
* Оптимизирована под **cosine similarity** с L2-нормализацией эмбеддингов
* Реализует **rerank-friendly** структуру — хорошо сочетается с cross-encoder'ами

---

## Специфика обучения

* Используется **contrastive + supervised multitask learning**
* Для каждой задачи задаётся кастомный prompt и режим supervision
* Эмбеддинги обучаются так, чтобы быть полезными и для retrieval (top-K search), и для semantic scoring

---

## Применение

* **Dense retrieval** — первичный поиск кандидатов по запросу
* **Reranking** — генерация input-эмбеддингов перед подачей в cross-encoder
* **LLM retrieval integration** — отбор кандидатов до генерации
* **Clustering, classification** — тематическая агрегация и семантическая группировка
