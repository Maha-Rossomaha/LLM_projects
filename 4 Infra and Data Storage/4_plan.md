# Инфраструктура и базы данных — Senior LLM Engineer

## 🧱 Общий контекст

Этот блок охватывает всё, что связано с хранением, поиском и масштабируемым обслуживанием эмбеддингов, документов и промежуточных результатов поиска. Он критичен для систем semantic search, RAG, и рекомендаций.

---

## 🔹 1. Vector Databases

### Основные технологии:
- **FAISS** (Flat, IVF, PQ, HNSW, OPQ, Sharding)
- **pgvector** (плагин PostgreSQL)
- **Qdrant / Milvus / Weaviate** (cloud / on-premise)
- **Pinecone / Vespa / Zilliz** (production-ready PaaS)

### Навыки:
- Выбор индекса под задачу: latency vs recall
- Хранение и обновление эмбеддингов
- Batch и streaming ingestion
- Sharding, compaction, TTL, background rebuilding
- Интеграция с RAG / hybrid search
- Фильтрация по метаданным (payload)

---

## 🔹 2. Elasticsearch / OpenSearch

### Навыки:
- Настройка кластеров (single/multi-node)
- Создание кастомных анализаторов и индексов
- Поддержка `dense_vector`, `script_score`
- Hybrid search: sparse (BM25) + dense
- DSL-запросы: bool, must/filter, scoring functions
- Настройка refresh, shard count, replicas

---

## 🔹 3. SQL / NoSQL

### PostgreSQL:
- Работа с JSONB, GIN/GiST индексами
- Оптимизация запросов (EXPLAIN ANALYZE)
- Partitioning по времени
- Использование `pgvector` для dense search

### MongoDB:
- Хранение semi-structured данных
- Индексы, TTL, шардирование

### Redis:
- Кеширование embedding-запросов
- Redis + RediSearch для vector lookup

---

## 🔹 4. ETL и потоковая обработка

### Навыки:
- ETL пайплайны: Kafka / RabbitMQ → VectorDB
- Использование Airflow / Prefect / Dagster
- Инкрементальное обновление эмбеддингов
- Интеграция ingestion → storage → indexing
- Snapshot-подходы для версионирования

---

## 🔹 5. Инфраструктура

- CI/CD пайплайны для деплоя (GitHub Actions, GitLab CI)
- Docker Compose для локальной разработки
- Helm + Kubernetes для продакшна
- Мониторинг (Prometheus, Grafana)
- Метрики: latency, update rate, hit-rate cache
- Billing-aware дизайн (ограничение запросов к Pinecone и пр.)

---

## 🔹 6. Latency-aware дизайн

- Sparse → Dense → Rerank (BM25 → dense → cross-encoder)
- Async обновление индексов
- Quantization (int8 vectors)
- Использование precomputed ANN индексов
- TTL и auto-deletion векторов по событиям

---

## 🔹 7. Безопасность и compliance

- Scoped access: tenant isolation, ACL
- GDPR/CCPA: удаление по запросу
- TTL и автосброс эмбеддингов
- Обфускация/анонимизация sensitive data
- Аудит доступа к embedding хранилищам

---

## 📌 Чеклист навыков

| Область                   | Навыки                                                                 |
|--------------------------|------------------------------------------------------------------------|
| Vector DB                | FAISS, pgvector, Qdrant, Pinecone — поиск, обновление, шардинг       |
| Hybrid Search            | ES/OpenSearch + dense/fusion search                                   |
| SQL/NoSQL                | PostgreSQL, MongoDB, Redis — модели данных, индексы, масштабирование |
| Инфраструктура           | CI/CD, Docker, Kubernetes, мониторинг, latency-оптимизация            |
| Обработка данных         | ETL, ingestion, трансформация, хранение embeddings                    |
| Безопасность и Privacy   | Удаление данных, TTL, доступ к приватным embedding                    |

---

## 📚 Ресурсы

### Vector Databases:
- [FAISS documentation](https://github.com/facebookresearch/faiss)
- [Qdrant docs](https://qdrant.tech/documentation/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Weaviate blog on hybrid search](https://weaviate.io/blog/hybrid-search-1)
- [Pinecone guide to vector search](https://docs.pinecone.io/docs/overview)

### Elasticsearch / OpenSearch:
- [ES dense vector docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- [OpenSearch knn search](https://opensearch.org/docs/latest/search-plugins/knn/)
- [Hybrid retrieval example](https://www.elastic.co/blog/introducing-hybrid-search)

### SQL / NoSQL:
- [PostgreSQL JSONB guide](https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-json/)
- [MongoDB Indexing](https://www.mongodb.com/docs/manual/indexes/)
- [Redis Search and Vector similarity](https://redis.io/docs/interact/search/)

### ETL & Infra:
- [Airflow official](https://airflow.apache.org/)
- [Docker + FAISS dev env](https://github.com/facebookresearch/faiss/issues/1736)
- [Vector indexing in production](https://sebastianraschka.com/blog/2023/approximate-nearest-neighbor-search.html)

### Security & Privacy:
- [GDPR and machine learning](https://arxiv.org/abs/1907.10320)
- [Secure vector search design (Qdrant)](https://qdrant.tech/documentation/concepts/security/)

---

