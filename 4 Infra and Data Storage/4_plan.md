# План компетенций: Инфраструктура и хранилища данных (Senior LLM)

> **Фокус:** проектирование и эксплуатация стойкого хранилища документов и эмбеддингов для поисковых / RAG‑/ рекомендательных систем.  
> Вопросы retrieval‑каскада, низко‑латентного inference и CI/CD разобраны в отдельных планах.

---

## I. Vector Databases

> 📝 Конспектов пока нет. Планируемые темы:
> - **Qdrant:** FAISS (Flat, IVF, PQ, OPQ, HNSW), pgvector, Milvus, Weaviate, Pinecone, Vespa.
> - Storage and Indexing, Collections/points/payload, Configuration and Deploy, Vector Search, Filtering and Payload Index, Hybrid and Sparse, Indexing Optimizer, Clusters/Shards/Replicas, Python client, RAG, Monitoring and SLO, Troubleshooting.
> - **Выбор индекса:** recall ↔ latency ↔ footprint; nlist≈√N, nprobe динамический, OPQ для памяти.
> - **Операции:** batch / streaming ingestion, metadata‑filtering, background re‑build, compaction.
> - **Шардирование и реплика:** hash‑id, semantic (k‑means), range; консистентность и merge partial‑top‑k.

---

## II. OpenSearch

📄 [Mappings and Analyzers](1%20Theory/2%20OpenSearch/1%20Mappings%20and%20Analyzers.md)
📄 [Indexing Bulk](1%20Theory/2%20OpenSearch/2%20Indexing%20Bulk.md)
📄 [Query DSL](1%20Theory/2%20OpenSearch/3%20Query%20DSL.md)
📄 [Scoring](1%20Theory/2%20OpenSearch/4%20Scoring.md)
📄 [Aggregations](1%20Theory/2%20OpenSearch/5%20Aggregations.md)
📄 [Security](1%20Theory/2%20OpenSearch/6%20Security.md)
📄 [Index State Management](1%20Theory/2%20OpenSearch/7%20Index%20State%20Management.md)
📄 [Snapshots](1%20Theory/2%20OpenSearch/8%20Snapshots.md)
📄 [Производительность и эксплуатация](1%20Theory/2%20OpenSearch/9%20%D0%9F%D1%80%D0%BE%D0%B8%D0%B7%D0%B2%D0%BE%D0%B4%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%BE%D1%81%D1%82%D1%8C%20%D0%B8%20%D1%8D%D0%BA%D1%81%D0%BF%D0%BB%D1%83%D0%B0%D1%82%D0%B0%D1%86%D0%B8%D1%8F.md)

---

## III. SQL / NoSQL слой

> 📝 Конспектов пока нет. Планируемые темы:
> - **PostgreSQL:** JSONB + GIN/GiST, partition by time, `pgvector` для маленьких корпусов.
> - **MongoDB:** flexible schema, TTL‑collection, Atlas Cluster.
> - **Redis + RediSearch:** read‑through caching, vector lookup, count‑min‑sketch stats.

---

## IV. Поточные и ETL пайплайны

> 📝 Конспектов пока нет. Планируемые темы:
> - **Ingestion:** Kafka / Pulsar → transformer workers → Vector DB.
> - **Distributed Compute — Spark:** DataFrame API, lazy eval, partitioning/repartition, shuffle, bucketing, broadcast‑join, window‑функции; чтение/запись Parquet (schema evolution, predicate pushdown, ZSTD/Snappy).
> - **Orchestrators:** Airflow / Prefect / Dagster — snapshot, incremental upsert, watermark‑based dedup.
> - **Версионирование:** dual‑write workflow, shadow index, atomic alias‑switch.

---

## V. Жизненный цикл индекса

> 📝 Конспектов пока нет. Планируемые темы:
> - **Drift‑детект:** PSI / KL на распределении расстояний, nClusters.
> - **Re‑index:** shadow build → canary traffic → alias cutover.
> - **TTL / expiry:** time‑boxed embeddings, auto‑purge by doc‑state.

---

## VI. Оптимизация latency

> 📝 Конспектов пока нет. Планируемые темы:
> - Pre‑computed ANN индексы, cache warm‑up, int8 quantization.
> - Adaptive `nprobe` / `efSearch`, early‑abort long postings, async refresh.
> - Tiered storage: NVMe (Hot) → HDD (Warm) → S3 (Cold snapshots).
> - **S3 101 (и MinIO):** модель S3 (bucket/ключ/версии), ACL vs bucket policy, IAM policy; Lifecycle/Glacier; Pre‑signed URLs; Multipart upload; Server‑side encryption (SSE‑S3 vs SSE‑KMS); MinIO как on‑prem аналог.

---

## VII. Безопасность и compliance

> 📝 Конспектов пока нет. Планируемые темы:
> - Multi‑tenant isolation, namespace ACL, row‑/shard‑level RBAC.
> - GDPR/CCPA: RTBF (delete API → tombstone → async scrub), audit‑trail.
> - Encryption AES‑GCM at‑rest, TLS in‑transit, secrets (KMS / Vault).
> - Privacy‑attacks mitigation: DP‑noise on vectors, access quotas.

---

## VIII. Мониторинг и наблюдаемость

> 📝 Конспектов пока нет. Планируемые темы:
> - **Метрики:** ingest lag, QPS, recall@k sample, p95 search‑latency, compaction CPU.
> - **Инструменты:** Prometheus + Grafana, OpenTelemetry traces, Loki logs.
> - Alert‑rules: ingest‑lag > SLA, shard‑hotspot, disk‑full, recall drop.

---

## IX. Дорожная карта компетенций

1. Поднять кластер Qdrant + Kafka ingestion → p95 latency < 100 мс при 10 M векторах.
2. Внедрить hybrid BM25 + dense fusion в OpenSearch, прирост nDCG ≥ 3%.
3. Реализовать shadow re‑index с alias‑switch без downtime.
4. Настроить PSI‑drift мониторинг и auto‑retrain алерт.
5. Обеспечить RTBF: 99% документов удаляются < 5 мин.

---

## X. Ресурсы

- **FAISS docs** — github.com/facebookresearch/faiss
- **Qdrant guide** — qdrant.tech/documentation
- **Weaviate hybrid search blog** — weaviate.io/blog/hybrid-search-1
- **OpenSearch K‑NN** — opensearch.org/docs/latest/search-plugins/knn/
- **pgvector** — github.com/pgvector/pgvector
- **Airflow** — airflow.apache.org
- **Dagster Incremental I/O** — docs.dagster.io
- **Data drift in embeddings** — evidentlyai.com/blog/embedding-drift-detection
- **GDPR & Machine Learning** — arxiv.org/abs/1907.10320