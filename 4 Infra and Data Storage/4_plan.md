# План компетенций: Инфраструктура и хранилища данных (Senior LLM)

> **Фокус:** проектирование и эксплуатация стойкого хранилища документов и эмбеддингов для поисковых / RAG‑/ рекомендательных систем. Вопросы retrieval‑каскада, низко‑латентного inference и CI/CD разобраны в отдельных планах.

---

## I. Vector Databases
- **Qdrant:** FAISS (Flat, IVF, PQ, OPQ, HNSW), pgvector, Qdrant / Milvus / Weaviate, Pinecone, Vespa.  
  🔗 [Storage and Indexing]()  
  🔗 [Collections, points and payload]()  
  🔗 [Configuration and Deploy]()  
  🔗 [Vector Search]()  
  🔗 [Filtering and Payload Index]()  
  🔗 [Hybrid and Sparse]()  
  🔗 [Indexing Optimizer]()  
  🔗 [Clusters, Shards and Replicas]()  
  🔗 [Python client]()  
  🔗 [RAG]()  
  🔗 [Monitoring and SLO]()  
  🔗 [Troubleshooting]()  

- **Выбор индекса:** recall ↔ latency ↔ footprint; правила nlist≈√N, nprobe динамический, OPQ для памяти.  
- **Операции:** batch / streaming ingestion, metadata‑filtering, background re‑build, compaction.  
- **Шардирование и реплика:** hash‑id, semantic (k‑means), range; консистентность и merge partial‑top‑k.

## II. OpenSearch
- **Documentation**  
  🔗 [Mappings and Analyzers]()  
  🔗 [Индексация/bulk/reindex]()  
  🔗 [Query DSL]()  
  🔗 [Scoring]()  
  🔗 [Aggregations]()  
  🔗 [Security]()  
  🔗 [Index State Management]()  
  🔗 [Snapshot & Restore]()  
  🔗 [Производительность и эксплуатация]()  

## III. SQL / NoSQL слой
- **PostgreSQL:** JSONB + GIN/GiST, partition by time, `pgvector` для маленьких корпусов.  
- **MongoDB:** flexible schema, TTL‑collection, Atlas Cluster.  
- **Redis + RediSearch:** read‑through caching, vector lookup, count‑min‑sketch stats.

## IV. Поточные и ETL пайплайны
- **Ingestion:** Kafka / Pulsar → transformer workers → Vector DB.  
- **Distributed Compute - Spark**
  - DataFrame API, lazy eval, partitioning/repartition, shuffle, bucketing, broadcast-join, window-функции; чтение/запись Parquet (schema evolution, predicate pushdown, ZSTD/ Snappy).
- **Orchestrators:** Airflow / Prefect / Dagster — snapshot, incremental upsert, watermark‑based dedup.  
- **Версионирование:** dual‑write workflow, shadow index, atomic alias‑switch.

## V. Жизненный цикл индекса
- **Drift‑детект:** PSI / KL на распределении расстояний, nClusters.  
- **Re‑index:** shadow build → canary traffic → alias cutover.  
- **TTL / expiry:** time‑boxed embeddings, auto‑purge by doc‑state.

## VI. Оптимизация latency
- Pre‑computed ANN индексы, cache warm‑up, int8 quantization.  
- Adaptive `nprobe` / `efSearch`, early‑abort long postings, async refresh.  
- Tiered storage: NVMe (Hot) → HDD (Warm) → S3 (Cold snapshots).
- S3 101 (и MinIO)
  - Модель S3: bucket/ключ/версии, ACL vs bucket policy, IAM policy для сервис-аккаунтов.
  - Lifecycle/Glacier: правила переходов, expiration, non-current versions cleanup.
  - Pre-signed URLs (upload/download) и ограничения по времени/правам.
  - Multipart upload (минимальный размер части, параллельная загрузка, возобновление).
  - Server-side encryption: SSE-S3 vs SSE-KMS, object lock/immutability.
  - MinIO как on-prem аналог (совместимость API) и checklist миграции.

## VII. Безопасность и compliance
- Multi‑tenant isolation, namespace ACL, row‑/shard‑level RBAC.  
- GDPR/CCPA: RTBF (delete API → tombstone → async scrub), audit‑trail.  
- Encryption AES‑GCM at‑rest, TLS in‑transit, secrets (KMS / Vault).  
- Privacy‑attacks mitigation: DP‑noise on vectors, access quotas.

## VIII. Мониторинг и наблюдаемость
- **Метрики:** ingest lag, QPS, recall@k sample, p95 search‑latency, compaction CPU.  
- **Инструменты:** Prometheus + Grafana, OpenTelemetry traces for query‑path, Loki logs.  
- Alert‑rules: ingest‑lag > SLA, shard‑hotspot, disk‑full, recall drop.

## IX. Дорожная карта компетенций
1. Поднять кластер Qdrant + Kafka ingestion → p95 latency < 100 мс при 10 M векторах.  
2. Внедрить hybrid BM25 + dense fusion в OpenSearch, прирост nDCG ≥ 3 %.  
3. Реализовать shadow re‑index с alias‑switch без downtime.  
4. Настроить PSI‑drift мониторинг и auto‑retrain алёрт.  
5. Обеспечить RTBF: 99 % документов удаляются < 5 мин.

## X. Ресурсы
- **FAISS docs** — github.com/facebookresearch/faiss.  
- **Qdrant guide** — qdrant.tech/documentation.  
- **Weaviate hybrid search blog** — weaviate.io/blog/hybrid-search-1.  
- **OpenSearch K‑NN** — opensearch.org/docs/latest/search-plugins/knn/.  
- **pgvector** — github.com/pgvector/pgvector.  
- **Airflow** — airflow.apache.org.  
- **Dagster Incremental I/O** — docs.dagster.io.  
- **Data drift in embeddings** — evidentlyai.com/blog/embedding-drift-detection.  
- **GDPR & Machine Learning** — arxiv.org/abs/1907.10320.

