# План компетенций: Инфраструктура и хранилища данных (Senior LLM)

> **Фокус:** проектирование и эксплуатация стойкого хранилища документов и эмбеддингов для поисковых / RAG‑/ рекомендательных систем. Вопросы retrieval‑каскада, низко‑латентного inference и CI/CD разобраны в отдельных планах.

---

## I. Vector Databases
- **Движки:** FAISS (Flat, IVF, PQ, OPQ, HNSW), pgvector, Qdrant / Milvus / Weaviate, Pinecone, Vespa.  
- **Выбор индекса:** recall ↔ latency ↔ footprint; правила nlist≈√N, nprobe динамический, OPQ для памяти.  
- **Операции:** batch / streaming ingestion, metadata‑filtering, background re‑build, compaction.  
- **Шардирование и реплика:** hash‑id, semantic (k‑means), range; консистентность и merge partial‑top‑k.

## II. Elasticsearch / OpenSearch
- Кластер: shards × replicas, ILM (Hot‑Warm‑Cold) для экономии.  
- Анализаторы, `dense_vector`, `script_score`, BM25 ↔ dense fusion.  
- Режим **hybrid search:** sparse + ANN retrieval → Rank‑Fusion (RRF / weighted sum).

## III. SQL / NoSQL слой
- **PostgreSQL:** JSONB + GIN/GiST, partition by time, `pgvector` для маленьких корпусов.  
- **MongoDB:** flexible schema, TTL‑collection, Atlas Cluster.  
- **Redis + RediSearch:** read‑through caching, vector lookup, count‑min‑sketch stats.

## IV. Поточные и ETL пайплайны
- **Ingestion:** Kafka / Pulsar → transformer workers → Vector DB.  
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

