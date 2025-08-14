# LLM в Search & Recommendations

## 1. Мотивация

Поисковые и рекомендательные системы сегодня всё чаще интегрируют Large Language Models (LLM), чтобы:

- **Улучшить релевантность**: переход от точного совпадения слов к пониманию смысла запроса.
- **Расширить функциональность**: поддержка сложных диалоговых запросов, многошагового поиска, генерации обоснований.
- **Персонализировать выдачу**: использование сигналов пользователя и контекста для адаптации результатов.
- **Интегрировать RAG**: объединение поиска и генерации для ответов на сложные вопросы.

---

## 2. Архитектура LLM-поиска и рекомендаций

### 2.1 Retrieval-слой

- **Dense Retrieval**: эмбеддинги из моделей вроде E5, BGE, GTE.
- **Sparse Retrieval**: BM25, SPLADE, BM25F.
- **Hybrid Retrieval**: объединение лексических и семантических сигналов.
- **ANN-индексация**: FAISS (IVF-PQ, HNSW), Qdrant, Milvus.

### 2.2 Reranking

- **Cross-encoder** (BGE-reranker, monoT5) — высокая точность, высокая стоимость.
- **Late Interaction** (ColBERT) — компромисс между скоростью и качеством.
- **Multi-stage rerank** — каскад из bi-encoder → ColBERT → cross-encoder.

### 2.3 RAG (Retrieval-Augmented Generation)

- Подбор контекста по запросу.
- Кластеризация top-K результатов.
- Отсечение нерелевантного контекста (marginal utility filtering).

### 2.4 Персонализация

- **Cold-start**: zero-shot модели, мета-признаки.
- **User embeddings**: симметричные модели user×item.
- **Feedback loop**: онлайн-обучение по кликам и реакциям.

---

## 3. Технические компоненты

- **Vector DB**: Qdrant, Milvus, Weaviate, Pinecone.
- **Search Engines**: Elasticsearch, OpenSearch (BM25 + dense vectors).
- **Feature Store**: хранение и версионирование признаков.
- **Orchestrators**: Airflow, Prefect для пайплайнов индексирования.

---

## 4. Метрики и оценка

- **Offline**: MRR, nDCG\@K, Recall\@K, MAP.
- **Online**: CTR, dwell-time, conversion rate.
- **Fairness**: disparity ratio, calibration.

---

## 5. Оптимизация и SLA

- **Tail latency mitigation**: hedged queries, adaptive nprobe.
- **Quantization**: INT8/4, PQ для снижения памяти.
- **Caching**: embedding cache, ANN result cache.

---

## 6. Применение LLM в рекомендациях

- **Semantic Matching**: сопоставление пользователя и объекта по смыслу.
- **Generative Recommendations**: создание описаний и подборок.
- **Context-Aware Recs**: учёт диалога и внешних сигналов (локация, время).

---

## 7. Вызовы и риски

- **Hallucinations**: необходимость контроля генерации.
- **Embedding Drift**: деградация качества при изменении модели.
- **Privacy**: защита пользовательских данных и RTBF.

---

## 8. Ресурсы для изучения

- Stanford CS276 (IR fundamentals)
- Pinecone Hybrid Search Guide
- ColBERTv2 (Bajaj et al., 2022)
- E5 Embeddings (Gao & Callan, 2023)
- SPLADE++ (Lin et al., 2022)
- TF-Ranking (Microsoft)
- Bandits for Recsys (O'Reilly, 2023)

