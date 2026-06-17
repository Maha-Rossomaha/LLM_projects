# План компетенций: Системы поиска и рекомендаций

> **Фокус:** навыки построения полноценных поисково‑рекомендательных систем на базе LLM‑retriever‑стека.  
> Темы инфраструктуры, CI/CD и хранилищ рассматриваются отдельно.

---

## I. Retrieval‑стек

- **Dense search**  
  📄 [Dense Search Intro](1%20Theory/1%20Retrieval/1%20Dense%20Search/1%20Dense%20Search%20Intro.md)  
  📄 [Cross vs Bi Encoder vs Twin Tower](1%20Theory/1%20Retrieval/1%20Dense%20Search/2%20Cross%20vs%20Bi%20Encoder%20vs%20Twin%20Tower.md)  
  📄 [Dimension, Pooling and Normalization](1%20Theory/1%20Retrieval/1%20Dense%20Search/4%20Dimension,%20Pooling%20and%20Normalization.md)  
  📄 [Vector Imbalance](1%20Theory/1%20Retrieval/1%20Dense%20Search/5%20Vector%20Imbalance.md)  
  📄 [Asymmetric Search](1%20Theory/1%20Retrieval/1%20Dense%20Search/6%20Asymmetric%20Search.md)  
  📄 [Embedding Drift](1%20Theory/1%20Retrieval/1%20Dense%20Search/7%20Embedding%20Drift.md)

- **Sparse signatures**  
  📄 [Direct and Inverted Indices Intro](1%20Theory/1%20Retrieval/2%20Sparse%20Search/0%20Direct%20and%20Inverted%20Indices%20Intro.md)  
  📄 [Inverted Index](1%20Theory/1%20Retrieval/2%20Sparse%20Search/0%20Inverted%20Index.md)  
  📄 [Sparse Search Intro](1%20Theory/1%20Retrieval/2%20Sparse%20Search/1%20Sparse%20Search%20Intro.md)  
  📄 [SPLADE](1%20Theory/1%20Retrieval/2%20Sparse%20Search/2%20Splade.md)  
  📄 [DocT5Query и Query Expansion](1%20Theory/1%20Retrieval/2%20Sparse%20Search/3%20DocT5Query%20%D0%B8%20Query%20Expansion.md)  
  📄 [Neural Sparse Retrieval](1%20Theory/1%20Retrieval/2%20Sparse%20Search/4%20Neural%20Sparse%20Retrieval.md)  
  📄 [Edge Cases](1%20Theory/1%20Retrieval/2%20Sparse%20Search/5%20Edge%20Cases.md)  
  📄 [Data Drift](1%20Theory/1%20Retrieval/2%20Sparse%20Search/6%20Data%20Drift.md)

- **Hybrid fusion**  
  📄 [Hybrid Search Basics](1%20Theory/1%20Retrieval/3%20Hybrid%20Fusion/1%20Hybrid%20Search%20Basics.md)  
  📄 [Rank Fusion](1%20Theory/1%20Retrieval/3%20Hybrid%20Fusion/2%20Rank%20Fusion.md)  
  📄 [Hybrid Fusion Practice](1%20Theory/1%20Retrieval/3%20Hybrid%20Fusion/3%20Hybrid%20Fusion%20Practice.md)

- **ANN indexing**  
  📄 [Multi Shard](1%20Theory/1%20Retrieval/4%20ANN/0%20Multi%20Shard.md)  
  📄 [HNSW](1%20Theory/1%20Retrieval/4%20ANN/1%20HNSW.md)  
  📄 [IVF](1%20Theory/1%20Retrieval/4%20ANN/2%20IVF.md)  
  📄 [PQ](1%20Theory/1%20Retrieval/4%20ANN/3%20PQ.md)  
  📄 [IVF-PQ](1%20Theory/1%20Retrieval/4%20ANN/4%20IVF%20PQ.md)  
  📄 [OPQ](1%20Theory/1%20Retrieval/4%20ANN/5%20OPQ.md)  
  📄 [ScaNN](1%20Theory/1%20Retrieval/4%20ANN/6%20ScaNN.md)  
  📄 [Metadata Filtering](1%20Theory/1%20Retrieval/4%20ANN/7%20Metadata%20Filtering.md)

- **Corpora Quality**  
  📄 [Deduplication Cleaning](1%20Theory/1%20Retrieval/5%20Corpus%20Quality/1%20Deduplication%20Cleaning.md)  
  📄 [Segmentation Long Docs](1%20Theory/1%20Retrieval/5%20Corpus%20Quality/2%20Segmentation%20Long%20Docs.md)  
  📄 [Small Corpora Retrieval](1%20Theory/1%20Retrieval/5%20Corpus%20Quality/3%20Small%20Corpora%20Retrieval.md)

---

## II. Reranking Cascade  
📄 [Reranking Intro](1%20Theory/2%20Reranking%20Cascade/1%20Reranking%20Intro.md)  
📄 [Base Layer](1%20Theory/2%20Reranking%20Cascade/2%20Base%20Layer.md)  
📄 [Late Interaction and ColBERT](1%20Theory/2%20Reranking%20Cascade/3%20Late%20Interaction.md)  
📄 [Cross Encoder Reranker](1%20Theory/2%20Reranking%20Cascade/4%20Cross%20Encoder%20Reranker.md)  
📄 [Multi Stage Rerank](1%20Theory/2%20Reranking%20Cascade/5%20Multi%20Stage%20Rerank.md)  
📄 [Rerank Signals](1%20Theory/2%20Reranking%20Cascade/6%20Rerank%20Signals.md)  
📄 [Rerank Metrics](1%20Theory/2%20Reranking%20Cascade/7%20Rerank%20Metrics.md)  
📄 [Latency Optimization](1%20Theory/2%20Reranking%20Cascade/8%20Latency%20Optimization.md)

---

## III. Learning‑to‑rank  
📄 [LTR Intro](1%20Theory/3%20Learning%20to%20Rank/1%20LTR%20Intro.md)  
📄 [Pointwise, Pairwise and Listwise](1%20Theory/3%20Learning%20to%20Rank/2%20Pointwise,%20Pairwise%20and%20Listwise.md)  
📄 [Cross Encoder](1%20Theory/3%20Learning%20to%20Rank/3%20Cross%20Encoder.md)  
📄 [RankNet](1%20Theory/3%20Learning%20to%20Rank/4.1%20RankNet.md)  
📄 [LambdaRank and LambdaMART](1%20Theory/3%20Learning%20to%20Rank/4.2%20LambdaRank%20and%20LambdaMart.md)  
📄 [ListNet and ListMLE](1%20Theory/3%20Learning%20to%20Rank/4.3%20ListNet%20and%20ListMLE.md)  
📄 [LightGBM and CatBoost Reranker](1%20Theory/3%20Learning%20to%20Rank/4.4%20LightGBM%20and%20CatBoost%20Reranker.md)  
📄 [Features Inputs](1%20Theory/3%20Learning%20to%20Rank/5%20Features%20Inputs.md)  
📄 [Distillation and Online Fine-Tuning](1%20Theory/3%20Learning%20to%20Rank/6%20Distillation%20and%20Online%20Fine-Tuning.md)

---

## IV. RAG и generative search  
📄 [RAG Basics](1%20Theory/4%20RAG%20and%20Generative%20Search/1%20RAG%20Basics.md)  
📄 [Chunking](1%20Theory/4%20RAG%20and%20Generative%20Search/2%20Chunking.md)  
📄 [Dynamic Context](1%20Theory/4%20RAG%20and%20Generative%20Search/3%20Dynamic%20Context.md)  
📄 [Conversational RAG](1%20Theory/4%20RAG%20and%20Generative%20Search/4%20Conversational%20RAG.md)  
📄 [Personalized Context](1%20Theory/4%20RAG%20and%20Generative%20Search/5%20Personalized%20Context.md)  
📄 [Answer Generation](1%20Theory/4%20RAG%20and%20Generative%20Search/6%20Answer%20Generation.md)  
📄 [Reliability and Security](1%20Theory/4%20RAG%20and%20Generative%20Search/7%20Reliability%20and%20Security.md)  
📄 [RAG Tools](1%20Theory/4%20RAG%20and%20Generative%20Search/8%20RAG%20Tools.md)

---

## V. Online‑feedback и bandits  
📄 [Bandits Explore-Exploit](1%20Theory/5%20Online%E2%80%91feedback%C2%A0and%20Bandits/1%20Bandits%20Explore-Exploit.md)

---

## VI. Tail‑latency  
📄 [Tail Latency in Retrieval Stack](1%20Theory/6%20Tail%20Latency/1%20Tail%20Latency%20in%20Retrieval%20Stack.md)  
📄 [Decomposition](1%20Theory/6%20Tail%20Latency/2%20Decomposition.md)  
📄 [Hedged Queries, Replication and Timeout Based Early Abort](1%20Theory/6%20Tail%20Latency/3%20Hedged%20Queries,%20Replication%20and%20Timeout%20Based%20Early%20Abort.md)  
📄 [Graceful Degradation](1%20Theory/6%20Tail%20Latency/4%20Graceful%20Degradation.md)  
📄 [Adaptive to SLA](1%20Theory/6%20Tail%20Latency/5%20Adaptive%20to%20SLA.md)  
📄 [LLM Streaming and Speculative Decoding](1%20Theory/6%20Tail%20Latency/6%20LLM%20Streaming%20and%20Speculative%20Decoding.md)  
📄 [Cache Layers](1%20Theory/6%20Tail%20Latency/7%20Cache%20Layers.md)  
📄 [Capacity Management](1%20Theory/6%20Tail%20Latency/8%20Capacity%20Management.md)

---

## VII. Embedding lifecycle  
📄 [Embeddings Drift Monitoring](1%20Theory/7%20Embedding%20Lifecycle/1%20Embeddings%20Drift%20Monitoring.md)  
📄 [Re-index and Re-embed Pipelines](1%20Theory/7%20Embedding%20Lifecycle/2%20Re-index%20and%20Re-embed%20Pipelines.md)  
📄 [Quality Guardrails](1%20Theory/7%20Embedding%20Lifecycle/3%20Quality%20Guardrails.md)  
📄 [Retrain Cadence](1%20Theory/7%20Embedding%20Lifecycle/4%20Retrain%20Cadence.md)

---

## VIII. Персонализация и cold‑start  
📄 [Basics](1%20Theory/8%20Personalization%20and%20Cold%20Start/1%20Basics.md)  
📄 [Recommendations Intro](1%20Theory/8%20Personalization%20and%20Cold%20Start/2%20Recommendations%20Intro.md)  
📄 [Cold Start Items](1%20Theory/8%20Personalization%20and%20Cold%20Start/3%20Cold%20Start%20Items.md)  
📄 [Cold Start Users](1%20Theory/8%20Personalization%20and%20Cold%20Start/4%20Cold%20Start%20Users.md)  
📄 [Two Tower, LightGCN and User-Item Symmetry](1%20Theory/8%20Personalization%20and%20Cold%20Start/5%20Two%20Tower,%20LightGCN%20and%20User-Item%20Symmetry.md)  
📄 [Recommendations Pipeline](1%20Theory/8%20Personalization%20and%20Cold%20Start/6%20Recommendations%20Pipeline.md)

---

## IX. Bias и fairness  
> 📝 Конспектов пока нет. Планируемые темы:
> - Популярность‑bias, exposure‑parity @K, calibration.
> - Митигаторы: re‑rank‑constraints, FairMatch, Δpop penalty.

---

## X. Метрики качества  
> 📝 Конспектов пока нет. Планируемые темы:
> - **Offline:** MRR, nDCG@K, Recall@K, MAP; bootstrap CI.
> - **Online:** CTR, dwell‑time, p50/p95 latency, Δbusiness metric; Sequential / CUPED / Bayesian A/B.
> - **Fairness metrics:** disparity ratio, fairness@K.
> - **RAG Metrics:** Faithfulness / Groundedness, Answer provenance, Redundancy / Diversity, Session-level метрики, Hallucination rate, Latency-aware метрики.

---

## XI. Дорожная карта компетенций

| Этап | Тема | Практика / Навыки | Результат |
| ---- | ---- | ----------------- | --------- |
| 1 | **Базовый поиск** | BM25 vs Dense (Sentence-BERT, E5, BGE), Recall/nDCG сравнение | Понимание разницы sparse vs dense |
| 2 | **Hybrid retrieval** | Weighted Sum, Reciprocal Rank Fusion, подбор весов | Рост качества за счёт гибрида |
| 3 | **ANN индексация** | FAISS IVF-PQ, HNSW, тюнинг `nlist`, `nprobe`, `M`, `ef` | Trade-off recall/latency, графики |
| 4 | **Reranking** | Bi-encoder → ColBERT → Cross-encoder | Каскад rerank с оптимизацией |
| 5 | **Learning-to-rank** | LightGBM Ranker, LambdaMART, pairwise/listwise, distillation | Освоение LTR, сравнение с нейронками |
| 6 | **RAG** | Retriever + reranker + LLM | End-to-end RAG pipeline |
| 7 | **Online feedback / Bandits** | CTR simulation, ε-greedy, Thompson Sampling, VW `--cb` | Управление explore–exploit |
| 8 | **Embedding lifecycle & Drift** | Shadow index, PSI/KL-div, alias switch | Zero-downtime reindex, drift monitor |
| 9 | **Персонализация / Cold-start** | MovieLens: Two-Tower, MF (ALS/BPR), LightGCN | Базовый рекомендатель, cold-start решения |
| 10 | **Bias & Fairness** | Popularity bias, FairMatch, disparity@K, calibration | Баланс качества и справедливости |
| 11 | **Tail latency & оптимизация** | Hedged queries, adaptive nprobe, p50/p95 latency | SLA-aware оптимизация |
| 12 | **Метрики офлайн/онлайн** | MRR, nDCG, Recall@K, bootstrap CI, CTR, A/B симуляции | Метрики качества и бизнес-эффект |

### 1. Базовый поиск
**Цель:** понять разницу между sparse и dense поиском.
- Взять датасет (MS MARCO, NQ, HotpotQA).
- Реализовать BM25 через rank_bm25 или elasticsearch.
- Подключить готовый эмбеддер (например, sentence-transformers/all-MiniLM).
- Сделать dense vs BM25 сравнение (Recall@10, nDCG@10).

### 2. Hybrid retrieval
**Цель:** научиться объединять сигналы.
- Реализовать Weighted Sum и Reciprocal Rank Fusion для BM25 + dense.
- Подобрать веса (grid search).
- Сравнить метрики: hybrid vs pure dense vs BM25.

### 3. ANN индексация
**Цель:** разобраться в FAISS / HNSW.
- Построить FAISS IVFPQ (играться с nlist, nprobe).
- Построить HNSW (играться с M, ef).
- Сравнить recall vs latency (при 100k и 1M документов).
- Сделать графики «recall@10 vs время ответа».

### 4. Reranking
**Цель:** увидеть прирост качества от reranker.
- Использовать bi-encoder для первичного поиска.
- Взять ColBERT (late interaction).
- Взять cross-encoder (например, cross-encoder/ms-marco-MiniLM-L-6-v2).
- Построить каскад: bi-encoder → ColBERT → cross.
- Измерить trade-off latency/качество.

### 5. Learning-to-rank
**Цель:** освоить классические модели.
- Взять фичи: BM25, dense sim, свежесть, длина документа.
- Обучить LightGBM Ranker (pairwise и listwise).
- Сравнить с нейронными reranker-ами.
- Попробовать online fine-tune: сэмплировать soft-labels из cross-encoder.

### 6. RAG (Retrieval-Augmented Generation)
**Цель:** собрать end-to-end RAG pipeline.
- Взять retriever (bi-encoder).
- Добавить reranker.
- Передавать top-K в LLM (например, Llama-3-Instruct).
- Реализовать dynamic context selection: кластеризация top-K и выбор релевантных.
- Замерить faithfulness (насколько ответы grounded).

### 7. Online feedback & Bandits
**Цель:** познакомиться с интерактивным улучшением.
- Смоделировать клики (CTR) на своих поисковых результатах.
- Реализовать ε-greedy и Thompson Sampling для выбора между вариантами ранжирования.
- Посмотреть динамику CTR.
- Попробовать Vowpal Wabbit --cb (contextual bandits).

### 8. Embedding lifecycle & Drift
**Цель:** научиться катить новые эмбеддинги без боли.
- Построить shadow-индекс на обновлённых эмбеддингах.
- Пустить часть запросов туда.
- Сравнить nDCG/latency.
- Использовать PSI/KL-div для оценки drift между старыми и новыми векторами.

### 9. Персонализация и cold-start
**Цель:** потрогать рекомендации руками.
- Взять MovieLens 1M.
- Обучить Two-Tower модель (user tower + item tower).
- Сравнить с Matrix Factorization (ALS/BPR).
- Решить cold-start item через meta-features (жанры).
- Решить cold-start user через popular-fallback.

### 10. Bias и fairness
**Цель:** понять этические аспекты.
- Измерить popularity bias: как часто популярные фильмы попадают в top-K.
- Реализовать re-rank с ограничением на diversity (FairMatch).
- Сравнить nDCG vs fairness@K.

### 11. Tail latency & оптимизация
**Цель:** познакомиться с системными аспектами.
- Реализовать hedged queries (дублировать запрос на два индекса, брать первый ответ).
- Попробовать adaptive nprobe в FAISS (меньше при высокой нагрузке).
- Оценить p50/p95 latency до и после оптимизации.

### 12. Метрики офлайн/онлайн
**Цель:** освоить измерения.
- Считать MRR, nDCG, Recall@K.
- Делать bootstrap доверительные интервалы.
- Симулировать A/B-тест (онлайн метрики CTR, dwell).