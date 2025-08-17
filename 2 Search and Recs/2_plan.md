# План компетенций: Системы поиска и рекомендаций (Senior LLM)

> **Фокус:** навыки построения полноценных поисково‑рекомендательных систем на базе LLM‑retriever‑стека. Темы инфраструктуры, CI/CD и хранилищ рассматриваются отдельно.

---

## I. Retrieval‑стек
- **Dense поиск:** Sentence‑Transformers, E5, BGE, DistilUSE; настройка размерности, pooling, L2‑нормализации.  
- **Sparse сигнатуры:** BM25, SPLADE, pruning stop‑words, lexical fallback.  
- **Hybrid fusion:** Reciprocal Rank Fusion (RRF), формула $w_{lex}·BM25 + w_{dense}·\cos$.  
- **ANN индексация:** IVF‑PQ (`nlist`, `nprobe`, PQ M/бит) и HNSW (`ef`, `M`), ScaNN, multi‑shard.  
- **Качество корпуса:** дедупликация (MinHash/SimHash), сегментация длинных документов (sliding window, text splitting).

## II. Каскад reranking
- **Late‑interaction** (ColBERT) vs **Cross‑encoder** (bge‑reranker) — trade‑off latency ↔ quality.  
- Мульти‑этап: bi‑encoder → ColBERT → cross; динамический $K/M$.  
- **Фичи для rerank:** dense sim, BM25, позиция, свежесть, user signals.

## III. Learning‑to‑rank
- **Модели:** LambdaMART, RankNet, ListNet, TF‑Ranking, LightGBM‑ranker.  
- **Pairwise vs listwise:** когда что.  
- **On‑line fine‑tune:** soft‑labels из кликов, knowledge distillation из cross‑encoder.  
- **Distillation:**  
  – *Hard-label distillation:* ученик учится предсказывать финальный выбор учителя (например, top‑1).  
  – *Soft-label distillation:* ученик приближает распределение логитов (softmax) учителя.  
  → Soft‑label distillation чаще даёт лучшие результаты, особенно при многоклассовых и ранжирующих задачах.

## IV. RAG и generative search
- Pipeline: retriever → reranker → LLM‑генератор.  
- **Dynamic context selection:** кластеризация top‑K, маржинальная полезность.  
- Multi‑hop / memory‑augmented RAG; answer re‑ranking.
- Conversational retrieval:
  - Multi-turn query refinement (пошаговое уточнение запросов с учётом истории диалога).
  - Context carry-over — сохранение и использование диалогового контекста для формирования следующего запроса в retriever.
  - Query planning: разбиение задачи на подзапросы и их последовательное выполнение.
  - Интеграция с reranker’ами и multi-hop RAG.
  - Метрики: groundedness на сессии, cumulative recall@K, доля релевантных уточнений.



## V. Online‑feedback и bandits
- **Explore‑Exploit:** Thompson Sampling, ε‑greedy, UCB — ротация кандидатов.  
- **Counterfactual LTR:** IPS, DLA, SNIPS, off‑policy evaluation.  
- **Инструменты:** Vowpal Wabbit `--cb`, Meta BanditPAM.

## VI. Tail‑latency
- Hedged queries, replication, timeout‑based early abort.  
- **Adaptive nprobe / efSearch** (IVF/HNSW) под SLA.  
- Спекулятивная генерация (draft + verifier), streaming LLM.  
- Кэш‑слои: embedding, ANN‑topK, rerank scores.

## VII. Embedding lifecycle
- Drift‑monitor: PSI, KL‑div, embedding norm shift.  
- Shadow‑index и alias‑switch для zero‑downtime re‑index.  
- Периодичность re‑train / re‑embed; хранение версий.

## VIII. Персонализация и cold‑start
- **Cold‑start items:** meta‑features, zero‑shot E5/BGE‑M3, graph‑propagation.  
- **Cold‑start users:** popular‑fallback, persona embeddings, federated warm‑up.  
- Two‑tower + LightGCN для user×item симметрии.

## IX. Bias и fairness
- Популярность‑bias, exposure‑parity @K, calibration.  
- Митигаторы: re‑rank‑constraints, FairMatch, Δpop penalty.

## X. Метрики качества
- **Offline:** MRR, nDCG@K, Recall@K, MAP; bootstrap CI.  
- **Online:** CTR, dwell‑time, p50/p95 latency, Δbusiness metric; Sequential / CUPED / Bayesian A/B.  
- **Fairness metrics:** disparity ratio, fairness@K.

## XI. Дорожная карта компетенций

| Этап | Тема                            | Практика / Навыки                                             | Результат                                 |
| ---- | ------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 1    | **Базовый поиск**               | BM25 vs Dense (Sentence-BERT, E5, BGE), Recall/nDCG сравнение | Понимание разницы sparse vs dense         |
| 2    | **Hybrid retrieval**            | Weighted Sum, Reciprocal Rank Fusion, подбор весов            | Рост качества за счёт гибрида             |
| 3    | **ANN индексация**              | FAISS IVF-PQ, HNSW, тюнинг `nlist`, `nprobe`, `M`, `ef`       |  Trade-off recall/latency, графики         |
| 4    | **Reranking**                   | Bi-encoder → ColBERT → Cross-encoder                          | Каскад rerank с оптимизацией              |
| 5    | **Learning-to-rank**            | LightGBM Ranker, LambdaMART, pairwise/listwise, distillation  | Освоение LTR, сравнение с нейронками      |
| 6    | **RAG**                         | Retriever + reranker + LLM                                    | End-to-end RAG pipeline                   |
| 7    | **Online feedback / Bandits**   | CTR simulation, ε-greedy, Thompson Sampling, VW `--cb`        | Управление explore–exploit                |
| 8    | **Embedding lifecycle & Drift** | Shadow index, PSI/KL-div, alias switch                        | Zero-downtime reindex, drift monitor      |
| 9    | **Персонализация / Cold-start** | MovieLens: Two-Tower, MF (ALS/BPR), LightGCN                  | Базовый рекомендатель, cold-start решения |
| 10   | **Bias & Fairness**             | Popularity bias, FairMatch, disparity\@K, calibration         | Баланс качества и справедливости          |
| 11   | **Tail latency & оптимизация**  | Hedged queries, adaptive nprobe, p50/p95 latency              | SLA-aware оптимизация                     |
| 12   | **Метрики офлайн/онлайн**       | MRR, nDCG, Recall\@K, bootstrap CI, CTR, A/B симуляции        | Метрики качества и бизнес-эффект          |

### 1. Базовый поиск

**Цель**: понять разницу между sparse и dense поиском.
- Взять датасет (MS MARCO, NQ, HotpotQA).  
- Реализовать BM25 через rank_bm25 или elasticsearch.  
- Подключить готовый эмбеддер (например, sentence-transformers/all-MiniLM).  
- Сделать dense vs BM25 сравнение (Recall@10, nDCG@10).

### 2. Hybrid retrieval

**Цель**: научиться объединять сигналы.
- Реализовать Weighted Sum и Reciprocal Rank Fusion для BM25 + dense.  
- Подобрать веса (grid search).  
- Сравнить метрики: hybrid vs pure dense vs BM25.

### 3. ANN индексация

**Цель**: разобраться в FAISS / HNSW.

- Построить FAISS IVFPQ (играться с nlist, nprobe).  
- Построить HNSW (играться с M, ef).  
- Сравнить recall vs latency (при 100k и 1M документов).  
- Сделать графики «recall@10 vs время ответа».

### 4. Reranking

**Цель**: увидеть прирост качества от reranker.

- Использовать bi-encoder для первичного поиска.  
- Взять ColBERT (late interaction).  
- Взять cross-encoder (например, cross-encoder/ms-marco-MiniLM-L-6-v2).  
- Построить каскад: bi-encoder → ColBERT → cross.  
- Измерить trade-off latency/качество.

### 5. Learning-to-rank

**Цель**: освоить классические модели.

- Взять фичи: BM25, dense sim, свежесть, длина документа.  
- Обучить LightGBM Ranker (pairwise и listwise).  
- Сравнить с нейронными reranker-ами.  
- Попробовать online fine-tune: сэмплировать soft-labels из cross-encoder.

### 6. RAG (Retrieval-Augmented Generation)

**Цель**: собрать end-to-end RAG pipeline.

- Взять retriever (bi-encoder).  
- Добавить reranker.  
- Передавать top-K в LLM (например, Llama-3-Instruct).  
- Реализовать dynamic context selection: кластеризация top-K и выбор релевантных.  
- Замерить faithfulness (насколько ответы grounded).

### 7. Online feedback & Bandits

**Цель**: познакомиться с интерактивным улучшением.

- Смоделировать клики (CTR) на своих поисковых результатах.  
- Реализовать ε-greedy и Thompson Sampling для выбора между вариантами ранжирования.  
- Посмотреть динамику CTR.  
- Попробовать Vowpal Wabbit --cb (contextual bandits).

### 8. Embedding lifecycle & Drift

**Цель**: научиться катить новые эмбеддинги без боли.

- Построить shadow-индекс на обновлённых эмбеддингах.  
- Пустить часть запросов туда.  
- Сравнить nDCG/latency.  
- Использовать PSI/KL-div для оценки drift между старыми и новыми векторами.

### 9. Персонализация и cold-start

**Цель**: потрогать рекомендации руками.

- Взять MovieLens 1M.  
- Обучить Two-Tower модель (user tower + item tower).  
- Сравнить с Matrix Factorization (ALS/BPR).  
- Решить cold-start item через meta-features (жанры).  
- Решить cold-start user через popular-fallback.

### 10. Bias и fairness

**Цель**: понять этические аспекты.

- Измерить popularity bias: как часто популярные фильмы попадают в top-K.  
- Реализовать re-rank с ограничением на diversity (FairMatch).  
- Сравнить nDCG vs fairness@K.

### 11. Tail latency & оптимизация

**Цель**: познакомиться с системными аспектами.

- Реализовать hedged queries (дублировать запрос на два индекса, брать первый ответ).  
- Попробовать adaptive nprobe в FAISS (меньше при высокой нагрузке).  
- Оценить p50/p95 latency до и после оптимизации.

### 12. Метрики офлайн/онлайн

**Цель**: освоить измерения.

- Считать MRR, nDCG, Recall@K.  
- Делать bootstrap доверительные интервалы.  
- Симулировать A/B-тест (онлайн метрики CTR, dwell).