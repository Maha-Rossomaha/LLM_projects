# План компетенций: Системы поиска и рекомендаций (Senior LLM)

> **Фокус:** навыки построения полноценных поисково‑рекомендательных систем на базе LLM‑retriever‑стека. Темы инфраструктуры, CI/CD и хранилищ рассматриваются отдельно.

---

## I. Retrieval‑стек
- **Dense поиск:** Sentence‑Transformers, E5, BGE, DistilUSE; настройка размерности, pooling, L2‑нормализации.  
- **Sparse сигнатуры:** BM25, SPLADE, pruning stop‑words, lexical fallback.  
- **Hybrid fusion:** Reciprocal Rank Fusion (RRF), формула $w_{lex}·BM25 + w_{dense}·\cos$.  
- **ANN индексация:** IVF‑PQ (`nlist`, `nprobe`, PQ M/бит) и HNSW (`ef`, `M`), ScaNN, multi‑shard.  
- **Качество корпуса:** дедуп (MinHash/SimHash), сегментация длинных документов (sliding window, text splitting).

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
1. Достигнуть nDCG@10 ≥ 0.45 на MTEB‑QA.  
2. Снизить p95 latency до 120 ms при K = 100 (IVF‑PQ, GPU).  
3. Внедрить dynamic bandits → +3 % CTR за квартал.  
4. Автоматизировать re‑index (shadow → alias) без достоев.

## XII. Ресурсы
- **Stanford CS 276 Notes** — учебник ранжирования.  
- **Bajaj et al., ColBERT‑v2 (2022)**.  
- **Lin et al., SPLADE++ (2022)**.  
- **Gao & Callan, E5 Embeddings (2023)**.  
- **Haystack 2 Docs** — RAG pipeline.  
- **Pinecone Hybrid Search Guide (2024)**.  
- **Microsoft TF‑Ranking repo**.  
- **“Bandits for Recsys”, O’Reilly (2023)**.

