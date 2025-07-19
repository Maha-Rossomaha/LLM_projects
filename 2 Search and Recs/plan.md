# Системы рекомендаций и поиск — Полный набор требований (Senior LLM)

## Retrieval (извлечение релевантных кандидатов)

### Что ты должен уметь:
- Semantic search (sentence-transformers, OpenAI, MTEB).
- Dense retrieval: FAISS, pgvector, Milvus, Qdrant, Weaviate.
- Hybrid search: BM25 + dense, fusion методов.
- Semantic IDs: dense hashing, quantization.
- Asymmetric search (query ≠ doc) — ColBERT, Twin-Tower.

### Знания:
- Tokenization, embedding space, triplet loss, similarity metrics (cosine, dot, IP).
- Архитектуры: Siamese, Dual Encoder, ColBERT, Sparse-Dense fusion.

### Ресурсы:
- [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) — для оценки моделей.
- [FAISS tutorial (Facebook)](https://github.com/facebookresearch/faiss/wiki) — официальное руководство.
- [ColBERTv2 (Stanford)](https://github.com/stanford-futuredata/ColBERT) — продвинутый dense retrieval.
- [Qdrant Docs](https://qdrant.tech/documentation/) — API и концепции production vector search.

---

## Reranking

### Что ты должен уметь:
- Cross-encoder (bge-reranker, ms-marco).
- Cascade / multi-stage ranking.
- Feature engineering для LTR.
- Loss-функции: RankNet, LambdaMART, ListNet.

### Знания:
- LTR модели: LightGBM, CatBoost, TF-Ranking, Reformer.
- Инструкционные rerankеры ("Pick the most relevant").

### Ресурсы:
- [Microsoft MS MARCO Dataset](https://microsoft.github.io/msmarco/) — основной бенчмарк.
- [bge-reranker](https://huggingface.co/BAAI/bge-reranker-large) — state-of-the-art reranker.
- [TF-Ranking (Google)](https://github.com/tensorflow/ranking) — реализация listwise/pairwise ranking.
- [LambdaMART в LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective) — документированный пример.

---

## RAG (retrieval-augmented generation)

### Что ты должен уметь:
- Построение RAG пайплайна (retriever → reranker → LLM).
- Обучение retriever+reader (InstructRAG).
- Prompting с retrieved контекстом.
- Multi-hop RAG, memory-augmented модели.

### Инфра:
- LangChain, LlamaIndex, Haystack.
- HuggingFace RAG, OpenAI APIs, Claude, Mistral, bge-m3.

### Ресурсы:
- [Haystack Docs](https://docs.haystack.deepset.ai/) — фреймворк для RAG.
- [LangChain Retrieval Cookbook](https://python.langchain.com/docs/modules/data_connection/retrievers/) — практические рецепты.
- [RAG от HuggingFace](https://huggingface.co/docs/transformers/model_doc/rag) — встроенная поддержка.
- [InstructRAG (Meta)](https://github.com/facebookresearch/instructor-embedding) — новая парадигма supervised retriever + reader.

---

## Feedback loops & Learning to Rank

### Что ты должен уметь:
- Логирование кликов, лайков, scrolls.
- Online learning, bandit feedback.
- Fine-tuning по пользовательским событиям.
- Метрики satisfaction / engagement uplift.

### Методологии:
- DLA, Inverse Propensity Weighting.
- A/B: CUPED, sequential, Bayesian AB.
- Hard negatives, лог-основы обучения.

### Ресурсы:
- [Bandits Book](https://www.andrew.cmu.edu/course/10-702/spring2020/notes/bandits.pdf) — теоретическая база.
- [Google RecSys YouTube Series](https://www.youtube.com/@GoogleRecSys) — практики Google.
- [CUPED Explained (Booking.com)](https://booking.ai/cuped-in-a-nutshell-4856babb3412) — корректировка дисперсии в AB.
- [Implicit Feedback LTR](https://arxiv.org/abs/1003.5956) — seminal paper от Hu et al.

---

## Engineering и latency

### Что ты должен уметь:
- Кэширование embeddings, батчинг, prefetch.
- Approximate search (IVF, HNSW, PQ).
- Async stack: FastAPI + asyncio.
- Профилирование latency, трейсинг.

### Инструменты:
- FastAPI, TorchScript, TensorRT.
- Prometheus, Grafana, OpenTelemetry.

### Ресурсы:
- [FAISS IVF+PQ Guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) — как выбрать index.
- [Async Python Patterns](https://realpython.com/async-io-python/) — подробный разбор.
- [OpenTelemetry Guide](https://opentelemetry.io/docs/) — стандарт для tracing.

---

## Контентные рекомендации

### Что ты должен уметь:
- Embeddings для item/user.
- Two-tower, user/item tower.
- Short-term/long-term интересы.
- Seq2Seq модели: Recformer, SASRec.

### Архитектуры:
- DeepFM, DIN, DLRM, DSSM, MIND, Graph-based.

### Ресурсы:
- [Recbole](https://github.com/RUCAIBox/RecBole) — библиотека с >60 моделей.
- [DLRM (Meta)](https://github.com/facebookresearch/dlrm) — industrial-scale модель.
- [SASRec paper](https://arxiv.org/abs/1808.09781) — self-attention в рекоммендациях.
- [GraphRec (SIGIR)](https://dl.acm.org/doi/10.1145/3209978.3210002) — моделирование связей.

---

## Метрики и оценка

### Что ты должен уметь:
- MRR, nDCG@K, Recall@K, MAP, Precision@K.
- Различие offline / online метрик.
- Distribution shift, CTR drop-off.
- Error annotation pipelines.

### Ресурсы:
- [Recommender Metrics Cheat Sheet](https://towardsdatascience.com/recommender-system-metrics-explained-examples-evaluation-369e9f00e6f5)
- [Evaluation metrics in IR](https://github.com/UKPLab/sentence-transformers/blob/master/examples/evaluation/README.md)
- [Interpreting CTR](https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-click-through-rate-ctr/) — на реальных данных.

---

## Data & Storage

### Что ты должен уметь:
- FAISS, Qdrant, Pinecone, Weaviate, pgvector.
- Elasticsearch, OpenSearch.
- Гибридные хранилища (pgvector + Postgres).
- Обновление embeddings без потери индекса.

### Ресурсы:
- [Pinecone Docs](https://docs.pinecone.io/) — production-ready API.
- [pgvector project](https://github.com/pgvector/pgvector) — SQL-native vector DB.
- [Weaviate 101](https://weaviate.io/developers/weaviate/current/index.html) — очень удобный SDK.

---

## LLM-specific Search Tricks

- Semantic Search + LLM Self-Reranking.
- Chain-of-Thought reasoning по retrieved контексту.
- Query Rewriting через LLM (например, с prompt: “Переформулируй как поисковый запрос”).
- Retrieval-Conditioned Generation + Source Attribution.

### Ресурсы:
- [ReAct prompting](https://arxiv.org/abs/2210.03629)
- [Self-RAG](https://arxiv.org/abs/2308.03281) — генерация retrieval-aware вопросов.
- [Query Rewriting for Retrieval](https://arxiv.org/abs/2305.13519)

---

## Поддержка продакшена

- CI/CD пайплайны для retriever / reranker.
- Мониторинг качества / деградации метрик.
- Canary rollout, A/B инфраструктура.
- Визуализация embedding space (например, через UMAP, t-SNE).

### Ресурсы:
- [Featureform](https://www.featureform.com/blog/embedding-monitoring-in-production) — про мониторинг эмбеддингов.
- [t-SNE, UMAP for vectors](https://distill.pub/2016/misread-tsne/) — как не напортачить с визуализацией.
- [Arize AI](https://arize.com/) — инструмент для мониторинга ML/LLM в продакшене.
