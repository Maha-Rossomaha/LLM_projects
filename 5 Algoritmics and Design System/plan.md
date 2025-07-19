# Algoritmics & System Design для LLM‑поиска и рекомендательных систем

Ниже — “расшивка” блока **Algoritmics & System Design** применительно к LLM‑поиску и рекомендательным системам (retrieval + reranking + RAG + hybrid). Структура: *ядерные алгоритмы* → *индексация и хранение* → *retrieval pipeline & ranking* → *оптимизация производительности* → *качество и эволюция* → *архитектурные паттерны* → *защита данных и надежность*. Внутри каждого: что знать (концепты), что уметь (практика/задачи), типичные вопросы / акценты для интервью.

---

## 1. Трансформеры и представления (Representation Layer)
**Что знать:** внутренняя математика self-attention (scaled dot-product, матричные умножения, сложность O(n²)), позиционное кодирование (absolute, rotary, ALiBi), различия encoder-only (BERT), decoder-only (GPT) и encoder-decoder схем, плюсы полной self-attention против рекуррентных сетей, ключевые бутылочные горлышки (memory bandwidth, KV-cache).  
**Что уметь:** объяснять trade-off между длиной последовательности, размером модели и латентностью; выбирать уровни адаптации (Adapter/LoRA слои) для retrieval; профилировать inference (torch.profiler, nvtx).  
**Интервью:** “почему Transformer быстрее RNN”, “как уменьшить квадратичную сложность”, “где узкие места при inference”.

## 2. Embeddings и взаимодействие (Early vs Late Interaction)
**Что знать:** bi-encoder (независимое кодирование), cross-encoder (совместное кодирование пары), late interaction (ColBERT: токеновые матрицы + MaxSim агрегирование); компромиссы качество/стоимость/память; влияние размерности и нормализации.  
**Что уметь:** проектировать многоступенчатый стек: (a) дешевый ANN на плотных эмбеддингах → (b) late interaction reranker → (c) узкий cross-encoder для top‑N; выбирать k на этапах.  
**Интервью:** “оптимальный k на каждом этапе?”, “как хранить и сжимать токеновые эмбеддинги ColBERT?”, “distillation cross→bi”.

## 3. Лексический сигнал + гибрид
**Что знать:** Probabilistic Relevance Framework, формула BM25 (k1, b), BM25F (полевой вес), term saturation, document length normalization; почему гибрид (BM25 + dense) устойчив к OOV, редким термам, морфологии.  
**Что уметь:** взвешивание: score = w_lex * norm(BM25) + w_dense * norm(cosine); поиск оптимальных весов (grid, Bayesian opt); ранговые фузии (RRF, Borda).  
**Интервью:** “как объединить ранги?”, “когда гибрид хуже чисто dense?”, “как нормализовать скор BM25 vs cosine?”.

## 4. Approximate Nearest Neighbor (ANN) — алгоритмы
**Что знать:** структуры: IVF / IVFPQ (инвертированные списки + Product Quantization), HNSW (многослойный навигационный граф), DiskANN, LSH / hashing, ScaNN; их сложность по памяти, update cost, recall/latency trade-offs, influence параметров (nlist, nprobe, efSearch).  
**Что уметь:** выбирать конфиг FAISS: nlist (≈√N), nprobe (runtime vs recall), PQ параметры (M, bits per subvector), OPQ rotation; сравнивать HNSW vs IVFPQ при разных QPS.  
**Интервью:** “почему HNSW лучше на малом QPS?”, “как уменьшить память при миллиарде векторов?”, “что дает OPQ?”.

## 5. Vector DB & распределённость
**Что знать:** поддерживаемые расстояния (L2, cosine, inner product), отличие специализированных движков (Pinecone, Weaviate, Milvus) от pgvector / Elasticsearch dense vectors; компромисс ACID/транзакций vs скорость; схемы шардирования: hash of id, semantic (k-means partition), range; репликация и консистентность.  
**Что уметь:** проектировать распределенный слой: fan-out запрос → частичные top‑k → глобальный merge; конфигурация шардинга так, чтобы минимизировать cross-shard traffic; коллокация метаданных и векторов.  
**Интервью:** “как предотвратить hotspot shard?”, “как агрегировать top‑k из шардов без сортировки всех результатов?”.

## 6. Индексация, обновления и Embedding Lifecycle
**Что знать:** embedding drift (изменение семантики при апдейте модели или доменных данных), необходимость переиндексации; стратегии dual-write (старый + новый индекс), alias switch, shadow mode.  
**Что уметь:** pipeline: (1) сбор нового индекса в стороне, (2) канареечный трафик, (3) сравнение метрик (MRR, nDCG, latency), (4) атомарное переключение алиаса, (5) retire старого индекса по SLA.  
**Интервью:** “как катить новую версию эмбеддингов без даунтайма?”, “как обнаружить drift автоматически?”.

## 7. Дедупликация и качество корпуса
**Что знать:** MinHash / LSH для Jaccard, SimHash для near-duplicates, к‑shingles (n-граммы символов/слов), trade-off ‘ложные срабатывания vs пропуски’; влияние дубликатов на переобучение и leakage.  
**Что уметь:** вычисление сигнатур, выбор количества хешей и бэндов (banding technique) для заданной вероятности; периодический дедуп в ingestion; хранение fingerprint-таблиц (LSM + компактные битовые подписи).  
**Интервью:** “как масштабировать MinHash на десятки миллиардов шинглов?”, “как хранить fingerprints эффективно?”.

## 8. Retrieval → Reranking каскад
**Что знать:** стандартный каскад: (a) первичный lexical/dense retrieval (top K≈1k–10k); (b) lightweight bi-encoder / late interaction → top M; (c) heavy cross-encoder rerank → top N; (d) LLM reasoning / answer synthesis; метрики и latency budget на каждом слое.  
**Что уметь:** разбивать бюджет миллисекунд (пример: 50% ANN, 30% rerank1, 20% rerank2); адаптивный K/M при перегрузке; удержание качества при деградации (пропуск heavy rerank).  
**Интервью:** “как уменьшить tail latency?”, “как адаптируешь K под нагрузкой?”.

## 9. Рерэнкеры: Cross-Encoder vs Late Interaction
**Что знать:** Cross-encoder — полное взаимодействие токенов, высокая точность, высокая стоимость; Late interaction (ColBERT) — хранит пер-токенные векторы, MaxSim, компромисс; distillation и knowledge transfer.  
**Что уметь:** динамический выбор модели по SLA; компрессия токеновых векторов (product quantization, pruning); distill cross → bi для ускорения.  
**Интервью:** “когда cross-encoder не окупается?”, “как сжать ColBERT embeddings?”.

## 10. Метрики и оценка (офлайн/онлайн)
**Что знать:** MRR, nDCG (логарифмическая скидка), Recall@K, Precision@K, MAP; отличие офлайн оценок (human relevance judgements) от онлайн (CTR, dwell, conversions); интерпретация дельт и доверительных интервалов.  
**Что уметь:** evaluation harness: фиксированный золотой набор, bootstrap доверительные интервалы, variance reduction (stratified sampling); калибровка гибридного весового коэффициента по корреляции с nDCG.  
**Интервью:** “почему nDCG лучше Precision@k?”, “как получить доверительный интервал для MRR?”.

## 11. Латентность и оптимизация inference
**Что знать:** источники задержек: attention матричное умножение, memory bandwidth, ANN traversal, network hops; техники: quantization (INT8/4), симметричная/асимметричная PQ, batch fusion, early exit (confidence margin), speculative decoding.  
**Что уметь:** измерять p50/p95/p99; adaptive nprobe (меньше при высокой загрузке), разделение warm/cold path, профилировать tail latency; оценивать влияние quantization на MRR/nDCG (A/B).  
**Интервью:** “что делаешь при росте tail latency?”, “как оценить влияние INT8 на качество?”.

## 12. Кэширование и память
**Что знать:** виды кэшей: (a) embedding cache (повторяющиеся user queries), (b) ANN result cache (top‑K doc ids), (c) feature cache (BM25 частоты, токенизации), (d) KV‑кэш для LLM. Политики: LRU, LFU, LFUDA, TinyLFU.  
**Что уметь:** версионирование embedding space для invalidation; composite keys (query text + модель + версия индекса); измерение hit ratio и его влияние на SLA.  
**Интервью:** “как избежать stale результатов после обновления документа?”, “какую политику выберешь при высокой кардинальности?”.

## 13. Масштабирование и распределённый дизайн
**Что знать:** шардирование индекса, репликация (read replicas для p99), федеративный поиск (fan‑out → partial merge через k-way heap), backpressure, rate limiting, hedged requests, circuit breaker.  
**Что уметь:** k-way merge O(k log N), динамическое отключение “дорогих” слоёв при деградации, балансировку по количеству векторов vs по QPS.  
**Интервью:** “алгоритм слияния top‑k из N шардов”, “как выдержать всплеск QPS ×10?”.

## 14. Data Freshness & Consistency
**Что знать:** eventual vs read-after-write, влияние staleness на пользовательские метрики, SLA на индексирование (например 95% документов индексированы <5 минут).  
**Что уметь:** Dual-write pipeline: документ → лог → async embedding → temp index → alias switch; метрики: ingest lag, time-to-first-availability.  
**Интервью:** “как измерить freshness?”, “что делать при лаге очереди > SLA?”.

## 15. Версионирование моделей
**Что знать:** семантические версии эмбеддеров; стратегии rollout: A/B, shadow, champion/challenger; embedding drift как триггер переиндексации.  
**Что уметь:** держать две версии индекса за фиче-флагом; собирать ΔnDCG, Δlatency; быстро откатывать alias.  
**Интервью:** “как безопасно откатить?”, “что логируешь для post‑mortem?”.

## 16. Обогащение признаков и Feature Store
**Что знать:** лексические features (BM25, term proximity), позиционные признаки, поле-специфические веса, пользовательские поведенческие сигналы (CTR, dwell), diversity/novelty, temporal decay.  
**Что уметь:** feature store с версионированием, TTL, on-demand вычисление редких признаков; offline → online feature parity тесты.  
**Интервью:** “какие признаки перед cross-encoder?”, “как добавить personalization сигнал?”.

## 17. Управление полнотекстовыми и векторными сигналами
**Что знать:** нормализация скорингов (z-score, min-max, rank-based), score fusion техники (weighted sum, Reciprocal Rank Fusion, Borda), calibration.  
**Что уметь:** рассчитывать корреляции между компонентными скоровыми сигналами; автоматический подбор весов (Bayesian Optimization, grid, coordinate descent).  
**Интервью:** “как нормализовать BM25 vs cosine?”, “когда использовать RRF?”.

## 18. Специфика RAG
**Что знать:** failure modes: retrieval leakage (несвязанные документы), redundancy, hallucination при слабом grounding; влияние длины контекста (context window budget vs token cost), chunking стратегии (fixed, semantic, sliding).  
**Что уметь:** динамический context selection (кластеризация top‑K → выбор по маргинальной полезности), reroll ответа при низком grounded confidence, сохранять provenance (doc ids + offsets).  
**Интервью:** “как сократить контекст без потери качества?”, “как снижать hallucinations?”.

## 19. Алгоритмическая корректность и деградация
**Что знать:** источники деградации: embedding drift, query drift, рост шума/дубликатов, data skew; статистические тесты (KL divergence, PSI) на распределениях расстояний и скор.  
**Что уметь:** regression guards (минимальные пороги Recall@K, nDCG@10) в CI для индекса; anomaly detection по latency/quality метрикам; алертинг (SLO burn rate).  
**Интервью:** “какой alert настроишь первым?”, “как детектировать drift без label’ов?”.

## 20. Надёжность и отказоустойчивость
**Что знать:** точки отказа: ANN shard, rerank service, feature store, LLM генератор; паттерны graceful degradation (отключить cross-encoder, fallback на bi-encoder), partial results merge, retry/timeout стратегии, idempotency.  
**Что уметь:** детерминированные таймауты per stage, реконфигурация каскада при деградации, chaos engineering (kill shard тесты).  
**Интервью:** “как система отвечает при деградации reranker?”, “какие метрики для SLO?”.

## 21. Производственный контроль и логирование
**Что знать:** какие логи: запрос, top‑K doc ids + скор, версия модели, latency каждого этапа, признаки; требования к PII (маскирование, hash). Tracing (OpenTelemetry) для межсервисных hops.  
**Что уметь:** строить explainability trace (score breakdown) и реплеер запросов; сохранять семплы для офлайн экспериментов (query + doc ids + labels).  
**Интервью:** “что лог не должен содержать?”, “как воспроизвести инцидент?”.

## 22. Security / Privacy
**Что знать:** риск инверсии эмбеддингов (membership inference, reconstruction), multi-tenant isolation, секреты и KMS, принципы минимизации данных, RTBF (Right To Be Forgotten).  
**Что уметь:** шифрование at-rest (AES-GCM) + TLS in-transit, per-tenant namespace / шард, быстрый delete path (tombstone + async reclaim), фильтры PII перед embedding.  
**Интервью:** “как удалить документ из миллиарда векторов быстро?”, “как предотвратить утечку tenant данных?”.

## 23. Типовые интервью-задачи (практиковать)
1. **k-way merge top‑k** из N потоков (min-heap).  
2. **Параметризация IVFPQ**: оценить память для (M, bits, N).  
3. **Реализация BM25** и влияние k1,b.  
4. **LSH / MinHash pipeline**: выбрать #hashes и #bands под требуемую вероятность.  
5. **Рерэнк каскад**: выбрать K/M/N под latency SLA.  
6. **Adaptive nprobe** формула по текущей загрузке.  
7. **Score fusion**: нормализация разношкальных скоров.  
8. **Shadow reindex**: схема alias switch.  
9. **Drift detection**: KL / PSI на распределении расстояний.  
10. **Tail latency mitigation**: hedged requests + timeouts.

## 24. Roadmap (10 недель)
**Неделя 1–2:** baseline: BM25 + bi-encoder + cross-encoder; сбор MRR, nDCG.  
**Неделя 3–4:** интеграция ColBERT / late interaction; измерить latency vs качество.  
**Неделя 5–6:** эксперименты ANN (IVFPQ vs HNSW) → таблица recall/latency/cost.  
**Неделя 7–8:** дедуп (MinHash) + drift мониторинг; shadow переиндексация.  
**Неделя 9–10:** tail latency оптимизация (hedged queries, adaptive nprobe), стабильный A/B.  
**Постоянно:** логирование + regression guardrails.

---

### Краткое резюме требований блока
1. Глубокое понимание трансформеров и вариантов взаимодействия (bi / cross / late).  
2. Проектирование гибридного retrieval каскада и оптимизация k на этапах.  
3. Освоение ANN структур (IVFPQ, HNSW), тюнинг параметров, memory/latency trade-offs.  
4. Управление жизненным циклом эмбеддингов, переиндексация без даунтайма, дедуп.  
5. Метрики качества (MRR, nDCG, Recall@K) и офлайн/онлайн оценка, regression guards.  
6. Латентность: quantization, adaptive nprobe, tail mitigation.  
7. Кэширование, шардирование, федеративный merge, отказоустойчивость, graceful degradation.  
8. Feature enrichment, score fusion и контроль сигналов.  
9. Специфика RAG: выбор контекста, уменьшение hallucinations, provenance.  
10. Security/Privacy: инверсия эмбеддингов, RTBF, multi-tenant isolation.

---

**Дальше** могу подготовить чеклист “proof-of-skill” проектов (репозитории / ноутбуки), или экспортнуть этот файл в PDF. Скажи, что нужно следующим шагом.

