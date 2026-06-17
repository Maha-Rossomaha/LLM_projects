# План компетенций: Рекомендательные системы

> **Фокус:** классические и современные методы рекомендаций — от коллаборативной фильтрации до полного продакшен-пайплайна.  
> Подробный 19-дневный учебный курс вынесен в `5_syllabus.md`.  
> Темы retrieval, reranking, ANN подробно разобраны в плане `2 Search and Recs`.  
> A/B-тесты и uplift modeling — в плане `6 ML System Design`.

---

## I. Введение в рекомендательные системы

📄 [Intro](1%20Theory/1%20Intro.md)
📄 [EDA and Baselines](1%20Theory/2%20EDA%20and%20Baselines.md)
📄 [Basics](1%20Theory/3%20Basics.md)
📒 [MovieLens EDA](2%20Practice/1%20MovieLens%20Practice/1%20EDA.ipynb)

---

## II. Коллаборативная фильтрация (Collaborative Filtering)

📄 [CF: Basics](1%20Theory/4%201%20Collaborative%20Filtering:%20Basics.md)
📄 [CF: Models](1%20Theory/4%202%20Collaborative%20Filtering:%20Models.md)
📒 [Collaborative Filtering](2%20Practice/2%20Collaborative%20Filtering.ipynb)
📒 [CF UserKNN](2%20Practice/1%20MovieLens%20Practice/21%20CF%20UserKNN.ipynb)

---

## III. Матричная факторизация (Matrix Factorization)

📄 [MF: SVD and ALS](1%20Theory/5%201%20Matrix%20Factorization:%20SVD%20and%20ALS.md)
📄 [MF: BPR](1%20Theory/5%202%20Matrix%20Factorization:%20BPR.md)
📒 [Matrix Factorization](2%20Practice/3%20Matrix%20Factorization.ipynb)

---

## IV. Контентные и гибридные подходы

📄 [Content Based and Hybrid](1%20Theory/6%20Content%20Based%20and%20Hybrid.md)
📄 [Ranking Task with BPR and WARP](1%20Theory/7%20Ranking%20Task%20with%20BPR%20and%20WARP.md)

---

## V. Графовые методы — LightGCN

📄 [LightGCN](1%20Theory/8%20LightGCN.md)

---

## VI. Sequential и Generative рекомендации

> 📝 Конспектов пока нет. Планируемые темы:

- **Sequential рекомендации — общая идея**
  - В отличие от классических методов, учитывается порядок взаимодействий пользователя (сессия, цепочка просмотров).
  - Зачем: интересы пользователя эволюционируют, последние действия важнее, чем старые.
  - Данные: сессии (сustomer journey), логи кликов по времени.

- **SASRec (Self-Attentive Sequential Recommendation)**
  - Применение causal self-attention (как в Transformer decoder) к последовательности item'ов.
  - Каждый item смотрит только на предыдущие (causal mask). Предсказание следующего item'а.
  - Плюсы: быстрее RNN, лучше захватывает long-range зависимости в сессии.
  - Сравнение с GRU4Rec, NextItNet.

- **BERT4Rec**
  - Bidirectional self-attention для рекомендаций: в отличие от SASRec, смотрит на оба направления.
  - Cloze task (masked item prediction): случайно маскируем item в последовательности и предсказываем.
  - Плюсы: более богатое представление контекста сессии.
  - Минусы: не авторегрессионен, сложнее для онлайн-инференса.

- **ARGUS (AutoRegressive Generative User Sequential modeling)**
  - Авторегрессионная генеративная модель последовательности действий пользователя.
  - По сути — GPT-подход: предсказание следующего item'а по истории.
  - Отличие от SASRec: более продвинутые техники (MoE, rotary embeddings, масштабирование до больших корпусов).
  - Связь с LLM: tokenization предметов как «слов», сессии как «тексты».
  - Практическое применение: next-item prediction, персонализация в реальном времени.

- **Другие sequential подходы**
  - GRU4Rec — GRU для sequence modeling в рекомендациях (первая волна deep learning).
  - NextItNet — residual causal CNN для sequence.
  - Lessons learned: attention > CNN > RNN для рекомендаций (похоже на NLP).

- **Generative RecSys и LLM**
  - P5 (Pretrain, Personalized Prompt, Predict): text-to-text подход, унифицирующий разные recsys-задачи через промпты.
  - TIGER: генеративная модель для retrieval.
  - Когда использовать LLM vs классические sequential модели: trade-off качество / latency / стоимость.

---

## VII. Candidate Generation (Retrieval) — генерация кандидатов

**Цель:** из миллионов товаров быстро отобрать ~100–1000 релевантных.  
**Главная метрика:** Recall@K — не упустить потенциально интересное.

- **Sparse retrieval:** BM25, инвертированный индекс, lexical search.
  - ✅ Конспекты: `2_plan.md` I → Sparse signatures.
- **Dense retrieval:** bi-encoder эмбеддинги + ANN-поиск (FAISS: IVF-PQ, HNSW).
  - ✅ Конспекты: `2_plan.md` I → Dense search + ANN.
- **Hybrid retrieval:** BM25 + dense fusion (Weighted Sum, Reciprocal Rank Fusion).
  - ✅ Конспекты: `2_plan.md` I → Hybrid fusion.
- **Multi-source candidates:** коллаборативные (CF/MF эмбеддинги), контентные, популярные, новинки, тренды. Объединение → дедупликация.
  - ⚠️ Частично: `5_plan.md` II–V. Нет сводного конспекта про merging стратегий.
- **Cold-start candidates:** отдельный трек для новых users (популярное, демография) и items (контентные фичи, explore).
  - ✅ Конспекты: `2_plan.md` VIII (Personalization and Cold Start).

---

## VIII. Filtering — фильтрация

**Цель:** выкинуть неподходящее до ранжирования.  
> 📝 Конспектов пока нет.

- **Бизнес-правила:** возрастные ограничения, региональная доступность, out-of-stock, лицензионные ограничения.
- **User-level дедупликация:** уже куплено/просмотрено/оценено — исключить из выдачи.
- **Чёрные списки:** скрытые/заблокированные пользователем товары, категории, продавцы.
- **Частотные ограничения:** не показывать один и тот же товар чаще N раз за сессию/день.

---

## IX. Feature Engineering — инженерия признаков

**Цель:** из сырых данных собрать признаки для модели ранжирования.  
> 📝 Конспектов пока нет.

- **User features:** демография (возраст, пол, регион), агрегаты поведения (любимый жанр, средний чек, частота покупок), давность последней активности, lifetime value.
- **Item features:** категория, цена, бренд, популярность (число просмотров/покупок), новизна (дней с добавления), рейтинг, текстовое описание.
- **Context features:** время суток, день недели, устройство (mobile/desktop), источник трафика, сезонность.
- **User-Item interaction features:** было ли показано раньше? сколько раз? когда последний раз? сколько похожих пользователь уже видел? позиция в списке кандидатов? embedding similarity (dot product user × item).
- **Embeddings as features:** выходы MF/LightGCN/Item2Vec как дополнительные колонки для бустинга (two-stage learning).
- **Feature Store:** Feast / Hopsworks — online ↔ offline parity, версионирование, TTL.
  - ⚠️ Упомянуто в `3_plan.md` XII и `6_plan.md` раздел 16.

---

## X. Scoring / Ranking — ранжирование

**Цель:** из ~100–500 кандидатов отранжировать top-5/10/20.  
**Главная метрика:** nDCG@K — качество порядка выдачи.  
> Основные конспекты: план `2`, разделы II (Reranking Cascade) и III (Learning to Rank).

- **Pointwise LTR:** LightGBM Ranker, CatBoost — предсказать вероятность клика для каждого кандидата отдельно.
- **Pairwise / Listwise LTR:** LambdaMART, ListNet, ListMLE — обучение на парах/списках.
- **Cross-Encoder:** BERT-подобная модель (user_context, item) → релевантность. Высокая точность, дорого.
- **Late Interaction (ColBERT):** MaxSim по токеновым эмбеддингам. Компромисс качество/стоимость.
- **Multi-stage cascade:** Bi-encoder (все) → ColBERT (top-500) → Cross-encoder (top-50). Latency budget на каждом слое.
- **Distillation:** cross-encoder → bi-encoder (ускорение инференса).

---

## XI. Post-processing — постобработка

**Цель:** финальные правки после ML-модели перед показом пользователю.  
> 📝 Конспектов пока нет.

- **Diversification:** MMR (Maximal Marginal Relevance), не более N из одной категории, intra-list similarity. Баланс relevance vs diversity.
- **Business rules:** гарантированные позиции (промо-товар на 1-м месте), исключение нежелательного, квоты по поставщикам.
- **Novelty / Serendipity:** принудительная вставка нового/неожиданного (например, 10% рекомендаций — товары без истории). Борьба с popularity bias на уровне выдачи.
- **Fairness constraints:** выравнивание exposure между seller'ами, жанрами, ценовыми сегментами. Disparate impact, calibration.
- **Bias mitigation:** Popularity bias (popular items доминируют → меньше exposure для новых), selection bias (видим только то, что пользователь выбрал), position bias (первые позиции кликают чаще), feedback loop (алгоритм сужает выдачу → пользователь видит только узкое → алгоритм ещё сильнее сужает).

---

## XII. Explore-Exploit — управление exploration

**Цель:** не зацикливаться на exploit (показывать только известное) — пробовать новое, собирать обратную связь.  
> 📝 Конспектов пока нет.

- **Простые стратегии:** ε-greedy (ε% случайных), UCB (Upper Confidence Bound), Thompson Sampling (байесовский подход).
- ✅ Конспекты: `2_plan.md` V (Bandits).
- **Contextual Bandits:** LinUCB — учитывает контекст (user/item features) при выборе руки.
- ✅ Конспекты: `2_plan.md` V.
- **В рекомендациях:** exploration новых товаров/категорий, exploration для новых пользователей, A/B-бандиты вместо фиксированных A/B-тестов.

---

## XIII. A/B-тесты, метрики и оценка качества

**Цель:** доказать, что изменения в модели/пайплайне улучшают бизнес-метрики.  
> Основные конспекты: план `6`, разделы I (AB Tests) и II (Uplift Modeling).

- **Offline метрики:** Precision@K, Recall@K, MRR, MAP@K, nDCG, Coverage, Personalization, Novelty. Bootstrap доверительные интервалы.
- **Online метрики:** CTR, конверсия, retention, dwell-time, engagement. SBS (Side-by-Side) сравнение.
- **A/B-тестирование:** traffic splitting, MDE (Minimum Detectable Effect), статзначимость (p-value, confidence intervals), duration estimation.
- **Продвинутые техники:** CUPED (variance reduction), sequential testing, Bayesian A/B.
- **Uplift modeling:** как измерить причинный эффект рекомендации на пользователя.

---

## XIV. Дорожная карта компетенций

| Этап | Тема | Где конспекты |
| ---- | ---- | ------------- |
| 1 | **Введение, EDA, Long Tail** | `5_plan.md` I |
| 2 | **Collaborative Filtering** (User/Item KNN) | `5_plan.md` II |
| 3 | **Matrix Factorization** (SVD, ALS, BPR) | `5_plan.md` III |
| 4 | **Контентные и гибридные** (Item2Vec, two-tower) | `5_plan.md` IV |
| 5 | **LightGCN** — графовые эмбеддинги | `5_plan.md` V |
| 6 | **Sequential** (SASRec, BERT4Rec, ARGUS) | `5_plan.md` VI |
| 7 | **Candidate Generation** (sparse/dense/hybrid) | `2_plan.md` I |
| 8 | **Filtering + Feature Engineering** | `5_plan.md` VIII–IX |
| 9 | **Scoring / Ranking** (LTR, cascade) | `2_plan.md` II–III |
| 10 | **Post-processing** (diversity, novelty, fairness) | `5_plan.md` XI |
| 11 | **Explore-Exploit** (bandits) | `2_plan.md` V, `5_plan.md` XII |
| 12 | **A/B-тесты и метрики** | `6_plan.md` I–II, `5_plan.md` XIII |
| 13 | **Продакшн-архитектура** | `5_plan.md` (заглушка), `3_plan.md` |

---

## XV. Ресурсы

- [Рекомендательные системы в современном мире / Хабр](https://habr.com/ru/companies/otus/articles/950650/)
- [Шпаргалка по рекомендательным системам / Хабр](https://habr.com/ru/articles/792994/)
- [Рекомендательные системы — IntSys MIPT](https://intsystems.github.io/ru/course/recommender_systems/index.html)
- [Метрики оценки для рекомендательных систем / Хабр](https://habr.com/ru/companies/otus/articles/732842/)
- [LightGCN: Simplifying and Powering GCN for Recommendation](https://arxiv.org/abs/2002.02126)
- [Многорукие бандиты в рекомендациях (Avito) / Хабр](https://habr.com/ru/companies/avito/articles/417571/)