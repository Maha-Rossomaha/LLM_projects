# Drift в sparse‑поиске

Крупный тайтл: сдвиги распределений в лексическом (sparse) поиске и как их мониторить, диагностировать и чинить.

---

## 1. Описание проблемы

**Drift в sparse‑поиске** — это изменение статистик корпуса и/или запросов, влияющее на веса лексических термов и итоговый скоринг (BM25/BM25+/DFI и др.). Аналог **embedding drift** в dense‑поиске, но проявляется в терминах словаря: document frequency (DF), term frequency (TF), средняя длина документа $avgdl$, доля стоп‑слов и распределение языков/скриптов.

**Почему важно:** сдвиги DF/TF ломают относительную информативность термов (через $idf_t$), меняют нормализацию длины и приводят к деградации Recall/nDCG, всплескам tail‑latency (из‑за расширения кандидатов) и росту шума.

---

## 2. Базовая структура

- **Словарь (vocabulary)**: множество термов $t \in V$.
- **Postings (inverted lists)**: для каждого $t$ список пар $(doc_{id}, tf_{t,d})$.
- **Глобальные статистики:**
  - $N$ — число документов.
  - $df_t$ — в скольких документах встречается $t$.
  - $idf_t$ — обратная частота документа (зависит от $N, df_t$).
  - $avgdl$ — средняя длина документа по полю.
- **Скоринговая функция** (напр. BM25/BM25+): суммирует вклад термов запроса $q$ по документу $d$.
- **Параметры модели:** $k_1, b$ (BM25), дополнительные константы (BM25+, BM25L), настройки анализатора (токенизация, лемматизация, стоп‑листы).

Drift затрагивает именно эти статистики и/или распределения токенов.

---

## 3. Ключевые формулы

**BM25 (классическая):** $$score(q,d) = \sum\limits_{t \in q} idf_t \cdot \frac{tf_{t,d} \cdot (k_1 + 1)}{tf_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}.$$

**IDF (с сглаживанием Робертсона):** $$idf_t = \log \frac{N - df_t + 0.5}{df_t + 0.5}.$$

**BM25+ (учёт свободного члена $\delta$):** 
$$
score_{+}(q,d) = \sum\limits_{t \in q} idf_t \cdot \frac{tf_{t,d} \cdot (k_1 + 1)}{tf_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)} + \delta \cdot idf_t.
$$

**Что дрейфует:**

- $df_t$ (и значит $idf_t$) — терм становится более/менее «распространённым».
- $tf_{t,d}$ — в «среднем» документе терм встречается чаще/реже.
- $|d|$ и $avgdl$ — смещение длины документов (напр., пришли длинные PDF).
- Распределения по языкам/скриптам — меняется активная часть словаря.

**Интуиция нормализаций:**

- Логарифм в $idf_t$ стабилизирует рост веса редких термов.
- $b$ управляет, насколько штрафуем длинные документы. При росте $avgdl$ без пересчёта веса длинных документов меняются системно.

---

## 4. Оптимизации (как смягчать влияние дрейфа)

1. **Регулярный пересчёт глобальных статистик** ($idf_t$, $avgdl$) — инкрементально по сегментам или батч‑джобом с alias‑switch.
2. **IDF‑clipping / floor**: ограничить $idf_t$ снизу/сверху, чтобы всплески DF не вызывали резких перетасовок рангов.
3. **BM25+ / BM25L**: свободный член $\delta$ снижает переобучение к длине документа на краях распределения.
4. **Адаптивные стоп‑списки**: автопрюнинг термов с $df_t/N > \tau_{hi}$; и/или агрессивное фильтрование очень редких мусорных токенов ($df_t/N < \tau_{lo}$).
5. **Min‑should‑match (MSM)**: повышать устойчивость к шуму при всплеске частых термов (требовать совпадения не всех термов запроса).
6. **Полевая нормализация (BM25F)**: если drift затрагивает разные поля неравномерно.
7. **SPLADE‑ветка**: если используете нейро‑sparse, применять sparsification‑прюнинг термов и контроль распределения весов $w_t$.

---

## 5. Edge cases и проблемы

- **Всплеск трендовых термов** (события, новости): быстрое падение $idf_t$ и «размытие» сигналов — падает точность.
- **Длинные документы**: смена $avgdl$ → меняется относительный вклад длины → cross‑корпусные перекосы.
- **Мультиязычие и код‑свитчинг**: сегмент‑специфический drift; общий словарь начинает «разбавляться» редкими скриптами.
- **Шумные токены**: артефакты OCR/парсинга, эмодзи/юникодные вариации → ложное увеличение $df_t$.
- **Словарная инфляция**: рост $|V|$ за счёт спама/ботов → разжижение весов.
- **Query drift**: смена распределения термов в запросах — $idf_t$ остаётся корпусным, но ощущаемая «информативность» в поиске для пользовательских сценариев меняется.

---

## 6. Сравнение с альтернативами

- **Embedding drift (dense)**: дрейф векторного пространства требует пере‑эмбеддинга документов/запросов; метрики — сдвиги расстояний/углов.
- **Sparse drift (lexical)**: переиндексация не всегда нужна, но требуются **пересчёт $idf_t/avgdl$**, актуализация стоп‑листов и калибровка $k_1, b, \delta$.
- **SPLADE‑style sparse**: дрейф может идти как по корпусу (DF/TF), так и по модели (распределение весов $w_t$) — нужны обе группы мониторингов.

---

## 7. Когда и как применять пересчёт 

**Пересчёт $idf\_t$ и $avgdl$ обязателен, если** выполнено одно из условий:

1. **Доля новых/изменённых документов** за период $> \alpha$ (напр., 5–10%).

2. **Сдвиг $avgdl$**:

$$
\frac{|\Delta avgdl|}{avgdl_{old}} > \beta \quad \text{напр.,\ 2-3\%}
$$

3. **Сводный IDF‑drift по запросным термам** превышает порог:

$$
D_{idf} = \sum_{t \in V_q} p_q(t) \cdot |idf_t^{new} - idf_t^{old}| > \gamma.
$$

где $p_q(t)$ — частоты термов по логам запросов (нормированные).

4. **Jensen–Shannon** между нормализованными $DF$‑распределениями (по топ‑$K$ активных термов):

$$
JSD(DF_{old}, DF_{new}) > \delta.
$$

5. **Всплеск доли стоп‑слов** в запросах/доках $> \tau$ — сигнал к обновлению стоп‑листа и MSM.


**Что пересчитывать:** $df_t, idf_t, avgdl$, стоп‑листы, пороги MSM, при необходимости — $k_1, b, \delta$ (калибровка на вал‑сете).

---

## 8. Практические советы

- Вести **две витрины статистик**: по корпусу и по запросам; сводить их в рисковую метрику $D_{idf}$.
- Пересчитывать статистики **инкрементально по сегментам**, а затем **alias‑switch** на свежие.
- Для новостных/соцсетевых корпусов настроить **ежедневный** (или почасовой) фоновый пересчёт; для стабильных — **еженедельный/ежемесячный**.
- Обязательно контролировать **качество парсинга** (OCR/HTML) — это главный источник ложного DF‑drейфа.
- При мультиязычии вести **пер‑языковые словари/статистики**.
- Хранить **версии** статистик и параметров BM25; логировать метки времени переключений.

---

## 9. Примеры кода 

Ниже — минимальные утилиты для мониторинга drift’а DF/IDF/TF и принятия решений о пересчёте. Предполагается, что у вас есть токенизированные документы.

```python
from collections import Counter, defaultdict
import math
from typing import List, Dict, Iterable, Tuple

# === Подсчёт базовых статистик: N, df_t, avgdl ===
def corpus_stats(tokenized_docs: Iterable[List[str]]):
    N = 0
    df = Counter()
    total_len = 0
    for doc in tokenized_docs:
        N += 1
        total_len += len(doc)
        df.update(set(doc))  # document frequency
    avgdl = total_len / max(N, 1)
    return N, df, avgdl

# === IDF по Робертсону ===
def idf_robertson(N: int, df: Dict[str, int]) -> Dict[str, float]:
    idf = {}
    for t, dft in df.items():
        idf[t] = math.log((N - dft + 0.5) / (dft + 0.5))
    return idf

# === Сводный IDF‑drift по запросам: D_idf ===
# q_term_freq — частоты термов в запросах (из логов), будут нормированы в p_q(t)
def aggregate_idf_drift(idf_old: Dict[str, float], idf_new: Dict[str, float],
                        q_term_freq: Dict[str, int]) -> float:
    total = sum(q_term_freq.values()) or 1
    D = 0.0
    for t, cnt in q_term_freq.items():
        p = cnt / total
        o = idf_old.get(t)
        n = idf_new.get(t)
        if o is None or n is None:
            continue
        D += p * abs(n - o)
    return D

# === Средний tf по терму (для TF‑shift) ===
def average_tf_per_term(tokenized_docs: Iterable[List[str]]):
    df = Counter()
    tf_sum = Counter()
    for doc in tokenized_docs:
        c = Counter(doc)
        for t, tf in c.items():
            df[t] += 1
            tf_sum[t] += tf
    avg_tf = {t: tf_sum[t] / df[t] for t in df}
    return avg_tf

# === JSD между нормализованными DF‑распределениями по топ‑K термам ===
def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    import math
    # p, q — распределения по одинаковому носителю; дополним нулями
    keys = set(p) | set(q)
    P = [p.get(k, 0.0) for k in keys]
    Q = [q.get(k, 0.0) for k in keys]
    def _kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                s += ai * math.log(ai / bi)
        return s
    M = [(pi + qi) / 2 for pi, qi in zip(P, Q)]
    # JSD = (KL(P||M) + KL(Q||M)) / 2
    return (_kl(P, M) + _kl(Q, M)) / 2

# === Подготовка DF‑распределений для JSD ===
def normalized_df_distribution(df: Dict[str, int], top_k: int = 5000) -> Dict[str, float]:
    items = sorted(df.items(), key=lambda x: x[1], reverse=True)[:top_k]
    total = sum(v for _, v in items) or 1
    return {t: v / total for t, v in items}

# === Решение о пересчёте ===
def should_recompute(old_docs, new_docs, q_term_freq,
                     alpha=0.05, beta=0.02, gamma=0.05, delta=0.02, top_k=5000):
    # old_docs/new_docs — это не сами корпуса, а выборки или снапшоты
    N_old, df_old, avgdl_old = corpus_stats(old_docs)
    N_new, df_new, avgdl_new = corpus_stats(new_docs)

    # 1) Сдвиг avgdl
    avgdl_shift = abs(avgdl_new - avgdl_old) / max(avgdl_old, 1e-9)

    # 2) IDF‑drift
    idf_old = idf_robertson(N_old, df_old)
    idf_new = idf_robertson(N_new, df_new)
    D_idf = aggregate_idf_drift(idf_old, idf_new, q_term_freq)

    # 3) JSD по DF
    P = normalized_df_distribution(df_old, top_k)
    Q = normalized_df_distribution(df_new, top_k)
    jsd = js_divergence(P, Q)

    decision = (avgdl_shift > beta) or (D_idf > gamma) or (jsd > delta)
    return {
        "avgdl_old": avgdl_old,
        "avgdl_new": avgdl_new,
        "avgdl_shift": avgdl_shift,
        "D_idf": D_idf,
        "jsd_df": jsd,
        "recompute": decision,
    }
```

**Комментарии к коду:**

- `aggregate_idf_drift` взвешивает $|\Delta idf_t|$ частотами термов в реальных запросах — так решение о пересчёте отражает пользовательский риск.
- `normalized_df_distribution` и `js_divergence` дают агрегатный взгляд на «форму» корпуса без перечисления всех термов.
- В проде реальный пересчёт делайте инкрементально (по сегментам), а переключение — через алиасы/версии.

---

## Псевдокод интеграции в индекс

```text
nightly_job():
  sample_old = read_snapshot("index_stats@T-1")
  sample_new = read_snapshot("ingestion_buffer@T")
  q_tf       = read_query_log_terms(T-24h)
  metrics    = should_recompute(sample_old, sample_new, q_tf)
  if metrics.recompute:
      build_new_stats_segments()
      switch_alias_to_new_stats()
      log_version_and_metrics()
```

---

## Чеклист тюнинга

-

---

## Метрики и мониторинг

- **Качество:** Recall\@K, nDCG\@K, MRR.
- **Скорость/ресурсы:** p50/p95 search latency, QPS, размер индекса, длина постингов.
- **Drift‑метрики:**
  - **DF shift:** $\Delta df_t/N$, $|\Delta idf_t|$, JSD/PSI по нормализованному $DF$.
  - **TF shift:** $\Delta \bar{tf}_t$ (средний TF по терму), срез по топ‑$K$.
  - **Length shift:** $\Delta avgdl$ и распределение длин.
  - **Query drift:** JSD между распределениями термов в запросах.
- **Алерты:**
  - $D_{idf} > \gamma$;
  - $|\Delta avgdl|/avgdl_{old} > \beta$;
  - JSD($DF$) $> \delta$;
  - всплеск доли стоп‑слов $> \tau$.

---

## Когда применять (сценарии)

- **Новости/соцсети/форумы**: быстрый drift, нужен частый пересчёт и агрессивные стоп‑листы.
- **Техподдержка/лог‑корпуса**: всплески редких токенов (коды ошибок, хэши) — фильтрация и стабилизация IDF.
- **Мультиязычные базы знаний**: вести пер‑языковые статистики и поля.
