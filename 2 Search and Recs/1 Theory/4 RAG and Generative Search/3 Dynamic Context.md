# Dynamic context selection

## 1. Проблема: ограничение на длину контекста

Большинство LLM имеют ограничение на максимальную длину входа — \(L_{\text{max}}\), например:

- GPT-3.5: 4k–16k токенов,
- Claude 2: 100k токенов,
- Gemini 1.5: до 1M токенов (в эксперименте).

Это означает:

- Нельзя просто передать все топ-K retrieved documents в модель, если они суммарно превышают \(L_{\text{max}}\).
- Нужно выбрать **подмножество** наиболее полезных документов.

---

## 2. Подходы к выбору документов (Dynamic Context Selection)

В RAG после retrieval топ-K документов мы часто имеем:

- \(D = \{d_1, d_2, ..., d_K\}\), где каждый \(d_i\) — чанк текста.
- \(q\) — запрос пользователя.
- \(L_{\text{max}}\) — ограничение на длину.

Цель: отобрать \(\hat{D} \subseteq D\), такие что:

- \(\sum_{d \in \hat{D}} \text{len}(d) \leq L_{\text{max}}\),
- но \(\hat{D}\) максимально релевантны и разнообразны.

**Методы выбора:**

### 2.1 Greedy по релевантности

- Ранжируем документы по score \(\text{sim}(q, d_i)\),
- Добавляем сверху вниз, пока не переполним окно.
- Простая эвристика, но часто дублирует одинаковые документы (низкая diversity).

### 2.2 Кластеризация top-K

#### Идея:

- Разнообразие важнее, чем просто топ по score.
- Кластеризуем retrieved документы \(D\) по эмбеддингам.
- Из каждого кластера берём наиболее релевантный документ.

#### Алгоритм:

1. Получаем эмбеддинги \(\vec{d}_i = \text{encoder}(d_i)\).
2. Применяем кластеризацию (\(k\)-means, spectral, HDBSCAN).
3. Из каждого кластера берём \(\arg\max_{d_j \in \text{cluster}} \text{sim}(q, d_j)\).

#### Плюсы:

- Повышает тематическое разнообразие (coverage).
- Может снижать redundancy.

#### Минусы:

- Вычислительно затратнее.
- Возможно попадание нерелевантного документа из слабого кластера.

### 2.3 Маржинальная полезность (Marginal Utility Selection)

#### Идея:

- Отбор документов не только по релевантности к запросу, но и по **новизне** по отношению к уже выбранным.
- Учитывает перекрытие контента.

#### Формула полезности:

$$
U(d_j | S) = \text{rel}(q, d_j) - \lambda \cdot \max_{d_k \in S} \text{sim}(d_j, d_k)
$$

где:

- \(S\) — текущее множество выбранных документов,
- \(\text{rel}(q, d_j)\) — релевантность документа \(d_j\) к запросу,
- \(\text{sim}(d_j, d_k)\) — семантическая схожесть с уже выбранными,
- \(\lambda\) — вес штрафа за дублирование.

#### Алгоритм:

1. Инициализируем \(S = \emptyset\).
2. Пока длина всех \(S\) < \(L_{\text{max}}\):
   - Вычисляем \(U(d_j | S)\) для всех \(d_j \notin S\).
   - Выбираем \(\arg\max U\), добавляем в \(S\).

#### Пример кода:

```python
from sklearn.metrics.pairwise import cosine_similarity

def marginal_utility_selection(query_emb, doc_embs, lengths, max_tokens, lambda_=0.5):
    selected, total_len = [], 0
    while total_len < max_tokens and len(selected) < len(doc_embs):
        best_doc, best_score = None, -float('inf')
        for i, emb in enumerate(doc_embs):
            if i in selected:
                continue
            rel = cosine_similarity([query_emb], [emb])[0,0]
            overlap = max([cosine_similarity([emb], [doc_embs[j]])[0,0] for j in selected], default=0)
            score = rel - lambda_ * overlap
            if score > best_score and total_len + lengths[i] <= max_tokens:
                best_doc, best_score = i, score
        if best_doc is None:
            break
        selected.append(best_doc)
        total_len += lengths[best_doc]
    return selected
```

---

## 3. Ограничения context window и как их обойти

### 3.1 Редактирование чанков

- **Truncation** — обрезка длинных чанков (после reranking).
- **Summarization** — сжатие длинных документов (T5, LLM, BART).

### 3.2 Фильтрация по источникам

- Исключение повторяющихся или менее доверенных источников.
- Ограничение на max по каждому домену/категории.

### 3.3 Динамическая адаптация под модель

- Один и тот же retrieved set можно подать в:
  - короткую модель (например, LLaMA 7B, контекст 4k),
  - длинную (например, Claude 100k),
  - и даже в несколько этапов (multi-step RAG).

### 3.4 Sliding inclusion

- В случае, если вся выборка не влезает, можно сделать итеративную генерацию:
  - сначала с первой частью контекста,
  - потом с дополнением новых документов,
  - далее агрегировать ответы.
