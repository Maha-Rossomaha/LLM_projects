# Hybrid Search Basics

## 1. Почему нужен гибрид (dense vs lexical)

- **Лексический поиск (BM25, TF-IDF)** хорошо работает, когда запрос и документ имеют **общие термы**. Он устойчив к шуму и не требует обучения, но не покрывает синонимы и парафразы.
- **Dense поиск (эмбеддинги, косинусная близость)** улавливает **семантику** (синонимы, перефразировки), но страдает от **Out-Of-Vocabulary (OOV)** слов, чисел, морфологических форм.
- **Гибрид** объединяет сильные стороны: лексический компонент гарантирует возврат точных совпадений, а dense — расширяет результат за счёт семантической близости.

Пример:

- Запрос: *«купить авто»*.
- BM25 найдёт документы с «купить», «авто».
- Dense найдёт также документы с «приобрести машину».
- Гибрид соединяет оба списка, повышая recall и устойчивость.

---

## 2. Score fusion (сумма, среднее, нормализация)

**Score fusion** = объединение результатов разных моделей через их **скор**.

1. **Сумма скоров:**
   $score_{hybrid}(d) = score_{lex}(d) + score_{dense}(d)$

2. **Среднее:**
   $score_{hybrid}(d) = \frac{score_{lex}(d) + score_{dense}(d)}{2}$

3. **Взвешенная сумма:**
   $score_{hybrid}(d) = w_{lex}·score_{lex}(d) + w_{dense}·score_{dense}(d)$

> Проблема: BM25 и cosine similarity имеют **разные масштабы**. Нужно нормализовать.

---

## 3. Нормализация

- **Min-max:**
  $norm(x) = \frac{x - min}{max - min}$

- **Z-score:**
  $norm(x) = \frac{x - \mu}{\sigma}$

- **Rank-based:** вместо исходных скоров каждому документу присваивается значение $1/rank$, где rank — позиция документа в выдаче (1 = лучший). Таким образом, документы из разных систем можно легко объединять: чем выше позиция, тем больший вклад. Этот подход лежит в основе Reciprocal Rank Fusion (RRF), где добавляется константа k для сглаживания: $score(d) = \sum_i 1 / (k + rank_i(d))$.


---

## 4. Формула $w\_{lex}·BM25 + w\_{dense}·\cos$

Общая схема гибридного поиска:

$score_{hybrid}(d) = w_{lex}·norm(BM25(d)) + w_{dense}·norm(cosine(d))$

- $w\_{lex}$ и $w\_{dense}$ — веса, подобранные вручную или через grid/Bayesian optimization.
- Обычно начинают с $w\_{lex} = 0.5, w\_{dense} = 0.5$ и подбирают по метрикам (nDCG, MRR).

---

## 5. Примеры расчёта 

### 5.1. Fusion BM25 + Dense

```python
import torch

# BM25 скоры для 3 документов
bm25_scores = torch.tensor([5.2, 2.8, 0.5])

# Dense cosine similarity ([-1, 1])
dense_scores = torch.tensor([0.72, 0.10, 0.55])

# Нормализация min-max
bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

# Веса
w_lex, w_dense = 0.6, 0.4

# Гибридные скоры
hybrid_scores = w_lex * bm25_norm + w_dense * dense_norm
print(hybrid_scores)
```

> Результат: тензор с объединёнными скорингами для каждого документа.

---

### 5.2. Rank-based fusion (Reciprocal Rank Fusion упрощённо)

```python
# Ранги документов (меньше = лучше)
bm25_ranks = torch.tensor([1, 2, 3])
dense_ranks = torch.tensor([2, 1, 3])

# Reciprocal Rank Fusion
rrf = 1.0 / (bm25_ranks + 60) + 1.0 / (dense_ranks + 60)
print(rrf)
```

> Такой метод не зависит от масштаба скоров, а только от рангов.

---

### 5.3. Подбор весов через grid search

```python
import numpy as np

def hybrid_score(bm25, dense, w):
    bm25_n = (bm25 - bm25.min()) / (bm25.max() - bm25.min())
    dense_n = (dense - dense.min()) / (dense.max() - dense.min())
    return w * bm25_n + (1-w) * dense_n

bm25 = torch.tensor([5.2, 2.8, 0.5])
dense = torch.tensor([0.72, 0.10, 0.55])

for w in np.linspace(0, 1, 6):
    print(w, hybrid_score(bm25, dense, w))
```

> Можно сравнить nDCG/MRR для разных $w$ и выбрать оптимальный.