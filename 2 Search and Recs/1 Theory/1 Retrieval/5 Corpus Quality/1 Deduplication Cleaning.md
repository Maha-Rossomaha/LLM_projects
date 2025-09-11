# Дедупликация и очистка корпуса для retrieval

Качественный корпус критически важен для retriever-систем: дубликаты, мусорные блоки и повторяющийся контент ухудшают метрики (nDCG, Recall), повышают нагрузку на индексацию и приводят к нерелевантной выдаче.&#x20;

---

## 1. MinHash и Jaccard-похожесть

### Что такое Jaccard:

Если $A$ и $B$ — множества n-грамм из двух документов:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

### Как работает MinHash:

1. Представим каждый документ как множество n-грамм (обычно 3-5).
2. Выбираем $k$ случайных (но детерминированных) хеш-функций $h_1, ..., h_k$.
3. Для каждой хеш-функции $h_i$ ищем минимум среди всех $h_i(n\text{-грамм})$.
4. Получаем вектор из $k$ чисел — это **MinHash-сигнатура**.

Пример:

* Док 1: {"собака лает", "лает на", "на прохожего"}
* После 3 хеш-функций: `[12, 4, 9]`
* Док 2: {"во дворе", "дворе лает", "лает собака"} → `[12, 4, 10]`

Совпадают 2 из 3 → Jaccard-приближение = 2/3 = 0.67

#### Что дальше?

* Сравниваем MinHash-вектора попарно → считаем долю совпадающих позиций.
* Если она > порога (например, 0.85) → считаем документы дубликатами.
* Можно использовать **MinHashLSH**, чтобы быстро находить пары с высокой схожестью без перебора всех пар.

### Пример  (datasketch):

```python
from datasketch import MinHash, MinHashLSH

# Пример документов
doc1 = "собака лает на прохожего во дворе"
doc2 = "во дворе на прохожего лает собака"

# Разделим на n-граммы (словные 2-граммы)
ngramms1 = set(zip(doc1.split(), doc1.split()[1:]))
ngramms2 = set(zip(doc2.split(), doc2.split()[1:]))

# Создаём MinHash объекты
m1, m2 = MinHash(num_perm=128), MinHash(num_perm=128)
for s in ngramms1: 
    m1.update(" ".join(s).encode('utf8'))
for s in ngramms2: 
    m2.update(" ".join(s).encode('utf8'))

# Приближённое сходство
print(f"Jaccard ≈ {m1.jaccard(m2):.3f}")
```

---

## 2. SimHash и Hamming расстояние

SimHash работает не с множествами, а с признаковым вектором.

### Суть:

1. Каждому токену (или n-грамме) сопоставляется псевдослучайный битовый вектор.
2. Каждый бит суммируется по всем токенам с весом (TF-IDF, frequency).
3. После суммирования:
   * если сумма по позиции > 0 → 1
   * иначе → 0
4. Получается бинарная сигнатура (например, 64-битная).

### Пример:

* "во дворе лает собака" → вектор: `101010111000...`
* "собака во дворе лает" → `101010111100...`
* Hamming-дистанция = число отличающихся битов (например, 2 из 64)

#### Что дальше?

* Можно использовать структуру **SimHashIndex** (например, в `simhash-py`), которая строит индексы по сигнатурам.
* Быстро ищем документы, отличающиеся не более чем на $k$ битов.

### Hamming-дистанция:

Количество несовпадающих битов между отпечатками.

### Пример (упрощённый):

```python
import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer

# Хеш-функция для токенов
def token_hash(token, dim=64):
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
    return np.array([1 if h >> i & 1 else -1 for i in range(dim)])

# SimHash-отпечаток
def simhash(text, dim=64):
    tfidf = TfidfVectorizer().fit([text])
    tokens = tfidf.get_feature_names_out()
    vec = np.zeros(dim)
    for token in tokens:
        vec += token_hash(token, dim)
    return (vec > 0).astype(int)

h1 = simhash("во дворе лает собака")
h2 = simhash("собака во дворе лает")

hamming = np.sum(h1 != h2)
print(f"Hamming distance: {hamming}")
```

---

## 3. Как применять MinHash / SimHash на практике

1. **Построить сигнатуры** для всех документов.
2. **Построить индекс:**

   * MinHash → LSH-бандинг
   * SimHash → группировка по Hamming расстоянию
3. **Фильтрация:**

   * Порог Jaccard (например, >0.9)
   * Порог Hamming (например, <5 битов)
4. **Убрать дубликаты:**

   * Хранить только один из пары (обычно самый длинный / с лучшей метаинформацией).
   * Или маркировать кластеры дубликатов.

---

## 4. TF-IDF + Cosine Similarity

Если корпус небольшой, можно использовать **TF-IDF-векторы** и вычислить косинусное сходство:

$$
\cos(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\|\|\vec{b}\|}
$$

### Пример:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = ["собака лает на прохожего", "во дворе лает собака"]
vectorizer = TfidfVectorizer().fit(corpus)
vecs = vectorizer.transform(corpus)

sim = cosine_similarity(vecs)
print(f"Cosine similarity: {sim[0,1]:.3f}")
```

> Подходит для кластеризации, сортировки дубликатов.

---

## 5. Удаление boilerplate

Boilerplate — повторяющиеся шаблонные части:

* футеры, copyright,
* рекламные блоки,
* одинаковые заголовки в документах СМИ.

### Методы:

* Правила (регулярки по типичным фразам).
* Частотный анализ: строки, встречающиеся в >X% документов.
* Алгоритмы типа Boilerpipe / Readability.
* Fine-tuned model для классификации фрагментов.

### Практика:

* Парсим документы на строки/абзацы.
* Строим TF-IDF по абзацам.
* Удаляем наиболее частотные (top-N percentile).

---

## 6. Кластеризация дубликатов

Цель: сгруппировать схожие документы, чтобы выбрать 1 представитель.

### Методы:

* Agglomerative clustering по cosine / Jaccard матрице.
* DBSCAN (особенно с SimHash).
* Faiss (для больших корпусов с dense embedding).

### Пример: кластеризация TF-IDF-векторов

```python
from sklearn.cluster import AgglomerativeClustering

n_clusters = 2
model = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
labels = model.fit_predict(vecs.toarray())
print(labels)
```

---

## 7. Практика на больших данных (Faiss + hashing)

### Workflow:

1. Получаем dense- или TF-IDF-вектора.
2. Снижаем размерность (SVD / PCA).
3. Индексируем в Faiss.
4. Находим top-K ближайших → фильтруем по порогу похожести.

### Альтернатива:

* Храним MinHash отпечатки и ищем совпадения по битам.
* Используем locality-sensitive hashing (LSH).

---

## Когда использовать какой метод

| Метод       | Тип данных        | Преимущества               | Недостатки                      |
| ----------- | ----------------- | -------------------------- | ------------------------------- |
| **MinHash** | Множество n-грамм | Хорошо приближает Jaccard  | Нужно хранить k значений        |
| **SimHash** | TF-IDF признаки   | Очень компактный (64 бита) | Слабо точен на коротких текстах |
| **TF-IDF**  | Векторы           | Точная косинусная метрика  | Требует O(N²) для перебора      |
