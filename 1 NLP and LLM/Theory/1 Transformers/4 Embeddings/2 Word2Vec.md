## Конспект: Word2Vec — устройство, суть и реализация на Python

URL:  
🔗 [Word2Vec](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

## 1. Зачем нужен Word2Vec
Word2Vec обучает плотные векторы слов, в которых семантически близкие слова располагаются рядом, что позволяет моделям учитывать смысловые связи и улучшать результаты в задачах кластеризации, классификации, перевода и поиска аналогий.

## 2. Архитектура модели

### 2.1 CBOW и Skip‑Gram
- **CBOW (Continuous Bag‑of‑Words)** предсказывает текущее слово по эмбеддингам контекста (окружающих слов) и минимизирует кросс‑энтропию при обучении.
- **Skip‑Gram** обучает модель предсказывать контекст по заданному слову, что даёт более точные эмбеддинги для редких слов.

### 2.2 Negative Sampling и Hierarchical Softmax
- **Hierarchical Softmax** строит дерево Хаффмана над словарём и вычисляет вероятность слова по пути в дереве, ускоряя обучение до O(log V).
- **Negative Sampling** заменяет полный softmax на оптимизацию бинарной классификации: для каждого положительного примера «слово–контекст» добавляются несколько случайных негативных пар.

## 3. Реализация с нуля
```python
import numpy as np

# Параметры
V, D = vocab_size, emb_dim
W = np.random.randn(V, D)      # эмбеддинги слов
Wc = np.random.randn(V, D)     # эмбеддинги контекста

# Softmax-функция
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Упрощённый Skip‑Gram без Negative Sampling
def train_skipgram(center, context, lr):
    v_c = W[center]
    scores = Wc @ v_c
    probs = softmax(scores)
    for i in range(V):
        label = 1 if i in context else 0
        e = probs[i] - label
        Wc[i] -= lr * e * v_c
        W[center] -= lr * e * Wc[i]
```

## 4. Использование Gensim
```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,        # 1 = Skip‑Gram, 0 = CBOW
    negative=10,
    epochs=5
)
vectors = model.wv
```  

## 6. Предобученные эмбеддинги
- GoogleNews-vectors (3 млн слов, 300 dim) можно загрузить через Gensim:
  ```python
  from gensim.models import KeyedVectors
  model = KeyedVectors.load_word2vec_format(
      'GoogleNews-vectors-negative300.bin', binary=True
  )
  ```
- TensorFlow Hub предоставляет модули с готовыми эмбеддингами.