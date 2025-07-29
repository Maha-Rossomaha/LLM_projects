# Конспект статьи «Global Vectors for Word Representation» (Pennington et al., 2014)

URL:  
🔗 [«Global Vectors for Word Representation»](https://aclanthology.org/D14-1162/)

## 1. Введение и мотивация
GloVe предлагает метод обучения векторных представлений слов на основе статистики совместного встречаемости слов в корпусе. В отличие от предшественников (Word2Vec CBOW/Skip-gram), фокусируется на глобальных соотношениях: соотношение вероятностей встречаемости слов фиксирует значимую лексическую информацию.

## 2. Математическая формулировка

### 2.1. Матрица совместной встречаемости
Пусть $X$ — матрица размера $V\times V$, где $X_{ij}$ — число, сколько раз слово $j$ встретилось в контексте слова $i$ (в окне фиксированного размера).  
Определим:
- $X_i = \sum_k X_{ik}$ — общее число контекстов для слова $i$.
- $P_{ij} = X_{ij} / X_i$ — условная вероятность встретить $j$ рядом с $i$.

### 2.2. Целевая функция
Идея: отношение $P_{ik}/P_{jk}$ отражает семантическую близость слов $i$ и $j$ относительно контекста $k$.  
Желаемая модель:  
$$
F(w_i, \tilde w_j, \tilde b_j, b_i) = w_i^T \tilde w_j + b_i + \tilde b_j \approx \log X_{ij}.
$$  
Минимизируем взвешенный квадрат ошибок:  
$$
J = \sum_{i,j=1}^V f(X_{ij}) \bigl(w_i^T\tilde w_j + b_i + \tilde b_j - \log X_{ij}\bigr)^2.
$$  
Здесь $f(x)$ — функция-вес, которая ограничивает вклад редких и слишком частых пар:
$$
f(x) =
\begin{cases}
(x/x_{\text{max}})^\alpha, & x < x_{\text{max}},\\
1, & x \ge x_{\text{max}}.
\end{cases}
$$  
Типичные значения: $x_{\text{max}}=100$, $\alpha=3/4$.

## 3. Разбор архитектуры

1. **Две матрицы эмбеддингов**  
   - $W\in\mathbb R^{V\times d}$ — эмбеддинги целевых слов.  
   - $\tilde W\in\mathbb R^{V\times d}$ — эмбеддинги контекстных слов.  
2. **Смещения**  
   - Векторы $b\in\mathbb R^V$ и $\tilde b\in\mathbb R^V$ для смещений (bias).  
3. **Обучение**  
   - Оптимизатор: AdaGrad.  
   - Обучаем все параметры совместно, используя стохастический градиентный спуск по ненулевым $X_{ij}$.  
   - Сохраняем лишь ненулевые пары $(i,j)$, чтобы не хранить всю матрицу $V\times V$.

После обучения итоговый эмбеддинг слова $i$ получается как сумма:  
$$
\mathbf{v}_i = w_i + \tilde w_i.
$$

## 4. Практические детали
- **Корпус**: обычно – Википедия, Common Crawl и т. п.  
- **Оконный размер**: 5–10 слов слева/справа.  
- **Пре- и постобработка**: нижний регистр, удаление пунктуации, частые слова могут исключаться.

## 5. Пример реализации на PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# 1. Собираем матрицу со-встречаемости
def build_cooccurrence(corpus, window_size=5):
    vocab = {w:i for i,w in enumerate(set(corpus))}
    inv_vocab = {i:w for w,i in vocab.items()}
    X = defaultdict(float)
    for idx, word in enumerate(corpus):
        w_i = vocab[word]
        start = max(0, idx - window_size)
        end   = min(len(corpus), idx + window_size + 1)
        for j in range(start, end):
            if j == idx: continue
            w_j = vocab[corpus[j]]
            X[(w_i, w_j)] += 1.0 / abs(j - idx)
    return X, vocab, inv_vocab

# 2. Определяем модель GloVe
class GloVe(nn.Module):
    def __init__(self, vocab_size, emb_dim, x_max=100, alpha=0.75):
        super().__init__()
        self.w  = nn.Embedding(vocab_size, emb_dim)
        self.w_tilde = nn.Embedding(vocab_size, emb_dim)
        self.b  = nn.Embedding(vocab_size, 1)
        self.b_tilde = nn.Embedding(vocab_size, 1)
        self.x_max = x_max
        self.alpha = alpha

    def forward(self, i_idx, j_idx, X_ij):
        w_i = self.w(i_idx)
        w_j = self.w_tilde(j_idx)
        b_i = self.b(i_idx).squeeze()
        b_j = self.b_tilde(j_idx).squeeze()
        f = torch.where(X_ij < self.x_max,
                        (X_ij / self.x_max)**self.alpha,
                        torch.ones_like(X_ij))
        pred = (w_i * w_j).sum(dim=1) + b_i + b_j
        loss = f * (pred - torch.log(X_ij))**2
        return loss.sum()

# 3. Подготовка данных и обучение
def train_glove(corpus, emb_dim=50, epochs=50, lr=0.05):
    X, vocab, inv_vocab = build_cooccurrence(corpus)
    pairs = list(X.items())
    model = GloVe(len(vocab), emb_dim)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for (i,j), X_ij in pairs:
            i_idx = torch.LongTensor([i])
            j_idx = torch.LongTensor([j])
            X_val = torch.FloatTensor([X_ij])
            optimizer.zero_grad()
            loss = model(i_idx, j_idx, X_val)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    embeddings = model.w.weight.data + model.w_tilde.weight.data
    return embeddings, vocab, inv_vocab

if __name__ == "__main__":
    sample_text = "we are what we repeatedly do excellence then is not an act but a habit".lower().split()
    emb, vocab, inv = train_glove(sample_text, emb_dim=20, epochs=100)
    print("Embedding for 'excellence':", emb[vocab["excellence"]])
```
