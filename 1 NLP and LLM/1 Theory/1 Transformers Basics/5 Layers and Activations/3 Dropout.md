# Dropout в LLM

## 1. Зачем нужен Dropout?

**Dropout** — техника регуляризации, направленная на снижение переобучения в нейросетях. Она случайным образом обнуляет часть нейронов при обучении, что вынуждает модель учиться более устойчивым и обобщающим признакам.

### Зачем это нужно в LLM?
- Большие языковые модели (LLM) имеют сотни миллионов или миллиарды параметров.
- Без регуляризации они могут переобучиться, особенно при fine-tuning на небольших датасетах.
- Dropout помогает уменьшить зависимость от конкретных весов и улучшает обобщающую способность модели.

---

## 2. Структура и реализация

### Основная идея:
При обучении каждый элемент входного тензора с вероятностью $p$ зануляется. Оставшиеся масштабируются на $1 / (1 - p)$ для сохранения математического ожидания.

```python
import torch.nn as nn

drop = nn.Dropout(p=0.1)  # 10% Dropout
```

### Внутренний механизм:
Если вход $x$, то:
$$
\text{Dropout}(x) = \frac{m \odot x}{1 - p}, \quad m \sim \text{Bernoulli}(1 - p)
$$

---

## 3. Где используется в LLM

- **FeedForward слой**: после первой линейной проекции и перед второй.
- **Attention**: после softmax и перед перемножением с value.
- **Embeddings**: в некоторых реализациях применяется embedding dropout.

Пример (упрощённый Transformer блок):

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ff_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.attn_drop(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.ff_drop(ff_out)
        x = self.norm2(x)
        return x
```

---

## 4. Разновидности Dropout

- **Standard Dropout** — классическое обнуление отдельных элементов.
- **Attention Dropout** — применяется к attention weights (после softmax).
- **Embedding Dropout** — применяется ко входным embedding-векторам.
- **DropPath / Stochastic Depth** — обнуление целых слоёв (чаще в vision transformers).
- **Variational Dropout** — одна маска на всю последовательность (для RNN/LLM).

---

## 5. Поведение в режимах обучения и инференса

| Режим                   | Dropout активен | Масштабирование входа | Маска генерируется |
|------------------------|------------------|------------------------|--------------------|
| `model.train()` (обучение) | Да               | $1 / (1 - p)$           | Да                 |
| `model.eval()` (инференс)  | Нет              | Нет                    | Нет                |

**Важно:** В режиме `eval` Dropout полностью отключён. Модель использует все значения, как есть.

---

## 6. Практические замечания

- При fine-tuning часто используется dropout=0.1 или 0.2.
- При inference необходимо вызывать `model.eval()` для отключения dropout.
- Значения $p > 0.3$ редко применяются в LLM, т.к. приводят к потере полезного сигнала.