# Слои нормализации в нейронных сетях

## 1. Зачем нужны слои нормализации

- **Стабилизация распределения активаций**  
  Во время обучения параметры разворачиваются так, что распределение входов в каждый слой меняется (internal covariate shift). Нормализация выравнивает это распределение, ускоряя и стабилизируя обучение.

- **Ускорение сходимости**  
  Выравнивание дисперсии и смещения входов позволяет использовать более высокие learning-rate и снижает зависимость от инициализации.

- **Регуляризация**  
  В ряде случаев (например, BatchNorm) вносит стохастичность через вычисление статистик по мини-батчу, действуя как лёгкий регуляризатор.

## 2. Batch Normalization

Нормализация по мини-батчу.

1. Вычисляем статистики по батчу $B$ размера $m$ для каждого канала/нейрона:
   $$
   \mu_{B} = \frac1m \sum_{i=1}^m x_i,\quad
   \sigma^2_{B} = \frac1m \sum_{i=1}^m (x_i - \mu_{B})^2.
   $$

2. Нормируем и масштабируем:
   $$
   \hat x_i = \frac{x_i - \mu_{B}}{\sqrt{\sigma^2_{B} + \epsilon}},\qquad
   y_i = \gamma\,\hat x_i + \beta,
   $$
   где $\gamma,\beta$ — обучаемые параметры масштаба и сдвига.

3. На inference используем скользящие усреднённые $\mu,\sigma^2$.

### Пример (PyTorch)

```python
import torch.nn as nn

# для полносвязного слоя
bn1d = nn.BatchNorm1d(num_features=hidden_size)
# для сверточного слоя
bn2d = nn.BatchNorm2d(num_features=num_channels)

# в прямом проходе
# x: [batch, hidden_size]
y = bn1d(x)
```

## 3. Layer Normalization

Нормализация по признакам внутри каждого примера.

1. Для одного примера $x\in\mathbb R^H$ считаем:
   $$
   \mu = \frac1H \sum_{i=1}^H x_i,\quad
   \sigma^2 = \frac1H \sum_{i=1}^H (x_i - \mu)^2.
   $$

2. Нормируем и масштабируем:
   $$
   \hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},\qquad
   y_i = \gamma_i\,\hat x_i + \beta_i,
   $$
   где $\gamma,\beta\in\mathbb R^H$ — обучаемые векторы.

- **Особенности**: 
  - Не зависит от размера батча.
  - Часто используется в Transformer’ах (Pre-LN и Post-LN).

### Пример (PyTorch)

```python
import torch.nn as nn

# нормализация по последнему измерению H
ln = nn.LayerNorm(normalized_shape=hidden_size)

# в прямом проходе
# x: [batch, seq_len, hidden_size]
y = ln(x)
```

## 4. RMS Normalization (RMSNorm)

Упрощённая версия LayerNorm без вычитания среднего.

1. Вычисляем root-mean-square:
   $$
   \mathrm{rms}(x) = \sqrt{\frac1H \sum_{i=1}^H x_i^2 + \epsilon}.
   $$

2. Нормируем и масштабируем:
   $$
   y_i = \frac{x_i}{\mathrm{rms}(x)}\,g_i,
   $$
   где $g\in\mathbb R^H$ — обучаемый вектор масштабов.

- **Плюсы**: 
  - Меньше операций (нет вычитания среднего), экономия памяти и времени
  - Лучше масштабируется (градиенты не обрубаются из-за mean subtraction)
- **Минусы**:
  - Нет центрирования - может быть вредным для высокодисперсных данных, сильно смещенных от нуля (эмбеддинги)
  - Хуже работает для маленьких моделей - они менее устойчивы к шуму

### Пример реализации (PyTorch)

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., H]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.g

# Использование
rmsnorm = RMSNorm(hidden_size)
y = rmsnorm(x)  # x: [batch, seq_len, hidden_size]
```

## 5. Сравнение и выбор

| Свойство              | BatchNorm     | LayerNorm       | RMSNorm       |
|-----------------------|---------------|-----------------|---------------|
| Зависи́мость от батча | да            | нет             | нет           |
| Вычитание среднего    | да            | да              | нет           |
| Масштабируемость      | хуже при малых батчах | стабильная | стабильная   |
| Вычислительная сложность | $O(mH)$       | $O(H)$          | $O(H)$        |
| Применение            | CNN, RNN      | Transformer, RNN| Transformer   |

- **BatchNorm** хорошо работает в CNN и при больших батчах.
- **LayerNorm** — стандарт для последовательных моделей и трансформеров.
- **RMSNorm** — облегчённый вариант для экономии ресурсов при почти равной эффективности.

## 6. Pre-Layer Normalization (Pre-LN)

Pre-Layer Normalization (Pre-LN) — это схема размещения слоя нормализации до основной подструктуры Transformer (Multi-Head Attention + Feed-Forward). В отличие от классического Post-LN (LayerNorm после Residual), Pre-LN ставит LayerNorm перед каждой «ядровой» операции и остаточным добавлением:  

### 6.1. Математическая формулировка

#### 6.1.1 Pre-LN Self-Attention блок

Для входа $x$ и параметров запроса, ключа, значения:
1. Нормализуем:  
   $$\hat x = \mathrm{LayerNorm}(x).$$
2. Считаем Q, K, V:  
   $$Q = \hat x W^Q,\quad K = \hat x W^K,\quad V = \hat x W^V.$$
3. Attention и остаток:  
   $$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\bigl(\tfrac{QK^T}{\sqrt{d_k}}\bigr)V,$$  
   $$y = x + \mathrm{Attn}(Q,K,V).$$

### 6.1.2. Pre-LN Feed-Forward блок

1. Нормализация входа:  
   $$\hat y = \mathrm{LayerNorm}(y).$$
2. Двухслойный MLP с активацией (например, GELU):  
   $$z = \hat y W_1 + b_1,\quad h = \mathrm{GELU}(z),\quad o = h W_2 + b_2.$$
3. Операция остатка:  
   $$\mathrm{Output} = y + o.$$

### 6.2. Преимущества и недостатки

- **Преимущества Pre-LN**  
  - Стабильные градиенты даже при большом числе слоёв.  
  - Отсутствие необходимости в «warm-up» фазе с малым LR.  
  - Быстрая сходимость и лучшие свойства при fine-tuning.

- **Недостатки Pre-LN**  
  - Немного меньшая эффективность на этапе inference из-за дополнительных нормализаций.  
  - В некоторых задачах может давать чуть хуже генерацию длинных зависимостей.

### 6.3 Реализация в PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreLNTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-LN Attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)

        # Pre-LN Feed-Forward
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x
```

## 7. Сравнение Pre-LN и Post-LN


| Свойство               | Pre-LN                        | Post-LN                       |
|------------------------|-------------------------------|-------------------------------|
| Стабильность градиентов| высокая (не нужна warm-up)    | требует warm-up learning-rate |
| Обучение глубоких сетей| проще, сходится быстрее       | может страдать vanishing/exploding градиенты |
| Нормализация выхода    | после каждого слоя            | только после Residual Block   |
| Пропускная способность | немного ниже на inference     | чуть быстрее (меньше норм)    |

![alt text](../../0%20images/image_0.png)