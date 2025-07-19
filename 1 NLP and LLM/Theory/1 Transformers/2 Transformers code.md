# Конспект «Self-Attention from Scratch» (Sebastian Raschka, 2023)

URL: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

## 1. Ментальная модель Self-Attention

1. **Входные эмбеддинги**  
   Пусть  
   $$
   X \in \mathbb{R}^{n 	\times d_{	{model}}}
   $$  
   где $n$ — длина последовательности, $d_{	ext{model}}$ — размер скрытого пространства.

2. **Линейные проекции**  
   Для каждой позиции вычисляем три представления:
   $$
   Q = X W^Q,\quad
   K = X W^K,\quad
   V = X W^V,
   $$
   где  
   $$
   W^Q, W^K, W^V \in \mathbb{R}^{d_{	{model}} 	\times d_k},
   $$  
   а $d_k$ — размерность пространства «ключей» и «запросов».  

---

## 2. Scaled Dot-Product Attention

1. **Счёт внимания**  
   $$
   S = \frac{Q K^T}{\sqrt{d_k}}
   \quad\in\;\mathbb{R}^{n \times n}.
   $$

2. **Softmax по строкам**  
   $$
   A = \mathrm{softmax}(S)
   $$

3. **Взвешенное суммирование**  
   $$
   \mathrm{Attention}(Q, K, V) = A V
   \quad\in\;\mathbb{R}^{n \times d_k}.
   $$

---

## 3. Реализация: «loops» vs матричное умножение

- **Наивная реализация (медленно)**  
  ```python
  scores = torch.zeros(n, n)
  for i in range(n):
      for j in range(n):
          scores[i, j] = (Q[i] * K[j]).sum() / math.sqrt(d_k)
  A = softmax(scores, dim=-1)
  output = A @ V
  ```
- **Векторизованная версия (быстро)**  
  ```python
  scores = (Q @ K.T) / math.sqrt(d_k)
  A = torch.softmax(scores, dim=-1)
  output = A @ V
  ```

---

## 4. Causal Masked Self-Attention (декодер)

Чтобы модель не «заглядывала в будущее», вводим маску  
$$
M_{i,j} =
\begin{cases}
  0,      & j \le i,\
  -\infty,& j > i,
\end{cases}
\quad M\in\{0,-\infty\}^{n	\times n}.
$$  
И считаем:
$$
\mathrm{MaskedAttention}(Q, K, V)
= \mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}} + M\Bigr)\,V.
$$

---

## 5. Multi-Head Attention

1. **Разбиение на $H$ голов**  
   Для каждой головы $h = 1,\dots,H$:
   $$
   Q_h = Q W_h^Q,\quad
   K_h = K W_h^K,\quad
   V_h = V W_h^V,
   $$
   где $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d_{	{model}} \times d_k}$.

2. **Вычисление по головам**  
   $$
   \mathrm{head}_h = \mathrm{Attention}(Q_h, K_h, V_h)
   \quad\in\;\mathbb{R}^{n 	\times d_k}.
   $$

3. **Конкатенация и финальная проекция**  
   $$
   \mathrm{MultiHead}(Q, K, V)
   = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H) W^O,
   $$
   где  
   $$
   W^O \in \mathbb{R}^{H d_k 	\times d_{	{model}}}.
   $$

---

## 6. Реализация SelfAttention в PyTorch

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]
        B, N, _ = x.size()
        # Линейные проекции и reshape
        Q = self.Wq(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        # Scaled dot-product
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        A = torch.softmax(scores, dim=-1)
        # Контекст
        context = (A @ V)
        context = context.transpose(1, 2).reshape(B, N, -1)
        return self.Wo(context)
```

---

## 7. Тестирование и визуализация

- **Проверка эквивалентности** циклов и матричного умножения:

  ```python
  assert torch.allclose(scores_loop, scores_vec)
  ```

- **Отображение тепловой карты**:

  ```python
  import seaborn as sns
  sns.heatmap(A[0, 0].detach().cpu())
  ```

---