# Position Encoding

Позиционное кодирование (Position Encoding) необходимо трансформерам для учета порядка токенов, поскольку они обрабатывают входные данные параллельно, без рекуррентной структуры.

---

## Absolute Positional Encoding

### Что это

Абсолютное позиционное кодирование задает уникальное положение каждого токена фиксированной функцией.

### Варианты:

- **Sinusoidal** (фиксированное): изначально предложено в "Attention Is All You Need" (2017).
- **Learned Embeddings**: позиции обучаются как обычные эмбеддинги.

### Формула (Sinusoidal):

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{\frac{2i}{d_{model}}}) \\
PE_{(pos, 2i+1)} = \cos(pos / 10000^{\frac{2i}{d_{model}}})
$$

### Где использовалось

- Sinusoidal: оригинальный Transformer
- Learned: BERT, GPT-2

### Преимущества

- Sinusoidal: безобучаемое, переносимое на длинные последовательности
- Learned: выше качество на коротких последовательностях

### Недостатки

- Ограничена длиной обучения (learned)
- Плохо экстраполирует (особенно learned)

### Пример реализации

```python
class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

```python
class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        return x + self.pe(pos)
```

---

## Relative Positional Encoding

### Что это

Закодированы **относительные расстояния** между токенами, а не абсолютные позиции.

### Где использовалось

- Transformer-XL
- DeBERTa

### Формула (примерно):

Изменяется attention score:

$$
A_{i,j} = Q_i K_j^T + Q_i R_{i-j}^T + u^T K_j + v^T R_{i-j}
$$

где $R_{i-j}$ — embedding относительной позиции.

### Преимущества

- Лучше экстраполирует на длины больше обучающей
- Учитывает локальные зависимости

### Недостатки

- Сложнее в реализации
- Не всегда переносимо между архитектурами

### Особенности

- **Attention bias**: добавка к score
- **Раздельное кодирование для $Q$ и $K$**: как в DeBERTa, вводится независимая позиционная компонента

### Пример реализации

```python
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_relative_position: int):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.rel_embedding = nn.Embedding(2 * max_relative_position + 1, d_model)

    def forward(self, length_q: int, length_k: int):
        # Создаём матрицу относительных позиций [length_q x length_k]
        range_q = torch.arange(length_q)[:, None]
        range_k = torch.arange(length_k)[None, :]
        relative_positions = range_k - range_q
        relative_positions = relative_positions.clamp(-self.max_relative_position, self.max_relative_position)
        relative_positions += self.max_relative_position  # чтобы быть в диапазоне индексов
        return self.rel_embedding(relative_positions)
```
---

## Rotary Positional Embedding (RoPE)

### Что это

Представление позиции в виде **вращения векторов** в комплексном пространстве.

### Где использовалось

- GPT-NeoX, LLaMA, ChatGLM

### Принцип

Вектора $Q$ и $K$ поворачиваются на угол, соответствующий позиции:

$$
\text{RoPE}(x, pos) = x \cdot R(pos)
$$

Реализация: чередующиеся координаты $(x_1, x_2)$ преобразуются по формуле поворота:

$$
\begin{bmatrix} x_1' \\ x_2' \end{bmatrix} = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

### Преимущества

- Отлично масштабируется на большие контексты
- Позиционность «встраивается» прямо в attention
- Совместим с KV-кешем

### Недостатки

- Неинтуитивна
- Нет явной интерпретируемости положения

### Пример реализации

```python
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, d_head: int, theta: int = 10_000):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d = d_head
        self.theta = theta
        
        # Углы theta_i
        freqs = theta ** (-2 * torch.arange(self.d // 2) / self.d)
        position_id = torch.arange(0, self.max_seq_len).float()
        
        # нужно получить матрицу m theta_i размера [max_seq_len, self.d] вида m theta_i
        # где m берется из position_id, а theta из freqs
        idx_theta = torch.einsum('i, j -> ij', position_id, freqs)
        
        # max_seq_len, d_head
        cos = idx_theta.cos()
        sin = idx_theta.sin()
        
        # нужно продублировать размерности. theta_i встерчается два раза подряд в синусах и косинусах
        # тут нам поможет torch.repeat_interleave
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        
        # 1, max_seq_len, 1, d_head
        self.register_buffer("sin", sin.unsqueeze(1).unsqueeze(0))
        self.register_buffer("cos", cos.unsqueeze(1).unsqueeze(0))
    
    @staticmethod
    def rotate_neg_vector(x: Float[torch.Tensor, "batch seq_len num_heads d_head"]):
        # На входе x = [x1, x2, x3, x4, ... x_{n-1}, x_n]
        # На выходе x' = [-x2, x1, -x4, x3, ..., -x_n, x_{n-1}]
        x_new = torch.empty_like(x)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_new = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        return x_new
    
    def forward(self, x: Float[torch.Tensor, "batch seq_len num_heads d_head"]):
        old_device = x.device
        seq_len = x.size(1)
        x_rotated = self.rotate_neg_vector(x)
        res = x * self.cos[:, :seq_len, :, :] + x_rotated * self.sin[:, :seq_len, :, :]
        return res
```

---

## Dynamic Positional Encoding

### Что это

Позиционность вычисляется **динамически**, на основе текущего контекста или длины запроса.

### Где использовалось

- ALiBi (Attention with Linear Biases)
- Используется в некоторых RL-подходах (Reinforcement Learning)

### ALiBi: Linear Biases

В attention добавляется линейный штраф в зависимости от расстояния:

$$
A_{i,j} = Q_i K_j^T - m_h \cdot |i - j|
$$

где 
- $i$ — позиция запроса (query)
- $j$ — позиция ключа (key)
- $m_h$ — индивидуальный наклон (slope) для каждой attention‑головы $h$  

Таким образом:
- ближайшие к $i$ токены получают меньший штраф
- дальние — сильный штраф


### Преимущества

- Почти не требует параметров
- Без обрезания позиции при длинных последовательностях
- Очень быстро и легко реализуется

### Недостатки

- Потенциально ограничена выразительность
- Не позволяет различать токены на одинаковом расстоянии от запроса

### Пример реализации
```python
class AlibiBias:
    def __init__(self, num_heads: int, max_seq_len: int):
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.slopes = self._get_slopes(num_heads)

        # Считаем bias по формуле: bias[i, j] = -slope * distance(i, j)
        # Shape: [num_heads, 1, max_seq_len]
        position_ids = torch.arange(max_seq_len).unsqueeze(0)
        rel_dist = position_ids - position_ids.T
        self.alibi = -self.slopes.view(num_heads, 1, 1) * rel_dist.abs().unsqueeze(0)

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor([start * ratio**i for i in range(n)])

        import math
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power = 2 ** math.floor(math.log2(n))
            return torch.cat([
                get_slopes_power_of_2(closest_power),
                self._get_slopes(2 * closest_power)[0::2][:n - closest_power]
            ])

    def get_bias(self, seq_len: int, device: torch.device):
        return self.alibi[:, :, :seq_len].to(device)  # [num_heads, seq_len, seq_len]
```
---

## Сравнительная таблица

| Метод                  | Годы  | Ключевые модели         | Плюсы                                                  | Минусы                                             |
| ---------------------- | ----- | ----------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| Absolute (Sinusoidal)  | 2017  | Transformer             | Простота, переносимость на большие длины               | Необучаемость, ограниченная гибкость               |
| Absolute (Learned)     | 2018 | BERT, GPT‑2             | Гибкость, точность в коротком контексте                | Плохо масштабируется, фиксированная длина          |
| Relative | 2018  | Transformer‑XL, DeBERTa | Учет расстояний, обобщаемость                          | Более сложная реализация, больше параметров        |
| RoPE                   | 2021  | LLaMA, GPT‑NeoX         | Быстро, совместимо с KV‑кешем, хорошее масштабирование | Нет явной интерпретации позиции                    |
| ALiBi                  | 2022  | T5‑X, Dolly             | Простота, линейность, экстраполяция                    | Потеря точной позиции, одна и та же для всех слоев |


