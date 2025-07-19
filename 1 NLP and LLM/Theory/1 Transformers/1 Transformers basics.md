# Формальное изложение механизма Transformer

URL: https://jalammar.github.io/illustrated-transformer/

## 1. Входные эмбеддинги и проекции Q, K, V

Пусть $X \in \mathbb{R}^{n \times d_{\mathrm{model}}}$ — матрица входных эмбеддингов для последовательности длины $n$, где $d_{\mathrm{model}}$ — размер скрытого пространства. Матрицы запросов $Q$, ключей $K$ и значений $V$ вычисляются как:
$$
Q = X W^Q,\quad
K = X W^K,\quad
V = X W^V,
$$
где $W^Q, W^K, W^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, а $d_k$ задаёт размерность пространства запросов и ключей. Обозначим $d_v = d_k$.

## 2. Scaled Dot-Product Attention

Для каждой головы внимания вводится матрица весов внимания:
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\Bigl(\frac{Q K^\top}{\sqrt{d_k}}\Bigr)\,V.
$$
Деление на $\sqrt{d_k}$ нормирует распределение, предотвращая сдвиг softmax при больших значениях скалярных произведений.

## 3. Masked Self-Attention в декодере

Для предотвращения «глядения в будущее» вводится маска $M \in \{0, -\infty\}^{n \times n}$:
$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\Bigl(\frac{Q K^\top}{\sqrt{d_k}} + M\Bigr)\,V,
$$
где $M_{i,j} = -\infty$ при $j > i$, что гарантирует автогрессивность генерации.

## 4. Multi-Head Attention

Механизм многоголового внимания реализуется как объединение $H$ параллельных голов:
$$
\mathrm{head}_h = \mathrm{Attention}(Q W_h^Q,\;K W_h^K,\;V W_h^V),\quad h=1,\dots,H,
$$
$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)\,W^O,
$$
где $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$ и $W^O \in \mathbb{R}^{H d_k \times d_{\mathrm{model}}}$.

## 5. Cross-Attention (Encoder–Decoder Attention)

В блоке внимания энкодер-декодер запросы формируются из скрытых представлений декодера $X_{dec}$, а ключи и значения — из выходов энкодера $X_{enc}$:
$$
Q = X_{dec} W^Q,\quad
K = X_{enc} W^K,\quad
V = X_{enc} W^V.
$$

$$
\mathrm{CrossAttention}(Q, K, V) 
= \mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr)\,V.
$$

## 6. Структура слоя Transformer

Каждый слой содержит:
1. **Multi-Head Attention** с residual connection и пред- или пост-LayerNorm.
2. **Position-wise Feed-Forward Network**:
$$
\mathrm{FFN}(x) = \mathrm{GELU}(x W_1 + b_1)\,W_2 + b_2.
$$
3. **LayerNorm** для стабильности градиентов и ускорения обучения.