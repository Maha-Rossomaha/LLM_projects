# Feed-Forward Layer в LLM (Transformer)

## 1. Назначение и роль
Feed-Forward Network (FFN) — один из ключевых компонентов в архитектуре Transformer (вместе с Attention и нормализациями). Он отвечает за **нелинейную трансформацию признаков** после attention-механизма и работает **независимо для каждого токена**.

**Зачем нужен:**
- усиливает выразительность модели (многоступенчатая нелинейная проекция);
- обеспечивает обработку информации вне зависимости от порядка токенов (в отличие от attention);
- служит локальной обработкой, в отличие от глобального attention.

## 2. Классическая структура FFN
В архитектуре Transformer FFN выглядит как:

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2
$$

Где:
- $x$ — вход (обычно размерности $d_{model}$),
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$,
- $d_{ff}$ — размер внутреннего слоя (обычно в 4 раза больше, чем $d_{model}$),
- активация: ReLU или GELU (чаще GELU в LLM).

## 3. Пример кода на PyTorch
```python
import torch.nn as nn

class TransformerFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)
```

## 4. Разновидности и улучшения FFN

### a. SwiGLU / Gated Linear Units
Изменяет FFN следующим образом:

$$
\text{FFN}(x) = W_2 \cdot (\text{SwiGLU}(W_1 x, V_1 x))
$$

где:
$$
\text{SwiGLU}(a, b) = \text{SiLU}(a) \cdot b
$$

Используется в PaLM, GPT-4, Mistral.

### b. MoE (Mixture-of-Experts)
Вместо одной FFN активируются $k$ из $N$ возможных подслоёв (экспертов):

- роутинг: top-k по значению gate(x);
- сохраняется sparsity (активно только ~10% параметров);
- значительно увеличивает параметры без роста latency.

Используется в GLaM, SwitchTransformer, Mixtral.

### c. FFN с Depthwise/Conv слоем
Добавляется depthwise 1D-свёртка между линейными слоями для локальной агрегации.

### d. Low-rank адаптация (LoRA) внутри FFN

FFN может быть адаптирован LoRA-слоями (как и Attention), если требуется parameter-efficient tuning.

## 5. Где используется FFN

FFN входит в каждый Transformer-блок в:
- BERT (encoder-only),
- GPT (decoder-only),
- T5 (encoder-decoder),
- LLaMA, Mistral, Falcon, Claude, Gemini и др.

Также применяется в:
- ViT (Vision Transformer),
- Audio Transformers,
- Multimodal LLMs (например, Flamingo).

## 6. Прочее

- FFN слои чаще всего работают **позиционно-независимо** — т.е. применяются отдельно к каждому токену (позиции).
- Основной bottleneck FFN — вторая проекция (из $d_{ff}$ в $d_{model}$), особенно при больших $d_{ff}$.
- В современных реализациях FFN **фьюзят** в один матричный умножитель или kernel для ускорения.