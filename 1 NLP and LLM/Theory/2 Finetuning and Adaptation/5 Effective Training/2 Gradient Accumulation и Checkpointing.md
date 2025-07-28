# Gradient Accumulation и Checkpointing

При обучении больших LLM моделей часто не хватает GPU‑памяти на желаемый batch size. В этом случае помогают два ключевых метода: **градиентное накопление (gradient accumulation)** и **чекпойнтинг активаций (activation checkpointing)**.

---

## Gradient Accumulation (накопление градиентов)

**Зачем нужно:** если память не позволяет прогонять большой batch сразу, мы можем делать несколько «мини-батчей» и аккумулировать градиенты.

### Как это работает:
1. Разбиваем один большой batch на $k$ мини-батчей.
2. На каждом шаге делаем forward и backward, **но не обновляем веса**.
3. Градиенты накапливаются.
4. После $k$ шагов — делаем **один шаг оптимизации**.

Это позволяет симулировать обучение с batch_size = $k \cdot \text{mini\_batch}$ без увеличения пикового потребления памяти.

### Пример (PyTorch):

```python
accum_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accum_steps  # делим loss
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Минусы:
- Увеличивает время одной эпохи
- Не помогает с памятью на forward (активации)

---

## Activation Checkpointing (чекпойнтинг активаций)

**Зачем нужно:** чтобы не хранить в памяти все активации для backward, а пересчитывать их при необходимости.

### Принцип работы:
1. Делим модель на блоки (например, по слоям).
2. Сохраняем только входы в эти блоки.
3. При backward повторно прогоняем forward для нужных блоков, чтобы восстановить активации.

Уменьшает использование памяти на активации, особенно при длинных последовательностях.

### Пример (PyTorch):

```python
from torch.utils.checkpoint import checkpoint

def custom_forward(*inputs):
    return model.layer(inputs[0])

outputs = checkpoint(custom_forward, inputs)
```

### Пример с HuggingFace:

```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.gradient_checkpointing_enable()
```

### Минусы:
- Увеличивает время обучения на 20–30%
- Требует аккуратного планирования блоков

---

## Вместе: Gradient Accumulation + Checkpointing

Обычно эти техники используются совместно:
- Checkpointing снижает пиковую память активаций
- Accumulation позволяет обучать с большим batch size

Вместе они позволяют обучать модели в условиях ограниченной памяти, без потери качества.

---

## Вывод

| Метод                  | Экономит память | Замедляет обучение | Сложность реализации |
|-----------------------|------------------|---------------------|------------------------|
| Gradient Accumulation | частично         | немного             | просто                |
| Checkpointing         | сильно           | умеренно            | средняя               |

