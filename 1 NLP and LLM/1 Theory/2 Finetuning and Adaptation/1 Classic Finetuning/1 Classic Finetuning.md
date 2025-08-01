# Классический fine-tuning (FP32, full finetune)

URL:  
🔗 https://huggingface.co/course/chapter3


## Что такое классический fine-tuning?

Классический fine-tuning — это полная дообучаемость всех параметров модели. Подразумевает обучение модели в формате FP32 (32-битные веса), без заморозки слоёв, adapter-слоёв, LoRA и других оптимизаций.

**Когда использовать:**

- Когда у вас достаточно данных и ресурсов.
- Когда требуется максимальное качество.
- Когда модель нужно подстроить глубоко под новую задачу (например, адаптация GPT к юридическим документам).

## Мотивация

- Модель была предобучена на обобщённых данных (например, Common Crawl).
- Задача пользователя может существенно отличаться по стилю, содержанию или домену.
- Тонкая настройка (fine-tuning) позволяет адаптировать внутренние представления модели под задачу, улучшая метрики и снижение галлюцинаций.

## Как это работает

Обучение модели как обычно, с нуля — но начиная не с рандомных весов, а с весов уже предобученной модели (pretrained checkpoint).

Формально:

$$
\theta^* = \arg\min_{\theta} \sum_{(x, y) \in D} \mathcal{L}(f_\theta(x), y)
$$

Где:

- $\theta$ — параметры модели
- $D$ — обучающая выборка (fine-tune dataset)
- $\mathcal{L}$ — функция потерь (например, cross-entropy для генерации)

## Инфраструктура и инструменты

- **Фреймворки:** `transformers`, `accelerate`, `Trainer`, `deepspeed`, `bitsandbytes` (если нужно ускорение)
- **Девайсы:** GPU с поддержкой FP32, желательно >= A100 40GB при работе с LLM
- **Оптимизаторы:** AdamW (наиболее популярен)
- **Scheduler:** линейный с warm-up, cosine decay

Пример с использованием `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
)

trainer.train()
```

## Типичные параметры обучения

- batch\_size: 4–32 (в зависимости от памяти)
- learning\_rate: 2e-5 — 5e-5 (иногда меньше)
- warmup\_steps: 500–2000
- weight\_decay: 0.01

## Подводные камни

- **Catastrophic forgetting** — модель "забывает" общее знание, если fine-tune на маленьком датасете.
- **Overfitting** — особенно если мало данных и нет регуляризации.
- **Слишком высокий LR** может привести к дестабилизации весов и снижению качества.
- **Невозможность откатить** — классический fine-tune затирает веса, не давая сравнить легко до/после.

## Оценка качества

- Метрики зависят от задачи: Rouge/BLEU (summarization), Accuracy/F1 (classification), Perplexity (LM).
- Обязательно сравнение с zero-shot и few-shot baseline.

## Use-case примеры

1. **GPT2 для генерации сказок на специфичном языке** — дообучается на корпусе славянских сказок.
2. **BERT для классификации юридических текстов** — fine-tune на размеченных контрактах.
3. **T5 для генерации SQL-запросов** — полное дообучение на датасете text2sql.

## Отличия от других методов адаптации

| Метод               | Параметры      | Память       | Качество       | Применение                   |
| ------------------- | -------------- | ------------ | -------------- | ---------------------------- |
| FP32 full fine-tune | Все            | Высокая      | Высокое        | Максимальная адаптация       |
| LoRA / QLoRA        | \~0.5–2% слоёв | Низкая       | Среднее        | Ограниченные ресурсы         |
| Prompt-tuning       | Только prompt  | Очень низкая | Низкое/среднее | Быстрая адаптация, eval only |