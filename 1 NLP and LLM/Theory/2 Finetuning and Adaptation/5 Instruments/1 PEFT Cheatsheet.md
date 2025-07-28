# Шпаргалка по PEFT (Parameter-Efficient Fine-Tuning)

## Что такое PEFT

PEFT (Parameter-Efficient Fine-Tuning) — это подходы, позволяющие адаптировать большие языковые модели (LLM) с минимальным числом изменяемых параметров, не обучая всю модель целиком.

### Зачем нужен PEFT:

* Экономия памяти и GPU
* Быстрая адаптация к новой задаче или домену
* Упрощение повторного использования модели

## Основные техники

| Метод| Идея| Где применяется|
| ------------- | ------------------------------------------------------- | ---------------------------- |
| LoRA          | Встраиваются низкоранговые матрицы в веса attention/MLP | Классика PEFT                |
| QLoRA         | LoRA + 4bit quantization через bitsandbytes             | Огромные модели на одном GPU |
| Prefix Tuning | Предобучаемые токены-префиксы добавляются к input       | Диалоговые задачи            |
| Prompt Tuning | Токены обучения играют роль learnable prompt            | Zero-shot сценарии           |
| Adapters      | Маленькие нейросети вставляются между слоями            | Универсальный способ         |

## Практика (на Hugging Face)

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
peft_model = get_peft_model(base_model, lora_cfg)
peft_model.print_trainable_parameters()
```

## Когда использовать PEFT

| Ситуация                                    | Подход                   |
| ------------------------------------------- | ------------------------ |
| Требуется минимизация VRAM и быстрый запуск | QLoRA                    |
| Есть ресурсы и нужен максимум качества      | LoRA с float16           |
| Нужна адаптация промптов, а не модели       | Prompt или Prefix Tuning |
| Планируется повторное использование слоёв   | Adapters                 |

## Советы по практике

* Использовать `BitsAndBytesConfig(load_in_4bit=True)` для QLoRA
* Профилировать обучение с помощью `accelerate`, `deepspeed`, `torch.compile()`
* Замораживать базовую модель через `model.requires_grad_(False)`

## Инструменты и окружение

* `peft`, `transformers`, `bitsandbytes`, `accelerate`, `deepspeed`
* Поддерживаются задачи: классификация, генерация текста, Seq2Seq, токенизация, QA и др.

## Ограничения

* Не все модели и задачи совместимы с QLoRA (например, не всегда доступна 4-bit quantization)
* PEFT может проигрывать full fine-tuning на больших датасетах, если ресурсы позволяют
