# QLoRA — полный и точный конспект

## Что такое QLoRA?
**QLoRA (Quantized LoRA)** — это метод эффективной адаптации больших языковых моделей, сочетающий:
- **Quantization** — 4-битное сжатие весов модели, позволяющее работать с LLM на обычных GPU;
- **LoRA** — дообучение с помощью низкоранговых адаптаций без изменения оригинальных весов.

Таким образом, QLoRA позволяет:
- загружать LLM в **int4** формате с минимальными потерями точности,
- обучать **LoRA-адаптацию** поверх квантованных весов,
- сводить использование GPU-памяти к минимуму при сохранении качества.

QLoRA делает возможным fine-tuning LLM объёмом 33B+ даже на 1 GPU (например, A100 40GB).

## Основная идея
В QLoRA не требуется обновлять веса модели напрямую. Вместо этого:
1. Квантованные веса модели (например, $W$) загружаются как read-only слои;
2. На каждую проекцию attention (обычно `q_proj` и `v_proj`) добавляются обучаемые LoRA-слои: 

$$
W' = \text{Quant}(W) + A B
$$

где:
- $\text{Quant}(W)$ — 4-битное представление весов, загружаемое с помощью `bitsandbytes`,
- $A, B$ — low-rank матрицы размера $d \times r$ и $r \times k$, где $r \ll d$.

В процессе обучения фиксированные квантованные веса используются только для прямого прохода, а градиенты текут только через LoRA.

## Компоненты QLoRA
1. **4-bit quantization** с алгоритмом **NF4 (NormalFloat4)** — квантование к квазинормальному распределению с высокой точностью.
2. **Double Quantization** — сами таблицы квантования тоже подвергаются 8-битному сжатию, экономя память.
3. **LoRA adaptation** — добавляются low-rank обучаемые матрицы (обычно в attention-слои).
4. **Paged Optimizer** — используется `paged_adamw_8bit`, позволяющий эффективно выгружать неактивные тензоры в RAM.

## Инструменты и инфраструктура
- `transformers` (HuggingFace) — модель и токенизатор
- `peft` — конфигурация и вставка LoRA-слоёв
- `bitsandbytes` — загрузка 4-bit весов и оптимизаторы
- `accelerate` / `deepspeed` — ускорение обучения
- `TRL` — для RLHF/SFT

### Пример инициализации модели с QLoRA
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

## Типичные параметры и best practices
- `r`: 8–64 — ранг low-rank разложения (обычно 16–64)
- `lora_alpha`: 16–128 — масштабирует градиенты $AB$
- `lora_dropout`: 0.05–0.1
- `bnb_4bit_quant_type`: "nf4" предпочтительнее "fp4"
- `bnb_4bit_compute_dtype`: bfloat16 (или float16 при отсутствии поддержки)
- `optimizer`: `paged_adamw_8bit` или `paged_lion_8bit`
- `batch_size`: зависит от GPU (пример: 16 с gradient accumulation)
- `learning_rate`: 2e-4 — 5e-4 (чуть выше, чем для full fine-tune)
- `lr_scheduler`: cosine + warmup_ratio ≈ 0.03–0.1

## Сравнение качества и эффективности
По результатам статьи Dettmers et al. (2023):
- QLoRA на 65B-модели достигает аналогичного качества (по BLEU, Exact Match и др.), что и full FP16 fine-tune;
- экономия VRAM до 70%, обучение на 1 GPU (A100 80GB) становится возможным;
- inference можно делать в 4bit + LoRA или после слияния (merging) LoRA в full-precision веса.

## Use-case примеры
1. **Инструкционное дообучение (Alpaca, OASST, DPO)** — под диалоговый стиль
2. **Медицина / финансы** — адаптация LLaMA на доменных датасетах
3. **SFT + RLHF** — последовательная настройка на пользовательские предпочтения
4. **Чат-боты на 24GB VRAM** — запуск LLaMA-2-13B + дообучение в consumer-среде

## Подводные камни
- Требуется архитектура с поддержкой 4bit через `bitsandbytes`
- Не все модели корректно указывают target_modules (нужен manual inspect)
- У некоторых моделей может не хватить precision — желательно использовать bfloat16
- Плохой merge LoRA-слоёв может испортить веса (если требуется экспорт)

## Отличия от других подходов
| Метод          | Вес модели   | VRAM         | Качество        | Обучаемые параметры |
|----------------|--------------|--------------|------------------|----------------------|
| FP32 fine-tune | FP32         | Очень высокая | Максимум       | Все                 |
| LoRA           | FP16         | Средняя       | Высокое         | ~1–2%               |
| QLoRA          | INT4 + LoRA  | Низкая        | Высокое         | ~1–2%               |
| Prompt-tuning  | FP32         | Очень низкая  | Среднее         | Только prompt       |

## Вывод
QLoRA — это мощный и доступный способ адаптировать LLM к конкретным задачам даже на одном GPU. Комбинируя:
- низкоразрядное квантование (NF4),
- LoRA-адаптацию,
- и современные оптимизаторы,

он делает fine-tuning возможным без огромных затрат. Особенно эффективен для:
- instruction tuning,
- пользовательской генерации,
- low-resource deployment с качеством, близким к full fine-tuning.

