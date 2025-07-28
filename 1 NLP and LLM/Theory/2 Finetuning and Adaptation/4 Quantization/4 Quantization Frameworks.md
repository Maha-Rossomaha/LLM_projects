# Фреймворки для квантования

## 1. bitsandbytes

**bitsandbytes** — это библиотека, ориентированная на эффективный inference и обучение с использованием low-bit квантования. Является частью экосистемы HuggingFace.

### Поддерживаемые режимы:

* **8bit quantization** (LLM.int8() в Transformers):

  * Квантование весов до INT8 во время загрузки модели.
  * Используется во многих LLM-инференсах как компромисс между точностью и производительностью.
* **4bit quantization (experimental):**

  * Поддерживает NF4 (NormalFloat4) и FP4 форматы.
  * Требует `load_in_4bit=True` + `bnb_4bit_config` в Transformers.
  * Часто используется в комбинации с QLoRA для обучения.

### Пример использования

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    load_in_8bit=True,  # для 8bit
    device_map="auto"
)
```

или для 4bit:

```python
from transformers import BitsAndBytesConfig

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
```

### Удобства

* Интеграция с HuggingFace Transformers.
* Поддержка `device_map="auto"` и `accelerate`.
* Совместимость с QLoRA и PEFT.
* Высокая производительность на A100, T4, RTX 30xx.

### Недостатки

* **4bit режим нестабилен** для некоторых моделей.
* Только weight-only квантование.
* Нет полной поддержки activation quantization.
* Ограничения на совместимость с non-GPU (например, CPU).

---
## 2. AutoGPTQ

**AutoGPTQ** — это специализированный фреймворк для выполнения **GPTQ-квантования**, ориентированный на inference больших языковых моделей (LLM) с 4-bit весами. Является де-факто стандартом для GPTQ и тесно интегрирован с HuggingFace Transformers.

### Поддерживаемые возможности:

* **Квантование весов в INT4** с использованием codebook и scale.
* **Group-wise quantization** (обычно по 128 весов).
* Совместимость с weight-only моделями.
* Экспорт квантованной модели в формате HuggingFace (`.safetensors`).
* Поддержка модели с prequantized `.json` конфигом + `.safetensors` весами.

### Пример использования (загрузка квантованной модели)

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_safetensors=True,
    quantize_config=None  # если уже квантована
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
```

### Удобства

* Простая загрузка квантованных моделей через `from_quantized()`.
* Поддержка множества моделей из `TheBloke` и других репозиториев.
* Возможность кастомного квантования с выбором group size, damp%, act-order и пр.
* Интеграция с text-generation-webui, llama.cpp, ExLlama.

### Недостатки

* Только weight-only квантование (активации остаются FP16/FP32).
* Не подходит для quant-aware training (QAT).
* Большие модели требуют длительного времени на квантование.
* Ограниченная поддержка non-GPT архитектур (например, T5, BERT).

---

## 3. llama.cpp + GGUF

**llama.cpp** — это высокоэффективная C/C++ реализация инференса LLaMA и других моделей, оптимизированная для CPU. Вместе с форматом **GGUF (Grokking General Unified Format)** он позволяет запускать LLM на Mac, Windows, Linux и даже Android.

### Основные особенности:

* Поддержка INT4, INT5, INT6, INT8 и FP16 моделей.
* Полностью автономная, не требует Python или CUDA.
* Использует AVX, AVX2, AVX512, NEON (на ARM), Metal (на Mac) для ускорения.
* Поддерживает многопоточность и стриминг токенов.
* Совместим с форматом GGUF, в который можно конвертировать модели с помощью `llama.cpp` tools.

### Пример запуска

```bash
./main -m ./models/llama-2-7b.Q4_K_M.gguf -p "Hello, how are you?"
```

### Удобства

* Работает на **CPU**: подходит для старых ПК, ноутбуков, телефонов.
* Поддерживает множество платформ: Mac (Metal), Linux, Windows, Android.
* Отличная оптимизация под x86 и ARM.
* Очень маленький footprint и бинарный размер.
* Множество обёрток: `llama-cpp-python`, `text-generation-webui`, `Ollama`, `LM Studio`, `KoboldCpp`, `Dalai`.

### Недостатки

* Не поддерживает fine-tuning / обучение.
* Кастомизация и расширение сложнее, чем в PyTorch/Transformers.
* Только weight-only модели, без QAT и quant-aware logic.
* Использует собственный формат GGUF, требует конвертацию.

### Где используется

* **LM Studio**, **Ollama**, **KoboldCpp**
* Локальный оффлайн-инференс LLM на MacBook / Windows-ноутбуке
* Разработка веб-интерфейсов с CPU-инференсом

---

## 4. LLM.int8() (HuggingFace Transformers)

`LLM.int8()` — это функция внутри библиотеки HuggingFace Transformers, реализующая **8-битное квантование** во время загрузки модели. Основана на интеграции с `bitsandbytes` и позволяет загружать большие LLM с существенно меньшими требованиями к памяти.

### Основные особенности:

* Квантование происходит **на лету** при загрузке модели (`load_in_8bit=True`).
* Используется **int8 weight-only** квантование.
* Автоматически активируется при наличии `bitsandbytes` и поддерживаемого GPU.
* Позволяет запускать модели до 30B на одной GPU с 16–24 ГБ VRAM.

### Пример использования

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
```

### Удобства

* Простая активация через `load_in_8bit=True`
* Полностью интегрирован в `transformers`
* Совместим с `accelerate`, `deepspeed`, `PEFT`, `LoRA`, `QLoRA`
* Работает с широким спектром моделей: GPT2, OPT, BLOOM, LLaMA, Mistral и др.

### Недостатки

* Только INT8 (нет INT4 / NF4)
* Только weight-only (активации остаются float)
* Требуется совместимый GPU (с поддержкой `bitsandbytes` + CUDA >= 11.6)
* Нестабильность на некоторых кастомных архитектурах

### Где используется

* Обучение с QLoRA и memory-efficient inference
* Прототипы с большим количеством моделей на одной GPU
* FastAPI / Gradio / streamlit-сервисы с LLM

---

## 5. AWQ (Activation-aware Weight Quantization)

**AWQ** — это быстрый метод постобучающего квантования, нацеленный на **веса модели** с учетом их влияния на **активации**, что помогает сохранить точность LLM при использовании 4-битных представлений.

Разработан как альтернатива GPTQ, но без необходимости сложной оптимизации выходов. Используется в llama.cpp, HuggingFace, web-интерфейсах.

### Основные особенности:

* Только **weight-only** quantization (активации не трогаются)
* Использует схему **WNAM** (Weight-only Non-uniform Asymmetric Mixed quantization):

  * INT4 индексы + индивидуальный codebook на группу весов
  * scale и zero-point — отдельные для каждой группы
* Учитывает **активации** при подборе весов (но не квантует их)

### Пример использования

```bash
# Квантование модели через CLI
python3 awq/quantize.py \
  --model llama2-7b \
  --wbits 4 \
  --version awq \
  --output_path ./llama2-7b-awq
```

или через PyTorch:

```python
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained(
    "llama2-7b-hf",
    quant_file="llama2-7b-awq"
)
```

### Удобства

* Быстрая конвертация (в разы быстрее GPTQ)
* Хорошая точность даже на INT4
* Поддержка в `text-generation-webui`, `llama.cpp`, `hf_transformers`
* Можно собирать `.gguf` для CPU-инференса

### Недостатки

* Только веса, активации остаются float
* Не поддерживает INT8 или комбинированные режимы
* В меньшей степени настраиваемый, чем GPTQ

### Где используется

* `llama.cpp` (AWQ → GGUF)
* `text-generation-webui` (вкладка awq)
* `autoawq` и CLI-инструменты
* inference на Mac/Windows/RTX с INT4
