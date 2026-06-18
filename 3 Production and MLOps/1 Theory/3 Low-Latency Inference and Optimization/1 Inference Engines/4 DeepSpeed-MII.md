# DeepSpeed-MII: Dynamic SplitFuse и инференс через DeepSpeed

## 1. Определение и мотивация

### 1.1 Проблема наивного подхода

К моменту появления DeepSpeed-MII (2023) основные inference-движки (vLLM, TGI) уже решили проблему фрагментации KV-кэша через PagedAttention. Однако оставался фундаментальный компромисс между **prefill** (обработка промпта) и **decode** (генерация токенов):

- **Prefill** — compute-bound. Промпт обрабатывается за один forward pass, GPU загружен на 100%, но latency одного запроса высокая.
- **Decode** — memory-bound. На каждом шаге генерируется один токен, GPU простаивает в ожидании данных из HBM.

При continuous batching prefill одного запроса блокирует decode всех остальных — пока большой промпт обрабатывается, GPU не может генерировать токены для уже запущенных запросов. В результате:

- **Tail latency растёт.** Один тяжёлый prefill задерживает decode для всех активных запросов.
- **GPU недогружен.** Во время decode большая часть вычислительной мощности простаивает.
- **Невозможно гарантировать SLA.** Время ответа сильно флуктуирует в зависимости от того, сколько prefill-запросов пришло одновременно.

### 1.2 Что предлагает DeepSpeed-MII

**DeepSpeed-MII (Model Implementations for Inference)** — библиотека от Microsoft, построенная поверх DeepSpeed, которая вводит принципиально другую стратегию шедулинга: **Dynamic SplitFuse** (DSF).

Ключевой вклад MII — не ещё одна реализация PagedAttention, а новый подход к распределению вычислительных ресурсов между prefill и decode внутри одного батча.

Возможности DeepSpeed-MII:

- **Dynamic SplitFuse** — длинные prefill разбиваются на несколько шагов, а освободившиеся compute-ресурсы отдаются decode.
- **Автоматический выбор стратегии** — MII сам определяет, как разбить prefill и сколько токенов decode обработать на каждом шаге.
- **Интеграция с DeepSpeed** — модель может быть сначала обучена через DeepSpeed, затем запущена на инференс через MII без конвертации.
- **Поддержка многих архитектур** — LLaMA, Mistral, Mixtral, OPT, BLOOM, GPT-NeoX, Falcon, Phi и другие.
- **Quantization** — INT8 и FP6 (через DeepSpeed-CFO).
- **Kernel injection** — автоматическая замена операций модели на оптимизированные CUDA-кернелы.
- **OpenAI-совместимый API** — `/v1/chat/completions` и `/v1/completions`.
- **GRPO Server** — поддержка on-policy RL-генераций (для обучения через Reinforcement Learning).

### 1.3 MII vs vLLM vs TGI: позиционирование

| Характеристика | DeepSpeed-MII | vLLM / TGI |
|---------------|---------------|------------|
| **Шедулинг** | Dynamic SplitFuse | Iteration-level batching (prefill → decode) |
| **KV-кэш** | ZeRO KV Cache (блочное управление) | PagedAttention |
| **Сценарий** | Высокая вариабельность длины промпта | Стабильная длина, максимальный throughput |
| **Prefill** | Разбивается на несколько шагов | За один проход (chunked prefill в новых версиях) |
| **Интеграция** | DeepSpeed (обучение → инференс) | Hugging Face / transformers |
| **Сложность** | Выше (зависимость от DeepSpeed) | Ниже (Docker-образ ready-to-use) |
| **RL-генерации** | Встроенная поддержка (GRPO Server) | Нет |

> **Когда выбирать MII:** если промпты сильно варьируются по длине (от 10 до 50K токенов), и tail latency критична. Если модель уже обучена через DeepSpeed (ZeRO-3, LoRA), MII запускает её без конвертации.

---

## 2. Основная концепция: Dynamic SplitFuse

### 2.1 Интуиция

Обычный continuous batching работает как конвейер:

```
Шаг 1: Prefill(req_A)                  # A — 1000 токенов промпта
Шаг 2: Decode(req_A) + Prefill(req_B)  # B — 2000 токенов промпта
Шаг 3: Decode(req_A) + Decode(req_B) + Prefill(req_C)
...
```

Проблема: пока prefill A обрабатывает 1000 токенов, decode для B и C не получает compute.

Dynamic SplitFuse меняет подход:

```
Шаг 1: Prefill(A, первые 256 токенов) + Decode(B, 10 токенов) + Decode(C, 5 токенов)
Шаг 2: Prefill(A, следующие 256 токенов) + Decode(B, 12 токенов) + Decode(C, 6 токенов)
Шаг 3: Prefill(A, следующие 256 токенов) + Decode(B, 8 токенов) + Decode(C, 10 токенов)
...
```

- Ни один prefill не занимает весь шаг целиком.
- Decode получает compute стабильно на каждом шаге.
- Длинные промпты не блокируют генерацию.

### 2.2 Формальное описание

На каждом шаге итерации MII решает задачу оптимального распределения токенов:

**Дано:**
- $B$ — максимальный batch size (токенов, а не запросов).
- $Q$ — очередь запросов. Каждый запрос $q$ имеет:
  - $P_q$ — длина промпта (известна).
  - $D_q$ — уже сгенерированные токены.
  - $T_q$ — max_new_tokens.
- Текущие active sequences $S$.

**На шаге $t$:**

1. Вычислить бюджет токенов: $M_t = \min(B,\; \text{доступная память})$.
2. Распределить $M_t$ между:
   - **Decode:** каждому активному запросу $s \in S$ гарантирован 1 токен decode. Всего $|S|$ токенов.
   - **Prefill:** оставшиеся токены $M_t - |S|$ распределяются между запросами из очереди $Q$.
3. Если запросов в очереди нет, все оставшиеся токены идут на decode (каждый запрос получает >1 токена — **multi-token decode**, ускорение за счёт batch-parallel execution).
4. Если декодируемый запрос завершился — немедленно замещается новым из очереди.

Ключевое отличие от классического continuous batching:

| | Classic | Dynamic SplitFuse |
|---|---|---|
| Prefill | Весь промпт за 1 шаг | Разбит на части |
| Decode | 1 токен/шаг/запрос | ≥1 токен/шаг/запрос |
| Латенси | Высокая при тяжёлом prefill | Сглаженная |
| GPU utilization | Неравномерная | Равномерная |

### 2.3 Multi-token decode

Когда в очереди нет prefill-запросов, MII не простаивает — он генерирует несколько токенов на запрос за один forward pass:

```python
# В обычном decode:
# out = model(input_ids[:, -1:])    # 1 токен на запрос

# В multi-token decode MII:
# out = model(input_ids[:, -K:])    # K токенов на запрос
# Это работает, потому что compute для decode не упирается в seqlen,
# а упирается в batch size. K одинаковых forward'ов по 1 токену
# считаются медленнее, чем 1 forward на K токенов (за счёт batch-parallel)
```

**Ограничение:** multi-token decode возможен только когда KV-кэш уже построен. На prefill этапе multi-token не применяется — там работает разбиение промпта.

---

## 3. Компоненты и механизм работы

### 3.1 Архитектура

```
┌────────────────────────────────────────────────────────────────┐
│                      DeepSpeed-MII                             │
├────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌────────────────────┐  ┌────────────────┐ │
│  │  MII Client   │  │  DeepSpeed-FastGen │  │  Model Backend │ │
│  │  (Python API) │  │  (C++/CUDA)        │  │                │ │
│  │               │  │                    │  │  ┌──────────┐  │ │
│  │  - create()   │  │  - Scheduler       │  │  │ ZeRO-KV  │  │ │
│  │  - generate() │  │  - Dynamic Split   │  │  │ Cache    │  │ │
│  │  - terminate()│  │    Fuse            │  │  └──────────┘  │ │
│  └───────────────┘  │  - Block Manager   │  │  ┌──────────┐  │ │
│                     │  - Kernel Inj.     │  │  │ Kernel   │  │ │
│                     └────────────────────┘  │  │ Injector │  │ │
│                                             │  └──────────┘  │ │
│                                             └────────────────┘ │
├────────────────────────────────────────────────────────────────┤
│                     DeepSpeed (runtime)                        │
│           ZeRO-3, NCCL, CUDA Graphs, Quantization              │
└────────────────────────────────────────────────────────────────┘
```

**MII Client** — высокоуровневый Python-интерфейс:

```python
import mii

# Создание pipeline (автоматически определяет конфигурацию)
pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")

# Генерация
result = pipe("Hello, how are you?", max_new_tokens=100)
print(result[0].generated_text)
```

**DeepSpeed-FastGen** — C++/CUDA runtime, реализующий ядро MII:

- **Scheduler** — реализация Dynamic SplitFuse.
- **Block Manager** — управление KV-кэшем (ZeRO KV Cache).
- **Kernel Injector** — замена операций модели (MLP, attention) на оптимизированные fused-кернелы.

### 3.2 ZeRO KV Cache

MII не использует PagedAttention в классическом понимании. Вместо этого он применяет **ZeRO KV Cache** — технику, заимствованную из DeepSpeed ZeRO-3:

- KV-кэш разбивается на блоки фиксированного размера (по умолчанию 64 токена).
- Блоки выделяются через block table, аналогично PagedAttention.
- **Дополнительно:** блоки могут быть выгружены в CPU (offload) при нехватке GPU-памяти — ZeRO-механизм позволяет автоматически управлять перемещением между GPU ↔ CPU.

ZeRO KV Cache опционально включает **KV-quantization** (INT8/FP6) — сжатие кэша без значительной потери качества.

### 3.3 Kernel Injection

MII автоматически заменяет операции модели на fused-кернелы при загрузке:

```python
# До: модель содержит стандартные torch-операции
# self.mlp = MLP(hidden_dim, intermediate_dim)
# self.attn = Attention(num_heads, head_dim)

# После kernel injection:
# self.mlp = FusedMLP(...)  — fused activation + matmul
# self.attn = FusedAttention(...)  — fused QKV projection + attention
```

Что заменяется:

| Операция | Исходная | Fused-кернел |
|----------|----------|--------------|
| QKV projection | 3 отдельных `nn.Linear` | Один fused matmul |
| MLP (gate/up/down) | 3 `nn.Linear` + activation | Fused MLP |
| Attention | `F.scaled_dot_product_attention` | Flash Attention + ZeRO KV Cache |
| LayerNorm | `F.layer_norm` | Fused LayerNorm |
| RoPE | Python-реализация | CUDA RoPE kernel |

Kernel injection не требует написания kernel'ов вручную — MII использует предустановленные шаблоны для каждой поддерживаемой архитектуры.

### 3.4 GRPO Server

Отдельная возможность MII — **GRPO Server** (Group Relative Policy Optimization), оптимизированный для RL-генераций:

```python
import mii

# Запуск GRPO-сервера с многократной генерацией
pipe = mii.pipeline(
    "meta-llama/Llama-3.1-8B",
    tensor_parallel=4,
    grpo_server=True,
    grpo_num_sequences=64,  # 64 генерации на промпт
)
```

**Зачем:** в RL (GRPO, PPO) на каждый промпт нужно сгенерировать 16–128 вариантов ответа. Обычные inference-движки не оптимизированы для этого сценария — они ожидают уникальные запросы. MII умеет размножать один промпт на $N$ последовательностей внутри одного forward pass, используя общий prefill (KV-кэш промпта вычисляется один раз и разделяется).

---

## 4. Примеры

### 4.1 Установка

```bash
pip install deepspeed-mii
```

MII устанавливается как pip-пакет. Зависимости (DeepSpeed, PyTorch, CUDA) должны быть установлены заранее.

> **Важно:** MII требует CUDA и совместимую GPU (NVIDIA Ampere или новее для полной поддержки kernel injection).

### 4.2 Базовое использование

```python
import mii

# Загрузка модели (скачивается с Hugging Face Hub)
pipe = mii.pipeline(
    "mistralai/Mistral-7B-v0.1",
    tensor_parallel=1,
    dtype="fp16",
)

# Простая генерация
response = pipe(
    "The capital of France is",
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
)
print(response[0].generated_text)
# "Paris. It is known for its art, fashion, and culture..."

# Пакетная генерация
batch = pipe(
    [
        "Explain quantum computing:",
        "What is machine learning?",
        "Write a poem about AI.",
    ],
    max_new_tokens=200,
)
for i, r in enumerate(batch):
    print(f"Response {i}: {r.generated_text}")
```

### 4.3 OpenAI-совместимый API

```bash
# Запуск REST-сервера
mii --model mistralai/Mistral-7B-v0.1 --port 8080
```

После запуска доступны эндпоинты:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",
)

response = client.chat.completions.create(
    model="mistral",
    messages=[
        {"role": "user", "content": "What is DeepSpeed?"},
    ],
    max_tokens=100,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 4.4 Tensor Parallelism

```python
# Запуск на 4 GPU
pipe = mii.pipeline(
    "meta-llama/Llama-2-70b-hf",
    tensor_parallel=4,      # разбиение на 4 GPU
    dtype="fp16",
    quantize="int8",         # INT8 квантизация
)
```

При tensor parallelism MII разбивает модель по слоям и по hidden dimension, используя ZeRO-3 для распределения весов.

### 4.5 GRPO-генерация (для RL)

```python
import mii

# Конфигурация для RL (много генераций на промпт)
pipe = mii.pipeline(
    "meta-llama/Llama-3.1-8B",
    grpo_server=True,
    grpo_num_sequences=32,   # 32 генерации
    grpo_max_length=2048,    # макс. длина генерации
)

# Один промпт — 32 разных completion
prompt = "Solve this math problem: 2 + 2 * 2"
outputs = pipe(prompt, num_return_sequences=32)

for i, out in enumerate(outputs):
    print(f"[{i}] {out.generated_text}")
```

**Как это работает внутри:** MII выполняет prefill для промпта один раз (вычисляет KV-кэш), затем размножает последовательности на этапе decode. KV-кэш промпта разделяется между всеми 32 последовательностями через блоки с ref_count.

### 4.6 Custom model (локальный путь)

```python
# Если модель сохранена локально
pipe = mii.pipeline(
    "/path/to/local/model",
    model_type="llama",  # явно указать архитектуру
)
```

MII пытается определить архитектуру по `config.json`. Если не получается — можно указать `model_type`.

---

## 5. Типичные ошибки и нюансы

### 5.1 Зависимость от версии DeepSpeed

MII жёстко привязан к конкретной версии DeepSpeed. При обновлении DeepSpeed может потребоваться обновление MII и наоборот:

```bash
# Проверить совместимость
pip install deepspeed-mii[ds_compat]  # устанавливает совместимую версию DS
```

Решение: не обновлять DeepSpeed и MII по отдельности — всегда вместе.

### 5.2 CUDA out of memory при первом запуске

MII при загрузке модели выделяет память под:
1. Веса модели (даже если они квантизированы).
2. KV-кэш (ZeRO KV Cache) — предварительно аллоцирует блоки.
3. Промежуточные буферы для kernel injection.

Если памяти недостаточно:

```python
pipe = mii.pipeline(
    "mistralai/Mistral-7B-v0.1",
    max_seq_length=2048,         # уменьшить макс. длину
    max_batch_tokens=4096,       # уменьшить бюджет токенов
    quantize="int8",             # включить квантизацию
)
```

### 5.3 Dynamic SplitFuse и очень короткие промпты

Если все промпты короткие (< 64 токенов), разбиение prefill неэффективно — overhead от управления блоками превышает выигрыш. В этом случае MII ведёт себя как обычный continuous batching, без преимущества DSF.

> **Ограничение:** DSF даёт наибольший выигрыш при смешанной нагрузке (короткие и длинные промпты). Если все запросы примерно одинаковой длины — vLLM или TGI могут быть быстрее.

### 5.4 Неподдерживаемые архитектуры

MII поддерживает не все модели из Hugging Face Hub. Список подтверждённых архитектур:

- LLaMA / LLaMA-2 / LLaMA-3 / LLaMA-3.1
- Mistral / Mixtral
- Falcon
- GPT-NeoX / GPT-J
- OPT
- BLOOM
- Phi-3
- Qwen-2
- Gemma

Для нетипичных архитектур (Mamba, RWKV) MII не подходит — kernel injection не сработает, и модель будет работать в режиме совместимости без оптимизаций.

### 5.5 Multi-token decode и качество

Multi-token decode (генерация нескольких токенов за один forward pass) не меняет распределения вероятностей — используется одна и та же модель, те же веса. Качество не страдает.

Однако при speculative decoding (который MII не поддерживает в явном виде) multi-token decode может давать менее уверенные результаты, т.к. модель не корректирует распределение после каждого токена.

### 5.6 MII не поддерживает LoRA напрямую

В отличие от vLLM (встроенный LoRA-сервинг) и TGI (через PEFT), MII не умеет динамически загружать LoRA-адаптеры во время инференса. Если нужен LoRA-сервинг — придётся смержить веса до загрузки:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("model")
model = PeftModel.from_pretrained(base, "lora-weights")
model = model.merge_and_unload()  # смержить LoRA в base
model.save_pretrained("/path/merged")

# Теперь загрузить merged модель через MII
pipe = mii.pipeline("/path/merged")
```

---

## 6. Когда применять

### Сценарии

- **Модель уже обучена через DeepSpeed (ZeRO-3).** MII запускает её на инференс с теми же оптимизациями — не нужно пересобирать чекпоинт для другого движка.
- **Высокая вариабельность длины промпта.** Если запросы приходят от 10 до 50K токенов — Dynamic SplitFuse сглаживает tail latency.
- **RL-генерации (GRPO, PPO).** Встроенная поддержка многократной генерации из одного промпта — значительное ускорение по сравнению с запуском N независимых инференсов.
- **Максимальное использование GPU.** Если нужно утилизировать GPU на 95%+ при смешанной нагрузке — DSF эффективнее iteration-level batching.

### Когда не подходит

- **Стабильная короткая нагрузка.** Если все промпты < 128 токенов — оверхед DSF не окупается, vLLM/TGI дают тот же throughput с меньшими накладными расходами.
- **Нужен LoRA-сервинг.** vLLM с поддержкой LoRA удобнее — не требует мёржить адаптеры заранее.
- **Модель не из поддерживаемых архитектур.** MII не запустит Mamba, RWKV или кастомную архитектуру с полной оптимизацией.
- **Простота эксплуатации.** TGI и vLLM имеют готовые Docker-образы, которые запускаются одной командой. MII требует установки DeepSpeed и совместимой CUDA-среды.

---

## 7. Вопросы для самопроверки

1. Чем Dynamic SplitFuse отличается от iteration-level batching в vLLM/TGI?
2. Почему multi-token decode возможен только когда KV-кэш уже построен?
3. Как MII обрабатывает GRPO-генерации — почему prefill выполняется один раз на N последовательностей?
4. В каком сценарии Dynamic SplitFuse не даёт выигрыша по сравнению с обычным continuous batching?
5. Какое ограничение MII по сравнению с vLLM в контексте LoRA-адаптеров?