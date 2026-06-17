# vLLM: PagedAttention и высокопроизводительный инференс

## 1. Что такое vLLM и зачем он нужен

### 1.1 Проблема наивного подхода

При генерации текста большие языковые модели (LLM) работают авторегрессивно — на каждом шаге:

1. Принимают уже сгенерированные токены как вход
2. Вычисляют распределение вероятностей для следующего токена
3. Сэмплируют следующий токен

Для модели с $N$ параметров и контекстом длины $L$:

* **KV-кэш** (key-value cache) хранит attention-состояния для всех предыдущих токенов
* Размер KV-кэша на один токен: $2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}}$
* Для LLaMA-7B: $\approx 1.6$ GB на 1K токенов контекста
* Для LLaMA-70B: $\approx 80$ GB на 1K токенов

**Проблема:** при параллельной обработке множества запросов с разной длиной контекста:

* Память выделяется с запасом (worst case)
* Фрагментация памяти при освобождении
* Невозможность эффективно использовать GPU при длинных последовательностях

### 1.2 Что предлагает vLLM

**vLLM** — open-source движок для инференса LLM, разработанный UC Berkeley. Ключевые инновации:

* **PagedAttention** — механизм управления KV-кэшем как страницами виртуальной памяти
* Эффективное управление памятью с минимальными потерями
* Поддержка attention-механизмов: MHA, MQA, GQA, grouped-query attention
* Tensor parallelism для распределённых вычислений
* OpenAI-совместимый API

**Результаты** (из оригинальной статьи):

| Метрика | Naive | vLLM | Улучшение |
|---------|-------|------|-----------|
| Throughput | 1.0x | 2.2–24x | до 24x |
| Memory utilization | ~20% | ~60–80% | ~3-4x |

---

## 2. PagedAttention: ключевая идея

### 2.1 Аналогия с виртуальной памятью

В традиционных ОС виртуальная память решает проблему фрагментации через paging:

* Память делится на страницы фиксированного размера (обычно 4KB)
* Физическая память выделяется постранично
* Неиспользуемые страницы могут быть выгружены на диск

**PagedAttention** применяет тот же принцип к KV-кэшу:

* KV-кэш хранится в виде **блоков фиксированного размера** (по 16 токенов)
* Блоки выделяются **по требованию** (on-demand)
* Неиспользуемые блоки освобождаются без фрагментации

### 2.2 Структура блока

```
┌─────────────────────────────────────────────┐
│              Block Table Entry              │
├─────────────────────────────────────────────┤
│  physical_block_num: int    # номер блока   │
│  block_size: int            # 16 токенов    │
│  token_offset: int          # позиция в seq │
│  ref_count: int             # счётчик ссылок│
│  is_full: bool              # заполнен?     │
└─────────────────────────────────────────────┘
```

### 2.3 Пример: обработка нескольких запросов

**Запрос A:** "Hello, world!" (13 токенов)
**Запрос B:** "The quick brown fox jumps" (7 токенов)

Без PagedAttention (наивный подход):
```
Request A: [block_0: 16 tokens allocated, 13 used, 3 wasted]
Request B: [block_1: 16 tokens allocated, 7 used, 9 wasted]
Total waste: 12 tokens
```

С PagedAttention:
```
Request A: [block_0: 16 tokens, 13 used]
Request B: [block_1: 16 tokens, 7 used]
           [block_2: shared, 9 tokens from A]
Total waste: 0 tokens (blocks reused efficiently)
```

### 2.4 Алгоритм PagedAttention

```python
def paged_attention(query, key_cache, value_cache, block_tables):
    """
    query: [batch_size, num_heads, head_dim]
    key_cache: [num_blocks, num_heads, block_size, head_dim]
    value_cache: [num_blocks, num_heads, block_size, head_dim]
    block_tables: [batch_size, seq_len] - маппинг логич.позиций в физ.блоки
    """
    output = zeros_like(query)
    
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            q = query[batch_idx, head_idx]  # [head_dim]
            
            # Проходим по всем логическим позициям
            for seq_pos in range(seq_len):
                # Получаем физический блок для этой позиции
                block_num = block_tables[batch_idx, seq_pos]
                block_offset = seq_pos % BLOCK_SIZE
                
                # Извлекаем K/V из блока
                k = key_cache[block_num, head_idx, block_offset]
                v = value_cache[block_num, head_idx, block_offset]
                
                # Вычисляем attention score
                score = dot_product(q, k) / sqrt(head_dim)
                output[batch_idx, head_idx] += softmax(scores) * v
    
    return output
```

---

## 3. Архитектура vLLM

### 3.1 Компоненты системы

```
┌─────────────────────────────────────────────────────────────┐
│                        vLLM Engine                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Scheduler │  │  Block Mgr  │  │   Worker Processes  │  │
│  │             │  │             │  │                     │  │
│  │ - Decoding  │  │ - Paged     │  │ - Model execution   │  │
│  │ - Batching  │  │   KV Cache  │  │ - CUDA kernels      │  │
│  │ - Scheduling│  │ - Memory    │  │ - Tensor parallel   │  │
│  └─────────────┘  │   alloc     │  └─────────────────────┘  │
│                   └─────────────┘                            │
├─────────────────────────────────────────────────────────────┤
│                    CUDA/ROCm Backend                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Scheduler

**Scheduler** отвечает за:

* Планирование запросов на выполнение
* Формирование батчей для эффективного использования GPU
* Управление последовательностями (sequence management)

**Три режима планирования:**

1. **Decoding** — для интерактивных запросов, минимальная latency
2. **Prefill** — предварительная обработка промптов
3. **Chunked prefill** — разбиение prefill на части для баланса throughput/latency

```python
# Пример конфигурации scheduler
scheduler_config = {
    "policy": "fcfs",  # или "priority"
    "max_num_seqs": 256,      # макс. запросов в батче
    "max_num_batched_tokens": 4096,  # макс. токенов в батче
    "prefill_chunk_size": 512,  # размер чанка для chunked prefill
}
```

### 3.3 Block Manager

**Block Manager** управляет физической памятью KV-кэша:

* Выделение/освобождение блоков
* Отслеживание использования блоков (ref counting)
* Обработка прерываний при нехватке памяти

```python
# Внутренняя структура block_manager
class BlockManager:
    def __init__(self, num_gpu_blocks, block_size=16):
        self.num_blocks = num_gpu_blocks
        self.block_size = block_size
        
        # Таблица отображения: sequence_id -> list of blocks
        self.block_tables: Dict[str, List[PhysicalBlock]] = {}
        
        # Свободные блоки
        self.free_blocks: Set[int] = set(range(num_gpu_blocks))
    
    def allocate(self, num_required_blocks):
        """Выделить блоки для новой последовательности"""
        if len(self.free_blocks) < num_required_blocks:
            # Eviction: вытеснение редко используемых блоков
            self._evict_blocks(num_required_blocks)
        
        blocks = []
        for _ in range(num_required_blocks):
            block_id = self.free_blocks.pop()
            blocks.append(PhysicalBlock(id=block_id))
        
        return blocks
    
    def free(self, sequence_id):
        """Освободить блоки последовательности"""
        for block in self.block_tables.pop(sequence_id, []):
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.add(block.id)
```

### 3.4 Worker Process

**Worker** выполняет вычисления:

* Загрузка модели на GPU
* Forward pass через модель
* Custom CUDA kernels для PagedAttention

---

## 4. Memory Management

### 4.1 Типы памяти в vLLM

| Тип | Описание | Пример (LLaMA-7B) |
|-----|----------|-------------------|
| Model weights | Параметры модели | ~14 GB |
| KV cache | Attention states | ~16 GB (max) |
| Activations | Промежуточные вычисления | ~2 GB |
| Temp buffers | Временные буферы | ~1 GB |

### 4.2 Расчёт необходимой памяти

Формула для KV-кэша:

$$
\text{KV cache size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_param} \times \text{max\_sequences} \times \text{max\_tokens\_per\_seq}
$$

Для LLaMA-7B (bfloat16):

```python
# Расчёт памяти KV-кэша
def calculate_kv_cache_memory(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    bytes_per_param: int = 2,  # bfloat16
    max_num_seqs: int = 256,
    max_tokens_per_seq: int = 2048,
):
    tokens_per_block = 16
    num_blocks = (max_num_seqs * max_tokens_per_seq) / tokens_per_block
    
    memory_per_token = 2 * num_layers * num_heads * head_dim * bytes_per_param
    total_memory = memory_per_token * max_num_seqs * max_tokens_per_seq
    
    # В гигабайтах
    return total_memory / (1024 ** 3)

# LLaMA-7B: ~16 GB для KV-кэша
print(f"KV cache: {calculate_kv_cache_memory():.2f} GB")
```

### 4.3 Memory profiling

vLLM автоматически профилирует память:

```python
from vllm import LLM, SamplingParams

# При инициализации происходит профилирование
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,  # 90% GPU памяти под KV cache
    max_model_len=4096,
)

# vLLM вычисляет:
# num_gpu_blocks = (total_gpu_memory * 0.9 - model_weights - activations) / block_size
```

### 4.4 Eviction policy

Когда память заканчивается:

1. **LRU (Least Recently Used)** — вытесняются давно используемые последовательности
2. **Swapping** — блоки могут быть перемещены в CPU RAM (опционально)

```python
# Конфигурация eviction
cache_config = {
    "gpu_memory_utilization": 0.9,
    "swap_space": 8,  # GB для CPU swap
}
```

---

## 5. Batching и Scheduling

### 5.1 Проблема variable-length sequences

При обработке запросов с разной длиной:

* Naive batching: padding до max length → wasted computation
* Static batching: фиксированные батчи → poor utilization

### 5.2 Continuous batching (iteration-level scheduling)

**Continuous batching** — формирование батчей на каждой итерации:

```
Time →
─────────────────────────────────────────────────────────────►

Iteration 1: [Req A: ████████░░░░] [Req B: ██████░░░░░░] [Req C: ████░░░░░░░░]
Iteration 2: [Req A: ████████████] [Req B: ████████░░░░] [Req D: ██████░░░░░░]
Iteration 3: [Req A: DONE        ] [Req B: ████████████] [Req D: ████████░░░░]
Iteration 4: [Req B: DONE        ] [Req D: ████████████] [Req E: ████░░░░░░░░]
```

**Преимущества:**

* Завершённые запросы заменяются новыми
* GPU всегда загружен
* Throughput значительно выше, чем static batching

### 5.3 Prefill vs Decode

**Prefill phase** — обработка входного промпта:

* Высокий параллелизм (все токены промпта обрабатываются параллельно)
* Heavy computation, но делается один раз

**Decode phase** — генерация токенов по одному:

* Низкий параллелизм (один токен за итерацию)
* Memory-bound операции

```python
# Конфигурация для разных сценариев
# Низкая latency (интерактивные чат-боты)
llm = LLM(
    model="...",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    enable_chunked_prefill=True,  # Разбивает prefill на части
    max_num_batched_tokens=512,   # Меньше = ниже latency
)

# Высокий throughput (батчевая обработка)
llm = LLM(
    model="...",
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    enable_chunked_prefill=False,
    max_num_batched_tokens=4096,  # Больше = выше throughput
)
```

### 5.4 Приоритизация запросов

```python
# Приоритет через metadata
from vllm import Request

# Запрос с высоким приоритетом
high_priority_req = Request(
    request_id="req_1",
    prompt="...",
    sampling_params=SamplingParams(temperature=0),
    priority=10,  # Чем выше, тем раньше
)

# Запрос с низким приоритетом
low_priority_req = Request(
    request_id="req_2",
    prompt="...",
    sampling_params=SamplingParams(temperature=0),
    priority=1,
)
```

---

## 6. Tensor Parallelism

### 6.1 Зачем нужен tensor parallelism

Для больших моделей (70B+ параметров) одна GPU не вмещает модель:

* LLaMA-70B в bfloat16: ~140 GB
* NVIDIA A100: 80 GB
* NVIDIA H100: 80 GB (или 188 GB с HBM3)

**Tensor parallelism** — разбиение модели по тензорным измерениям:

* Attention: разбиение по heads
* MLP: разбиение по hidden dimension

### 6.2 Tensor Parallelism в vLLM

```python
# Запуск на 4 GPU с tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 GPU
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# vLLM автоматически:
# 1. Разбивает веса по GPU
# 2. Настраивает коммуникацию (NCCL)
# 3. Распределяет вычисления
```

### 6.3 Архитектура распределённых вычислений

```
┌─────────────────────────────────────────────────────────────┐
│                    Tensor Parallelism                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   GPU 0          GPU 1          GPU 2          GPU 3        │
│   ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐     │
│   │ W_0 │        │ W_1 │        │ W_2 │        │ W_3 │     │
│   │     │◄──────►│     │◄──────►│     │◄──────►│     │     │
│   │     │  NCCL  │     │  NCCL  │     │  NCCL  │     │     │
│   └─────┘        └─────┘        └─────┘        └─────┘     │
│       │            │            │            │             │
│       └────────────┴────────────┴────────────┘             │
│                        AllReduce                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 Пример: Attention с tensor parallelism

Для модели с 8 heads и TP=4:

```
Input Q: [batch, 8, head_dim]
         ↓ partition
Q_0: [batch, 2, head_dim] → GPU 0
Q_1: [batch, 2, head_dim] → GPU 1
Q_2: [batch, 2, head_dim] → GPU 2
Q_3: [batch, 2, head_dim] → GPU 3

Каждая GPU считает partial attention для своих heads
AllReduce на выходе для получения финального результата
```

---

## 7. Установка и базовое использование

### 7.1 Установка

```bash
# Через pip (требует CUDA 12.1+)
pip install vllm

# Или из исходников для максимальной производительности
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# Проверка установки
python -c "import vllm; print(vllm.__version__)"
```

### 7.2 Базовый пример

```python
from vllm import LLM, SamplingParams

# Инициализация модели
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# Параметры сэмплирования
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# Генерация
prompts = [
    "The capital of France is",
    "To make pizza, you need to",
    "The theory of relativity states that",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("---")
```

### 7.3 OpenAI-совместимый API сервер

```bash
# Запуск сервера
vllm serve meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# Или через Python
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

**API endpoints:**

```bash
# Completions API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "The capital of France is",
        "max_tokens": 100,
        "temperature": 0.7
    }'

# Chat completions API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100
    }'
```

### 7.4 Интеграция с OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM не требует API key
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=200,
)

print(response.choices[0].message.content)
```

---

## 8. Продвинутые возможности

### 8.1 Guided decoding

Ограничение вывода форматом (JSON, regex и т.д.):

```python
from vllm import SamplingParams

# JSON constrained generation
sampling_params = SamplingParams(
    max_tokens=256,
    guided_decoding_backend="outlines",  # или "lm-format-enforcer"
)

# Определяем схему
import json
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    },
    "required": ["name", "age"]
}

outputs = llm.generate(
    prompts=["Generate a user profile:"],
    sampling_params=sampling_params,
    guided_decoding_backend="outlines",
)
```

### 8.2 LoRA support

```python
from vllm import LLM, LoraConfig

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_lora_rank=16,
)

# Загрузка LoRA адаптера
lora_config = LoraConfig(
    loraui_adapter_path="/path/to/lora/adapter",
    loraui_base_model_name="meta-llama/Llama-2-7b-hf",
)

# Генерация с LoRA
outputs = llm.generate(
    prompts=["Hello"],
    sampling_params=SamplingParams(max_tokens=100),
    lora_request=lora_config,
)
```

### 8.3 Prefix caching

Кэширование общих префиксов:

```python
# Промпты с общим префиксом
prompts = [
    "[SYSTEM_PROMPT] Summarize: The quick brown fox...",
    "[SYSTEM_PROMPT] Summarize: A lazy dog...",
    "[SYSTEM_PROMPT] Summarize: Machine learning is...",
]

# vLLM автоматически:
# 1. Вычисляет хэш префикса [SYSTEM_PROMPT]
# 2. Кэширует KV для префикса
# 3. Переиспользует при обработке всех запросов
```

### 8.4 Speculative decoding

Ускорение генерации через предсказание:

```python
# Требует дополнительную "speculative" модель
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    speculative_model="meta-llama/Llama-2-1.3b-hf",  # Меньшая модель
    num_speculative_tokens=5,  # Сколько токенов предсказывать
)
```

---

## 9. Мониторинг и отладка

### 9.1 Метрики Prometheus

```bash
# Включение метрик
vllm serve model --enable-metrics

# Метрики доступны на /metrics
curl http://localhost:8000/metrics
```

**Ключевые метрики:**

| Метрика | Описание |
|---------|----------|
| `vllm:num_requests_running` | Активные запросы |
| `vllm:num_tokens_generated_total` | Всего сгенерировано токенов |
| `vllm:gpu_cache_usage_perc` | Использование KV-кэша |
| `vllm:time_to_first_token_seconds` | TTFT (latency) |
| `vllm:time_per_output_token_seconds` | TPOT (per-token latency) |

### 9.2 Логирование

```python
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

llm = LLM(model="...")

# Подробные логи планировщика
# DEBUG: Scheduling decisions, block allocation
# INFO: Request throughput, token counts
```

### 9.3 Профилирование

```python
import torch
from vllm import LLM, SamplingParams

# Profiling с PyTorch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    outputs = llm.generate(["Hello world"] * 100, SamplingParams(max_tokens=50))

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

---

## 10. Бенчмарки и производительность

### 10.1 Сравнение throughput

**Конфигурация теста:**

* Model: LLaMA-2-7B
* Input: 512 tokens
* Output: 128 tokens
* Hardware: NVIDIA A100 80GB

| Engine | Throughput (req/s) | Throughput (tok/s) |
|--------|-------------------|-------------------|
| Naive HF | 2.3 | 290 |
| HF + Dynamic Batching | 8.1 | 1,040 |
| vLLM | 24.2 | 3,100 |
| TensorRT-LLM | 28.5 | 3,650 |

### 10.2 Факторы производительности

```python
# Параметры, влияющие на throughput

# 1. Batch size
llm = LLM(
    model="...",
    max_num_seqs=256,           # Больше = выше throughput
    max_num_batched_tokens=4096,
)

# 2. GPU memory utilization
llm = LLM(
    model="...",
    gpu_memory_utilization=0.95,  # Больше = больше KV cache
)

# 3. Chunked prefill
llm = LLM(
    model="...",
    enable_chunked_prefill=True,  # Лучше для смешанной нагрузки
    max_num_batched_tokens=512,
)

# 4. CUDA graphs
llm = LLM(
    model="...",
    enforce_eager=False,  # Включает CUDA graphs
)
```

### 10.3 Latency vs Throughput tradeoff

```
High Throughput (batch_size=256):
┌────────────────────────────────────────────────────────────┐
│ Batch of 256 requests                                     │
│ Processing time: 1000ms                                   │
│ Time per request: 1000ms / 256 ≈ 4ms                      │
│ Total tokens/sec: 256 * 128 = 32,768 tok/s                │
└────────────────────────────────────────────────────────────┘

Low Latency (batch_size=1):
┌────────────────────────────────────────────────────────────┐
│ Single request                                            │
│ Processing time: 50ms                                     │
│ Time per request: 50ms                                    │
│ Total tokens/sec: 128 tok/s                               │
└────────────────────────────────────────────────────────────┘
```

---

## 11. Практические рекомендации

### 11.1 Конфигурация для разных сценариев

**Интерактивный чат-бот:**

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    enable_chunked_prefill=True,
    max_num_batched_tokens=512,  # Меньше для никой latency
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["<|eot_id|>"],
)
```

**Batch processing (summarization):**

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    enable_chunked_prefill=False,
    max_num_batched_tokens=8192,  # Больше для высокого throughput
)

sampling_params = SamplingParams(
    temperature=0.0,  # Детерминированно для batch
    max_tokens=256,
)
```

**Длинные контексты (RAG):**

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.85,  # Больше памяти под длинные seqs
    max_model_len=16384,  # Поддержка длинных контекстов
    block_size=32,  # Больший размер блока
)
```

### 11.2 Оптимизация памяти

```python
# 1. Используйте квантизацию
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    quantization="awq",  # или "gptq", "squeezellm"
    tensor_parallel_size=4,
)

# 2. Настройте swap для больших батчей
llm = LLM(
    model="...",
    gpu_memory_utilization=0.8,  # Оставить место для батчей
    swap_space=16,  # GB CPU swap
)

# 3. Prefill chunking для длинных промптов
llm = LLM(
    model="...",
    enable_chunked_prefill=True,
    max_num_batched_tokens=1024,  # Размер чанка
    prefill_chunk_size=1024,
)
```

### 11.3 Обработка ошибок

```python
from vllm import LLM, SamplingParams
from vllm.errors import VLLM_VALUE_ERROR, InvaildTokenStopError

llm = LLM(model="...")

try:
    outputs = llm.generate(prompts, sampling_params)
except VLLM_VALUE_ERROR as e:
    # Проблема с входными данными
    print(f"Invalid input: {e}")
except InvaildTokenStopError as e:
    # Проблема с stop tokens
    print(f"Stop token issue: {e}")
except Exception as e:
    # Другие ошибки
    print(f"Unexpected error: {e}")
```

---

## 12. Интеграция с ML-пайплайнами

### 12.1 LangChain

```python
from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate

llm = VLLM(
    model="meta-llama/Llama-2-7b-hf",
    trust_remote_code=True,
    max_model_len=4096,
    tensor_parallel_size=1,
)

template = """Question: {question}

Answer: Let's think step by step:"""

prompt = PromptTemplate.from_template(template)
chain = prompt | llm

result = chain.invoke({"question": "What is 2 + 2?"})
print(result)
```

### 12.2 Ray Serve

```python
import ray
from ray import serve
from vllm import LLM

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class vLLMDeployment:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name, tensor_parallel_size=1)
    
    def generate(self, prompt: str, **kwargs) -> str:
        from vllm import SamplingParams
        sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

# Запуск
ray.init()
serve.run(vLLMDeployment.bind("meta-llama/Llama-2-7b-hf"))
```

### 12.3 Kubernetes deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "80Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "80Gi"
        args:
        - "--model"
        - "meta-llama/Llama-2-7b-hf"
        - "--gpu-memory-utilization"
        - "0.9"
        - "--max-model-len"
        - "4096"
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-inference
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 13. Ограничения и известные проблемы

### 13.1 Текущие ограничения

| Ограничение | Описание | Workaround |
|-------------|----------|------------|
| Attention | Только MHA/MQA/GQA | Ожидание FlashAttention-3 |
| Quantization | Ограниченная поддержка | AWQ/GPTQ, нет INT8 для attention |
| Multi-node | Требует NCCL | Для 70B+ моделей |
| Dynamic batching | Не perfect | Chunked prefill |

### 13.2 Troubleshooting

**CUDA OOM:**

```python
# Уменьшить использование памяти
llm = LLM(
    model="...",
    gpu_memory_utilization=0.7,  # Вместо 0.9
    max_model_len=2048,  # Вместо 4096
)
```

**Низкий throughput:**

```python
# Проверить настройки
llm = LLM(
    model="...",
    enforce_eager=False,  # Включить CUDA graphs
    max_num_seqs=256,
    max_num_batched_tokens=4096,
)
```

**Медленная инициализация:**

```python
# Для больших моделей
llm = LLM(
    model="...",
    download_dir="/path/to/cache",  # Кэш модели
    enforce_eager=False,  # Компиляция
)
```

---

## 14. Чек-лист

### Установка и настройка

- [ ] Установлен vLLM (`pip install vllm`)
- [ ] Проверена совместимость CUDA версии
- [ ] Достаточно GPU памяти (проверить `nvidia-smi`)

### Конфигурация

- [ ] Выбран правильный `gpu_memory_utilization` (0.8–0.95)
- [ ] Настроен `max_model_len` под задачу
- [ ] Включён `enable_chunked_prefill` для интерактивных сценариев
- [ ] Настроен `tensor_parallel_size` для больших моделей

### Производительность

- [ ] Проведён бенчмарк throughput/latency
- [ ] Настроен batch size для оптимальной утилизации
- [ ] Включены CUDA graphs (`enforce_eager=False`)
- [ ] Настроен prefix caching для повторяющихся промптов

### Мониторинг

- [ ] Настроены метрики Prometheus
- [ ] Логируется TTFT и TPOT
- [ ] Отслеживается использование GPU памяти
- [ ] Настроен алертинг на OOM

### Production

- [ ] Настроен health check endpoint
- [ ] Настроен graceful shutdown
- [ ] Подготовлен Dockerfile
- [ ] Настроен autoscaling (HPA или KEDA)
- [ ] Документированы лимиты модели

---

## 15. Источники

* **Оригинальная статья:** "Efficient Memory Management for Large Language Model Serving Using PagedAttention" (Kwon et al., 2023)
* **GitHub:** https://github.com/vllm-project/vllm
* **Документация:** https://docs.vllm.ai/
* **Blog post:** https://vllm.ai/