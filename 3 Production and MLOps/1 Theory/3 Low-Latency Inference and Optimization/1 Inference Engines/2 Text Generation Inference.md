# Text Generation Inference (TGI): инференс LLM от Hugging Face

## 1. Определение и мотивация

### 1.1 Проблема наивного подхода

Наивный инференс LLM через Hugging Face `transformers` сталкивается с теми же проблемами, что и любой другой движок:

- **KV-кэш** потребляет огромное количество GPU-памяти. Для LLaMA-7B — ~0.5 MB на токен, для LLaMA-70B — ~2.6 MB на токен. На 4K контекста и 256 запросов память под KV-кэш исчисляется десятками гигабайт.

- **Continuous batching** отсутствует. `transformers.pipeline` обрабатывает запросы последовательно или фиксированными батчами. GPU простаивает, когда короткие запросы завершаются раньше длинных.

- **Нет управления памятью.** KV-кэш выделяется непрерывными блоками с запасом под `max_new_tokens`. Фрагментация неизбежна при параллельной обработке.

- **Отсутствуют оптимизации** вроде Flash Attention, квантизации, tensor parallelism, speculative decoding «из коробки».

### 1.2 Что предлагает TGI

**Text Generation Inference (TGI)** — open-source движок для инференса LLM от Hugging Face. Запущен в 2023 году как решение для продакшн-нагрузок на HF Inference Endpoints.

Ключевые возможности:

- **Continuous batching** (iteration-level scheduling) — батч переформировывается на каждом шаге decode.
- **PagedAttention** — управление KV-кэшем через блочную индексацию (с версии 2.0, реализация собственной, не от vLLM).
- **Flash Attention 2** — оптимизированный CUDA-кernel для attention.
- **Квантизация** — GPTQ, AWQ, bitsandbytes (8/4 bit), Marlin, EETQ, FP8.
- **Tensor Parallelism** — разбиение модели по нескольким GPU через NCCL.
- **Speculative Decoding** — ускорение через меньшую «assistant»-модель.
- **Prefix Caching** — автоматическое кэширование общих префиксов.
- **Streaming** — вывод токенов по мере генерации через SSE.
- **Message API** — поддержка chat-формата (system/user/assistant).
- **OpenAI-совместимый API** — `/v1/chat/completions` и `/v1/completions`.

### 1.3 TGI vs vLLM: коротко

| Характеристика | TGI | vLLM |
|---------------|-----|------|
| Разработчик | Hugging Face | UC Berkeley |
| PagedAttention | Собственная (с 2.0) | Своя оригинальная |
| Квантизация | GPTQ, AWQ, bitsandbytes, Marlin, EETQ, FP8 | GPTQ, AWQ, FP8, SqueezeLLM |
| Flash Attention | FA v2 (через `flash-attn`) | FA v2, flashinfer |
| LoRA | Да (через `peft`) | Да |
| Guided decoding | Outlines / lm-format-enforcer | Outlines / lm-format-enforcer |
| Интеграция с HF Hub | Нативная (модели скачиваются через `huggingface_hub`) | Через `transformers` |
| Docker-образ | Официальный (рекомендуемый способ) | Есть |
| OpenAI API | `/v1/chat/completions` + `messages` | `/v1/chat/completions` |

> TGI — выбор, когда важна интеграция с экосистемой Hugging Face (Hub, Inference Endpoints, Datasets, PEFT). vLLM часто даёт чуть более высокий throughput на чистом инференсе без квантизации.

---

## 2. Архитектура TGI

### 2.1 Компоненты

```
┌──────────────────────────────────────────────────────────┐
│                        TGI Server                        │
├──────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │   Router     │   │  Scheduler   │   │   Workers    │  │ │  │              │   │              │   │              │  │
│  │  (HTTP/gRPC) │   │              │   │  (GPU procs) │  │
│  │              │   │ - Continuous │   │              │  │
│  │ - /generate  │   │   batching   │   │ - Model exec │  │
│  │ - /chat      │   │ - Prioritize │   │ - FA2 kernels│  │
│  │ - /metrics   │   │ - Queue mgmt │   │ - TP comms   │  │
│  └──────────────┘   └──────────────┘   └──────────────┘  │
├──────────────────────────────────────────────────────────┤
│          CUDA Backend (Flash Attention + Paged)          │
└──────────────────────────────────────────────────────────┘
```

**Router** — HTTP/gRPC-сервер, принимающий запросы:
- Парсинг входных данных (prompt, chat history, sampling parameters).
- Проксирование запросов к worker'ам через gRPC.
- SSE-стриминг (каждый декодированный токен отправляется клиенту по мере готовности).
- Экспорт Prometheus-метрик.

**Scheduler** — планировщик итераций:
- Управление очередью запросов.
- Формирование батча на каждом шаге (continuous batching).
- Распределение памяти под KV-кэш (PagedAttention block manager).

**Workers** — GPU-процессы:
- Загрузка модели (включая квантизированные веса).
- Выполнение forward pass.
- Коммуникация при tensor parallelism (NCCL).

### 2.2 Router и gRPC-коммуникация

В отличие от vLLM, где scheduler и worker живут в одном процессе (или в нескольких через Ray), TGI использует **gRPC** для связи router → worker:

```
Client HTTP ──► Router ──gRPC──► Worker (GPU)
   (SSE stream) ◄─── gRPC stream ◄───
```

**Зачем gRPC:**
- Эффективная бинарная сериализация (protobuf) — меньше overhead, чем HTTP/JSON.
- Bidirectional streaming — токены передаются от worker к router по мере генерации, router сразу шлёт их клиенту через SSE.
- Лёгкое масштабирование — можно запустить несколько worker'ов за одним router.

### 2.3 Continuous Batching в TGI

TGI использует iteration-level scheduling, как и vLLM, но с особенностями:

1. **Prefill** — все токены промпта обрабатываются за один forward pass (без chunked prefill в базовой версии, появился позже).
2. **Decode** — на каждом шаге генерируется по одному токену на запрос.
3. **Завершённые запросы** — немедленно заменяются новыми из очереди.

```python
# Упрощённая логика scheduler TGI
class TGIScheduler:
    def __init__(self, max_batch_size=256, max_total_tokens=4096):
        self.queue = []
        self.active = []
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens

    def step(self):
        # 1. Добавляем новые запросы из очереди в активный батч
        while self.queue and len(self.active) < self.max_batch_size:
            req = self.queue.pop(0)
            self.active.append(PreparedRequest(req))

        # 2. Выполняем один шаг decode/prefill
        outputs = self.model_forward(self.active)

        # 3. Убираем завершённые, добавляем новые
        for i, (req, out) in enumerate(zip(self.active, outputs)):
            if req.is_finished:
                req = self.queue.pop(0) if self.queue else None
                self.active[i] = req

        return outputs
```

---

## 3. Установка и запуск

### 3.1 Docker (рекомендуемый способ)

```bash
# Запуск LLaMA-2-7B на одной GPU
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id meta-llama/Llama-2-7b-hf \
    --max-total-tokens 4096

# Docker run с квантизацией AWQ
docker run --gpus all \
    -p 8080:80 \
    -e HF_TOKEN=$HF_TOKEN \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id TheBloke/Llama-2-7B-AWQ \
    --quantize awq \
    --max-total-tokens 4096
```

**Параметры запуска:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--model-id` | (обязательный) | ID модели на HF Hub |
| `--max-total-tokens` | 2048 | Макс. длина контекста (prompt + generation) |
| `--max-batch-size` | 32 | Макс. запросов в батче |
| `--max-batch-prefill-tokens` | 4096 | Макс. токенов в prefill-батче |
| `--quantize` | None | `gptq`, `awq`, `bitsandbytes`, `marlin`, `eetq`, `fp8` |
| `--num-shard` | 1 | Количество GPU для TP |
| `--max-input-length` | — | Макс. длина промпта |
| `--cuda-graphs` | — | Включить CUDA graphs |

### 3.2 Установка через pip

```bash
pip install text-generation

# Запуск сервера через Python
text-generation-launcher \
    --model-id meta-llama/Llama-2-7b-hf \
    --port 8080
```

### 3.3 Hugging Face Inference Endpoints

TGI — это движок, на котором работают HF Inference Endpoints. При создании endpoint'а на Hugging Face выбирается конфигурация GPU, и TGI запускается автоматически.

```
HF Hub → Deploy → Inference Endpoints → Выбрать GPU → Endpoint на TGI
```

---

## 4. API и примеры использования

### 4.1 OpenAI-совместимый API (рекомендуемый)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",  # TGI не требует API key локально
)

# Chat Completion
response = client.chat.completions.create(
    model="tgi",  # любое значение
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=200,
    stream=True,  # SSE streaming
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 4.2 TGI Client (huggingface_hub)

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="http://localhost:8080",
)

# Текстовая генерация
output = client.text_generation(
    prompt="The capital of France is",
    max_new_tokens=10,
    temperature=0.7,
    stream=True,
)

for token in output:
    print(token, end="", flush=True)

# Chat
messages = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is attention?"},
]

stream = client.chat_completion(
    messages=messages,
    max_tokens=200,
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.get("content", ""), end="")
```

### 4.3 Streaming через cURL

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tgi",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5:"}
        ],
        "max_tokens": 50,
        "stream": true
    }'
```

Ответ приходит чанками `data: {"choices":[{"delta":{"content":"1"}}]}` по SSE.

### 4.4 Text Generation Inference (TGI) native endpoint

```bash
# POST /generate — текстовая генерация
curl http://localhost:8080/generate \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "The meaning of life is",
        "parameters": {
            "max_new_tokens": 20,
            "temperature": 0.9,
            "top_p": 0.95
        }
    }'

# POST /generate_stream — стриминг
curl http://localhost:8080/generate_stream \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "Explain gravity simply:",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.8
        }
    }'
```

### 4.5 Генерация с detokenization

TGI поддерживает **detokenization** (декодирование токенов в текст) на стороне сервера. Это значит, что клиент получает уже читаемый текст, а не ID токенов. Все примеры выше работают именно так.

Однако можно запросить и "сырые" токены:

```bash
curl http://localhost:8080/generate \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "Hello",
        "parameters": {
            "max_new_tokens": 5,
            "decoder_input_details": true,
            "details": true
        }
    }'
```

Ответ с `details: true` включает:
- `prefill` — список токенов промпта с их logprobs.
- `tokens` — сгенерированные токены с logprobs.
- `generated_text` — финальный текст.

---

## 5. Ключевые оптимизации

### 5.1 PagedAttention

С версии 2.0 TGI использует собственную реализацию PagedAttention, решающую проблему фрагментации KV-кэша точно так же, как vLLM:

- KV-кэш разбивается на блоки (по умолчанию 16 токенов).
- Block table связывает логические позиции последовательности с физическими блоками на GPU.
- Блоки выделяются on-demand и освобождаются атомарно.

**Разница с vLLM:** vLLM перехватывает вызовы `torch` через monkey-patch оригинального attention (подменяет `F.scaled_dot_product_attention` на `paged_attention`). TGI использует интеграцию на уровне модели через `transformers` — модель грузится через `AutoModelForCausalLM`, но forward pass модифицирован так, чтобы использовать PagedAttention-кэш.

```python
# TGI модифицирует forward модели
# Вместо:
# cache = cache_class(...)      # непрерывный тензор
# out = model(input_ids, past_key_values=cache)

# TGI использует:
# class PagedCache:
#   blocks: [num_blocks, 2, num_heads, block_size, head_dim]
#   block_table: Dict[seq_id, List[int]]
#   ref_counts: List[int]
# out = model(input_ids, past_key_values=PagedCache(block_table))
```

### 5.2 Flash Attention 2

TGI использует **Flash Attention 2** — fused-трик для вычисления attention, который:
1. Избегает материализации матрицы `S = QK^T` (память $O(N^2)$ → $O(N)$).
2. Использует tiling по query и key-блокам — обрабатывает матрицы частями, храня их в shared memory.
3. Пересчитывает softmax на лету (online softmax).

**Как включить:**

```bash
docker run --gpus all \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id meta-llama/Llama-2-7b-hf \
    --max-total-tokens 4096
    # Flash Attention 2 включается автоматически,
    # если `flash-attn` установлен в контейнере.
```

### 5.3 Квантизация

TGI поддерживает несколько методов квантизации — ниже разница между ними.

| Метод | Точность | Скорость | Память | Когда выбирать |
|-------|----------|----------|--------|---------------|
| **GPTQ** | 4-bit | Высокая | Низкая | Когда нужно сжать модель без потери структуры |
| **AWQ** | 4-bit | Очень высокая | Низкая | Лучший баланс скорость/качество |
| **bitsandbytes** | 8/4-bit | Средняя | Средняя | Для экспериментов (NF4, FP4) |
| **Marlin** | 4-bit | Максимальная | Минимальная | Только GPTQ-веса, kernel-оптимизация |
| **EETQ** | 8-bit | Высокая | Средняя | INT8, почти без потери качества |
| **FP8** | 8-bit | Высокая | Средняя | H100/H200 только |

**Пример с AWQ:**

```bash
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id TheBloke/Llama-2-13B-AWQ \
    --quantize awq \
    --max-total-tokens 4096
```

**Как TGI загружает квантизированные веса:**

1. Определяет метод квантизации по конфигу модели (config.json, quantize_config.json).
2. Загружает веса через специальный слой, который заменяет `nn.Linear`:
   - GPTQ → `GPTQLinear` — хранит веса в INT4, при вычислении распаковывает в FP16 и умножает.
   - AWQ → `AWQLinear` — хранит INT4 + scaling factors, распаковывает в FP16 с учётом scale.
3. Включение `--quantize` с явным указанием метода (например, `--quantize awq`) переопределяет автоопределение.

### 5.4 Tensor Parallelism

```bash
# Запуск на 4 GPU
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id meta-llama/Llama-2-70b-hf \
    --num-shard 4 \
    --max-total-tokens 4096
```

TGI разбивает модель по heads (для attention) и по hidden dimension (для MLP), используя NCCL для AllReduce.

**Что происходит внутри:**

1. При старте TGI загружает всю модель в CPU RAM, затем разбивает веса на `num_shard` частей и отправляет каждую часть на свою GPU.
2. Каждая GPU держит 1/4 весов модели.
3. При forward pass:
   - **Attention:** каждая GPU считает attention для своих heads, затем результаты конкатенируются (AllGather).
   - **MLP:** каждая GPU считает свою часть MLP, затем результаты суммируются (AllReduce).
4. KV-кэш распределён: каждая GPU хранит блоки для своих heads.

### 5.5 Speculative Decoding

TGI поддерживает speculative decoding через **assistant-модель**:

```bash
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id meta-llama/Llama-2-70b-hf \
    --assistant-model-id meta-llama/Llama-2-7b-hf \
    --num-assistant-tokens 5
```

**Как это работает:**

1. Маленькая модель (assistant, 7B) генерирует $K=5$ токенов быстро.
2. Большая модель (target, 70B) проверяет эти 5 токенов за один forward pass.
3. Если все 5 токенов приняты — big model сгенерировала 5 токенов за 1 forward pass (ускорение ~5×).
4. Если какой-то токен отвергнут — большая модель пересчитывает с этого места.

> Ускорение максимально, когда assistant-модель хорошо согласована с target-моделью (дообучена на тех же данных, с тем же токенизатором). Если модели из разных семей — rejection rate растёт, ускорение падает.

### 5.6 Prefix Caching

TGI кэширует KV-кэш для общих префиксов. Если несколько запросов начинаются одинаково, KV-кэш префикса вычисляется один раз и разделяется:

```python
# Запросы с общим системным промптом
prompts = [
    "[SYS] You are a helpful model. [USER] What is Python?",
    "[SYS] You are a helpful model. [USER] Explain quantum computing.",
    "[SYS] You are a helpful model. [USER] Write a poem.",
]

# TGI вычисляет KV-кэш для [SYS] один раз
# После первого запроса: KV-кэш [SYS] — в кэше, ref_count = 3
# Для второго и третьего: KV-кэш берётся из кэша, вычисляется только часть после префикса
```

**Включение:** `--prefix-caching` (по умолчанию включено в новых версиях).

> **Ограничение:** кэш работает только для точных совпадений префикса. Если у двух запросов отличается один токен — кэш не переиспользуется.

---

## 6. Мониторинг

### 6.1 Prometheus метрики

```bash
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id meta-llama/Llama-2-7b-hf \
    --max-total-tokens 4096

# Метрики на /metrics
curl http://localhost:8080/metrics
```

**Ключевые метрики:**

| Метрика | Тип | Описание |
|---------|-----|----------|
| `tgi_queue_size` | Gauge | Размер очереди ожидающих запросов |
| `tgi_batch_size` | Gauge | Текущий размер батча |
| `tgi_batch_max_tokens` | Gauge | Макс. токенов в батче |
| `tgi_request_count` | Counter | Всего запросов |
| `tgi_request_success` | Counter | Успешных запросов |
| `tgi_request_failure` | Counter | Упавших запросов |
| `tgi_request_duration_ms` | Histogram | Время выполнения запроса |
| `tgi_token_count` | Counter | Всего сгенерировано токенов |
| `tgi_tokens_per_second` | Gauge | Текущий throughput |
| `tgi_cuda_memory_free` | Gauge | Свободная GPU-память |
| `tgi_cuda_memory_used` | Gauge | Используемая GPU-память |
| `tgi_cuda_memory_cached` | Gauge | Кэшированная GPU-память |

### 6.2 Health check

```bash
# Health endpoint
curl http://localhost:8080/health

# Возвращает:
# {"status": "healthy", "model_id": "meta-llama/Llama-2-7b-hf", "device": "cuda:0", "version": "3.0.1"}
```

---

## 7. Пример полного развёртывания

### 7.1 Docker Compose

```yaml
version: "3.8"
services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:3.0.1
    ports:
      - "8080:80"
    environment:
      - HF_TOKEN=${HF_TOKEN:-}
    volumes:
      - /path/to/cache:/data  # кэш моделей
    command:
      - --model-id
      - meta-llama/Llama-2-7b-hf
      - --max-total-tokens
      - "4096"
      - --max-batch-size
      - "64"
      - --quantize
      - awq
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 7.2 Пример клиента с обработкой ошибок

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class TGIClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """Генерация текста с retry и таймаутом."""
        response = httpx.post(
            f"{self.base_url}/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    **kwargs,
                },
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=30.0 + max_new_tokens * 0.1,
                write=10.0,
            ),
        )
        response.raise_for_status()
        return response.json()["generated_text"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def chat(self, messages: list, max_new_tokens: int = 200, **kwargs) -> str:
        """Chat completion с повторными попытками."""
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "tgi",
                "messages": messages,
                "max_tokens": max_new_tokens,
                **kwargs,
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=30.0 + max_new_tokens * 0.1,
                write=10.0,
            ),
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


client = TGIClient("http://localhost:8080")

# Пример с chat format
answer = client.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
])
print(answer)
# "The capital of France is Paris."
```

---

## 8. Типичные ошибки и нюансы

### 8.1 Длинные промпты и OOM

TGI проверяет `max_input_length` и `max_total_tokens` **до начала генерации**. Если промпт превышает лимит — возвращается ошибка 413 (Payload Too Large), а не OOM.

```bash
# Настроить лимиты при запуске
--max-input-length 2048
--max-total-tokens 4096
```

### 8.2 CUDA out of memory при большом batch

Если batch слишком велик, TGI может упасть с OOM при prefill. Решение — уменьшить `--max-batch-size` или `--max-batch-prefill-tokens`.

```bash
--max-batch-size 16
--max-batch-prefill-tokens 2048
```

### 8.3 Несовместимость квантовых форматов

Не все модели совместимы со всеми методами квантизации. GPTQ-веса нельзя использовать с AWQ-кернелом и наоборот. TGI проверяет совместимость при загрузке:

```bash
# Ошибка: модель в GPTQ, а указан --quantize awq
# TGI: "Model is quantized with GPTQ, but awq is requested"
# Решение: указать --quantize gptq или убрать флаг
```

### 8.4 Зависание при длинных генерациях

Если stream=False и клиент не выставил read timeout, соединение может висеть, пока модель генерирует все токены.

**Решение:** всегда выставлять таймаут, пропорциональный `max_new_tokens`.

---

## 9. Когда применять

### Сценарии

- **Интеграция с Hugging Face Hub.** Если модель уже на HF Hub (или своя в формате transformers), TGI подхватывает её без конвертации.
- **Inference Endpoints.** Если деплой через HF Inference Endpoints — TGI используется под капотом.
- **Чат-боты.** Встроенная поддержка messages API и SSE streaming.
- **Быстрый старт.** Docker образ готов к использованию без сборки.

### Когда не подходит

- **Если нужен максимальный throughput** — vLLM часто даёт чуть выше (10–30% на чистых моделях без квантизации).
- **Если моделей нет на HF Hub** или они требуют custom code (например, Mamba) — TGI не поддерживает все архитектуры.
- **Если нужен LoRA-сервинг с сотнями адаптеров** — vLLM справляется лучше.

---

## 10. Связи с другими конспектами

- [vLLM](1%20vLLM.md) — альтернативный inference engine, сравнение архитектурных решений.
- LLM Quantization (GPTQ, AWQ, Marlin) — детали методов, упомянутых в разделе 5.3 (конспект в разработке).
- Flash Attention 2 — как работает fused attention (конспект в разработке).
- [KV cache](../../2%20Optimization/KV-cache.md) — что хранится в KV-кэше и почему он так важен.

---

## 11. Вопросы для самопроверки

1. Чем TGI отличается от vLLM в архитектуре (Router через gRPC, подход к PagedAttention)?
2. Почему TGI использует gRPC для связи router → worker, а не HTTP?
3. Как работает speculative decoding в TGI и от чего зависит ускорение?
4. В чём разница между GPTQ, AWQ и Marlin с точки зрения TGI? Почему нельзя указать `--quantize marlin` для AWQ-модели?
5. Что произойдёт, если отправить в TGI промпт длиннее `max_input_length`?