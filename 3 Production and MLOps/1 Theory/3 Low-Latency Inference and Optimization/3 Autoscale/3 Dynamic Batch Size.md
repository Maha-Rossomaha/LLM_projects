# Dynamic Batch Size: адаптивный размер батча

## 1. Зачем менять размер батча динамически

Фиксированный `max_batch_size` — компромисс:

- **Слишком маленький.** GPU недогружен. При низкой нагрузке это не страшно, но при пике throughput ниже возможного.
- **Слишком большой.** OOM при prefill (если пришло много длинных промптов) или рост tail latency (каждый decode-шаг обрабатывает больше токенов).

Промпты могут быть от 10 до 100K токенов. Батч из 32 коротких промптов помещается в 16 GB. Батч из 32 промптов по 4K — OOM. **Динамический batch size** подстраивается под текущую ситуацию.

## 2. Стратегии

### 2.1 По числу токенов (token budget)

Ограничение не по числу запросов, а по суммарному числу токенов в батче:

```python
class TokenBudgetBatcher:
    def __init__(self, max_budget_tokens: int = 4096):
        self.max_budget = max_budget_tokens

    def build_batch(self, requests: list[Request]) -> list[Request]:
        """Собрать батч, не превышающий budget токенов."""
        batch = []
        total = 0
        for req in sorted(requests, key=lambda r: len(r.prompt_tokens)):
            cost = len(req.prompt_tokens) + req.max_new_tokens
            if total + cost <= self.max_budget:
                batch.append(req)
                total += cost
        return batch
```

**Вариант:** сортировать по возрастанию длины промпта — сначала короткие, потом длинные. Так в батч попадёт максимум запросов.

### 2.2 По доступной памяти

Перед каждым prefill проверять, сколько памяти свободно:

```python
import torch

class MemoryAwareBatcher:
    def __init__(self, memory_reserve_gb: float = 4.0):
        self.reserve = memory_reserve_gb * 1024**3  # резерв в байтах

    def max_batch_for_prompt(self, prompt_length: int) -> int:
        """Сколько запросов данной длины поместится в текущую память."""
        free = torch.cuda.mem_get_info()[0]  # свободно в байтах
        available = free - self.reserve

        # Примерная стоимость одного запроса (KV-кэш + активации)
        cost_per_req = prompt_length * KV_BYTES_PER_TOKEN * 3  # ×3 запас

        return max(1, int(available / cost_per_req))
```

**Ограничение:** `mem_get_info()` может быть неточным из-за CUDA memory caching allocator. PyTorch не возвращает память в CUDA — он держит её в кэше для переиспользования.

### 2.3 Адаптивный подбор на лету (экспериментальный)

Начинать с консервативного batch size, увеличивать, пока latency растёт линейно, и уменьшать при признаках насыщения:

```python
class AdaptiveBatcher:
    def __init__(self, min_batch: int = 1, max_batch: int = 64):
        self.current = min_batch
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.latency_history: list[float] = []

    def record_batch(self, batch_size: int, latency_ms: float):
        self.latency_history.append(latency_ms)
        if len(self.latency_history) < 20:
            return

        # Скользящее среднее latency на запрос
        recent = self.latency_history[-20:]
        avg_lat_per_req = sum(recent) / len(recent) / batch_size

        if avg_lat_per_req > 100:  # больше 100 мс на запрос — перегруз
            self.current = max(self.min_batch, self.current // 2)
        elif avg_lat_per_req < 30 and self.current < self.max_batch:
            self.current = min(self.max_batch, self.current + 1)
```

### 2.4 На основе размера KV-кэша (точный расчёт)

Если движок поддерживает PagedAttention и сообщает utilisation блоков, можно рассчитывать batch size точно:

```python
class PagedAttentionAwareBatcher:
    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size

    def available_capacity(self, used_blocks: int) -> int:
        return (self.total_blocks - used_blocks) * self.block_size

    def estimate_blocks(self, prompt_tokens: int, max_new_tokens: int) -> int:
        total_tokens = prompt_tokens + max_new_tokens
        return (total_tokens + self.block_size - 1) // self.block_size
```

## 3. Компромисс: latency vs throughput

Чем больше батч, тем выше throughput (больше токенов/с), но тем выше latency каждого отдельного запроса:

| Batch size | Throughput (токенов/с) | p50 latency | p99 latency |
|-----------|----------------------|-------------|-------------|
| 1 | 25 | 40 мс | 42 мс |
| 4 | 80 | 45 мс | 55 мс |
| 16 | 200 | 70 мс | 120 мс |
| 64 | 320 | 200 мс | 600 мс |

Динамический batch size должен балансировать: увеличивать батч, пока p99 latency не превысит SLA.

## 4. Реализация в production-движках

- **vLLM:** `max_num_seqs` (макс. число запросов) и `max_num_batched_tokens` (макс. токенов). Dynamic SplitFuse в DeepSpeed-MII по сути делает то же самое на уровне токенов.
- **TGI:** `max_batch_size` (запросы) + `max_batch_prefill_tokens` (токены). При привышении — возврат 429 Too Many Requests.
- **TensorRT-LLM:** `max_batch_size` настраивается в engine. Dynamic batching через inflight batching.

## 5. Типичные ошибки

- **Фиксированный batch size.** Одинаково плох для коротких промптов (недогруз) и длинных (OOM).
- **Сортировка по убыванию длины.** Если в батч сначала класть длинные промпты, короткие могут не поместиться, хотя могли бы.
- **Игнорировать разницу prefill/decode.** Prefill считает все токены промпта. Decode — один токен на запрос. Batch size для prefill и decode должен считаться отдельно.
- **Слишком частые изменения.** Если batch size меняется каждый запрос, overhead от пересчёта превышает выгоду.

## 6. Вопросы для самопроверки

1. Почему фиксированный max_batch_size — плохое решение для LLM-инференса?
2. Как token budget учитывает разную длину промптов?
3. В чём проблема использования `torch.cuda.mem_get_info()` для расчёта batch size?
4. Какой компромисс между latency и throughput при увеличении batch size?
5. Как production-движки (vLLM, TGI) реализуют динамический batch size?