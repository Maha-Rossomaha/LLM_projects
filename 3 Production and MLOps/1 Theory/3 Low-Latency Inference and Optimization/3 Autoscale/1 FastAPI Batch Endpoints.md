# FastAPI batch endpoints для LLM-инференса

## 1. Зачем нужен batch-эндпоинт

LLM эффективнее всего работают с батчами. Один forward pass на батч из 8 запросов выполняется всего в 2–3 раза медленнее, чем на одном запросе. Если каждый запрос приходит через отдельный HTTP-вызов, выгода от батчинга теряется.

**Batch-эндпоинт** — эндпоинт, который принимает список запросов и возвращает список ответов, обрабатывая все за один проход модели.

## 2. Реализация на FastAPI

### 2.1 Простой batch-эндпоинт

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class BatchRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: int = 200
    temperature: float = 0.7

class BatchResponse(BaseModel):
    generated_texts: List[str]
    tokens_generated: List[int]

@app.post("/v1/generate-batch", response_model=BatchResponse)
async def generate_batch(req: BatchRequest):
    """Обработать несколько промптов за один forward pass."""
    # Токенизация всего батча
    inputs = tokenizer(req.prompts, padding=True, return_tensors="pt").to("cuda")

    # Один forward pass
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )

    # Декодирование
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    tokens_gen = (outputs != inputs.input_ids).sum(dim=1).tolist()

    return BatchResponse(generated_texts=generated, tokens_generated=tokens_gen)
```

### 2.2 Асинхронная очередь с батчингом

Когда запросы приходят по одному, их можно накапливать и отправлять батчем:

```python
import asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    request_id: str
    max_tokens: int = 200

class BatchProcessor:
    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 50):
        self.max_batch = max_batch_size
        self.timeout = timeout_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._process())
        self._pending: dict[str, asyncio.Future] = {}

    async def submit(self, req: InferenceRequest) -> str:
        future = asyncio.get_event_loop().create_future()
        self._pending[req.request_id] = future
        await self.queue.put(req)
        return await future

    async def _process(self):
        while True:
            batch: list[InferenceRequest] = []
            deadline = asyncio.get_event_loop().time() + self.timeout

            # Ждём первый запрос
            req = await self.queue.get()
            batch.append(req)

            # Накапливаем
            while len(batch) < self.max_batch and asyncio.get_event_loop().time() < deadline:
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout=0.005)
                    batch.append(req)
                except asyncio.TimeoutError:
                    pass

            # Выполняем батч
            results = await self._inference(batch)

            # Отдаём результаты
            for req, result in zip(batch, results):
                future = self._pending.pop(req.request_id)
                future.set_result(result)

    async def _inference(self, batch: list[InferenceRequest]) -> list[str]:
        prompts = [r.prompt for r in batch]
        max_tokens = max(r.max_tokens for r in batch)

        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)


processor = BatchProcessor(max_batch_size=32, timeout_ms=50)

@app.post("/v1/generate")
async def generate(req: InferenceRequest):
    """Отдельный запрос — отправляется в батч-процессор."""
    result = await processor.submit(req)
    return {"generated_text": result, "request_id": req.request_id}
```

**Ключевые параметры:**

| Параметр | Значение | Влияние |
|----------|----------|---------|
| `max_batch_size` | 16–64 | Чем больше, тем выше throughput, но больше памяти |
| `timeout_ms` | 20–100 | Компромисс между latency накопления и размером батча |

**Выбор timeout_ms:** если SLA = 500 мс, а время инференса батча = 200 мс, timeout должен быть ≤ 300 мс. Иначе запрос выпадет из SLA ещё до начала обработки.

### 2.3 Graceful degradation при переполнении

Если батч-процессор перегружен, вместо отбрасывания запроса можно снизить качество (уменьшить `max_new_tokens`):

```python
@app.post("/v1/generate")
async def generate(req: InferenceRequest):
    queue_size = processor.queue.qsize()
    if queue_size > 100:
        # Перегрузка — уменьшаем max_tokens
        req.max_tokens = min(req.max_tokens, 50)
    elif queue_size > 50:
        # Умеренная нагрузка
        req.max_tokens = min(req.max_tokens, 100)

    result = await processor.submit(req)
    return {"generated_text": result}
```

## 3. Продвинутые техники

### 3.1 Priority batching

Запросы с высоким приоритетом (например, платные пользователи) обрабатываются вне очереди:

```python
from dataclasses import dataclass, field
from enum import IntEnum
import heapq

class Priority(IntEnum):
    HIGH = 0
    NORMAL = 1
    LOW = 2

@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    timestamp: float = field(compare=False)
    request: InferenceRequest = field(compare=False)
    future: asyncio.Future = field(compare=False)

class PriorityBatchProcessor(BatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = []  # heapq

    async def submit(self, req: InferenceRequest, priority: Priority = Priority.NORMAL):
        future = asyncio.get_event_loop().create_future()
        heapq.heappush(self.queue, PrioritizedRequest(
            priority=priority,
            timestamp=asyncio.get_event_loop().time(),
            request=req,
            future=future,
        ))
        return await future
```

## 4. Типичные ошибки

- **Слишком большой max_batch_size.** При prefill большого батча можно получить OOM. Лучше подбирать экспериментально.
- **Отсутствие timeout.** Если timeout_ms = 0, батч собирается только при заполнении — в часы низкой нагрузки запросы висят вечно.
- **Смешивание prefill и decode в одном батче.** Для авторегрессивных моделей prefill и decode — разные типы forward pass. Batch-эндпоинт должен это учитывать (см. конспект Continuous Batching).
- **Игнорирование padding.** Если промпты разной длины, `tokenizer(prompts, padding=True)` добавляет pad-токены. Они тратят compute. Нужно следить за разбросом длин.

## 5. Вопросы для самопроверки

1. Почему batch-эндпоинт эффективнее, чем N отдельных вызовов модели?
2. Как выбрать timeout_ms для батч-процессора при SLA = 300 мс?
3. В чём проблема смешивания prefill и decode в одном batch-эндпоинте?
4. Как адаптивно снижать качество при перегрузке?