# Concurrency Gate: ограничение параллельных запросов к GPU

## 1. Зачем нужно ограничивать конкурентность

GPU — последовательное устройство с точки зрения инференса. Одна операция на GPU выполняется за раз (в рамках одного CUDA-стрима). Отправка нескольких параллельных запросов приводит к:

- **Конкуренции за HBM bandwidth.** Все активные decode-шаги одновременно читают веса и KV-кэш из HBM. Полоса пропускания делится между ними, каждый decode замедляется.
- **Росту latency.** Для 4 параллельных decode latency может вырасти не в 4, а в 6–8 раз из-за коллизий чтения.
- **OOM.** Каждый запрос держит свой KV-кэш на GPU. N параллельных запросов с контекстом 4K — это N × 2 GB для LLaMA-7B.

**Concurrency gate** — механизм, ограничивающий число одновременно обрабатываемых запросов. Если все слоты заняты, новый запрос ждёт в очереди.

## 2. Реализации

### 2.1 Семафор (простой gate)

Самый простой способ — `asyncio.Semaphore`:

```python
import asyncio
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Максимум 4 параллельных инференса
inference_semaphore = asyncio.Semaphore(4)

async def generate_with_gate(prompt: str) -> str:
    async with inference_semaphore:
        return await model.generate(prompt)

@app.post("/generate")
async def generate(prompt: str):
    try:
        result = await asyncio.wait_for(
            generate_with_gate(prompt),
            timeout=30.0,  # таймаут ожидания слота
        )
        return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Service busy, try again")
```

**Проблема:** равномерный семафор не различает типы запросов. Короткая генерация на 10 токенов может ждать длинной на 10K.

### 2.2 Адаптивный gate по памяти

Ограничение не по числу запросов, а по доступной GPU-памяти:

```python
import torch

class MemoryGate:
    def __init__(self, max_memory_usage: float = 0.85):
        self.max_memory = max_memory_usage  # доля от общей памяти
        self.lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self.lock:
                used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if used < self.max_memory:
                    return  # можно принимать запрос

            # Память переполнена — ждём освобождения
            await asyncio.sleep(0.1)

    def release(self):
        """Вызывается после завершения запроса."""
        pass  # освобождение памяти произойдёт само при удалении тензоров
```

**Проблема:** `torch.cuda.memory_allocated()` возвращает значение с задержкой. Можно случайно принять запрос, когда памяти уже нет.

### 2.3 Token-aware gate (учёт длины контекста)

Ограничение на основе предсказанного размера KV-кэша каждого запроса:

```python
@dataclass
class RequestBudget:
    prompt_length: int
    max_new_tokens: int

    @property
    def estimated_kv_bytes(self) -> int:
        """Приблизительный размер KV-кэша для этого запроса."""
        tokens = self.prompt_length + self.max_new_tokens
        return tokens * KV_BYTES_PER_TOKEN  # константа для модели

class TokenAwareGate:
    def __init__(self, total_kv_budget: int):
        self.total_budget = total_kv_budget
        self.used_budget = 0
        self.lock = asyncio.Lock()

    async def acquire(self, budget: RequestBudget) -> bool:
        async with self.lock:
            if self.used_budget + budget.estimated_kv_bytes <= self.total_budget:
                self.used_budget += budget.estimated_kv_bytes
                return True
            return False

    def release(self, budget: RequestBudget):
        self.used_budget -= budget.estimated_kv_bytes
```

**Ограничение:** оценка приблизительная — реальный KV-кэш может быть меньше или больше. Но для практических целей точности достаточно.

### 2.4 Динамический gate (по latency)

Если p99 latency превышает SLA, gate автоматически уменьшает число разрешённых запросов:

```python
from collections import deque
import time

class DynamicGate:
    def __init__(self, max_concurrent: int = 8, p99_target_ms: int = 1000):
        self.max_concurrent = max_concurrent
        self.current = max_concurrent
        self.p99_target = p99_target_ms
        self.latencies: deque[float] = deque(maxlen=100)
        self.lock = asyncio.Lock()

    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    async def adjust(self):
        """Периодически пересчитывать лимит."""
        if len(self.latencies) < 50:
            return  # недостаточно данных

        sorted_lat = sorted(self.latencies)
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        async with self.lock:
            if p99 > self.p99_target * 1.2:
                self.current = max(1, self.current - 1)
            elif p99 < self.p99_target * 0.7 and self.current < self.max_concurrent:
                self.current += 1

    async def acquire(self):
        while True:
            async with self.lock:
                if self.current > 0:
                    self.current -= 1
                    return
            await asyncio.sleep(0.05)
```

## 3. Где размещать gate

```
Client → Load Balancer → FastAPI → [Concurrency Gate] → Model
                              ↓
                        Queue (если gate занят)
```

Gate должен быть **перед вызовом модели**, но **после HTTP-парсинга и валидации**. Не имеет смысла отклонять запрос до того, как мы его распарсили — мы уже потратили время на приём соединения.

Варианты размещения:

| Уровень | Плюсы | Минусы |
|---------|-------|--------|
| В middleware FastAPI | Централизованно, все эндпоинты | Не видит тип запроса |
| Внутри batch-процессора | Учитывает тип запроса | Только для batch-эндпоинта |
| На уровне модели (внутри движка) | Максимально точно | Зависит от движка (vLLM, TGI) |

## 4. Типичные ошибки

- **Слишком высокий лимит.** Если GPU может комфортно обработать 4 запроса, а gate пропускает 8 — latency каждого вырастет в 3–4 раза.
- **Таймаут меньше ожидания в очереди.** Если очередь рассчитана на 10 запросов, а timeout = 5 секунд, запросы будут падать по timeout, даже если GPU просто перегружен.
- **Не отличать тип запроса.** Prefill и decode имеют разную стоимость. Если gate считает их одинаковыми, короткие decode могут страдать от тяжёлых prefill.
- **Глобальный gate для нескольких моделей.** Если на одной GPU работают две модели (например, ранкер и LLM), gate должен учитывать обе, а не быть «на модели».

## 5. Вопросы для самопроверки

1. Почему 4 параллельных запроса на GPU могут быть медленнее, чем 4 последовательных?
2. Чем token-aware gate лучше простого семафора?
3. Как динамический gate выбирает лимит на основе p99 latency?
4. Почему concurrency gate должен быть перед моделью, а не после неё?
5. Как учесть разную стоимость prefill и decode в concurrency gate?