# Continuous Batching: как батчить запросы разной длины

## 1. Определение и мотивация

### 1.1 Проблема статического батчинга

Наивный батчинг для LLM выглядит так:

```python
# Статический батч
batch = [prompt_1, prompt_2, prompt_3]  # все пришли одновременно

# Выровнять по длине
max_len = max(len(p) for p in batch)
padded = [p + [pad_token] * (max_len - len(p)) for p in batch]

# Запустить — все живут до max_new_tokens
outputs = model.generate(padded, max_new_tokens=256)
```

Проблемы:

- **Pad-токены тратят compute.** GPU умножает матрицы и считает attention для фиктивных токенов.
- **Время жизни батча фиксировано.** Батч не завершится, пока самый длинный запрос не сгенерирует все токены. Короткие запросы ждут.
- **Запросы, пришедшие после старта, ждут следующего батча.** Если формировать батч раз в N миллисекунд, часть времени GPU простаивает.

Для авторегрессивной генерации статический батчинг особенно неэффективен: decode шаги идут последовательно, и хвост батча (запросы, уже сгенерировавшие EOS) продолжает занимать место.

### 1.2 Идея continuous batching

**Continuous batching** (он же iteration-level scheduling, блочное батчирование) — техника, при которой батч переформировывается на **каждом шаге** генерации:

```
Статический батч:
┌─────────────────────────────────────┐
│ Prefill(A,B,C) → Decode(A,B,C) × N  │
└─────────────────────────────────────┘

Continuous batching:
t=0: Prefill(A)
t=1: Decode(A) + Prefill(B)
t=2: Decode(A,B) + Prefill(C)
t=3: Decode(A,B,C)
t=4: A завершён → Decode(B,C) + Prefill(D)
t=5: Decode(B,C,D)
...
```

Ключевое отличие: запросы **не ждут** формирования полного батча. Они присоединяются к активному батчу на ближайшем шаге. Завершившиеся — немедленно заменяются новыми.

---

## 2. Как это работает

### 2.1 Базовая логика

На каждом шаге scheduler решает две задачи:

1. **Кого добавить** — один или несколько запросов из очереди переводятся в active set (prefill).
2. **Кого убрать** — запросы, сгенерировавшие EOS, удаляются из active set.

```python
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int = 32):
        self.queue: list[Request] = []      # ожидающие prefill
        self.active: list[Request] = []     # в процессе decode
        self.max_batch_size = max_batch_size

    def step(self):
        # 1. Добавить новые запросы, пока есть место
        while self.queue and len(self.active) < self.max_batch_size:
            req = self.queue.pop(0)
            self._prefill(req)             # forward pass для промпта
            self.active.append(req)

        if not self.active:
            return                          # нечего делать

        # 2. Один decode-шаг для всех активных
        outputs = self._decode(self.active) # [batch_size, vocab_size]

        # 3. Обработать завершённые
        for i in range(len(self.active) - 1, -1, -1):
            self.active[i].tokens.append(sample(outputs[i]))
            if self.active[i].is_finished:  # EOS или max_new_tokens
                del self.active[i]

    def _prefill(self, req: Request):
        """Обработать весь промпт за один forward pass."""
        with torch.no_grad():
            out = self.model(req.prompt_tokens)
        req.kv_cache = extract_kv_cache(out)

    def _decode(self, batch: list[Request]) -> torch.Tensor:
        """Один шаг генерации для всех запросов в батче."""
        input_ids = torch.tensor([r.last_token for r in batch])  # [B, 1]
        with torch.no_grad():
            out = self.model(input_ids, past_key_values=[r.kv_cache for r in batch])
        return out.logits[:, -1, :]
```

**Два типа forward pass в одном батче недопустимы.** Если в active set есть запросы на decode (ожидают 1 токен) и запросы на prefill (нуждаются в полноценном forward), их нельзя смешивать в один тензор — разная размерность.

Поэтому на практике шаги чередуются:

```
Шаг с decode:
  Вход: [t_A, t_B]  — один токен от каждого активного запроса
  Выход: [logits_A, logits_B]

Шаг с decode + prefill:
  Вход: [t_A, t_B, p_C_1, p_C_2, ..., p_C_P] — токены decode + весь промпт C
  Выход: [logits_A, logits_B, logits_C_last]
```

Pytorch это поддерживает через `padding` тензоров до одинаковой длины (см. раздел 3.1).

### 2.2 Partitioned batching (разделение prefill и decode)

Проблема: prefill считает много токенов за раз (compute-bound), decode — один токен (memory-bound). Если запустить их в одном forward pass, большие prefill-тензоры «задавят» маленькие decode:

```
Batch с одним prefill на 2000 токенов и 8 decode по 1 токену:
  - 2000 + 8 = 2008 токенов
  - 99.6% compute уходит на prefill
  - decode ждёт, хотя мог бы уже вернуть результат
```

Решение — partitioned batching: prefill и decode считаются **отдельными forward passes**, после чего результаты агрегируются:

```
Шаг:
  1. Decode forward (8 токенов) — быстро, memory-bound
  2. Prefill forward (2000 токенов) — медленно, compute-bound
  3. Разослать результаты клиентам
```

Теперь decode не ждёт prefill — он выполняется первым и отправляется клиентам сразу. Это важное отличие от наивной реализации, где один big-batch блокирует всё.

> Partitioned batching — ключевое отличие production-движков (vLLM, TGI, TensorRT-LLM) от «игрушечных» реализаций.

### 2.3 Chunked prefill

Даже с partitioned batching prefill на 50K токенов займёт сотни миллисекунд. За это время decode-запросы будут ждать (хотя и не так долго, как без partitioning).

**Chunked prefill** разбивает длинный промпт на чанки:

```python
def chunked_prefill(model, prompt_tokens, chunk_size=512):
    kv_cache = []
    for i in range(0, len(prompt_tokens), chunk_size):
        chunk = prompt_tokens[i:i + chunk_size]
        with torch.no_grad():
            out = model(chunk, past_key_values=kv_cache)
        kv_cache = out.past_key_values
    return kv_cache
```

Каждый чанк — отдельный forward pass. Между чанками scheduler может вставить decode-шаги для других запросов:

```
Шаг 1: Decode(A,B) + Prefill(C, чанк 1/4)
Шаг 2: Decode(A,B) + Prefill(C, чанк 2/4)
Шаг 3: C готов к decode → Decode(A,B,C) + Prefill(D, чанк 1/2)
...
```

**Компромисс:** чем меньше чанк, тем чаще scheduler может переключаться на decode, но тем выше overhead от множества маленьких forward passes (launch overhead CUDA-кернелов).

---

## 3. Проблемы и нюансы

### 3.1 Padding в mixed batch

Когда prefill и decode выполняются в одном forward pass, тензоры нужно привести к одной длине. Самый короткий путь — pad decode-токенов до длины prefill-чанка:

```python
def make_mixed_batch(decode_tokens, prefill_chunk):
    """
    decode_tokens: list[int] — по одному токену на запрос
    prefill_chunk: list[int] — чанк промпта
    """
    max_len = len(prefill_chunk)
    batch = []
    for t in decode_tokens:
        padded = [t] + [pad_id] * (max_len - 1)  # pad до max_len
        batch.append(padded)
    batch.append(prefill_chunk)
    return torch.tensor(batch)
```

**Проблема:** pad-токены всё равно тратят compute — attention считается для них. Но их количество равно `(num_decode * (chunk_size - 1))`, что обычно значительно меньше, чем при статическом батчинге (где pad добавляется к prefill).

Некоторые движки (vLLM) решают эту проблему через **специальный attention kernel**, который маскирует pad-токены на уровне CUDA-кернела, а не на уровне матриц (zero overhead).

### 3.2 Выбор max_batch_size

Слишком большой батч → OOM на prefill (память под KV-кэш и промежуточные активации). Слишком маленький → GPU недогружен.

```python
# Эмпирический расчёт
def estimate_max_batch(
    gpu_memory_gb=80,
    model_size_gb=35,
    kv_per_token_gb=0.0026,  # LLaMA-70B
    avg_seq_len=2048,
    safety_margin=0.15,
):
    available = gpu_memory_gb - model_size_gb
    kv_per_seq = kv_per_token_gb * avg_seq_len
    max_batch = available / kv_per_seq * (1 - safety_margin)
    return int(max_batch)
```

На практике `max_batch_size` подбирается экспериментально: стартовать с консервативного значения, увеличивать, пока не возникнет OOM или рост tail latency не превысит SLA.

### 3.3 Starvation (голодание)

Если очередь prefill-запросов не иссякает, decode-запросы могут никогда не получить compute — scheduler всё время будет добавлять новые prefill.

**Решение:** приоритет decode над prefill. Гарантировать, что на каждом шаге хотя бы 1 токен decode генерируется для каждого активного запроса:

```python
def schedule(self):
    # Приоритет 1: decode для всех активных (минимум 1 токен)
    decode_tokens = max(1, self.decode_budget // len(self.active))

    # Приоритет 2: prefill для новых (остаток бюджета)
    prefill_budget = self.token_budget - decode_tokens * len(self.active)
    ...
```

### 3.4 Scheduling latency

На каждом шаге scheduler принимает решение: кого добавить, кого убрать. При 256+ запросах overhead от Python-циклов становится заметным (1–5 мкс на запрос → 1–2 мс на шаг).

Production-движки реализуют scheduler на C++/CUDA (vLLM, TensorRT-LLM) или в отдельном потоке (TGI), чтобы scheduling не блокировал GPU.

---

## 4. Сравнение стратегий

| Параметр | Static batching | Continuous batching | + Chunked prefill |
|----------|----------------|-------------------|-------------------|
| GPU utilization | Низкая (простои) | Высокая | Максимальная |
| Tail latency | Высокая (ждут longest) | Средняя | Сглаженная |
| Overhead | Низкий | Средний (scheduling) | Выше (чанкинг) |
| Сложность реализации | Низкая | Высокая | Очень высокая |
| Когда эффективен | Offline batch | Real-time, разная длина | Очень длинные промпты |

---

## 5. Вопросы для самопроверки

1. Чем static batching отличается от continuous batching с точки зрения времени жизни батча?
2. Почему prefill и decode нельзя смешивать в один forward pass без padding?
3. Как partitioned batching решает проблему «prefill блокирует decode»?
4. Зачем нужен chunked prefill и какой компромисс он создаёт?
5. Как scheduler предотвращает starvation decode-запросов?
6. Почему production-движки пишут scheduler на C++, а не на Python?

## 6. Типичные ошибки

- **Смешивать prefill и decode в один forward pass без partitioned batching.** Decode ждёт завершения prefill — теряется смысл continuous batching.
- **Игнорировать memory-bound decode.** Если батч собран, но decode медленный — проблема не в батчинге, а в HBM bandwidth.
- **Слишком маленький chunk в chunked prefill.** Каждый чанк — отдельный launch kernel. 64 токена на чанк создают больше overhead, чем выигрыш от переключения на decode.
- **Не выставлять timeout на prefill.** Если один запрос с промптом в 100K токенов пришёл — chunked prefill разобьёт его, но даже с разбивкой это займёт заметное время. Нужен лимит на max_input_length.