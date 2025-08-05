# Оптимизация latency и throughput PEFT-моделей в продакшене

## Цель
Снизить latency и повысить throughput LLM-сервиса с использованием PEFT-адаптации (например, LoRA или p-tuning), не жертвуя критически качеством и стабильностью. Ориентир — production-API с высокими требованиями к SLA.

---

## 1. Bottlenecks PEFT-моделей при inference

В отличие от full-моделей, PEFT-модели могут страдать от следующих узких мест:

- **Нефьюзенные LoRA-адаптеры** → дополнительное матричное умножение
- **Поддержка динамического выбора адаптера (multi-LoRA)** → замедление маршрутизации
- **Модели с prefix/p-tuning** → увеличенная sequence length (т.к. префиксы добавляются в input)
- **Невекторизованный inference (без batching)** → подрыв throughput
- **GPU-утечки из-за неправильной выгрузки адаптеров**

---

## 2. Техники оптимизации

### A. Инфраструктурный уровень
- **Batching**: агрегировать входящие запросы в батчи (например, через `vLLM`, `TGI`, `DeepSpeed-MII`)
- **Token streaming**: минимизирует user-perceived latency
- **KV caching**: особенно важно при длинных запросах (past key/value reuse)
- **Async-обработка**: FastAPI + asyncio / Uvicorn workers / gunicorn worker-per-core

### B. Оптимизация модели
- **Fused LoRA kernels**: использовать сборки с поддержкой LoRA-in-fused матриц (Exllama, vLLM, `bitsandbytes`)
- **Compiling (torch.compile / torchdynamo / TensorRT)**: особенно эффективно при фиксированной длине input/output
- **Quantization (INT8, GPTQ, AWQ)**: снижает latency и потребление памяти (снижает quality, но приемлемо)

### C. Управление адаптерами
- **Merge LoRA в base-модель**, если:
  - Адаптация финализирована и не требуется гибкости
  - Нужно максимизировать скорость
  - Используется только 1 адаптер
- **Оставить адаптеры отдельно**, если:
  - Много моделей на одном сервере (multi-tenant)
  - Часто hot-reload адаптеров (фича A/B)

---

## 3. Компромиссы: качество vs latency

| Компромисс                           | Готовность принять |
|--------------------------------------|---------------------|
| Merge LoRA с возможной деградацией   | Да, если прирост >15% latency |
| INT8 квантование                     | Да, только если нет критичных ошибок |
| Truncate длинных inputs              | Да, при длине >2k токенов     |
| Урезать длину output                 | Да, через `max_new_tokens`    |

---

## 4. Мониторинг и логирование

### Что логировать:
- `request_duration_ms`
- `generation_latency_ms`
- `queue_time_ms`
- `model_loading_time`
- `adapter_switch_time`
- `gpu_memory_allocated`, `gpu_memory_reserved`

### Что мониторить:
- **P95 latency**
- **Throughput (req/sec)**
- **CUDA OOM / timeout ошибки**
- **GPU utilization / memory fragmentation**
- **Batch fill rate** (особенно в vLLM/TGI)

---

## Вывод
> Для production‑сервиса с PEFT-моделью ключ к низкой latency — **fused реализация + квантование + батчинг + правильное управление адаптерами**. Важно логировать и мониторить каждый шаг inference path, чтобы отслеживать деградацию. PEFT — мощный инструмент, но требует продуманной интеграции в API‑платформу.

