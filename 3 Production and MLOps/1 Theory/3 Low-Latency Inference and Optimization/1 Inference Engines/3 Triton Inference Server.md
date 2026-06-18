# Triton Inference Server

## 1. Определение и мотивация

### 1.1 Проблема наивного подхода

В продакшне редко используется одна модель. Типичный ML-сервис включает:

- **Препроцессинг** (токенизация, нормализация, извлечение признаков).
- **Основную модель** (LLM, BERT-ранкер, ResNet-encoder).
- **Постпроцессинг** (фильтрация, агрегация, форматирование ответа).

При наивном подходе каждый компонент запускается отдельным микросервисом:

```
Client → Preprocessing (Python) → Model (Python) → Postprocessing (Python) → Client
```

Проблемы:

- **Лишние копирования данных.** Каждый переход между сервисами — сериализация/десериализация (JSON, protobuf) и transfer по сети. Для больших тензоров (эмбеддинги, изображения) это доминирующий источник задержки.
- **Разные фреймворки.** Модель может быть обучена в PyTorch, а препроцессинг написан на Python с `numpy`. Наивный подход требует поднимать отдельные рантаймы, каждый со своими зависимостями.
- **Нет динамического батчирования.** Если препроцессинг умеет обрабатывать по одному запросу, а модель — батчами, приходится либо писать свой batching-слой, либо мириться с низким throughput.
- **Гетерогенное оборудование.** Одна модель эффективнее на GPU, другая — на CPU. Управлять этим вручную сложно.
- **Нет единого мониторинга.** Для каждого сервиса нужно отдельно настраивать метрики, логи, health checks.

### 1.2 Что предлагает Triton

**Triton Inference Server** — open-source инференс-сервер от NVIDIA, который решает эти проблемы:

- **Единый рантайм для любых фреймворков:** TensorRT, ONNX Runtime, PyTorch (TorchScript), TensorFlow, OpenVINO, Python (для кастомной логики).
- **Динамическое батчирование (dynamic batching):** запросы автоматически накапливаются в батч перед отправкой модели.
- **Одновременное выполнение нескольких моделей** на одном GPU с разделением памяти.
- **Model Ensembles и BLS (Business Logic Scripting):** композиция моделей в пайплайн без написания отдельного микросервиса.
- **Поддержка CPU, GPU и multi-GPU:** гибкое назначение инстансов модели на устройства.
- **Prometheus-метрики** для мониторинга latency, throughput, utilization.
- **OpenAI-совместимый API** (с Triton 24.08+).

Triton **не является LLM-движком** вроде vLLM или TGI. Он не реализует PagedAttention, continuous batching или Flash Attention. Его задача — быть универсальной прослойкой между клиентом и моделью, которая берёт на себя routing, batching, управление памятью и мониторинг.

### 1.3 Triton vs vLLM/TGI: кратко

| Характеристика | Triton | vLLM / TGI |
|---|---|---|
| Назначение | Универсальный инференс-сервер | LLM-специфичный движок |
| LLM-оптимизации | Нет встроенных (но можно подключить TensorRT-LLM бэкенд) | PagedAttention, Flash Attention, continuous batching |
| Поддержка фреймворков | TensorRT, ONNX, PyTorch, TF, OpenVINO, Python | Только PyTorch через трансформеры |
| Dynamic batching | Да (аппаратный, для любой модели) | Нет (continuous batching только для LLM) |
| Model Ensembles | Да (нативные пайплайны) | Нет |
| Когда выбирать | Сервис с композицией моделей, гетерогенные пайплайны | Чистый LLM-инференс |

> Triton и vLLM/TGI — не конкуренты, а комплементарные инструменты. Triton управляет пайплайном и батчингом, а внутри него может быть запущена LLM через TensorRT-LLM бэкенд.

---

## 2. Архитектура

### 2.1 Компоненты

```
┌──────────────────────────────────────────────────────────────┐
│                          Triton Server                       │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ HTTP    │  │ gRPC     │  │ C API    │  │ Metrics       │  │
│  │ (8000)  │  │ (8001)   │  │          │  │ (8002)        │  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └───────────────┘  │
│       └────────────┼─────────────┘                           │
│                    ▼                                         │
│          ┌──────────────────┐                                │
│          │  Scheduler       │                                │
│          │  - Sequence      │                                │
│          │  - Dynamic       │                                │
│          │  - Ensemble      │                                │
│          └────────┬─────────┘                                │
│                   ▼                                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                 Model Repository                     │    │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐    │    │
│  │  │ PyTorch│  │TensorRT│  │  ONNX  │  │  Python  │    │    │
│  │  └────────┘  └────────┘  └────────┘  └──────────┘    │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

- **HTTP/gRPC endpoints** — приём запросов от клиентов. HTTP использует REST API (JSON + binary data), gRPC — protobuf с более эффективной сериализацией.
- **Scheduler** — распределяет запросы по инстансам моделей. Поддерживает несколько политик: sequence (сохранение порядка запросов), dynamic batching (накопление батчей), ensemble (пайплайны моделей).
- **Model Repository** — файловая система или хранилище с конфигурациями и весами моделей. Каждая модель описывается YAML-файлом `config.pbtxt`.
- **Backends** — рантаймы, выполняющие модель. Triton поддерживает C++ и Python бэкенды.

### 2.2 Model Repository: структура

Каждая модель — отдельная папка в model repository:

```
model_repository/
├── text_encoder/
│   ├── 1/                      # version 1
│   │   ├── model.onnx
│   │   └── config.pbtxt        # опционально, если есть глобальный config
│   └── config.pbtxt            # конфигурация модели
├── reranker/
│   ├── 1/
│   │   └── model.pt            # TorchScript
│   └── config.pbtxt
└── ensemble_pipeline/
    └── config.pbtxt            # ensemble без весов, только routing
```

Номер папки (`1/`, `2/`) — версия модели. Triton автоматически загружает последнюю версию по умолчанию и может переключать версии через API без перезапуска.

### 2.3 config.pbtxt: конфигурация модели

```protobuf
# config.pbtxt для модели ONNX
name: "text_encoder"
backend: "onnxruntime"

max_batch_size: 64  # максимальный размер батча

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]  # -1 означает variable-length
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [-1]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [768]
  }
]

instance_group [
  {
    count: 2           # два инстанса модели
    kind: KIND_GPU     # на GPU
    gpus: [0]          # на GPU 0
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 200  # ждать 200 мкс перед отправкой батча
  preferred_batch_size: [16, 32, 64] # предпочтительные размеры батча
}
```

**instance_group** — ключевой параметр. Он определяет:

- **`count: 2`** — два параллельных инстанса модели. Каждый инстанс — полная копия модели со своим участком GPU-памяти. Если модель небольшая, несколько инстансов повышают throughput за счёт параллельного выполнения.
- **`kind: KIND_GPU`** — выполнение на GPU. Альтернатива: `KIND_CPU`.
- **`gpus: [0]`** — привязка к конкретному GPU. Можно оставить пустым для автоматического выбора.

**dynamic_batching** — управление батчингом:

- **`max_queue_delay_microseconds: 200`** — максимальное время ожидания перед отправкой батча. Если за 200 мкс набралось меньше запросов, чем `preferred_batch_size`, батч всё равно отправляется с тем, что есть (это называется **rarefaction** — разрежение батча).
- **`preferred_batch_size: [16, 32, 64]`** — предпочтительные размеры батча. Triton старается сформировать батч одного из этих размеров. Если запросов не хватает, по истечении `max_queue_delay` отправляется меньший батч.

---

## 3. Dynamic batching в деталях

### 3.1 Как работает dynamic batching

```
Время →
Запросы: [A] [B] [C]          [D] [E] [F]   [G] [H]  [I]
          │   │   │            │   │   │     │   │    │
          ▼   ▼   ▼            ▼   ▼   ▼     ▼   ▼    ▼
Batching: ──────Batch 1────── ─────Batch 2────── Batch 3
          [A,B,C] (3 req)     [D,E,F] (3 req)   [G,H,I]
                                    │
                              200 мкс задержки
```

**Механизм:**

1. Клиент отправляет запрос. Triton помещает его в очередь модели.
2. Включается таймер (`max_queue_delay_microseconds`).
3. Если за время таймера набралось `preferred_batch_size` запросов — батч отправляется немедленно.
4. Если таймер истёк, а достаточное количество не набрано — батч отправляется с тем, что есть (rarefaction).
5. Пустой батч никогда не отправляется.

### 3.2 Rarefaction (разрежение)

**Rarefaction** — ситуация, когда батч отправляется с размером меньше предпочтительного. Triton сам решает, отправлять ли неполный батч или ждать дальше, на основе параметров:

- `max_queue_delay_microseconds` — чем больше, тем выше вероятность полного батча, но выше latency.
- `preferred_batch_size` — чем больше разрыв между минимальным и максимальным значением, тем чаще будет rarefaction.

**Пример:** `preferred_batch_size = [32]`, `max_queue_delay = 1000` (1 ms). Если за 1 ms пришло 10 запросов — батч из 10 будет отправлен. Загрузка GPU составит 10/32 = 31%.

### 3.3 Dynamic batching vs Continuous batching

| | Dynamic batching (Triton) | Continuous batching (vLLM, TGI) |
|---|---|---|
| Механизм | Накопление запросов в очередь → один батч на модель | Переформирование батча на каждом decode-шаге |
| Для каких моделей | Любые (encoder, decoder, CNN) | Только авторегрессивные decoder |
| Latency | Добавляет `max_queue_delay` к TTFT | Не добавляет задержку (батч формируется из уже активных запросов) |
| Throughput | Высокий при равномерной нагрузке | Высокий при любой нагрузке |

---

## 4. Model concurrency и Instance Groups

### 4.1 Несколько инстансов на одном GPU

Если модель небольшая, она не утилизирует GPU полностью. Triton позволяет запустить **несколько копий модели** (`count > 1`), которые будут выполняться параллельно:

```
GPU 0 ┌──────────────────────────────────────────────────┐
      │  Instance 0 (reranker)  │  Instance 1 (reranker) │
      │                         │                        │
      │  ┌───┐ ┌───┐ ┌───┐      │  ┌───┐ ┌───┐ ┌───┐     │
      │  │ Q │ │ K │ │ V │      │  │ Q │ │ K │ │ V │     │
      │  └───┘ └───┘ └───┘      │  └───┘ └───┘ └───┘     │
      └─────────────────────────┴────────────────────────┘
                         shared model weights (read-only)
```

**Важно:** веса модели загружаются в GPU-память **один раз** и разделяются между инстансами (copy-on-write). Дополнительная память тратится только на активации и временные буферы каждого инстанса.

**Когда это полезно:**

- Модель маленькая (BERT-base, 110M параметров — ~0.5 GB в FP16). Один инстанс не загрузит GPU. Два-четыре инстанса повышают throughput в 2–3x.
- Модель CPU-only (XGBoost, линейная регрессия) — несколько инстансов на разных CPU-ядрах.
- Модель с низкой latency (5 ms на запрос) — несколько инстансов снижают contention.

### 4.2 Разные модели на одном GPU

Triton позволяет запускать разные модели на одном GPU, разделяя память:

```protobuf
# Модель A: encoder (занимает 3 GB)
instance_group [{ count: 1, kind: KIND_GPU }]

# Модель B: reranker (занимает 2 GB)
instance_group [{ count: 2, kind: KIND_GPU }]
```

Обе модели работают одновременно. Triton использует CUDA streams для параллельного выполнения: модели A и B могут выполняться конкурентно, если хватает вычислительных ресурсов GPU.

**Ограничения:**

- Суммарная память всех моделей не должна превышать GPU memory.
- CUDA kernel launch — serialized by default. Реальное распараллеливание происходит только при использовании разных CUDA streams.
- Если суммарная загрузка превышает 100% GPU — возникает contention и latency растёт.

---

## 5. Model Ensembles и BLS

### 5.1 Ensembles: пайплайн из моделей

**Ensemble** — статический пайплайн, определённый в конфиге. Он не содержит весов, только описание того, как данные передаются между моделями.

```protobuf
# ensemble_pipeline/config.pbtxt
name: "search_pipeline"
platform: "ensemble"
max_batch_size: 64

ensemble_scheduling {
  step [
    {
      model_name: "text_encoder"
      model_version: 1
      input_map: {
        key: "INPUT_TOKENS"
        value: "input_ids"
      }
      output_map: {
        key: "EMBEDDINGS"
        value: "query_embedding"
      }
    },
    {
      model_name: "reranker"
      model_version: -1  # последняя версия
      input_map: {
        key: "QUERY_EMBEDDING"
        value: "query_embedding"
      }
      output_map: {
        key: "SCORES"
        value: "final_scores"
      }
    }
  ]
}
```

**Как работает ensemble:**

1. Клиент отправляет один запрос на `search_pipeline`.
2. Triton отправляет `input_ids` в `text_encoder` (шаг 1).
3. После завершения шага 1, `reranker` получает эмбеддинг из `text_encoder` (шаг 2).
4. Triton возвращает клиенту `final_scores`.

Ensemble **не требует отдельного микросервиса**. Весь routing и согласование форматов данных происходит внутри Triton, без копирования данных через CPU.

**Ограничения ensemble:**

- Только статические DAG (directed acyclic graph). Ветвления (if/else), циклы и динамический routing невозможны.
- Размер батча фиксирован на всех шагах. Нельзя на первом шаге обработать батч из 64, а на втором — из 1.
- Нет обработки ошибок на уровне шагов (если один шаг упал, весь ensemble падает).

### 5.2 BLS: динамические пайплайны

**Business Logic Scripting (BLS)** — Python-бэкенд, который позволяет писать произвольную логику пайплайна. Это надстройка над ensemble API, дающая полный контроль над потоком данных.

```python
# model.py для Python-бэкенда с BLS
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            # Извлекаем входные данные
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "query_text"
            )
            
            # Вызываем другую модель через BLS API
            infer_request = pb_utils.InferenceRequest(
                model_name="text_encoder",
                requested_output_names=["embeddings"],
                inputs=[input_tensor]
            )
            infer_response = infer_request.exec()
            
            # Проверяем ошибку
            if infer_response.has_error():
                raise pb_utils.TritonModelException(
                    infer_response.error().message()
                )
            
            # Извлекаем результат
            embedding = pb_utils.get_output_tensor_by_name(
                infer_response, "embeddings"
            )
            
            # Постпроцессинг на Python
            normalized = embedding.as_numpy() / np.linalg.norm(
                embedding.as_numpy()
            )
            
            # Формируем финальный ответ
            output_tensor = pb_utils.Tensor("normalized_embedding", normalized)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output_tensor])
            )
        
        return responses
```

**BLS даёт:**

- Условную логику: if/else, switch, фильтрацию результатов.
- Динамическое создание запросов к другим моделям.
- Произвольный постпроцессинг на Python (numpy, scipy, собственные функции).
- Обработку ошибок на каждом шаге.

**Цена BLS:** данные проходят через Python (CPU), что добавляет latency и накладные расходы на сериализацию. Для максимальной производительности ensemble без Python всегда быстрее.

**Ensemble vs BLS — что выбрать:**

| Критерий | Ensemble | BLS |
|---|---|---|
| Производительность | Максимальная (всё на GPU) | Ниже (данные через Python/CPU) |
| Гибкость | Только static DAG | Любая логика |
| Условные переходы | Нет | Да |
| Динамический батчинг | Единый на весь пайплайн | Можно менять на каждом шаге |
| Когда использовать | Стабильный предсказуемый пайплайн | Нужна логика, динамика, обработка ошибок |

---

## 6. Performance tuning

### 6.1 Выбор scheduler

Triton поддерживает три типа scheduler:

| Scheduler | Описание | Когда использовать |
|---|---|---|
| Default (sequence) | Запросы выполняются в порядке поступления, без батчирования | Простая модель, низкая нагрузка |
| Dynamic batching | Запросы накапливаются в батч | Модель эффективна на батчах (encoder, CNN) |
| Ensemble | Композиция моделей | Пайплайн из нескольких моделей |

Scheduler выбирается автоматически на основе конфига. Если указан `dynamic_batching` — используется dynamic batching. Если `ensemble_scheduling` — ensemble. Иначе — default.

### 6.2 Rate limiter

**Rate limiter** — механизм, предотвращающий перегрузку модели слишком большим количеством параллельных запросов.

```protobuf
# config.pbtxt с rate limiter
name: "reranker"
max_batch_size: 64

model_transaction_policy {
  decoupled: false  # один запрос → один ответ
}

rate_limiter {
  resources [
    {
      name: "GPU_compute"
      global: false   # per-instance resource
      count: 1
    }
  ]
  priority: 1  # 0 = highest
}
```

**Зачем нужен rate limiter:** если модель может обрабатывать максимум 4 параллельных запроса (например, из-за памяти или compute limits), а клиентов — 100, без rate limiter'а запросы будут падать с OOM или timeout. Rate limiter ставит избыточные запросы в очередь.

**`resources`** — перечисление ресурсов, которые занимает инстанс модели:

- `global: false` — ресурс выделяется на каждый инстанс (per-instance). Если `global: true` — ресурс общий для всех инстансов модели.
- `count: 1` — каждый запрос занимает 1 единицу ресурса. Можно выставить `count > 1`, если один запрос использует несколько единиц ресурса (например, долгий запрос с `count: 5`).

### 6.3 Параметры запуска (опции сервера)

```bash
# Запуск tritonserver с оптимальными параметрами
tritonserver \
    --model-repository=/models \
    --http-thread-count=8 \              # треды на HTTP endpoint
    --grpc-infer-allocation-pool-size=8 \ # пул grpc-буферов
    --pinned-memory-pool-byte-size=$((8*1024*1024*1024)) \  # 8 GB pinned memory
    --cuda-memory-pool-byte-size=0:$((40*1024*1024*1024))   # 40 GB CUDA pool на GPU 0
```

- **`http-thread-count`** — количество тредов для обработки HTTP-запросов. Должно быть не меньше количества CPU-ядер. Больше = лучше параллелизм, но больше overhead на переключение контекста.
- **`pinned-memory-pool-byte-size`** — размер закреплённой (pinned) памяти CPU. Запросы сначала буферизируются в CPU, затем передаются на GPU. Pinned memory ускоряет transfer CPU→GPU (до 12 GB/s vs 6 GB/s для pageable).
- **`cuda-memory-pool-byte-size`** — размер CUDA memory pool (cache allocations). Формат: `<GPU_ID>:<size>`. Большой пул уменьшает overhead на `cudaMalloc`/`cudaFree`, но резервирует память, которая не доступна другим приложениям.

### 6.4 Protocol buffer vs JSON

Triton поддерживает два протокола для HTTP:

- **JSON** (по умолчанию): данные кодируются как base64-строки внутри JSON. Простота, но overhead на кодирование/декодирование + больший размер.
- **Binary (Tensor Content Extension)**: данные передаются в сыром двоичном виде в теле HTTP. Эффективнее: без base64, без лишнего парсинга.

```bash
# Запрос с binary data
curl -X POST http://localhost:8000/v2/models/reranker/infer \
    -H "Inference-Header-Content-Length: 100" \
    --data-binary @request.bin
```

Для продакшна **всегда используйте gRPC**, а не HTTP. gRPC использует protobuf — компактное бинарное представление с явной типизацией. Сравнение:

| | HTTP (JSON) | HTTP (binary) | gRPC |
|---|---|---|---|
| Размер запроса (эмбеддинг 768 fp32) | ~5 KB | ~3 KB | ~3 KB |
| Overhead на сериализацию | Высокий | Средний | Низкий |
| Читаемость | Да | Нет | Нет |
| Streaming | SSE | Нет | Да |

---

## 7. Установка и запуск

### 7.1 Docker (рекомендуемый способ)

```bash
# Pull образа (с бэкендами: tensorrt, onnx, pytorch, python)
docker pull nvcr.io/nvidia/tritonserver:24.08-py3

# Запуск сервера
docker run --gpus all --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models

# Проверка здоровья
curl -v http://localhost:8000/v2/health/ready
```

**Порты:**

- `8000` — HTTP endpoint для инференса
- `8001` — gRPC endpoint
- `8002` — Prometheus метрики

### 7.2 Triton + LLM через TensorRT-LLM

Triton не умеет инференсить LLM напрямую — для этого нужен **TensorRT-LLM бэкенд**:

```bash
# сборка Triton с TensorRT-LLM
docker build -t triton_trtllm \
    -f docker/Dockerfile.trtllm \
    --build-arg TRTLLM_VER=0.10.0 .

# запуск
docker run --gpus all --rm \
    -p 8000:8000 \
    -v /models:/models \
    triton_trtllm \
    tritonserver --model-repository=/models
```

**TensorRT-LLM** — отдельная библиотека от NVIDIA для оптимизации LLM. Она включает:

- PagedAttention (in-flight batching).
- Flash Attention 2.
- INT4/INT8/FP8 квантизацию.
- Tensor parallelism.
- In-flight batching (аналог continuous batching).

В этой связке Triton отвечает за HTTP/gRPC API, dynamic batching (если нужно), мониторинг — а TensorRT-LLM — за сам инференс.

### 7.3 Клиент

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(
    url="localhost:8001",
    verbose=False
)

# Подготовка входного тензора
input_ids = np.array([[101, 2054, 2003, 1037, 3007]], dtype=np.int64)
input_tensor = grpcclient.InferInput("input_ids", input_ids.shape, "INT64")
input_tensor.set_data_from_numpy(input_ids)

attention_mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
mask_tensor = grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
mask_tensor.set_data_from_numpy(attention_mask)

# Вызов модели
response = client.infer(
    model_name="text_encoder",
    inputs=[input_tensor, mask_tensor]
)

# Получение результата
embedding = response.as_numpy("embeddings")
print(f"Embedding shape: {embedding.shape}")
```

---

## 8. Мониторинг

### 8.1 Prometheus метрики

Triton экспортирует метрики на порту `8002`:

```bash
curl http://localhost:8002/metrics
```

**Ключевые метрики:**

| Метрика | Тип | Описание |
|---|---|---|
| `nv_inference_request_success` | Counter | Всего успешных запросов |
| `nv_inference_request_failure` | Counter | Упавших запросов |
| `nv_inference_count` | Counter | Всего инференсов |
| `nv_inference_execution_count` | Counter | Количество запусков модели (с учётом батчей) |
| `nv_inference_request_duration_us` | Histogram | Распределение latency |
| `nv_inference_queue_duration_us` | Histogram | Время в очереди |
| `nv_inference_compute_input_duration_us` | Histogram | Время на подготовку входа |
| `nv_inference_compute_infer_duration_us` | Histogram | Время инференса |
| `nv_inference_compute_output_duration_us` | Histogram | Время на обработку выхода |
| `nv_gpu_utilization` | Gauge | Загрузка GPU % |

**Важная метрика: `nv_inference_queue_duration_us`.** Если время в очереди растёт, а `compute_infer` — нет, проблема в rate limiter или нехватке инстансов. Если растёт и очередь, и compute — модель стала узким местом.

### 8.2 Health endpoints

```bash
# Live (жив ли процесс)
curl http://localhost:8000/v2/health/live

# Ready (готов принимать запросы)
curl http://localhost:8000/v2/health/ready

# Статус всех моделей
curl http://localhost:8000/v2/models
```

---

## 9. Типичные ошибки и нюансы

### 9.1 Shape mismatch на входе

**Проблема:** модель в ONNX ожидает `[batch, seq_len, dim]`, а клиент отправляет `[seq_len, batch, dim]`.

**Решение:** проверять dims в `config.pbtxt` и порядок осей на клиенте. Для ONNX модели Triton не транспонирует данные автоматически.

### 9.2 Memory pool исчерпан

**Симптом:** `CUDA OOM` при старте второй модели.

**Причины:**
- `cuda-memory-pool-byte-size` выставлен слишком большим — вся память зарезервирована, для второй модели не осталось.
- На одной GPU запущено слишком много `instance_group` инстансов.

**Решение:** уменьшить `cuda-memory-pool-byte-size` или `instance_group.count`.

### 9.3 Rarefaction при низкой нагрузке

**Симптом:** батчи маленькие (2–3 запроса) при `preferred_batch_size = [64]`.

**Причина:** `max_queue_delay_microseconds` слишком мал (например, 100 мкс). Triton не успевает накопить достаточное количество запросов.

**Решение:** увеличить `max_queue_delay_microseconds` до 500–1000 (0.5–1 ms). Подходит, когда задержка в 1 ms допустима по SLA.

### 9.4 Запросы падают с 503

**Симптом:** при высокой нагрузке клиенты получают `503 Service Unavailable`.

**Причина:** rate limiter или превышение `max_batch_size`. Triton кладёт избыточные запросы в очередь с ограниченной ёмкостью. Если очередь переполнена — новые запросы получают 503.

**Решение:** увеличить `max_batch_size`, `instance_group.count` или отключить rate limiter (риск OOM).

### 9.5 Ensemble падает целиком при ошибке одной модели

**Симптом:** если `text_encoder` в ensemble упал (например, OOM), весь пайплайн возвращает ошибку.

**Решение:** использовать BLS вместо ensemble для сценариев, где нужна graceful degradation.

---

## 10. Когда выбирать Triton

### Triton — хороший выбор, если:

1. **Сервис использует несколько моделей.** Encoder + reranker + постпроцессинг — один Triton вместо трёх микросервисов.
2. **Модели на разных фреймворках.** PyTorch encoder и ONNX reranker работают в одном сервере.
3. **Нужно динамическое батчирование.** В TGI/vLLM батчинг завязан на авторегрессивную генерацию. Для encoder'ов и CNN нужен dynamic batching.
4. **Гетерогенное оборудование.** Encoder на GPU, постпроцессинг на CPU — Triton управляет назначением.
5. **Продакшн-требования к мониторингу.** Prometheus метрики и health checks встроены.

### Triton — избыточен, если:

1. **Только одна LLM.** vLLM или TGI дадут более высокий throughput с меньшей сложностью настройки (57% vs 24x в бенчмарках).
2. **Простой сервис без композиции.** Одна модель, один endpoint, нет пайплайна — Triton добавит сложности без выгоды.
3. **Эксперименты и прототипирование.** Для быстрых экспериментов Triton — излишняя инфраструктура.
