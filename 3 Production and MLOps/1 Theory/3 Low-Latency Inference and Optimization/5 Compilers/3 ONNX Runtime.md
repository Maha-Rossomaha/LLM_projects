# ONNX Runtime: кроссплатформенный движок для инференса

## 1. Определение и мотивация

### 1.1 Проблема фрагментации рантаймов

В production ML-сервисе модель может быть обучена в PyTorch, а запускаться на устройствах разных типов:

- **GPU (NVIDIA):** максимальная скорость через CUDA.
- **GPU (AMD, Intel):** та же модель, другой бэкенд.
- **CPU:** дешёвый инференс эмбеддингов и ранжирования.
- **Mobile / Edge:** ограниченные ресурсы.

Держать под каждый тип устройства отдельную реализацию — накладно. **ONNX Runtime** решает эту проблему: модель экспортируется в ONNX (промежуточный формат), а ORT запускает её на любом устройстве через execution providers.

### 1.2 Что даёт ONNX Runtime

- **Единый формат модели.** Один ONNX-файл работает на всех бэкендах.
- **Execution providers.** Переключение между CUDA, TensorRT, OpenVINO, CoreML, DirectML, CPU без изменения кода.
- **Графовые оптимизации.** Fusion, constant folding, dead code elimination на уровне ONNX-графа.
- **Кроссплатформенность.** Windows, Linux, macOS, ARM, WebAssembly.
- **Минимальная dependency.** Для инференса не нужен PyTorch.

## 2. Экспорт модели в ONNX

### 2.1 Из PyTorch

```python
import torch
import torch.onnx

model = MyModel().cuda().eval()
dummy = torch.randn(1, 512, device="cuda")

torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,  # вычислить константы на этапе экспорта
)
```

**Параметры:**

| Параметр | Описание |
|----------|----------|
| `dynamic_axes` | Какие размерности могут меняться (batch, seqlen) |
| `opset_version` | Версия ONNX opset (чем новее, тем больше операций) |
| `do_constant_folding` | Вычислить константные подграфы заранее |
| `input_names/output_names` | Имена тензоров в графе |

**Проблемы при экспорте:**
- `torch.einsum`, `torch.where` с динамическими условиями — не экспортируются.
- Контроль потока (if по данным) фиксируется на этапе trace.
- `torch.amp.autocast` может сломать граф (рекомендуется экспорт в FP32, квантизация — в ORT).

### 2.2 Из TensorFlow / Keras

```python
import tf2onnx

# Конвертация SavedModel → ONNX
tf2onnx.convert.from_saved_model(
    saved_model_path,
    output_path="model.onnx",
    inputs_as_nchw=["input"],  # каналы на первом месте
)
```

### 2.3 Визуализация ONNX

```bash
pip install onnx onnxruntime
python -c "
import onnx
model = onnx.load('model.onnx')
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
"
# netron.app — веб-визуализатор ONNX-графа
```

## 3. Execution providers

Execution provider (EP) — бэкенд, на котором выполняется ONNX-граф. ORT поддерживает:

| Provider | Устройство | Когда выбирать |
|----------|-----------|----------------|
| `CPUExecutionProvider` | Любой CPU | Для CPU-инференса, как fallback |
| `CUDAExecutionProvider` | NVIDIA GPU | Быстрый CUDA-инференс без TensorRT |
| `TensorrtExecutionProvider` | NVIDIA GPU | Максимальная скорость (через TensorRT) |
| `OpenVINOExecutionProvider` | Intel CPU/GPU/NPU | Для Intel-железа |
| `CoreMLExecutionProvider` | Apple Silicon | Для macOS / iOS |
| `DirectMLExecutionProvider` | Windows GPU | Для Windows без CUDA |
| `AzureExecutionProvider` | Azure Cloud | Для serverless Azure |

```python
import onnxruntime as ort

# GPU inference через TensorRT (если доступен)
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)
```

**Приоритет:** ORT выбирает первый доступный провайдер. Если TensorRT не установлен — переходит к CUDA. Если CUDA нет — к CPU.

### 3.1 CUDAExecutionProvider

Стандартный EP для NVIDIA GPU. Использует cuDNN и cuBLAS. Без дополнительных компиляций — просто загружает ONNX и выполняет через CUDA.

```python
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8 GB
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }),
    ],
)
```

**Когда использовать:** когда TensorRT не нужен (модель не поддерживает или нет времени на AOT-компиляцию). Быстрее PyTorch eager за счёт графовых оптимизаций, но медленнее TensorRT.

### 3.2 TensorrtExecutionProvider

Запускает ONNX через TensorRT под капотом. Требует установленного TensorRT.

```python
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("TensorrtExecutionProvider", {
            "trt_max_workspace_size": 4 * 1 << 30,  # 4 GB
            "trt_fp16_enable": True,
            "trt_int8_enable": True,
            "trt_int8_calibration_table_name": "calibration.cache",
            "trt_dla_enable": False,
        }),
    ],
)
```

**Когда использовать:** когда нужна максимальная скорость на NVIDIA GPU, но удобнее работать через единый ONNX-формат.

### 3.3 CPUExecutionProvider

Не требует GPU. Подходит для эмбеддингов, ранжирования и маленьких моделей.

```python
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
    providers_options=[{
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 2,
    }],
)
```

## 4. Session options

```python
import onnxruntime as ort
import numpy as np

opts = ort.SessionOptions()

# ——— Графовые оптимизации ———
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Варианты:
#   ORT_DISABLE_ALL     — без оптимизаций
#   ORT_ENABLE_BASIC    — базовые (constant folding)
#   ORT_ENABLE_EXTENDED — расширенные (fusion)
#   ORT_ENABLE_ALL      — все доступные (layout optimization, NCHW→NHWC)

# ——— Потоки ———
opts.intra_op_num_threads = 4    # потоков внутри одной операции
opts.inter_op_num_threads = 2    # потоков для параллельного выполнения узлов

# ——— Память ———
opts.enable_cpu_mem_arena = True   # переиспользование CPU-памяти
opts.enable_mem_pattern = True     # pattern memory pool

# ——— Логирование ———
opts.log_severity_level = 3  # 0:VERBOSE, 1:INFO, 2:WARNING, 3:ERROR, 4:FATAL
opts.logid = "my-model"

# ——— Оптимизация для конкретного устройства ———
opts.optimized_model_filepath = "model_optimized.onnx"  # сохранить оптимизированный граф

session = ort.InferenceSession("model.onnx", opts, providers=["CUDAExecutionProvider"])
```

### 4.1 Intra-op vs Inter-op threads

| Тип | Что делает | Когда увеличивать |
|-----|-----------|-------------------|
| `intra_op_num_threads` | Параллелит одну операцию (например, matmul) | Для CPU, когда операция большая |
| `inter_op_num_threads` | Параллелит узлы графа (независимые ветки) | Для CPU, когда граф разветвлённый |

На GPU оба параметра не имеют значения — GPU сам управляет параллелизмом.

## 5. Инференс

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

# Получить имена входов/выходов
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Подготовить вход (numpy, не torch!)
input_np = np.random.randn(1, 512).astype(np.float32)

# Запуск
outputs = session.run(
    [output_name],
    {input_name: input_np},
)

# outputs — список numpy-массивов
print(outputs[0].shape)
```

**Важно:** ORT принимает numpy-массивы, не torch-тензоры. Конвертация torch → numpy добавляет overhead (~10–50 мкс). Для production можно передавать буферы через OrtValue для нулевого копирования.

### 5.1 OrtValue (zero-copy)

```python
# Создать OrtValue из существующего буфера (без копирования)
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

# numpy array → OrtValue
input_val = ort.OrtValue.ort_value_from_numpy(input_np)

# Inference
outputs = session.run_with_ort_values(
    [output_name],
    {input_name: input_val},
)
```

### 5.2 Batch inference

```python
# Батч из N запросов
batch_np = np.random.randn(8, 512).astype(np.float32)

# Всё так же — session.run принимает батч
outputs = session.run([output_name], {input_name: batch_np})
```

**Важно:** максимальный batch size ограничен ONNX-графом (параметр `max_batch_size` при экспорте или dynamic_axes).

## 6. Сравнение: ONNX Runtime vs конкуренты

| | ONNX Runtime | PyTorch eager | TensorRT |
|---|---|---|---|
| Подготовка | Экспорт ONNX | Нет | AOT-компиляция (часы) |
| NVIDIA GPU | CUDA / TensorRT EP | Нативная | Нативная (plan file) |
| AMD GPU | ROCm EP | Нет | Нет |
| Intel GPU | OpenVINO EP | Нет | Нет |
| Apple Silicon | CoreML EP | MPS backend | Нет |
| CPU | Да | Да | Нет |
| Dynamic shapes | Да | Да | Ограниченно |
| INT8 quant | Через QDQ | Нет | Да (калибровка) |
| C++ runtime | Да | Нет | Да |

**Когда выбирать ONNX Runtime:**
- Модель должна работать на разных устройствах (CPU/GPU/Intel/Apple).
- Нужно переключаться между бэкендами без перекомпиляции.
- Хочется графовые оптимизации без AOT-компиляции (как в TensorRT).
- Production для BERT/embedding/reranker моделей.

## 7. Типичные ошибки

- **Не указать providers.** По умолчанию ORT использует CPU — GPU не задействуется. Всегда передавать `providers=["CUDAExecutionProvider"]` или выше.
- **non-Numpy входы.** ORT принимает только numpy. PyTorch-тензор нужно сконвертировать: `tensor.cpu().numpy()`.
- **Динамические оси без dynamic_axes.** Если при экспорте не указать dynamic_axes, batch size фиксирован (например, только batch=1).
- **Слишком высокий graph_optimization_level.** ORT_ENABLE_ALL может сломать граф для некоторых моделей. Начинать с ORT_ENABLE_EXTENDED.
- **OrtValue без zero-copy для batch > 1.** Каждый вызов `ort.OrtValue.ort_value_from_numpy` копирует данные. Для batch лучше создать один OrtValue и переиспользовать.

## 8. Вопросы для самопроверки

1. Как execution providers позволяют запускать одну модель на разных устройствах?
2. Чем CUDAExecutionProvider отличается от TensorrtExecutionProvider?
3. Какие графовые оптимизации выполняет ONNX Runtime?
4. В чём разница между intra_op и inter_op threads?
5. Как передать данные в ORT без копирования (zero-copy)?
6. Почему при экспорте в ONNX нужно указывать dynamic_axes?