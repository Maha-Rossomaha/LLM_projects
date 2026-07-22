# TensorRT: AOT-компиляция и оптимизация для NVIDIA GPU

## 1. Определение и мотивация

### 1.1 Для чего нужен TensorRT

PyTorch в eager mode тратит значительную часть времени на launch overhead, не делает fusion операций и не оптимизирует память. **TensorRT** — AOT-компилятор от NVIDIA, который берёт обученную модель (в формате ONNX или через TensorRT API) и создаёт **plan file** — бинарный исполняемый файл, максимально эффективный для конкретной GPU.

**Типичное ускорение относительно PyTorch eager:**

| Модель | Без оптимизаций | + TensorRT FP16 | + TensorRT INT8 |
|--------|----------------|-----------------|-----------------|
| ResNet-50 | 1× | 3–4× | 6–8× |
| BERT-base | 1× | 2–3× | 4–5× |
| LLaMA-7B | 1× | 1.5–2× | — |

TensorRT не предназначен для обучения — только для инференса.

### 1.2 Почему не использовать просто PyTorch

- **Launch overhead.** При eager execution каждый вызов torch.nn.functional.relu — отдельный CUDA-кернел. В TensorRT всё компилируется в один fused-кернел.
- **Нет kernel autotuning.** PyTorch использует стандартные реализации cuDNN/cuBLAS. TensorRT перебирает десятки вариантов кернелов для каждой операции и выбирает самый быстрый для данной GPU.
- **Нет оптимизации памяти.** TensorRT анализирует lifetimes тензоров и переиспользует память, снижая пиковое потребление на 20–40%.
- **Нет калибровочной квантизации.** PyTorch не умеет в INT8-квантизацию с калибровкой (post-training quantization). TensorRT делает это автоматически.

## 2. Архитектура

```
PyTorch / TF → ONNX → TensorRT Parser → Network Definition
                                               │
                                               ▼
                                     Builder (optimization)
                                               │
                                               ▼
                                     Plan File (binary)
                                               │
                                               ▼
                                     Runtime (C++ / Python)
```

### 2.1 Network Definition

TensorRT представляет модель как граф операций (layers) с тензорами. Граф можно построить:

1. **Через ONNX parser** — самый распространённый способ.
2. **Через TensorRT API (C++/Python)** — ручное построение графа для кастомных моделей.
3. **Через TensorRT-LLM** — специализированный слой для LLM с поддержкой PagedAttention, Flash Attention, continuous batching.

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for err in range(parser.num_errors):
            print(parser.get_error(err))
```

### 2.2 Builder и оптимизация

Builder принимает Network Definition и превращает его в оптимизированный plan. Этап компиляции:

```python
config = builder.create_builder_config()

# ——— Precision ———
config.set_flag(trt.BuilderFlag.FP16)  # FP16 inference

# ——— Memory limit ———
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))  # 2 GB

# ——— Dynamic shapes ———
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 128), (8, 512), (32, 1024))  # min, opt, max
#   ───────── ───── ────  ──────  ──────────
#   имя входа  min   opt   max
config.add_optimization_profile(profile)

# ——— INT8 calibration ———
if use_int8:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator  # калибровочный датасет

# ——— Build ———
plan = builder.build_serialized_network(network, config)
with open("model.plan", "wb") as f:
    f.write(plan)
```

### 2.3 Optimisation profile

**Optimisation profile** — обязательный элемент для моделей с динамическими размерами (batch size, seqlen). Задаётся три точки: min, opt, max.

- **min:** минимальный размер входа (batch=1, seqlen=128).
- **opt:** типичный размер, под который engine оптимизирован максимально.
- **max:** максимальный размер (выше — ошибка).

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --fp16 \
        --minShapes=input:1x128 \
        --optShapes=input:8x512 \
        --maxShapes=input:32x1024
```

Если реальный batch size выходит за пределы [min, max] — engine падает с ошибкой. Нужно перекомпилировать с новым profile.

## 3. Ключевые оптимизации

### 3.1 Layer fusion

TensorRT объединяет последовательные операции в один fused-кернел:

```
Исходный граф:
  Conv2D → Bias → ReLU → MaxPool → Conv2D → Bias → ReLU

После fusion:
  FusedConvRelu → FusedConvRelu  (2 кернела вместо 6)
  или даже:
  FusedConvBiasReluMaxPool → FusedConvBiasRelu
```

Каждый fusion устраняет одну запись в HBM и один launch. Для моделей с сотнями слоёв экономия составляет десятки миллисекунд.

### 3.2 Kernel autotuning

Для каждой операции TensorRT перебирает все доступные реализации (cuDNN, cuBLAS, собственные кернелы) с разными параметрами:

- Размер блока (block size): 64, 128, 256, 512.
- Размер warp'а.
- Unrolling factor.
- Использование shared memory vs registers.

Перебор происходит на этапе компиляции (build). Результат кешируется в plan file — runtime использует выбранный вариант без замеров.

```bash
# trtexec показывает результаты autotuning
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 --verbose
# В выводе: "Tactic: 1234" — ID выбранной реализации для каждого слоя
```

### 3.3 Precision calibration (INT8 / FP8)

TensorRT поддерживает три режима точности:

| Режим | Размер весов | Размер активаций | Ускорение | Потери качества |
|-------|-------------|-----------------|-----------|-----------------|
| FP32 | 4 байта | 4 байта | 1× | — |
| FP16 | 2 байта | 2 байта | 2–3× | Минимальные |
| INT8 | 1 байт | 1 байт | 4–6× | Заметные (нужна калибровка) |

**INT8 калибровка:** TensorRT прогоняет калибровочный датасет (обычно 500–5000 семплов) через модель в FP16, собирает гистограммы активаций и подбирает scaling factors для минимизации потери качества.

```python
class Calibrator(trt.IInt8Calibrator):
    def __init__(self, dataloader, batch_size=32):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.batch_idx = 0

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Вернуть батч калибровочных данных."""
        if self.batch_idx >= len(self.dataloader):
            return None
        batch = self.dataloader[self.batch_idx]
        self.batch_idx += 1
        # Преобразовать в numpy и вернуть список тензоров
        return [batch[name].numpy() for name in names]

    def read_calibration_cache(self):
        return None  # можно кешировать результаты

    def write_calibration_cache(self, cache):
        with open("calibration.cache", "wb") as f:
            f.write(cache)
```

**FP8 (H100+):** TensorRT 9.0+ поддерживает FP8 — 8-битный формат с динамической экспонентой. По качеству близок к FP16, но требует в 2 раза меньше памяти и bandwidth.

### 3.4 Memory planning

TensorRT анализирует lifetimes всех промежуточных тензоров и переиспользует память:

```
До планировщика:
  T1 = conv(x)    # alloc T1 (10 MB)
  T2 = relu(T1)   # alloc T2 (10 MB), T1 всё ещё жив
  T3 = pool(T2)   # alloc T3 (5 MB), T1 жив, T2 жив
  out = fc(T3)    # alloc out (1 MB)
  # Пиковое потребление: 26 MB

После планировщика:
  T1 = conv(x)    # alloc T1 (10 MB)
  T2 = relu(T1)   # T2 использует тот же адрес, что T1 (T1 больше не нужен)
  T3 = pool(T2)   # T3 — новый адрес (T2 завершился)
  out = fc(T3)    # T3 освобождён, out на его месте
  # Пиковое потребление: 15 MB
```

Экономия памяти 20–50% в зависимости от модели.

## 4. TensorRT-LLM: инференс LLM

**TensorRT-LLM** — надстройка над TensorRT для LLM. Поддерживает:

- **PagedAttention** (блочное управление KV-кэшем).
- **Flash Attention 2** (fused attention kernel).
- **Continuous batching** (inflight batching).
- **Tensor parallelism** (разбиение модели на несколько GPU).
- **Pipeline parallelism** (разбиение по слоям).
- **Quantization** (FP16, FP8, INT4 AWQ/GPTQ).
- **Speculative decoding** (Medusa, Lookahead).

```python
# TensorRT-LLM API (пример)
from tensorrt_llm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel=1,
    dtype="float16",
    max_batch_size=32,
    max_seq_len=8192,
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

outputs = llm.generate(["Explain quantum computing"], params)
print(outputs[0].outputs[0].text)
```

TensorRT-LLM — движок, аналогичный vLLM и TGI, но с максимальной интеграцией с TensorRT и NVIDIA GPU.

## 5. Экспорт моделей в ONNX для TensorRT

PyTorch → ONNX — самый надёжный путь к TensorRT:

```python
import torch
import torch.onnx

model = MyModel().cuda().eval()
dummy = torch.randn(1, 512, device="cuda")

torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,
)
```

**Проблемы экспорта:**
- Не все PyTorch-операции поддерживаются ONNX (например, `torch.einsum`).
- Контроль потока (if/for) фиксируется на этапе trace.
- Рекомендуется использовать `torch.onnx.export` с `dynamic_axes` для гибкости.

## 6. Runtime: запуск plan file

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

# Загрузка plan file
with open("model.plan", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# Создание context (execution context)
context = engine.create_execution_context()

# Выделение памяти
inputs, outputs, bindings = [], [], []
for binding in engine:
    shape = context.get_binding_shape(binding)
    size = trt.volume(shape) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))

    # Аллокация device memory
    device_mem = cuda.mem_alloc(size * dtype.itemsize)
    bindings.append(int(device_mem))

    if engine.binding_is_input(binding):
        inputs.append({"name": binding, "mem": device_mem, "shape": shape, "dtype": dtype})
    else:
        outputs.append({"name": binding, "mem": device_mem, "shape": shape, "dtype": dtype})

# Инференс
def infer(input_numpy):
    # Копировать вход на GPU
    cuda.memcpy_htod(inputs[0]["mem"], input_numpy)

    # Выполнить engine
    context.execute_v2(bindings)

    # Копировать выход с GPU
    output = numpy.empty(outputs[0]["shape"], dtype=outputs[0]["dtype"])
    cuda.memcpy_dtoh(output, outputs[0]["mem"])
    return output
```

TensorRT Runtime можно запускать из C++ без Python — это стандарт для production.

## 7. Типичные ошибки

- **Нет optimisation profile.** Если batch size или seqlen меняются, а profile не задан — engine падает.
- **Слишком маленький optShapes.** TensorRT оптимизирует для точки opt. Если реальный размер далёк от opt, производительность ниже ожидаемой.
- **Калибровка INT8 на неподходящем датасете.** Если калибровочные данные не похожи на реальные — loss quality заметный.
- **Компиляция для одной GPU, запуск на другой.** Plan file содержит кернелы для конкретной архитектуры (sm_80, sm_90). На GPU другой архитектуры — не запустится.
- **Не использовать TensorRT-LLM для LLM.** Стандартный TensorRT без TensorRT-LLM не поддерживает PagedAttention и continuous batching — для LLM он даёт меньший выигрыш.

## 8. Вопросы для самопроверки

1. Что такое optimisation profile и зачем он нужен?
2. Как работает INT8 калибровка в TensorRT? Почему важен калибровочный датасет?
3. Какие оптимизации выполняет TensorRT (layer fusion, kernel autotuning, memory planning)?
4. Чем TensorRT-LLM отличается от стандартного TensorRT?
5. Почему plan file несовместим между разными архитектурами GPU?
6. Как экспортировать модель из PyTorch в ONNX для TensorRT?