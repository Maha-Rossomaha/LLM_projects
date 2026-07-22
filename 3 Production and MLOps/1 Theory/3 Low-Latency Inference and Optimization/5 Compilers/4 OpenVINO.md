# OpenVINO: оптимизация инференса для Intel-платформ

## 1. Определение и мотивация

### 1.1 Для чего нужен OpenVINO

Intel-процессоры широко распространены в дата-центрах и на edge-устройствах. Однако их CPU не имеют CUDA-кернов, а Intel GPU (ARC, Iris) — не NVIDIA. Стандартные фреймворки (PyTorch, TensorFlow) не оптимизированы под Intel-архитектуру.

**OpenVINO (Open Visual Inference and Neural Network Optimization)** — набор инструментов от Intel для оптимизации и запуска моделей глубокого обучения на Intel CPU, GPU, NPU и FPGA.

Типичное ускорение относительно PyTorch eager на Intel CPU:

| Модель | PyTorch CPU | + OpenVINO CPU |
|--------|-------------|----------------|
| ResNet-50 | 1× | 3–4× |
| BERT-base | 1× | 2.5–3.5× |
| Stable Diffusion | 1× | 2–3× |

OpenVINO не предназначен для NVIDIA GPU (хотя может работать через ONNX Runtime EP). Его ниша — Intel.

### 1.2 Когда OpenVINO актуален

- **CPU-инференс.** Если нет GPU или модель маленькая (эмбеддинги, ранкеры, BERT).
- **Intel ARC GPU.** Бюджетная альтернатива NVIDIA для инференса.
- **Edge-устройства.** Intel NUC, Atom, Xeon-D — OpenVINO хорошо оптимизирован для маломощных процессоров.
- **OpenVINO Model Server (OVMS).** Аналог Triton для Intel.

## 2. Архитектура

```
PyTorch / TF / ONNX → Model Optimizer (MO) → IR (XML + BIN)
                                                    │
                                                    ▼
                                            OpenVINO Runtime
                                                    │
                                          CPU / GPU / NPU / FPGA
```

### 2.1 Model Optimizer (MO)

**Model Optimizer** конвертирует модель из исходного формата в OpenVINO IR (Intermediate Representation):

```bash
# Из PyTorch (через ONNX)
python -m openvino.tools.mo \
    --input_model model.onnx \
    --output_dir ./ir \
    --input_shape [1,512] \
    --data_type FP16 \
    --compress_to_fp16

# Из TensorFlow SavedModel
mo \
    --saved_model_dir ./saved_model \
    --output_dir ./ir \
    --input_shape [1,512]
```

**Формат IR:** два файла:
- `model.xml` — граф вычислений (узлы, связи, атрибуты).
- `model.bin` — веса модели (в FP16 по умолчанию).

### 2.2 OpenVINO Runtime

Runtime загружает IR и выполняет модель на целевом устройстве:

```python
import openvino as ov
import numpy as np

core = ov.Core()

# Загрузка IR
model = core.read_model("ir/model.xml")

# Компиляция под устройство
compiled = core.compile_model(model, "CPU")

# Инференс
output = compiled([input_numpy])[compiled.output(0)]
```

**Поддерживаемые устройства:**

| Имя устройства | Тип | Когда использовать |
|---------------|-----|-------------------|
| `CPU` | Intel CPU (любой) | Базовый инференс, нет GPU |
| `GPU` | Intel GPU (ARC, Iris, UHD) | Есть Intel GPU |
| `NPU` | Intel NPU | Для AI-ускорителей (Meteor Lake+) |
| `MULTI` | CPU + GPU | Автоматическое распределение |
| `AUTO` | Любое | Автовыбор устройства |

### 2.3 OpenVINO Model Server (OVMS)

**OVMS** — аналог Triton Inference Server для Intel:

```bash
docker run -d \
    -p 8000:8000 \
    -v /models:/models \
    openvino/model_server:latest \
    --model_path /models/bert-base \
    --model_name bert \
    --port 8000 \
    --shape auto  # динамические размеры
```

OVMS поддерживает gRPC, REST, метрики Prometheus, continuous batching, model versioning.

## 3. Ключевые оптимизации

### 3.1 Automatic batching

OpenVINO автоматически определяет оптимальный batch size на основе загрузки устройства:

```python
config = {
    "PERFORMANCE_HINT": "THROUGHPUT",  # или "LATENCY" или "CUMULATIVE_THROUGHPUT"
    "NUM_STREAMS": "AUTO",             # автоматическое число стримов
}
compiled = core.compile_model(model, "CPU", config)
```

**THROUGHPUT:** OpenVINO накапливает запросы в батч автоматически. **LATENCY:** минимальная задержка, batch=1. **CUMULATIVE_THROUGHPUT:** баланс.

### 3.2 Dynamic shapes

OpenVINO поддерживает переменную длину входов без перекомпиляции:

```python
# При компиляции указать динамические размеры
model.reshape({0: ov.PartialShape([-1, -1])})  # [batch, seqlen] — динамические

# Теперь можно передавать входы разной длины
output_short = compiled([np.random.randn(1, 128)])
output_long = compiled([np.random.randn(1, 4096)])
```

**Ограничение:** dynamic shapes могут снизить производительность на 10–20% относительно фиксированных размеров.

### 3.3 INT8 квантизация (POT / NNCF)

OpenVINO поддерживает два инструмента для INT8:

**POT (Post-training Optimization Tool):**

```bash
pot \
    -m ir/model.xml \
    -o ir/quantized \
    --quantize default \
    --dataset ./calibration_data \
    --engine simplified
```

**NNCF (Neural Network Compression Framework):**

```python
import nncf
import openvino as ov

model = core.read_model("ir/model.xml")

# INT8 квантизация с калибровкой
quantized_model = nncf.quantize(
    model,
    calibration_dataset,
    subset_size=300,
    preset=nncf.QuantizationPreset.MIXED,  # mixed: INT8 для весов, FP16 для sensitive layers
)

# Сохранить
ov.save_model(quantized_model, "ir/quantized/model.xml")
```

**Результат:** модель в 2 раза меньше (веса INT8), скорость +50–100%, loss качества < 1%.

### 3.4 Throughput mode

OpenVINO может обрабатывать несколько запросов параллельно через несколько стримов (streams):

```python
config = {
    "NUM_STREAMS": "4",           # 4 параллельных стрима
    "INFERENCE_NUM_THREADS": "8",  # 8 потоков на стрим
}
```

**Как это работает:** CPU разбивается на 4 виртуальных ядра (по 2 физических ядра на стрим). Каждый стрим обрабатывает свой запрос. Если запросов меньше — стримы простаивают.

## 4. Пример: BERT с OpenVINO

```python
import openvino as ov
import numpy as np
from transformers import AutoTokenizer

# Токенизация
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(
    "OpenVINO is a toolkit for model optimization",
    return_tensors="np",
    padding=True,
)

# Загрузка и компиляция модели
core = ov.Core()
model = core.read_model("models/bert-base/ir/model.xml")
compiled = core.compile_model(model, "CPU")

# Инференс
output = compiled({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "token_type_ids": inputs["token_type_ids"],
})

# Результат
logits = output[compiled.output(0)]
predicted_class = np.argmax(logits, axis=-1)
print(f"Class: {predicted_class}")
```

**Бенчмарк (Intel Xeon Ice Lake, 16 ядер):**

| Backend | Latency (мс) | Throughput (req/s) |
|---------|-------------|-------------------|
| PyTorch CPU | 45 | 22 |
| ONNX Runtime CPU | 35 | 28 |
| OpenVINO CPU | 18 | 55 |
| OpenVINO CPU INT8 | 12 | 83 |

OpenVINO на CPU быстрее PyTorch в 2.5× и ONNX Runtime в 2×.

## 5. OpenVINO для LLM

OpenVINO поддерживает LLM через `optimum-intel`:

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"

# Загрузка и конвертация в OpenVINO IR
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,  # конвертировать PyTorch → OpenVINO
    load_in_8bit=True,  # INT8 квантизация
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Инференс
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**Ограничение:** OpenVINO для LLM уступает TensorRT-LLM и vLLM на NVIDIA GPU. Но на Intel CPU или ARC GPU — лучший вариант.

## 6. Сравнение: OpenVINO vs альтернативы

| | OpenVINO CPU | PyTorch CPU | ONNX Runtime CPU | TensorRT |
|---|---|---|---|---|
| Скорость | 2–4× vs PyTorch | 1× | 1.2–1.5× | Нет CPU |
| INT8 | POT / NNCF | Нет | QDQ | Только GPU |
| Dynamic shapes | Да | Да | Да | Ограниченно |
| Intel GPU | Да | Нет | Через EP | Нет |
| NVIDIA GPU | Нет | Да | Через EP | Да |
| Edge | Да (NUC, Atom) | Нет | Ограниченно | Нет |
| Model Server | OVMS | Нет | ORT Server | Triton |

**Когда выбирать OpenVINO:**
- Инференс на Intel CPU без GPU.
- Intel ARC GPU (бюджетный инференс).
- Edge-устройства (Intel NUC, Xeon-D).
- Нужна INT8 квантизация на CPU (POT/NNCF).

**Когда не подходит:**
- NVIDIA GPU (лучше TensorRT).
- Нужна максимальная скорость для LLM (vLLM / TensorRT-LLM).
- Нет Intel-железа (AMD, ARM).

## 7. Типичные ошибки

- **OpenVINO на NVIDIA GPU.** Не работает. Используйте ONNX Runtime + CUDA или TensorRT.
- **Забыть про PERFORMANCE_HINT.** По умолчанию OpenVINO выбирает LATENCY. Если нужен throughput — передать `PERFORMANCE_HINT: THROUGHPUT`.
- **Не настроить NUM_STREAMS.** Для CPU-инференса 1 стрим = недоиспользование ядер. Оптимально: `NUM_STREAMS: "AUTO"`.
- **Квантизация без калибровки.** NNCF без калибровочного датасета даёт заметный loss качества.
- **Dynamic shapes с фиксированным batch.** Если batch всегда 1, не нужно делать его динамическим — это добавляет overhead.

## 8. Вопросы для самопроверки

1. Какие устройства поддерживает OpenVINO (CPU, GPU, NPU)?
2. Как OpenVINO отличается от ONNX Runtime на CPU?
3. Как работает automatic batching в OpenVINO?
4. Чем POT отличается от NNCF для INT8 квантизации?
5. Как настроить OpenVINO для максимального throughput на CPU?
6. В каком сценарии OpenVINO не подходит?