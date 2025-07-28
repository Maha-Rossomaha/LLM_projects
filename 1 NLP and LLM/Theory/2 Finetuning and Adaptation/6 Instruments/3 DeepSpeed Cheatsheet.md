# Шпаргалка по DeepSpeed

## Что такое DeepSpeed

DeepSpeed — это фреймворк от Microsoft для масштабируемого и эффективного обучения больших моделей (в том числе LLM) с поддержкой оптимизаций памяти, скорости и стоимости.

### Назначение и мотивация

Стандартный PyTorch DistributedDataParallel требует много ручной настройки и неэффективен при обучении очень больших моделей. DeepSpeed предлагает готовые решения:

- оптимизация памяти (offloading, zero redundancy)
- масштабирование на десятки и сотни GPU
- интеграция с 3D parallelism (data, tensor, pipeline)
- снижение затрат на обучение и inference (int8/4bit, quantization-aware training)

Используется в проектах с масштабом от десятков до миллиардов параметров.

## Основные компоненты

| Компонент                | Назначение                                                       |
| ------------------------ | ---------------------------------------------------------------- |
| ZeRO (v1/v2/v3)          | Убирает дублирование параметров, градиентов, оптимизаторов       |
| DeepSpeed Engine         | Обёртка модели и оптимизатора, управляет распределением ресурсов |
| Config-файл (`.json`)    | Централизованная настройка: память, fp16, ZeRO, offload и т.д.   |
| DeepSpeed CLI            | `deepspeed train.py` с указанием конфига и аргументов            |
| Offloading               | Перенос параметров и градиентов на CPU или NVMe                  |
| Activation checkpointing | Снижение памяти за счёт recompute активаций                      |

## Пример конфига (`ds_config.json`)

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 2,
  "fp16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu" },
    "allgather_partitions": true
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  }
}
```

## Минимальный пример запуска

```bash
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

Внутри `train.py`:

```python
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    config="ds_config.json"
)
```

## Когда использовать DeepSpeed

| Сценарий                          | Подходит?                               |
| --------------------------------- | --------------------------------------- |
| Обучение модели >1B параметров    | Да                                      |
| Нехватает GPU памяти              | Да, через offload + ZeRO                |
| Нужно ускорить training loop      | Да, за счёт оптимизаций в ядре          |
| Используется Accelerate           | Да, DeepSpeed интегрирован в Accelerate |
| Нужно inference на CPU/GPU в 4bit | Да, через quantization + offload        |

## Продвинутые возможности

- ZeRO Stage 3: разбиение *всех* параметров модели по девайсам
- CPU/NVMe offload: до 10× экономия памяти
- 3D parallelism: Tensor + Pipeline + Data
- Quantization Aware Training
- Sparse Attention / Mixture of Experts (MoE)
- Профилировка памяти и скорости (`deepspeed.runtime.utils`)

## Ограничения и особенности

- Требует json-конфиг, нельзя всё задать в коде
- Иногда сложно отлаживать ошибки из-за обёрток
- Плохо работает без `deepspeed.initialize()`
- Возможны несовместимости с кастомными слоями без поддержки ZeRO

## Интеграция с Hugging Face

- `transformers.Trainer` поддерживает `deepspeed` напрямую (аргумент `--deepspeed`)
- Через `accelerate` можно подключать `deepspeed_config.json`
- Некоторые модели (например, BLOOM, OPT) уже имеют готовые конфиги

## Что учесть в продакшне

- Сохранять `ds_config.json` рядом с кодом (версионирование)
- Проверить, совместим ли оптимизатор с ZeRO Stage > 1
- Не забывать `model.eval()` и `with torch.no_grad()` для inference
- Использовать `gradient_clipping` и `gradient_accumulation` из конфига
- Профилировать загрузку GPU/CPU через DeepSpeed logging или `deepspeed.monitor`

