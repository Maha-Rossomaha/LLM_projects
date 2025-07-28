# Шпаргалка по Accelerate (Hugging Face)

## Что такое Accelerate

`accelerate` — это легковесный инструмент от Hugging Face для упрощения запуска тренировки и инференса моделей на различных устройствах: CPU, GPU, Multi-GPU, TPU и кластерах без переписывания кода.

### Назначение и мотивация

Обучение и запуск больших языковых моделей требуют грамотного распределения нагрузки, настройки mixed precision, поддержки разных девайсов и интеграции с фреймворками типа DeepSpeed или FSDP. Эти задачи сложно и неудобно решать вручную — особенно если код должен работать одинаково на CPU, одной или нескольких GPU, кластерах или TPU.

`accelerate` позволяет писать универсальный код, который автоматически адаптируется под конкретное окружение. Он скрывает сложности `torch.distributed`, управления девайсами, синхронизации и mixed precision — разработчик концентрируется на модели и логике обучения, не думая о low-level инфраструктуре.

Позволяет:
- писать код один раз, запускать везде
- автоматизировать `DistributedDataParallel`, mixed precision и gradient accumulation
- использовать DeepSpeed, FSDP, Megatron без ручной настройки

## Основные возможности

| Возможность | Описание |
|------------|----------|
| `accelerate config` | CLI для настройки окружения (тип устройства, mixed precision и т.д.) |
| `Accelerator()` | Обёртка для модели, данных и оптимизатора |
| Поддержка DeepSpeed / FSDP / Megatron | Упрощает подключение этих фреймворков |
| Mixed Precision | Поддержка fp16 / bf16 тренировки |
| Gradient Accumulation | Автоматическая разбивка градиентов по шагам |
| `accelerate launch` | Запуск скрипта с распределённой конфигурацией |

## Базовый пример

```python
from accelerate import Accelerator
accelerator = Accelerator()

model, optimizer, train_dataloader = ...
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

## Комбинации и продвинутые сценарии

- **Mixed precision**: указывается при создании `Accelerator(fp16=True)` или через CLI
- **DeepSpeed**: используется `accelerate config` → `deepspeed_config.json`
- **FSDP**: аналогично DeepSpeed, можно выбрать стратегию через CLI
- **Multi-GPU**: автоматическая настройка через `accelerate launch --multi_gpu`
- **Gradient checkpointing**: можно использовать совместно с `accelerator.backward()`

## Советы и best practices

- Всегда вызывай `.prepare()` для модели, оптимизатора и лоадеров
- Используй `accelerator.print()` вместо `print()` для корректной печати при DDP
- Поддерживает интеграцию с `Trainer`, `transformers`, `deepspeed`, `bitsandbytes`
- Хорошо сочетается с `torch.compile` и `torch.profiler` для оптимизации

## Когда использовать Accelerate

| Сценарий | Подходит? |
|----------|-----------|
| Один GPU | Да, особенно с fp16/bf16 |
| Multi-GPU на одной машине | Да, автоматизация DDP |
| Multi-node кластер | Да, с настройкой через CLI |
| TPU | Поддерживается (через PyTorch/XLA) |
| Hugging Face Trainer | Поддерживается, интеграция по умолчанию |

## Альтернативы

| Инструмент | Отличия |
|-----------|----------|
| DeepSpeed | Более низкоуровневый, нужен конфиг, больше контроля |
| FSDP | Модуль PyTorch, больше гибкости, но и больше ручной работы |
| Torch DDP | Базовый уровень, требуется ручная настройка |

## Ограничения

- Не покрывает весь функционал DeepSpeed/FSDP (но упрощает 80% случаев)
- В проде может потребоваться ручная настройка логирования и запуска
- При нестандартной архитектуре возможны конфликты с prepare()

## Что учесть в продакшне

- Хранить `accelerate.yaml` в репозитории (версионирование)
- Оборачивать всё логирование через `accelerator.log()` или свой wrapper
- Не смешивать `accelerate` и ручной `torch.distributed` без понимания конфликта контекста
- Профилировать отдельно для CPU/GPU узких мест

