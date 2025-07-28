# Distributed Data Parallel (DDP)

Обучение больших моделей требует распределения нагрузки между несколькими GPU. Один из самых популярных и эффективных способов сделать это в PyTorch — использовать **Distributed Data Parallel (DDP)**.

---

## Что такое DDP

**DDP** — это способ распараллеливания модели по нескольким GPU/нодам с помощью копирования модели на каждый девайс и синхронизации градиентов после каждого backward.

В отличие от `DataParallel` (устаревший способ), **DDP работает по принципу “один процесс на одну GPU” (single-process, single-device)** и использует более эффективную коммуникацию градиентов на уровне C++ через `torch.distributed`.

---

## Зачем нужен DDP

- **Ускорение обучения**: каждый процесс обучает модель на своей части данных (data parallelism).
- **Масштабируемость**: можно запускать обучение на нескольких машинах.
- **Оптимизация использования памяти**: каждый процесс работает только со своей копией данных и модели.
- **Стабильность и производительность**: меньше накладных расходов и больше совместимость с `torch.cuda.amp`, `checkpointing`, `accelerate` и т.п.

---

## Как работает DDP

1. Инициализируется `torch.distributed` backend (обычно `nccl` для GPU).
2. Модель копируется на каждую GPU.
3. Каждая копия получает свою часть данных.
4. После `backward()` градиенты синхронизируются **автоматически** между копиями модели.
5. `optimizer.step()` происходит локально, но после синхронизации градиентов — веса остаются согласованными.

> Градиенты синхронизируются **во время backward**, а не после — это важно для производительности.

---

## Пример DDP в PyTorch

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataloader = get_dataloader(rank, world_size)
    optimizer = torch.optim.Adam(ddp_model.parameters())

    for batch in dataloader:
        outputs = ddp_model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

---

## Плюсы и минусы

**Плюсы:**

- Высокая производительность и масштабируемость
- Поддержка mixed precision, gradient checkpointing
- Простота интеграции с PyTorch экосистемой

**Минусы:**

- Требует запуска через `torchrun` или `torch.multiprocessing.spawn`
- Более сложный дебаг (много процессов)
- Нужно аккуратно разносить данные между процессами (использовать `DistributedSampler`)

---

## Когда использовать DDP

- Если модель уже не помещается на одну GPU из-за объёма параметров, DDP не поможет — он лишь дублирует модель на каждую видеокарту. Но если модель помещается, а узким местом становится batch size или время обучения, то DDP может сильно ускорить процесс за счёт параллелизма по данным
- Когда нужно сократить **время обучения** без потери качества
- При работе в кластере с несколькими машинами и несколькими GPU

