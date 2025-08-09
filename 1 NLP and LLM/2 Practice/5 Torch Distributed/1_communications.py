import os

import torch
import torch.distributed as dist

from utils import get_backend


def print_rank_0(message):
    """
    Функция, которая выводит сообщения только на 0 (главном) процессе
    После этого происходит синхронизация процессов - остальные ждут, пока главный допечатает
    """
    local_rank = dist.get_rank()
    if local_rank == 0:
        print(message)
    dist.barrier()


def blocking_send_to_last():
    """
    Все процессы, кроме последнего, посылают последнему свой ранг.
    Последний процесс получает ранги всех остальных процессов и складывает их.
    """
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    send_value = torch.Tensor([local_rank]).long()
    if local_rank != world_size- 1:
        dist.send(send_value, dst=world_size-1)
    else:
        sum_ranks = 0
        recv_tensor = torch.tensor([0]).long()
        for _ in range(world_size-1):
            dist.recv(tensor=recv_tensor)
            sum_ranks += recv_tensor.item()
        print(sum_ranks)
    dist.barrier()
    print_rank_0("Успешно послали свои ранги последнему процессу")


def cyclic_send_recv():
    """
    Циклический обмен тензорами между процессами (послать следующему, считать от прошлого)
    """
    values_to_send = [10, 20, 30, 40]
    values_to_recv = [40, 10, 20, 30]
    send_tensor = torch.Tensor([values_to_send[dist.get_rank()]])
    recv_tensor = torch.zeros_like(send_tensor)
    
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    
    work_send = dist.isend(send_tensor, dst=(local_rank+1)%world_size)
    work_recv = dist.irecv(recv_tensor, src=(local_rank-1+world_size)%world_size)

    work_send.wait()
    work_recv.wait()
    print(f'Process {local_rank} sent to {(local_rank+1)%world_size} and received from {(local_rank-1+world_size)%world_size}')
    dist.barrier()
    print_rank_0("Процессы успешно получили тензоры соседних процессов!")


def group_comms():
    """
    На каждом ранге гененрируется случайный тензор.
    1. С помощью операции all_reduce происходит поиск минимального значения среди всех local_tensor
    2. С помощью all_gather собираются все local_tensor на всех процессах и ищется минимальное значение
    """
    local_tensor = torch.rand(1)
    print(f'Число на процессе {dist.get_rank()} было {local_tensor.item()}')
    dist.barrier() 
    dist.all_reduce(local_tensor, op=dist.ReduceOp.MIN)

    print(f'Минимальное значение на {dist.get_rank()} процессе: {local_tensor.item()}') # all_reduce вычисляет ReduceOp и СОХРАНЯЕТ результат в каждом процессе
    dist.barrier()

    local_tensor = torch.rand(1)
    print(f'Новое число на {dist.get_rank()} процессе - {local_tensor.item()}')
    dist.barrier() 
    all_tensors = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, local_tensor)
    print_rank_0('\n')
    all_tensors = torch.stack(all_tensors)
    min_value = all_tensors.min().item()
    print_rank_0(all_tensors)
    print_rank_0(min_value)
    

if __name__ == "__main__":
    # torchrun --nproc-per-node 4 1_communications.py
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=get_backend(), rank=local_rank, world_size=world_size)

    print_rank_0("Это сообщение должно быть выведено всего один раз")
    blocking_send_to_last()
    cyclic_send_recv()
    group_comms()