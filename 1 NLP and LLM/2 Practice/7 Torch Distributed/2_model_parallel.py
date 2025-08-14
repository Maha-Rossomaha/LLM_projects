import os

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import get_backend, sync_module_params, gen_random_tensor_at_0


def send_2d_tensor(tensor, dst):
    """
    Иногда нам хочется послать тензор, но в принимающем процессе мы заранее не знаем, какого он размера.
    Эта функция вначале посылает двумерный тензор shape размерностей тензора, а потом уже его содержимое
    """
    dist.send(torch.tensor(tensor.shape).long(), dst=dst)
    dist.send(tensor, dst=dst)


def recv_2d_tensor(src):
    """
    Эта функция
    1. Принимает двумерный тензор размерностей
    2. Создает тензор нужных размерностей для приема данных
    3. Принимает данные в этот тензор и возвращает его
    """
    shape_tensor = torch.zeros(2).long()
    dist.recv(shape_tensor, src=src)
    received_tensor = torch.zeros(tuple(shape_tensor.tolist())).float()
    dist.recv(received_tensor, src=src)
    return received_tensor 


class PipeliningLinearLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(16, 32)
        self.ln2 = nn.Linear(32, 64)
    
    def forward(self, x):
        if dist.get_rank() == 0:
            x = self.ln1(x)
            send_2d_tensor(x, 1)
            return None

        elif dist.get_rank() == 1:
            x = recv_2d_tensor(0)
            return self.ln2(x)
            
    def forward_full_rank_0(self, x):
        if dist.get_rank() == 0:
            return self.ln2(self.ln1(x))
        return None


def test_pipelining():
    """
    Входной тензор есть только у 0го процесса
    1. На 0м процессе применяется первый линейный слой к входному тензору
    2. Посылается результат на 1й процесс
    3. На 1м процессе принимается результат и возвращается. На 0м процессе возвращается None
    """
    pp_layer = PipeliningLinearLayer()
    sync_module_params(pp_layer)
    pp_input = gen_random_tensor_at_0(7, 16)
    pp_output = pp_layer(pp_input)
    
    if dist.get_rank() == 1:
        send_2d_tensor(pp_output, 0)
    else:
        pp_output = recv_2d_tensor(1)
        assert torch.allclose(pp_output, pp_layer.forward_full_rank_0(pp_input)), 'Error'
        print("Успешно отработал пайплайнинг")
    dist.barrier()


def test_tensor_parallel():
    """
    Tensor Parallel для линейного слоя
    1. Матрицу A разбивается по процессам по последней размерности
    2. Делается матричные умножения на X
    3. Результаты конкатенируютя обратно
    """
    A = torch.rand(16, 32)
    dist.broadcast(A, 0)
    X = torch.rand(7, 16)
    dist.broadcast(X, 0)

    world_size = dist.get_world_size()
    A_chunks = torch.chunk(A, world_size, -1)

    Y_local = X @ A_chunks[dist.get_rank()]

    Y = [torch.zeros_like(Y_local) for _ in range(world_size)]
    dist.all_gather(Y, Y_local)
    
    if dist.get_rank() == 0:
        Y = torch.cat(Y, dim=1)
        Y_REF = X @ A

        assert torch.allclose(Y_REF, Y)
        print("Успешно отработал tensor parallel")
    


if __name__ == "__main__":
    # torchrun --nproc-per-node 2 2_model_parallel.py
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=get_backend(), rank=local_rank, world_size=world_size)
    test_pipelining()
    test_tensor_parallel()