## Embedding Layer

URL:  
🔗 [Embedding layer tutorial: A comprehensive guide to neural network representations](https://www.byteplus.com/en/topic/400368)

## 1. Зачем нужен Embedding Layer
- Embedding layer преобразует дискретные категориальные данные, такие как слова, в плотные векторы, что позволяет моделям выявлять семантические связи между элементами.
- Без embedding приходилось использовать one-hot представление, которое неэффективно из-за высокой размерности и разреженности.

## 2. Внутреннее устройство
- Embedding layer хранит таблицу весов формы `(vocab_size, embedding_dim)`, где каждая строка — вектор токена.
- При вызове слой осуществляет операцию look-up: выбирает строки таблицы по входным индексам, формируя выходную тензорную последовательность векторов.
- Параметры embedding обучаются совместно с другими весами модели методом backpropagation.

## 3. Пример собственной реализации на PyTorch
```python
import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, x):
        return self.weight[x]
```
- Здесь `self.weight` инициализируется случайными значениями и оптимизируется в процессе обучения.
- Метод `forward` выполняет извлечение векторов путем индексирования тензора весов по входным токенам.


