# Scalable Nearest Neighbors (ScaNN)

ScaNN (Scalable Nearest Neighbors) — библиотека от Google Research для сверхбыстрого поиска ближайших соседей в больших коллекциях (сотни миллионов и миллиарды векторов). Оптимизирована под низкую латентность и высокую точность, особенно на CPU с AVX2/AVX512 и на TPU.

---

## 1. Идея

* Сочетает несколько техник: **кластеризацию (partitioning)**, **продуктовую квантизацию (PQ)**, **тщательную переоценку кандидатов (reordering)** и **скоростные SIMD-инструкции**.
* Основная стратегия: быстро отбрасываем большую часть базы → оставляем небольшой набор кандидатов → пересчитываем их расстояния более точно.

---

## 2. Архитектура

1. **Partitioning (IVF-подобный этап):**

   * Вектора разбиваются на кластеры (обычно k-means).
   * Запрос ищет только в $n_{probe}$ ближайших кластерах.

2. **Scoring (PQ / tree):**

   * Внутри кластера расстояния аппроксимируются через PQ или specialized tree structures.
   * Используются SIMD-инструкции для ускорения.

3. **Reordering (Refinement):**

   * Для top-R кандидатов пересчитываются точные расстояния по float-векторам.
   * Это даёт баланс: быстрое отсеивание + высокая точность.

4. **Training:**

   * Обучение центров кластеров и PQ-кодировщиков на репрезентативном сэмпле.

---

## 3. Параметры

* **num_leaves (аналог nlist):** число кластеров. Большее значение → точнее, но дороже по памяти и обучению.
* **num_leaves_to_search (аналог nprobe):** число сканируемых кластеров. Большее значение → выше recall, но выше latency.
* **training_sample_size:** сколько точек берём для обучения центров и PQ-кодировщиков.
* **reorder_k:** сколько кандидатов после грубого поиска пересчитываем точно. Позволяет сильно повысить качество при небольших накладных расходах.

---

## 4. Сложность и память

* Память: хранит PQ-коды (десятки байт/вектор) + центры кластеров + часть исходных векторов (для refine).
* Поиск: $O(\frac{N}{num\_leaves} \cdot num\_leaves\_to\_search)$ для кандидатов + пересчёт top-R.
* На практике: миллионы запросов/с на CPU при миллионах векторов.

---

## 5. Достоинства

* Оптимизирован под производственный high-QPS поиск.
* Отличная скорость на CPU (SIMD) и TPU.
* Высокий recall даже при жёстком ограничении latency.

---

## 6. Недостатки

* Меньше фич и гибкости, чем у FAISS.
* Слабее коммьюнити, меньше туториалов.
* Обновления векторов дороже, чем в HNSW.
* На GPU менее распространён (основной упор — CPU/TPU).

---

## 7. Практические советы

* Для миллиардов векторов используйте большое `num_leaves` (≥10k) и адаптивный `num_leaves_to_search`.
* Настраивайте `reorder_k` (например, 1000) — это сильно повышает точность без серьёзного роста latency.
* При cosine-поиске нормализуйте вектора заранее.
* Для интеграции с PyTorch можно подготовить эмбеддинги как `torch.Tensor` и конвертировать в numpy.

---

## 8. Примеры кода 

> В PyTorch считаем эмбеддинги на GPU/CPU, затем конвертируем в `numpy` (ScaNN ожидает `numpy.ndarray`). Для cosine используем нормализацию и метрику `dot_product`.

### 8.1. Базовый ScaNN + PyTorch-эмбеддинги (cosine/IP)

```python
import torch
import numpy as np
import scann  # pip install scann

# Параметры
in_dim = 256   # размер входа в энкодер
emb_dim = 128  # размер эмбеддинга (d)
N = 1_000_000  # размер базы
k = 10

# Простейшая PyTorch-модель эмбеддера
class Emb(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# Инициализация модели
model = Emb(in_dim, emb_dim).eval()

# Генерим базу признаков и считаем эмбеддинги (CPU для примера)
x_features = torch.randn(N, in_dim)
with torch.no_grad():
    xb = model(x_features)

# Cosine-поиск: L2-нормализация базы и запросов
xb = torch.nn.functional.normalize(xb, dim=1)
xb_np = xb.detach().cpu().numpy().astype(np.float32)

# Строим ScaNN-индекс
searcher = (scann.scann_ops_pybind.builder(xb_np, k, "dot_product")
            .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(1000)
            .build())

# Запросы из PyTorch
xq_features = torch.randn(1000, in_dim)
with torch.no_grad():
    xq = model(xq_features)
    xq = torch.nn.functional.normalize(xq, dim=1)

neighbors, distances = searcher.search_batched(xq.detach().cpu().numpy().astype(np.float32))
print(neighbors[:2])
```

### 8.2. Инкрементальное обновление индекса (переобучение offline → hot‑swap)

```python
import numpy as np
import scann

# Предположим, что мы перерасчитали эмбеддинги (xb_new) той же размерности
xb_new = np.load("/path/xb_new.npy").astype(np.float32)  # форма (N, d)

# Строим новый индекс со схожими параметрами (offline)
new_searcher = (scann.scann_ops_pybind.builder(xb_new, 10, "dot_product")
                .tree(num_leaves=2000, num_leaves_to_search=80, training_sample_size=300000)
                .score_ah(2, anisotropic_quantization_threshold=0.2)
                .reorder(800)
                .build())

# Дальше — атомарная замена хэндла индекса в вашем сервисе (alias switch)
# current_searcher = new_searcher
```

### 8.3. Батчевый оффлайн‑поиск (эмбеддинги из PyTorch → NumPy)

```python
import torch
import numpy as np
import scann

# Пусть есть обученная torch‑модель model и подготовленный searcher
# Преобразуем большой массив признаков батчами и ищем top‑k для каждого

def torch_to_numpy_batches(tensor, bs=8192):
    for i in range(0, tensor.size(0), bs):
        yield tensor[i:i+bs].detach().cpu().numpy().astype(np.float32)

with torch.no_grad():
    corpus_feats = torch.load("/path/corpus_feats.pt")           # [M, in_dim]
    queries_feats = torch.load("/path/queries_feats.pt")         # [Q, in_dim]

    corpus_emb = torch.nn.functional.normalize(model(corpus_feats), dim=1)
    queries_emb = torch.nn.functional.normalize(model(queries_feats), dim=1)

# Индекс строим один раз по всей базе
corpus_np = corpus_emb.cpu().numpy().astype(np.float32)
searcher = (scann.scann_ops_pybind.builder(corpus_np, 20, "dot_product")
            .tree(num_leaves=4000, num_leaves_to_search=100, training_sample_size=min(400000, corpus_np.shape[0]))
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(1000)
            .build())

# Поиск по запросам батчами
results_idx = []
results_dist = []
for q_np in torch_to_numpy_batches(queries_emb, bs=4096):
    I, D = searcher.search_batched(q_np)
    results_idx.append(I)
    results_dist.append(D)

I = np.vstack(results_idx)
D = np.vstack(results_dist)
print(I.shape, D.shape)  # (Q, k)
```

---

## 9. Чеклист тюнинга

* Подбирать `num_leaves` ≈ √N.
* Начать с `num_leaves_to_search` = 50–100.
* Использовать `reorder_k` = 500–1000 для повышения recall.
* Проверять recall/latency на валидации под ваши SLA.
* При росте базы — периодически переобучать индекс.
  Чеклист тюнинга
* Подбирать `num_leaves` ≈ √N.
* Начать с `num_leaves_to_search` = 50–100.
* Использовать `reorder_k` = 500–1000 для повышения recall.
* Проверять recall/latency на валидации под ваши SLA.
* При росте базы — периодически переобучать индекс.
