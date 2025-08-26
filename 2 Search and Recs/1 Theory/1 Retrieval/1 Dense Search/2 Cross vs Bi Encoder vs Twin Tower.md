# Cross vs Bi-encoder vs Twin-tower

Сравнение базовых архитектур dense‑retrieval: **bi‑encoder** (раздельное кодирование запроса и документа с общими весами), **twin‑tower** (раздельное кодирование с разными весами) и **cross‑encoder** (совместное кодирование пары).

---

## 1. Краткая интуиция

* **Bi‑encoder**: заранее считаем эмбеддинги документов $d = f(x)$ (одна и та же сеть для query/doc). При запросе считаем $q = f(y)$ и ищем ближайших соседей. Быстро и масштабируемо, но теряется часть взаимодействий.
* **Twin‑tower (two‑tower)**: запрос и документ кодируются разными энкодерами $q = f_q(y)$, $d = f_d(x)$. Это гибче: query‑tower можно оптимизировать под короткие тексты, doc‑tower — под длинные. Но требует больше параметров и обучения.
* **Cross‑encoder**: формируем пару $[y; x]$ и подаём её целиком в модель; она сразу выдаёт $score = f_{cross}([y; x])$. Качество выше всего, но инференс дорогой.
* **Гибрид**: bi/twin‑tower для ANN‑retrieval (top‑K), cross‑encoder пересортировывает top‑R.

---

## 2. Формализация

**Bi‑encoder (shared weights)**

* $q = f(y) \in \mathbb{R}^d$, $d = f(x) \in \mathbb{R}^d$.
* $s(q,d) = q^\top d$ или cosine.

**Twin‑tower (separate weights)**

* $q = f_q(y)$, $d = f_d(x)$, энкодеры разные.
* Обучение: те же contrastive / triplet / KD, но гибкость выше.

**Cross‑encoder**

* $score = f_{cross}([y;x])$.
* Обучение: pointwise/pairwise/listwise.

---

## 3. Потоки инференса

* **Bi‑encoder**: offline считаем все $d_i$, online: кодируем $q$, ищем через ANN.
* **Twin‑tower**: аналогично bi, но с отдельными сетями (сложнее хранить/дообучать).
* **Cross‑encoder**: онлайн прогон каждой пары $[q,d]$, дорого.

```
Hybrid pipeline:
[Query y] --f_q--> [q] --ANN--> [top-K docs] --f_cross--> [reranked top-R]
```

---

## 4. Обучение и лоссы

**Bi / Twin‑tower**

* InfoNCE / in‑batch negatives.
* Hard negatives (BM25‑ или семантические).
* KD от cross‑encoder.

**Cross‑encoder**

* Pointwise: BCE/регрессия.
* Pairwise: margin ranking.
* Listwise: softmax на списке.

---

## 5. Метрики и нормализация

* Для cosine: L2‑нормализация эмбеддингов.
* Для IP без нормализации: следить за распределением норм.
* Cross‑encoder: calibration (temperature/Platt scaling).

---

## 6. Скорость, память, стоимость

| Критерий          | Bi‑encoder           | Twin‑tower               | Cross‑encoder        |
| ----------------- | -------------------- | ------------------------ | -------------------- |
| Online‑latency    | Очень низкая         | Очень низкая             | Высокая              |
| Offline‑стоимость | Высокая (индексация) | Высокая                  | Низкая               |
| Память            | $N \cdot d$        | $N \cdot d$            | Не храним эмбеддинги |
| Качество          | Среднее              | Выше bi на спец. задачах | Максимум             |
| Масштаб           | Миллиарды            | Миллиарды                | Только топ‑R         |

---

## 7. Особые случаи и риски

* **Asymmetric search**: twin‑tower полезен, когда запросы и документы сильно различаются.
* **Long docs**: чанкинг + doc‑tower.
* **Мультиязычие/домены**: twin‑tower позволяет учесть разные распределения.
* **Фильтры/метаданные**: комбинировать с ANN + pre/post‑filter.

---

## 9. Примеры кода

### 9.1. Bi‑encoder: InfoNCE с in‑batch negatives
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

d = 256
T = 0.05  # temperature

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)

q_enc = Encoder(768, d)
d_enc = Encoder(768, d)

# Положительные пары (y_i, x_i)
batch = 128
y = torch.randn(batch, 768)
x = torch.randn(batch, 768)

q = q_enc(y)                 # [B, d]
d = d_enc(x)                 # [B, d]
S = q @ d.t() / T            # [B, B] сходства
labels = torch.arange(batch)
loss = F.cross_entropy(S, labels)  # InfoNCE: правильный ответ — диагональ
loss.backward()
```

### 9.2. Построение ANN (FAISS) + первичный поиск
```python
import faiss

# Индексация документов (offline)
index = faiss.IndexFlatIP(d)
# d_embeds: np.ndarray [N, d], L2-нормализованные
# index.add(d_embeds)

# Online: запрос → эмбеддинг → top-K
# q_embed: np.ndarray [1, d]
# D, I = index.search(q_embed, 1000)  # K=1000
```

### 9.3. Cross‑encoder для rerank top‑R
```python
class CrossScorer(nn.Module):
    def __init__(self, in_q=768, in_d=768, hid=512):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_q + in_d, hid), 
            nn.ReLU(), 
            nn.Linear(hid, 1)
        )
    def forward(self, q_tok, d_tok):  # q_tok/d_tok — любые фиксаторы признаков
        x = torch.cat([q_tok, d_tok], dim=-1)
        return self.scorer(x).squeeze(-1)  # [B]

cross = CrossScorer()

# Возьмём R кандидатов и посчитаем скоры
R = 100
q_tok = torch.randn(R, 768)
d_tok = torch.randn(R, 768)
with torch.no_grad():
    scores = cross(q_tok, d_tok)         # [R]
    reranked = scores.argsort(descending=True)
```

### 9.4. Дистилляция: учим bi‑encoder под сигналы cross‑encoder
```python
# Пусть есть teacher_scores для пар (y_i, x_j) из топ-К
teacher_scores = torch.randn(batch, batch)
student_scores = S  # из примера 9.1 (q @ d^T / T)
kd_loss = F.mse_loss(student_scores, teacher_scores)
(total_loss = loss + 0.1 * kd_loss).backward()
```

---

## 10. Тюнинг гибридного пайплайна
- Выбор $K$ (кандидаты из ANN) и $R$ (rerank cross): обычно $K \in [200, 2000]$, $R \in [20, 200]$.
- A/B: подбирать $K,R,\tau$ (temperature), пороги.
- ANN‑параметры: $nlist/nprobe$ (IVF), $M/ef$ (HNSW), num\_leaves/num\_leaves\_to\_search (ScaNN).
- Дистилляция: повышает recall при том же latency (bi перенимает сигналы cross).

---


## 10. Чеклист выбора
* **Real‑time, миллиарды документов** → bi‑encoder или twin‑tower + ANN.
* **Разные распределения query/doc (короткие vs длинные)** → twin‑tower.
* **Нужна максимальная точность при малом потоке** → cross‑encoder.
* **Баланс** → гибрид: $K$ кандидатов через bi/twin, rerank cross.

---

## 11. Резюме
* **Bi‑encoder**: скорость и масштаб, но теряются тонкие взаимодействия.
* **Twin‑tower**: чуть тяжелее, но даёт гибкость при асимметричных данных.
* **Cross‑encoder**: качество выше всего, но дорогой инференс.
* **Практика**: чаще используется гибрид bi/twin → cross.