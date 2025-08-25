# Cross vs Bi-encoder

Сравнение двух базовых архитектур dense‑retrieval: **bi‑encoder** (раздельное кодирование запроса и документа) и **cross‑encoder** (совместное кодирование пары).

---

## 1. Краткая интуиция
- **Bi‑encoder**: заранее считаем эмбеддинги документов $d = f_{doc}(x)$, кладём их в ANN‑индекс (например, FAISS/HNSW). При запросе считаем $q = f_{query}(y)$ и ищем ближайших соседей по метрике (cosine/IP/L2). Это очень быстро и масштабируемо, подходит для миллиардов объектов, но модель теряет часть информации о взаимодействии query–doc.
- **Cross‑encoder**: формируем пару $[y; x]$ и подаём её целиком в модель; она сразу выдаёт $score = f_{cross}([y; x])$. Такой подход учитывает токен‑к‑токен взаимодействия и даёт наивысшее качество, но работает медленно, так как требуется прогон через модель для каждого кандидата.
- **Гибрид**: bi‑encoder извлекает top‑$K$ кандидатов из ANN (быстро), а cross‑encoder пересортировывает top‑$R$ ($R \ll K$), добиваясь баланса скорость ↔ качество.

---

## 2. Формализация
**Bi‑encoder**
- Кодеры: $q = f_q(y) \in \mathbb{R}^d$, $d = f_d(x) \in \mathbb{R}^d$.
- Сходство: $s(q,d) = \begin{cases} q^\top d & \text{(IP)}\\ \frac{q^\top d}{\|q\|\,\|d\|} & \text{(cosine)}\\ -\|q-d\|_2^2 & \text{(L2)}\end{cases}$.
- Обучение: contrastive (InfoNCE), triplet, margin ranking, дистилляция от cross‑encoder.

**Cross‑encoder**
- Параметризуем $score = f_{cross}([y;x])$.
- Обучение: pointwise (BCE/регрессия), pairwise (margin ranking), listwise (softmax на списке кандидатов).

---

## 3. Потоки инференса
- **Bi‑encoder (offline + online)**: предвычислить $d_i$, построить ANN (FAISS/HNSW/IVFPQ/ScaNN); online: $q$ → top‑$K$.
- **Cross‑encoder (online)**: для каждого кандидата считать $f_{cross}([y;x_i])$ → дорого; обычно только rerank top‑$R$ из предыдущего шага.

```
Hybrid pipeline:
[Query y] --f_q--> [q] --ANN--> [top-K docs] --f_cross--> [reranked top-R]
```

---

## 4. Обучение и лоссы
**Bi‑encoder**
- **InfoNCE / In‑batch Negatives**: для батча положительных пар $(y_i, x_i)$ строим матрицу S с $S_{ij}=s(f_q(y_i), f_d(x_j))$; оптимизируем $\text{CE}(i, \text{softmax}(S_{i,*}/\tau))$.
- **Hard negatives**: добавлять трудные отрицательные примеры (BM25‑негативы, семантически близкие, «клик‑но‑нет»).
- **KD от cross‑encoder**: минимизировать MSE/CE между $s(q,d)$ и $score_{cross}(y,x)$ на тех же кандидатах.

**Cross‑encoder**
- **Pointwise**: $\text{BCE}(f_{cross}([y;x]), \text{label})$.
- **Pairwise**: $\max(0, m - f_{cross}([y;x^+]) + f_{cross}([y;x^-]))$.
- **Listwise**: softmax/CE по списку кандидатов.

---

## 5. Метрики и нормализация
- Для cosine: L2‑нормализация эмбеддингов ($q \leftarrow q/\|q\|$, $d \leftarrow d/\|d\|$), метрика IP.
- Для IP без нормализации: следить за распределением норм (смещает ранжирование).
- Cross‑encoder выдаёт необработанные скоры; полезна калибровка (temperature, Platt scaling), если нужен порог классификации.

---

## 6. Скорость, память, стоимость
| Критерий | Bi‑encoder | Cross‑encoder |
|---|---|---|
| Online‑latency | Очень низкая (ANN) | Высокая (O(R·T\_model)) |
| Offline‑стоимость | Высокая (индексация) | Низкая |
| Память | Храним эмбеддинги ($N\cdot d$) | Не храним, но тратим CPU/GPU на онлайне |
| Качество | Ниже cross, зависит от эмбеддера | Выше всего |
| Масштаб | Миллиарды (multi‑shard) | Десятки–сотни кандидатов на запрос |

---

## 7. Тюнинг гибридного пайплайна
- Выбор $K$ (кандидаты из ANN) и $R$ (rerank cross): обычно $K \in [200, 2000]$, $R \in [20, 200]$.
- A/B: подбирать $K,R,\tau$ (temperature), пороги.
- ANN‑параметры: $nlist/nprobe$ (IVF), $M/ef$ (HNSW), num\_leaves/num\_leaves\_to\_search (ScaNN).
- Дистилляция: повышает recall при том же latency (bi перенимает сигналы cross).

---

## 8. Особые случаи и риски
- **Asymmetric search**: разные распределения $q$ и $d$ → обязательно L2‑норма и проверка метрики.
- **Long docs**: чанкинг + doc‑level rerank.
- **Мультиязычие/домены**: адаптация/дообучение, контроль дрейфа (Embedding Drift).
- **Фильтры/метаданные**: pre‑/post‑filter на уровне векторной БД.

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

## 10. Чеклист выбора
- **Нужен real‑time и миллиарды документов** → bi‑encoder (+ ANN, multi‑shard), cross как reranker.
- **Нужна максимальная точность при малом потоке** → cross‑encoder напрямую.
- **Хочется качества cross при скорости bi** → дистилляция + гибрид $K\to R$.
- **Сложные фильтры/бизнес‑правила** → гибрид с метаданными и post‑/pre‑filter.

---

## 11. Резюме
- **Bi‑encoder**: скорость и масштаб, но часть «тонких» взаимодействий теряется.
- **Cross‑encoder**: максимум качества, но цена — latency.
- **Практика**: двухэтапный pipeline с дистилляцией и грамотным тюнингом $K, R$ даёт наилучший баланс recall/latency.

