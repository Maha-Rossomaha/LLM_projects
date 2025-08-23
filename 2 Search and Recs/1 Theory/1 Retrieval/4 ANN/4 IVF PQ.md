# IVF-PQ

IVFPQ сочетает кластеризацию (IVF) и продуктовую квантизацию (PQ) для ускорения и сжатия поиска ближайших соседей при почти неизменном качестве.

---

## 1. Идея и обозначения
- База: $X = \{x_i \in \mathbb{R}^d\}_{i=1}^N$, запрос $q \in \mathbb{R}^d$.
- Разбиение пространства на $nlist$ кластеров центрами $\{c_j\}_{j=1}^{nlist}$.
- Внутри каждой ячейки храним **PQ-код** вектора (обычно по **остатку**):
  $$r_i = x_i - c_{\text{assign}(x_i)}.$$
- **PQ**: разбиваем $r_i$ на $M$ под-векторов длины $d/M$ и кодируем каждый индексом ближайшего центра из своего кодбука размера $2^{nbits}$. Итоговый код — $M\cdot nbits$ бит.
- При поиске сканируем только $nprobe$ ближайших центров, а расстояние до $x_i$ аппроксимируем **асимметрично** (ADC):
  $$\operatorname{dist}(q, x_i) \approx \sum_{m=1}^M \mathrm{LUT}_m\big[\mathrm{code}_m(r_i)\big],$$
  где $\mathrm{LUT}_m$ — таблица расстояний между $q^{(m)}$ и центрами $m$‑го подпространства.

---

## 2. Архитектура индекса
1) **Coarse quantizer** (обычно IndexFlatL2/IP) — выбирает ближайшие центры $c_j$.
2) **Inverted lists** — для каждого кластера список $(\text{id}, \text{PQ-код остатка})$.
3) **PQ-кодировщик** — набор из $M$ кодбуков по $2^{nbits}$ центров каждый.
4) (Опционально) **OPQ** — линейное преобразование $R\in\mathbb{R}^{d\times d}$ перед PQ, чтобы равномернее распределить информацию по подпространствам.

---

## 3. Обучение
1) **Кластеризация**: обучаем $nlist$ центров $\{c_j\}$ на сэмпле $X_{train}$; размер $|X_{train}|$ обычно $10^5$–$10^6$.
2) **Остатки**: для каждого $x$ из $X_{train}$ находим ближайший $c_j$, считаем $r = x - c_j$.
3) **PQ-кодбуки**: обучаем $M$ кодбуков (k-means по каждому подпространству над остатками). При OPQ предварительно ищем $R$ и применяем $R x$ на этапе обучения и индексации.

> В FAISS `IndexIVFPQ` по умолчанию кодирует **остатки** (residual encoding), что заметно повышает точность при том же бюджете памяти.

---

## 4. Поиск (query-time)
1) Нормализуем под метрику (для cosine — L2-нормировка и метрика IP).  
2) Ищем $nprobe$ ближайших центров $c_j$ (cheap).  
3) Для каждого $j$ сканируем его inverted list:  
   – строим $M$ LUT-таблиц расстояний для $q$ (дешево: $M\cdot 2^{nbits}$ значений),  
   – для каждого кандидата суммируем $M$ ячеек LUT по его кодам, получаем аппрокс. расстояние,  
   – поддерживаем top-$k$ в мин‑куче.
4) (Опционально) **Refine**: пересчитать точные расстояния для top-$R$ кандидатов по исходным векторам (индекс IVFPQR).

---

## 5. Параметры и их влияние
- $nlist$: число кластеров. Большее $nlist$ → короче списки → меньше сканируемых кандидатов; но дороже тренировка и память на центры. Стартовая эвристика: $nlist \approx \sqrt{N}$.
- $nprobe$: число просматриваемых ячеек. Большее $nprobe$ → выше recall, но больше латентность. Типичные значения: 16–128.
- $M$: число субквантайзеров PQ. Большее $M$ → тоньше аппроксимация (лучше recall), но дороже память на код (линейно) и чуть больше CPU.
- $nbits$: бит на под-вектор (4/5/6/8). Большее $nbits$ → крупнее кодбуки, лучше качество, выше LUT и код.
- $d$: размерность, обычно $d$ кратно $M$ (равные подпространства).
- OPQ: матрица $R$ повышает качество при том же $M, nbits$.

---

## 6. Оценка памяти
- Код на вектор: $M\cdot nbits/8$ байт.
- ID: 4–8 байт на вектор.
- Центры: $nlist\cdot d \cdot 4$ байта.
- Иные накладные: границы списков, статистика, выравнивание.

**Пример**: $d=768,\ N=10^7,\ nlist=4096,\ M=96,\ nbits=8$.  
Код: $96$ байт/вектор; ID: $8$ байт → $\approx 104$ байт/вектор ⇒ $\approx 1.04$ ГБ на $10^7$ кодов (+ центры и накладные). Экономия относительно IVFFlat ($\approx 30$ ГБ) — на порядок.

---

## 7. Качество и ускорение
- Повышаем **recall**: растим $nprobe$, $M$, $nbits$; добавляем OPQ; включаем refine топ-$R$ кандидатов.
- Снижаем **latency**: уменьшаем $nprobe$; увеличиваем $nlist$ (если позволяет память/обучение); используем предвычисленные таблицы/низкие $nbits$; перенос на GPU.

---

## 8. Практические советы
1) Нормализация: для cosine — нормализовать и базу, и запросы; метрика IP.  
2) Тренировка: брать репрезентативный сэмпл, не смешивать сильно разные домены.  
3) Дрифт: при обновлении эмбеддингов — shadow‑индекс, канареечный трафик, alias‑switch.  
4) Адаптивный $nprobe$: снижать при высокой нагрузке, повышать при низкой.  
5) OPQ: почти «бесплатный» рост качества при фиксированном коде.  
6) Refine: IVFPQR даёт заметный буст качества при умеренной надбавке к времени.

---

## 9. Примеры кода

### 9.1. Базовый IVFPQ под cosine (IP)
```python
import numpy as np
import faiss

d = 768
nlist = 4096
M, nbits = 96, 8
k = 10

# База
xb = np.random.randn(1_000_000, d).astype('float32')
faiss.normalize_L2(xb)

# Квантайзер и IVFPQ (метрика IP эквивалентна cosine при L2-норме)
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)

# Тренировка на сэмпле
train = xb[np.random.choice(xb.shape[0], size=200_000, replace=False)]
index.train(train)

# Загрузка и поиск
index.add(xb)
index.nprobe = 64
xq = np.random.randn(1000, d).astype('float32')
faiss.normalize_L2(xq)
D, I = index.search(xq, k)
```

### 9.2. IVFPQ + OPQ (IndexPreTransform)
```python
import numpy as np
import faiss

d = 768; nlist = 4096; M, nbits = 96, 8

# OPQ матрица размера d
opq = faiss.OPQMatrix(d, M)  # разобьёт пространство согласованно с M

# Базовый IVFPQ
quantizer = faiss.IndexFlatIP(d)
ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)

# Оборачиваем в преобразование: x -> OPQ(x) -> IVFPQ
index = faiss.IndexPreTransform(opq, ivfpq)

# Тренировка (OPQ и IVFPQ обучаются совместно)
train = np.random.randn(200_000, d).astype('float32')
faiss.normalize_L2(train)
index.train(train)

# Добавление/поиск как обычно
xb = np.random.randn(1_000_000, d).astype('float32')
faiss.normalize_L2(xb)
index.add(xb)
index.nprobe = 64
xq = np.random.randn(1000, d).astype('float32')
faiss.normalize_L2(xq)
D, I = index.search(xq, 10)
```

### 9.3. IVFPQR (refine top-R)
```python
import numpy as np
import faiss

d = 512; nlist = 2048; M, nbits = 64, 8

quantizer = faiss.IndexFlatL2(d)
base = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

# IVFPQR: хранит PQ-коды + небольшой реконструктор для доуточнения top-R
R = 32  # сколько кандидатов уточнять
index = faiss.IndexIVFPQR(base, d, M, nbits, R)

# Тренировка
train = np.random.randn(150_000, d).astype('float32')
index.train(train)

# Загрузка
xb = np.random.randn(5_000_000, d).astype('float32')
index.add(xb)

# Поиск
index.nprobe = 32
xq = np.random.randn(1000, d).astype('float32')
D, I = index.search(xq, 10)
```

### 9.4. Перенос на GPU (при наличии)
```python
import faiss

# gpu_id = 0
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 0, index)  # index — любой CPU Index

# Тот же API поиска
D, I = index_gpu.search(xq, 10)
```

---

## 10. Типичные ошибки
- Несогласованная метрика: cosine без нормализации или L2 с нормализованными базовыми векторами → деградация качества.
- Слишком маленький $nlist$ для больших $N$ → длинные inverted lists, рост latency.
- Недообучение PQ/OPQ (маленький $|X_{train}|$) → сильная потеря recall.
- Отсутствие переиндексации при drift эмбеддингов → падение качества.

---

## 11. Чеклист тюнинга
- Старт: $nlist \approx \sqrt{N}$, $nprobe=32$, $M=64$–$96$, $nbits=8$.
- Если память жмёт: уменьшать $M$ или $nbits$; включить OPQ; рассмотреть IVFPQR для возврата качества.
- Если latency высока: увеличить $nlist$; снижать $nprobe$; вынести на GPU; включить адаптивный $nprobe$.
- Если recall низок: растить $nprobe$; включить OPQ; поднять $M$ или $nbits$; добавить refine.
