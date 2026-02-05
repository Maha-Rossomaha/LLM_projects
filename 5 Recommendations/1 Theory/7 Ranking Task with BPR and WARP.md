# Ranking Task в рекомендациях: BPR и WARP (pairwise‑оптимизация)

## 0) Почему “ranking task” — отдельная постановка
Во многих рекомендательных задачах цель — не предсказать точное значение (рейтинг/вероятность), а **построить хороший top‑K список**.

В implicit‑данных у нас почти всегда:
- **позитивы**: пользователь взаимодействовал (клик/покупка/лайк),
- **unknown**: пользователь не взаимодействовал (не видел, не дошёл, не было экспозиции).

Поэтому “точность по MSE” плохо отражает качество рекомендаций. Pairwise‑подходы напрямую учат модель **поднимать позитивы выше негативов/unknown**.

Далее рассмотрим две классики:
- **BPR** — логистическая pairwise‑вероятность правильного порядка,
- **WARP** — оптимизация top‑K через аппроксимацию ранга (связь с precision@k).

---

## 1) Общая схема pairwise‑обучения (для обоих методов)
### 1.1 Триплеты
Для каждого пользователя `u`:
- $i ∈ I_u^+$ — позитивный айтем,
- $j ∈ I \backslash I_u^+$ — негативный/unknown (иногда “явно плохой”, если есть).

Строим триплет `(u, i, j)` и требуем:

$$
\hat{x}_{ui} > \hat{x}_{uj}
$$

где $\hat{x}_{ui}$ — скор модели.

### 1.2 Типовой backbone (пример: Matrix Factorization)
- эмбеддинг пользователя: `p_u ∈ ℝ^k`,
- эмбеддинг айтема: `q_i ∈ ℝ^k`.

Скор:

$$
\hat{x}_{ui}=p_u^\top q_i \quad (\text{иногда } +b_u+b_i)
$$

Разница скоров для триплета:

$$
\hat{x}_{uij}=\hat{x}_{ui}-\hat{x}_{uj}
$$

Именно $\hat{x}_{uij}$ оптимизируется pairwise‑лоссами.

---

## 2) BPR (Bayesian Personalized Ranking)
### 2.1 Интуиция
BPR учит вероятность того, что пользователь предпочитает `i` перед `j`:

$$
P(i \succ_u j\mid \Theta)=\sigma(\hat{x}_{uij}(\Theta))
$$

где:

$$
\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\hat{x}_{uij}=\hat{x}_{ui}-\hat{x}_{uj}.
$$

Если $\hat{x}_{uij}$ большая положительная, сигмоида близка к 1 → порядок правильный “с высокой вероятностью”.

---

### 2.2 Байесовская постановка (почему это называется Bayesian)
BPR вводит **апостериорную вероятность параметров** при условии, что для каждого пользователя наблюдаемая структура предпочтений верная:

$$
P(\Theta\mid >_u) \propto P(>_u \mid \Theta)\,P(\Theta)
$$

- $P(>_u \mid \Theta)$ — вероятность того, что модель с параметрами `Θ` объясняет правильный порядок,
- $P(\Theta)$ — априор на параметры.

Дальше делаются стандартные допущения независимости:
- пользователи независимы,
- предпочтения внутри пользователя раскладываются на независимые триплеты.

Тогда

$$
P(> \mid \Theta)=\prod_{(u,i,j)\in D_s} P(i \succ_u j\mid \Theta)
=\prod_{(u,i,j)\in D_s} \sigma(\hat{x}_{uij}).
$$

где `D_s` — множество триплетов (user, positive, negative).

Априор часто берут гауссовский:

$$
P(\Theta)\sim \mathcal{N}(0,\Sigma_\Theta)
$$

что в лог‑форме даёт L2‑регуляризацию.

---

### 2.3 Оптимизируемый функционал (BPR‑OPT)
Максимизация апостериора эквивалентна максимизации лог‑апостериора:

$$
\max_{\Theta}\; \sum_{(u,i,j)\in D_s} \ln\sigma(\hat{x}_{uij}) - \lambda_\Theta\|\Theta\|^2.
$$

В виде функции потерь (минимизация):

$$
\mathcal{L}_{BPR}= -\sum_{(u,i,j)\in D_s} \ln\sigma(\hat{x}_{uij}) + \lambda_\Theta\|\Theta\|^2.
$$

**Смысл**:
- первый член поднимает $\hat{x}_{ui}$ относительно $\hat{x}_{uj}$,
- регуляризация не даёт параметрам “раздуваться” (иначе сигмоида насыщается и модель может переобучаться).

---

### 2.4 Градиентная форма обновлений (для MF)
Пусть $\hat{x}_{ui}=p_u^\top q_i$. Тогда:

$$
\hat{x}_{uij}=p_u^\top(q_i-q_j)
$$

Производная ключевого члена:

$$
\frac{d}{dx}\big(-\ln\sigma(x)\big)=\sigma(-x)=1-\sigma(x).
$$

Обозначим $\gamma=\sigma(-\hat{x}_{uij})$. Тогда (без регуляризации):

$$
\frac{\partial \mathcal{L}}{\partial p_u}=\gamma\,(q_j-q_i),\quad
\frac{\partial \mathcal{L}}{\partial q_i}=-\gamma\,p_u,\quad
\frac{\partial \mathcal{L}}{\partial q_j}=\gamma\,p_u.
$$

Интуитивно:
- если позитив и негатив перепутаны (Δ маленькая или отрицательная) → `γ` большая → сильный апдейт.

---

### 2.5 Связь BPR и AUC
AUC можно понимать как долю пар (positive, negative), которые модель упорядочила верно. BPR максимизирует **логистически сглаженную** версию этого принципа:
- вместо жёсткого индикатора $I(\hat{x}_{ui} > \hat{x}_{uj})$ используется гладкая $\sigma(\hat{x}_{uij})$.

Отсюда распространённая интуиция: **BPR ≈ оптимизация AUC** (в мягкой форме).

---

### 2.6 Практика: откуда берутся негативы
`j` почти всегда — не истинный негатив, а unknown.
Стратегии:
- uniform sampling,
- popularity sampling,
- in‑batch negatives,
- hard negatives.

Качество BPR сильно зависит от негатив‑семплинга.

---

### 2.7 Мини‑пример на Python (BPR для MF, SGD)
```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bpr_step(P, Q, u, i, j, lr=0.05, reg=0.01):
    """Один SGD‑шаг BPR для MF: x_ui = p_u^T q_i."""
    pu = P[u]
    qi = Q[i]
    qj = Q[j]

    x_uij = pu @ (qi - qj)
    gamma = sigmoid(-x_uij)  # 1 - sigmoid(x_uij)

    # grad ascent for log-sigma, but we implement as updates directly
    P[u] += lr * ( gamma * (qi - qj) - reg * pu )
    Q[i] += lr * ( gamma * pu        - reg * qi )
    Q[j] += lr * ( -gamma * pu       - reg * qj )


def sample_negative(rng, n_items, user_pos_set):
    j = rng.integers(0, n_items)
    while j in user_pos_set:
        j = rng.integers(0, n_items)
    return j

# toy train loop
rng = np.random.default_rng(0)
user_pos = [ {1,2}, {2,3,4}, {0} ]
n_users, n_items, k = 3, 6, 8
P = 0.01 * rng.standard_normal((n_users, k))
Q = 0.01 * rng.standard_normal((n_items, k))

for epoch in range(50):
    for u in range(n_users):
        if not user_pos[u]:
            continue
        i = rng.choice(list(user_pos[u]))
        j = sample_negative(rng, n_items, user_pos[u])
        bpr_step(P, Q, u, i, j)

scores = Q @ P[0]
print('user0 top:', np.argsort(-scores)[:3])
```

---

## 3) WARP (Weighted Approximate‑Rank Pairwise)
### 3.1 Мотивация
BPR оптимизирует “в среднем по парам” и часто коррелирует с AUC.
Но в рекомендациях нас нередко больше интересуют **top‑K метрики** (Precision@K/Recall@K/NDCG@K).

WARP пытается приблизить оптимизацию top‑K через **ранг** позитивного айтема.

---

### 3.2 Идея: margin + ранк‑взвешивание
Рассмотрим триплет `(u, i, j)` и хотим, чтобы позитив был выше негативного **с зазором** (margin):

$$
\hat{x}_{ui} \ge \hat{x}_{uj} + 1
$$

Эквивалентно:

$$
1 - \hat{x}_{ui} + \hat{x}_{uj} \le 0.
$$

Если условие нарушено, используем hinge‑потерю:

$$
\ell_{hinge}(u,i,j)=\max(0,\; 1 - \hat{x}_{ui} + \hat{x}_{uj}).
$$

Но ключ WARP не только в hinge, а в том, что апдейт **взвешивается оценкой ранга** позитивного айтема.

---

### 3.3 Как WARP аппроксимирует ранг
Для фиксированных `(u,i)` (позитивная пара) WARP сэмплирует негативы `j` до тех пор, пока не найдёт “нарушителя”:

$$
\hat{x}_{uj} > \hat{x}_{ui} - 1
$$

Пусть мы сделали `N` попыток, прежде чем нашли нарушителя.
Интуиция:
- если нарушителя нашли быстро (маленький `N`), значит, у позитивного айтема **плохой ранг** (много негативов выше него) → нужен сильный апдейт,
- если нарушителя долго не находится (большой `N`), значит, позитив уже высоко → апдейт должен быть маленьким.

Оценка ранга обычно берётся как:

$$
\hat{r} \approx \left\lfloor \frac{Y-1}{N} \right\rfloor
$$

где `Y` — число айтемов.

Дальше применяем ранк‑взвешивание:

$$
\mathcal{L}_{WARP} = L(\hat{r})\cdot \max(0, 1 - \hat{x}_{ui} + \hat{x}_{uj}).
$$

Функция `L(k)` выбирается так, чтобы сильнее штрафовать ошибки на верхних позициях. Частые варианты:

- логарифм:
$$
L(k)=\log(k)
$$

- гармоническая сумма:
$$
L(k)=\sum_{t=1}^{k}\frac{1}{t}
$$

Гармоническая форма сильнее похожа на “focus на top‑K”.

---

### 3.4 Почему WARP связан с precision@k
Precision@K зависит от того, **сколько релевантных попало в верхние позиции**.
WARP делает апдейты тем больше, чем хуже оценённый ранг позитивного айтема (особенно если он должен быть “вверху”).

То есть WARP “сильнее переживает” ошибки на верхних позициях, чем ошибки в хвосте — отсюда связь с оптимизацией top‑K.

---

### 3.5 Алгоритм Online WARP (разбор по шагам)
Для каждого обучающего примера `(u,i)`:

1) Вычислить скор позитивной пары: $f(u,i)=\hat{x}_{ui}$.
2) Инициализировать `N=0`.
3) Повторять:
   - сэмплировать негативный айтем `j` (который не является позитивом),
   - посчитать $f(u,j)=\hat{x}_{uj}$,
   - `N = N + 1`,
   - остановиться, если найден нарушитель:

$$
\hat{x}_{uj} > \hat{x}_{ui} - 1
$$

   или если `N` достигло лимита (например `Y-1`).

4) Если нарушитель найден:
   - оценить ранг $\hat{r}=\lfloor (Y-1)/N \rfloor$,
   - сделать градиентный шаг по лоссу:

$$
L(\hat{r})\cdot \max(0, 1 - \hat{x}_{ui} + \hat{x}_{uj}) + \lambda\|\Theta\|^2.
$$

Смысл:
- **сложные** (top‑ошибки) получают **большие** веса,
- лёгкие (уже хорошо ранжируемые) почти не двигают модель.

---

## 4) BPR vs WARP — краткое сравнение
### BPR
- Логистический лосс `-log σ(Δ)`.
- Мягкая оптимизация pairwise порядка.
- Хорошо коррелирует с AUC.
- Часто более гладкое/стабильное обучение.

### WARP
- Margin + hinge.
- Аппроксимация ранга через число сэмплов до нарушителя.
- Сильнее заточен на top‑K (precision@K).
- Чувствителен к негатив‑семплингу и лимитам `N` (важно для скорости).

Оба метода применимы к разным backbone‑моделям (MF, kNN, FM, NN), потому что им нужен только скор $\hat{x}(u,i)$.

---

## 5) Мини‑пример на Python: один шаг WARP для MF
```python
import numpy as np

def warp_rank_weight(r):
    # пример: гармонический вес
    # L(r) = sum_{t=1}^r 1/t
    return np.sum(1.0 / np.arange(1, r + 1)) if r > 0 else 0.0


def warp_step(P, Q, u, i, user_pos_set, lr=0.05, reg=0.01, margin=1.0, max_trials=1000, seed=None):
    rng = np.random.default_rng(seed)
    n_items = Q.shape[0]

    pu = P[u]
    qi = Q[i]
    x_ui = pu @ qi

    # sample negatives until find a violator
    N = 0
    j = None
    x_uj = None

    while N < max_trials:
        cand = rng.integers(0, n_items)
        if cand in user_pos_set:
            continue
        N += 1
        score = pu @ Q[cand]
        if score > x_ui - margin:
            j = cand
            x_uj = score
            break

    if j is None:
        return  # no update

    # estimate rank
    Y = n_items
    r_hat = int((Y - 1) / N)
    w = warp_rank_weight(r_hat)

    # hinge loss active by construction: 1 - x_ui + x_uj > 0
    qj = Q[j]

    # gradients for hinge: d/dp_u (1 - p·qi + p·qj) = (qj - qi)
    pu_old = pu.copy()
    qi_old = qi.copy()
    qj_old = qj.copy()

    P[u] -= lr * ( w * (qj_old - qi_old) + reg * pu_old )
    Q[i] -= lr * ( w * (-pu_old)         + reg * qi_old )
    Q[j] -= lr * ( w * (pu_old)          + reg * qj_old )

# toy
rng = np.random.default_rng(0)
n_users, n_items, k = 3, 10, 8
P = 0.01 * rng.standard_normal((n_users, k))
Q = 0.01 * rng.standard_normal((n_items, k))
user_pos = [ {1,2}, {2,3,4}, {0} ]

for epoch in range(50):
    for u in range(n_users):
        if not user_pos[u]:
            continue
        i = rng.choice(list(user_pos[u]))
        warp_step(P, Q, u, i, user_pos[u], max_trials=100, seed=int(rng.integers(0, 1e9)))

scores = Q @ P[0]
print('user0 top:', np.argsort(-scores)[:5])
```

Замечания:
- это учебный пример; в реальных реализациях делают более аккуратный семплинг и оптимизируют вычисления,
- `max_trials` контролирует время обучения.

---

## 6) Практические советы
1) Негатив‑семплинг — ключевой рычаг качества (особенно для WARP).
2) Для BPR следить за:
   - регуляризацией,
   - нормами эмбеддингов,
   - насыщением сигмоиды.
3) Для WARP следить за:
   - лимитами на число сэмплов,
   - тем, насколько “сложные” негативы попадаются,
   - стабильностью при больших каталогах.
4) В прод‑retrieval оба метода хорошо сочетаются с ANN (инференс через dot product).
