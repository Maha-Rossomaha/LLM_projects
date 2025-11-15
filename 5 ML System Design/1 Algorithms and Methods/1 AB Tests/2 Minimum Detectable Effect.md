# Power Analysis и MDE — полный конспект с практикой на Python

> Цель: уметь **спланировать A/B-эксперимент до запуска**: задать требуемую мощность (power), выбрать уровень значимости (alpha), задать чувствительность (MDE), посчитать **объём выборки** и оценить **длительность** теста с учётом трафика.
> Инструменты: `statsmodels`, `scipy`, немного NumPy.

---

## 1. Интуиция

**MDE (Minimum Detectable Effect)** — минимальный эффект (разница), который эксперимент способен обнаружить с заданной мощностью ($1 - \beta$) при уровне $\alpha$.

* Чем **меньше MDE** и/или **меньше шум (дисперсия)** — тем **меньше** нужен объём.
* Чем **жёстче $\alpha$** (меньше) и/или **выше power** — тем **больше** нужен объём.
* Для бинарных метрик шум ~ `p(1−p)` (максимум при p $\approx$ 0.5).
* Для средних — шум ~ дисперсия метрики ($\sigma^2$). Снижение $\sigma^2$ (CUPED, стратификация) уменьшает N.

---

## 2. Два базовых класса задач

### 2.1. Бинарная метрика (CR/CTR) — тест разности долей

**Модель.** Пусть $p_A$ — базовая конверсия, $p_B = p_A + \Delta$.  
**MDE (абсолютный)**: $\Delta_{abs} = p_B − p_A$.  
**MDE (относительный)**: $\Delta_{rel} = (p_B − p_A) / p_A$. Тогда $p_B = p_A · (1 + \Delta_{rel})$.

**Эффект-сайз (Cohen’s h)** для пропорций:
$h = 2·\arcsin(\sqrt{p_B}) − 2·\arcsin(\sqrt{p_A})$.
`statsmodels` считает размер выборки по h (эквивалент Z-тесту пропорций).

**Когда использовать.** Конверсия на пользователя, CTR на пользователя (важно: единица анализа та же, что рандомизации).

### 2.2. Непрерывная метрика (ARPU, время) — тест разности средних

**Модель.** $μ_A$ и $μ_B$ — средние, $\sigma$ — std (если дисперсии не равны — Welch, но в планировании часто берут «типичную» $\sigma$).  

**Эффект-сайз (Cohen’s d):** $d = (\mu_B − \mu_A)/\sigma$.

**Когда использовать.** ARPU, средний чек, средняя длительность, если распределение умеренно скошено (или берём лог-преобразование).

---

## 3. MDE: абсолютный vs относительный

* **Абсолютный** (в п.п. для долей, в единицах метрики для средних): удобно, если есть конкретный порог окупаемости.
* **Относительный** (%): удобнее коммуницировать (например, «хотим ловить ≥ +3% к CR»).

Примеры:

* Бинарная задача: $p_A = 0.10$, хотим $\Delta_{rel} = +5\% ⇒ p_B = 0.105 ⇒ \Delta_{abs} = +0.005$ (0.5 п.п.).
* Непрерывная задача: $μ_A = 100 ₽$, $\Delta_{abs} = +3 ₽$; если $\sigma=30 ₽ ⇒ d = 3/30 = 0.1$.

---

## 4. Расчёт размера выборки в `statsmodels`

### 4.1. Бинарная метрика (две доли)

```python
import math
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

def n_per_group_for_proportions(
    p_baseline: float,
    mde_relative: float = None,
    mde_absolute: float = None,
    alpha: float = 0.05,                                
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided"
) -> float:
    """
    Вернёт размер выборки на группу (для контрольной). 
    Для тестовой — умножить на ratio.
    """
    if (mde_relative is None) == (mde_absolute is None):
        raise ValueError("Укажите ровно один из mde_relative или mde_absolute")

    if mde_relative is not None:
        p_alt = p_baseline * (1.0 + mde_relative)
    else:
        p_alt = p_baseline + mde_absolute

    # Cohen's h
    effect_size = proportion_effectsize(p_baseline, p_alt)  
    
    analysis = NormalIndPower()
    n_ctrl = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )
    return math.ceil(n_ctrl)
```

**Пример использования:**

```python
# База 10%, хотим ловить +5% относительно (то есть 10% → 10.5%)
nA = n_per_group_for_proportions(
    p_baseline=0.10,
    mde_relative=0.05,
    alpha=0.05,
    power=0.8
)
nB = nA  # ratio=1
print(nA, nB)
```

**Замечания.**

* При p ближе к 0.5 N максимален (шум больше).
* Двусторонний тест требует большего N, чем односторонний (при прочих равных).
* При A/B/n скорректируйте $\alpha$ (Bonferroni/Holm) или проводите подтверждающий A/B для победителя — это меняет N.

### 4.2. Непрерывная метрика (разность средних)

```python
import math
from statsmodels.stats.power import TTestIndPower

def n_per_group_for_means(
    mu_baseline: float,
    mde_abs: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided"
) -> float:
    """
    Предполагаем независимые выборки. 
    Используем Cohen's d = delta / sigma.
    """
    d = mde_abs / sigma
    analysis = TTestIndPower()
    n_ctrl = analysis.solve_power(
        effect_size=d,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )
    return math.ceil(n_ctrl)
```

**Пример:**

```python
# ARPU ~ 100, σ ~ 40 (оценено по пилоту), хотим MDE_abs = +3
nA = n_per_group_for_means(
    mu_baseline=100, 
    mde_abs=3,
    sigma=40, 
    alpha=0.05, 
    power=0.8
)
print(nA)
```

**Если распределение сильно правохвостое (логнормаль):**

* Планируйте тест на **логах**: MDE задавайте как относительный множитель (например, +3%), эквивалентно разнице средних логов $\approx$ ln(1.03).
* Sigma берите как std логов.

---

## 5. Длительность эксперимента и баланс «MDE ↔ время»

Пусть нужно `nA` и `nB` пользователей. Ежедневный «видящий» трафик — $DAU_{testable}$. Доли трафика: $w_A$ и $w_B$ (обычно 50/50).

Оценка дней:
$\text{days} \approx \max(\frac{n_A}{DAU_{testable}·w_A}, \frac{n_B}{DAU_{testable}·w_B})$.

```python
import math

def estimate_days(
    nA: int, 
    nB: int, 
    dau_testable: int, 
    wA: float = 0.5, 
    wB: float = 0.5
):
    days_A = nA / (dau_testable * wA)
    days_B = nB / (dau_testable * wB)
    return math.ceil(max(days_A, days_B))

# Пример:
# нужно по 60_000 на группу, есть 80_000 тестопригодных пользователей в день, распределяем 50/50
print(estimate_days(60_000, 60_000, 80_000, 0.5, 0.5))  
# ~2 дня (идеально), на практике заложите буфер
```

**Компромиссы:**

* Меньший MDE ⇒ больше N ⇒ дольше.
* Больше power или меньше $\alpha$ ⇒ больше N.
* Увеличение доли трафика на тест ⇒ короче длительность, но выше риск для продукта.
* A/B/n: трафик делится на больше вариантов ⇒ растёт длительность (или MDE).

---

## 6. Как метрика влияет на power

### 6.1. Бинарная: зависимость от baseline p

Шум Var $\approx$ p(1−p). При p=0.5 максимален ⇒ нужен больший N. Если p очень мало/много, дисперсия меньше ⇒ N меньше.

```python
import numpy as np

def sweep_n_vs_baseline(
    mde_rel=0.05, 
    alpha=0.05, 
    power=0.8
):
    ps = np.linspace(0.02, 0.5, 25)
    ns = []
    for p in ps:
        ns.append(n_per_group_for_proportions(
            p_baseline=p, 
            mde_relative=mde_rel, 
            alpha=alpha, 
            power=power
        ))
    return ps, ns

ps, ns = sweep_n_vs_baseline()
# Выведите таблицу ps→ns или постройте график (matplotlib).
```

### 6.2. Непрерывная: роль $\sigma$

$N$ зависит от $d = \Delta/\sigma$. При фиксированном $\Delta$ уменьшение $\sigma$ в 2 раза снижает $N$ примерно в 4 раза (квадратичный эффект).  

**Варианты снижения $\sigma$:**

* **CUPED/регрессионная корректировка** (использование ковариат, например, предтестовые значения),
* стратификация/блокировка,
* устойчивые метрики (лог-трансформация, усечение/винзоризация).

**Правило:** если корректировка уменьшает дисперсию в $k$ раз, требуемый $N$ уменьшается примерно в $k$ раз.

Удобно моделировать это коэф. `vrf = (1 − R2)` — доля неизъяснённой дисперсии.

```python
def n_with_variance_reduction(n_raw: int, r2: float) -> int:
    # r2 = объясненная доля дисперсии корректировкой (0..1)
    return math.ceil(n_raw * (1 - r2))

# Если CUPED даёт R2 ≈ 0.3, то нужный N падает на 30%
print(n_with_variance_reduction(100_000, r2=0.3))  # 70_000
```

### 6.3. Кластеризация (дизайн-эффект)

Если рандомизируем **кластера** (например, регионы) или анализируем зависимые наблюдения, учитываем **ICC** (intra-cluster correlation):  
* $\text{Design Effect (DE)} \approx 1 + (m − 1)·ICC$, где $m$ — средний размер кластера.  
* Эффективный объём = $N_{сырые} / DE$ ⇒ чтобы сохранить power, **умножаем** $N$ на $DE$.

```python
def inflate_by_design_effect(
    n_raw: int,
    m: float, 
    icc: float
) -> int:
    deff = 1 + (m - 1) * icc
    return math.ceil(n_raw * deff)

print(inflate_by_design_effect(50_000, m=5, icc=0.02))  
```

---

## 7. Дополнительно: счётчики/рейты (Пуассон) и heavy-tail

### 7.1. Счётчики/рейты (ошибки/инциденты на пользователя)

Если число событий на пользователя ~ Пуассон($\lambda$), можно планировать по **отношению скоростей** (rate ratio). При экспозиции $T$ на пользователя $Var(\lambda) \approx \lambda/T$.

Упрощённый подход: перейти к **непрерывной метрике «события на пользователя»** и применить блок 4.2, оценив $\sigma$ по пилоту. Для редких событий часто лучше агрегировать **по пользователю** за окно, тогда t-тест по средним работает приемлемо.

### 7.2. Сильно правохвостые метрики (выручка)

* Планируйте тест на **лог-метрике** (или используйте винзоризацию/обрезку хвостов).
* В расчёте N берите $\sigma$ лог-метрики; MDE задавайте как относительный множитель (например, $+3\% ⇒ delta\_log \approx ln(1.03)$).

---

## 8. Power-кривые и обратные задачи

`statsmodels` умеет решать четыре варианта: найти $N$, power, effect_size или alpha при известных трёх. Это удобно для «что-если».

### 8.1. Примеры для долей

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np

analysis = NormalIndPower()
pA = 0.10
pB = 0.105
h = proportion_effectsize(pA, pB)

# 1) Найти N на группу
n = analysis.solve_power(
    effect_size=h, 
    alpha=0.05,
    power=0.8, 
    ratio=1.0, 
    alternative="two-sided"
)

# 2) Посчитать power при фиксированном N
power = analysis.power(
    effect_size=h, 
    nobs1=n, 
    alpha=0.05, 
    ratio=1.0, 
    alternative="two-sided"
)

# 3) Посчитать минимальный h (MDE) при фиксированном N и power
# Возвращает effect_size; его можно преобразовать обратно в pB численным подбором.
targets = np.linspace(0.002, 0.02, 10)  
# сетка абсолютных MDE
```

### 8.2. Примеры для средних

```python
from statsmodels.stats.power import TTestIndPower

tt = TTestIndPower()
d = 0.1  # Cohen's d
n = tt.solve_power(
    effect_size=d, 
    alpha=0.05, 
    power=0.8, 
    ratio=1.0, 
    alternative="two-sided"
)
pw = tt.power(
    effect_size=d, 
    nobs1=n, 
    alpha=0.05, 
    ratio=1.0, 
    alternative="two-sided"
)
```

---

## 9. Множественные варианты и последовательные тесты

* **A/B/n:** если вы сразу тестируете несколько вариантов, эффективный $\alpha$ на сравнение снижается (Bonferroni/Holm), что **увеличивает N**. Альтернатива — отбор победителя и подтверждающий A/B (двухэтапный процесс).
* **«Подглядывание» (peeking):** преждевременная остановка и цикличные проверки искажают $\alpha/p$. Если нужно раннее решение — используйте **групповые последовательные дизайны** или байес-мониторинг (это отдельная тема). На этап планирования закладывайте фиксированную длительность.

---

## 10. Итоговые шаблоны 

### 10.1. Планирование для CR

```python
# 1) Вводные
pA = 0.12           # baseline CR
mde_rel = 0.04      # хотим ловить +4% относ.
alpha = 0.05
power = 0.8
ratio = 1.0

# 2) N
nA = n_per_group_for_proportions(
    p_baseline=pA,
    mde_relative=mde_rel,
    alpha=alpha, 
    power=power, 
    ratio=ratio
)
nB = math.ceil(nA * ratio)

# 3) Длительность (примерная)
dau_testable = 120_000
days = estimate_days(nA, nB, dau_testable, wA=0.5, wB=0.5)

print({"nA": nA, "nB": nB, "days": days})
```

### 10.2. Планирование для ARPU (лог-метрика)

```python
import numpy as np

# 1) По пилоту оценили sigma лог-ARPU
sigma_log = 0.65
mde_rel = 0.03
delta_log = np.log(1.0 + mde_rel)

nA = n_per_group_for_means(
    mu_baseline=0.0,  # не используется
    mde_abs=delta_log,
    sigma=sigma_log,
    alpha=0.05, 
    power=0.8
)
print(nA)
```

### 10.3. Учёт variance reduction (CUPED)

```python
# Если по пилоту/бэктесту R2≈0.25
n_raw = 100_000
n_adj = n_with_variance_reduction(n_raw, r2=0.25)
print(n_adj)  # 75_000
```

### 10.4. Учёт кластеризации (ICC)

```python
n_raw = 50_000
m = 10        # средний размер кластера
icc = 0.02
n_eff = inflate_by_design_effect(n_raw, m, icc)
print(n_eff)
```

---

## 11. Частые практические решения и проверки

1. **Оцените $\sigma$ или baseline p по пилоту** (или по ретроспективным данным), чтобы план был реалистичным.
2. **Фиксируйте параметры до старта**: $\alpha$, power, MDE, длительность, единицу анализа, доли трафика.
3. **Проверяйте SRM** (соотношение выборок) и качество логов.
4. **Стратегия на heavy-tail**: лог-метрики или винзоризация, заранее это пропишите.
5. **Guardrails**: если у них высокая дисперсия, заранее поймите, какой MDE для них вы способны уловить (иногда требуется больше длительности).
6. **A/B/n**: учитывайте деление трафика и множественные сравнения — лучше планировать в два шага.
7. **Вариантность трафика по дням**: планируйте целое число недель или включайте сезонность.

---

### Быстрая памятка по API (`statsmodels`)

* Бинарные:

  * `proportion_effectsize(p1, p2)` → Cohen’s h
  * `NormalIndPower().solve_power(effect_size=h, ...)`
* Средние:

  * `TTestIndPower().solve_power(effect_size=d, ...)`
* Обратные задачи: `.power(...)`, `.solve_power(...)` с другими неизвестными.