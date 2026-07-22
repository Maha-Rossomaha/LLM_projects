# Speculative Decoding: ускорение генерации через маленькую модель

## 1. Определение и мотивация

### 1.1 Проблема memory-bound decode

Каждый decode-шаг LLM упирается в пропускную способность HBM, а не в FLOPs:

- Считается один токен.
- Загружаются веса всего слоя (десятки GB).
- Загружается KV-кэш (до единиц GB).

Результат: decode в 10–100 раз медленнее prefill в пересчёте на токен. Для LLaMA-70B на H100:

- Prefill: 2000 токенов за ~30 мс → ~67K токенов/с.
- Decode: 1 токен за ~40 мс → ~25 токенов/с.

**Пропускная способность decode — узкое горлышко.** Ускорить его можно, только если генерировать несколько токенов за один forward pass.

### 1.2 Идея speculative decoding

**Speculative decoding** (спекулятивная генерация) использует маленькую «assistant»-модель, чтобы быстро набросать черновик, а большая модель его проверяет.

Алгоритм:

1. **Assistant** (маленькая модель, 0.5–7B) генерирует $K$ токенов стандартным decode — быстро, потому что веса маленькие.
2. **Target** (большая модель, 70B+) принимает эти $K$ токенов как один промпт и выполняет prefill — считает вероятности для всех $K$ позиций за один forward pass.
3. **Rejection sampling:** для каждой позиции target сравнивает своё распределение $p_t(x)$ с распределением assistant $q_t(x)$. Если $p_t(x) \leq q_t(x)$ — токен принимается. Если $p_t > q_t$ — токен может быть отвергнут с вероятностью $1 - p_t/q_t$.
4. Если токен отвергнут — target ресэмплит из своего распределения для этой позиции, генерация продолжается с этого места.

```
Обычный decode:
Шаг 1: Target → t1  (40 мс)
Шаг 2: Target → t2  (40 мс)
Шаг 3: Target → t3  (40 мс)
...
Всего на 10 токенов: 400 мс

Speculative decoding (K=5):
Шаг 1: Assistant → [d1, d2, d3, d4, d5]  (10 мс)
Шаг 2: Target → проверка 5 токенов (prefill, 30 мс)
       Принято: [d1, d2, d3, d4], отвергнут: d5
       Target ресэмплит → t5
Итого: 5 токенов за ~40 мс → ~125 токенов/с
```

### 1.3 Ключевой момент: матожидание принятых токенов

Ускорение speculative decoding определяется долей принятых assistant-токенов:

$$
\text{Ускорение} = \frac{K + 1}{1 + K \cdot (1 - \alpha)}
$$

где $K$ — число черновиков, $\alpha$ — acceptance rate (доля принятых токенов).

- $\alpha = 0.8$, $K = 5$: ускорение в ${6} / {1 + 5 \cdot 0.2} = 3$ раза.
- $\alpha = 0.5$, $K = 5$: ускорение в ${6} / {1 + 5 \cdot 0.5} = 1.7$ раза.
- $\alpha = 0.95$, $K = 10$: ускорение в ${11} / {1 + 10 \cdot 0.05} \approx 7.3$ раза.

Acceptance rate зависит от согласованности assistant и target. Если модели из одной семьи (например, LLaMA-3-8B как assistant для LLaMA-3-70B) — acceptance rate 0.7–0.9. Если модели из разных семей — может упасть до 0.3–0.5.

---

## 2. Как это работает

### 2.1 Rejection sampling — строгое обоснование

Пусть $p(x)$ — распределение target, $q(x)$ — распределение assistant для одной позиции. Алгоритм:

1. Сэмплируем $x \sim q(x)$.
2. Сэмплируем $r \sim U[0, 1]$.
3. Если $r \leq \min(1, p(x)/q(x))$ — принимаем $x$.
4. Иначе — сэмплируем $x \sim \hat{p}(x) = \text{norm}\big(\max(0, p(x) - q(x))\big)$.

**Утверждение:** итоговое распределение совпадает с $p(x)$. Speculative decoding **не меняет качество генерации**, а только ускоряет её.

Доказательство интуитивно: вероятность принять токен $x$ равна $q(x) \cdot \min(1, p(x)/q(x)) = \min(q(x), p(x))$. Вероятность отвергнуть — $1 - \sum_x \min(q(x), p(x))$. После rejection сэмплинг из $\max(0, p - q)$ восстанавливает $p$:

$$
P_{\text{final}}(x) = \min(q, p) + \frac{\max(0, p - q)}{\sum_x \max(0, p - q)} \cdot \big(1 - \sum_x \min(q, p)\big) = p(x)
$$

> **Следствие:** модель можно ускорять speculative decoding без re-training, fine-tuning или изменения архитектуры. Качество идентично обычному decode.

### 2.2 Выбор K (числа черновиков)

Слишком маленькое $K$: assistant мало помогает (большая часть ускорения не используется).

Слишком большое $K$: вероятность ошибки assistant растёт, rejection происходит раньше, тратится compute на вычисление вероятностей для неиспользованных позиций.

Оптимальное $K$ можно оценить через acceptance rate:

```python
def optimal_k(acceptance_rate, assistant_cost_ratio):
    """
    acceptance_rate: float, 0..1
    assistant_cost_ratio: float, время шага assistant / время шага target
    """
    best_k, best_speedup = 1, 0
    for k in range(1, 20):
        speedup = (k + 1) / (1 + k * (1 - acceptance_rate) + assistant_cost_ratio * k)
        if speedup > best_speedup:
            best_k, best_speedup = k, speedup
    return best_k, best_speedup
```

На практике $K = 5$ — хороший старт для большинства пар assistant/target.

### 2.3 Stella: drafter вместо assistant

Классический speculative decoding требует двух отдельных моделей. Это неудобно:

- Два набора весов на GPU (или переключение).
- Разные tokenizer'ы (нужна синхронизация).
- Дополнительная память.

**Stella (Self-speculative decoding)** — вариант, где одна и та же модель используется и как assistant, и как target, но на разных слоях:

- Assistant = первые $N$ слоёв модели (ранний выход, early exit).
- Target = полная модель.

Ранний выход даёт меньшее качество (как маленькая модель), но использует те же веса и тот же tokenizer.

### 2.4 Lookahead decoding (без assistant)

**Lookahead decoding** — speculative decoding без отдельного assistant. Идея:

1. На шаге $t$ модель генерирует не 1, а $K$ токенов через **Jacobian-итерацию** (n-граммный backtracking).
2. Первый токен — обычный decode. Второй — модель получает на вход первый и предсказывает второй. Третий — второй и предсказывает третий. И так далее.
3. Все $K$ токенов проверяются через обычный forward pass модели (как в speculative decoding).

**Проблема:** без assistant модель хуже предсказывает свои же будущие токены, чем специализированная маленькая модель. Acceptance rate обычно 0.3–0.5, что даёт скромное ускорение.

**Когда полезен:** когда нет подходящей assistant-модели. Например, для кастомной архитектуры, для которой не существует маленькой версии.

---

## 3. Компоненты и механизм работы

### 3.1 Архитектура speculative decoding в production

```
1. Запрос приходит в систему
2. KV-кэш assistant обновляется (prefill для обоих моделей)
3. Цикл:
   a. Assistant decode: генерирует K токенов (по одному)
   b. Target prefill: обрабатывает все K токенов за один forward pass
   c. Rejection sampling: принимает/отвергает токены
   d. KV-кэш target обновляется (только принятые токены)
   e. KV-кэш assistant синхронизируется с target (для следующей итерации)
4. Вернуть принятые токены
```

**Управление KV-кэшем — ключевая сложность:**

- Assistant и target имеют KV-кэши разного размера (разное число слоёв/голов).
- После rejection часть KV-кэша target не используется (отвергнутые токены не добавляются).
- KV-кэш assistant нужно синхронизировать: он не должен содержать токены, которых нет в target.

### 3.2 Parallel decoding

Speculative decoding можно расширить на **параллельную генерацию нескольких черновиков**:

```
Assistant генерирует 3 варианта продолжения:
  Draft 1: [a, b, c, d, e]
  Draft 2: [a, b, x, y, z]
  Draft 3: [a, f, g, h, i]

Target проверяет все 3 варианта:
  Draft 1: [✓, ✓, ✓, ✗, -]
  Draft 2: [✓, ✓, ✗, -, -]
  Draft 3: [✓, ✗, -, -, -]

Итог: [a, b, c] — 3 принятых токена за один prefill
```

**Выигрыш** — больше шансов найти длинную принимаемую последовательность. **Цена** — в K раз больше compute на prefill target.

Parallel decoding полезен, когда assistant нестабилен (низкий acceptance rate с одним черновиком).

### 3.3 Medusa: multiple heads

**Medusa** — модификация, где модель дообучается предсказывать $K$ следующих токенов одновременно (через $K$ дополнительных голов на последнем слое).

```python
# Обычная модель:
# head(x) → logits для следующего токена

# Medusa:
# head_1(x) → logits для токена t+1
# head_2(x) → logits для токена t+2
# ...         ...
# head_K(x) → logits для токена t+K
```

Все $K$ голов обучаются на одном датасете (next-token prediction, но со сдвигом). На инференсе:

1. Модель генерирует $K$ вероятностных распределений для следующих $K$ токенов.
2. Из них сэмплируются $M$ наиболее вероятных последовательностей (beam search / tree attention).
3. Target (та же модель, но без Medusa-голов) проверяет кандидатов через один prefill.

Medusa не требует отдельного assistant, но требует дообучения.

---

## 4. Практические аспекты

### 4.1 Выбор assistant

| Критерий | Хороший assistant | Плохой assistant |
|----------|-----------------|-----------------|
| Acceptance rate | > 0.7 | < 0.4 |
| Размер | 0.5–7B | > 20B |
| Tokenizer | Тот же, что у target | Разный |
| Архитектура | Та же семья (LLaMA → LLaMA) | Другая семья (GPT → LLaMA) |
| Дообучение | Distilled на target | Без дообучения |

**Без дообучения** acceptance rate для assistant из той же семьи — 0.6–0.8. **С distillation** (дообучение assistant предсказывать распределение target) — 0.8–0.95.

### 4.2 Overhead

Speculative decoding добавляет:

1. **Память:** assistant-модель (или её веса) на GPU. Для LLaMA-3-8B — ещё ~16 GB.
2. **Задержка assistant:** каждый decode-шаг assistant быстрее target, но не мгновенный. При K=5 assistant тратит 5 × 5 мс = 25 мс.
3. **Синхронизация KV-кэша:** после rejection нужно выровнять кэши assistant и target. Это копирование памяти (до единиц MB).
4. **Rejection sampling:** вычисление $\min(1, p/q)$ для каждого токена. Overhead пренебрежимо мал (< 1 мкс на токен).

**Когда overhead не окупается:**
- Если модель маленькая (assistant примерно равна target по размеру) — выигрыша нет, только overhead.
- Если acceptance rate < 0.4 — speculative decoding может быть медленнее обычного decode.
- Если latency не критична, а throughput — да (offline batch speculative decoding не даёт прироста).

### 4.3 Совместимость с continuous batching

Speculative decoding можно комбинировать с continuous batching, но это усложняет scheduler:

- В батче могут быть запросы с speculative decoding (требуют assistant) и без.
- Assistant-модель должна успевать генерировать черновики для всех speculative-запросов в батче.
- После rejection часть запросов «откатывается» на несколько токенов назад — scheduler должен уметь пересчитывать KV-кэш.

**Решение в production:** выделять отдельные GPU (или partition) для speculative decoding. Или использовать speculative decoding только для запросов с низким SLA по latency.

---

## 5. Типичные ошибки

- **Игнорировать overhead assistant.** Каждый decode-шаг assistant тоже занимает время. Если assistant всего в 2 раза быстрее target (например, LLaMA-3-8B vs LLaMA-3-70B при маленьком batch), ускорение speculative decoding может быть < 2×.
- **Не синхронизировать KV-кэш.** Если assistant продолжает генерировать с устаревшим KV-кэшем (после rejection), его распределения расходятся с target, acceptance rate падает.
- **Использовать assistant из другой tokenizer-семьи.** Если assistant и target токенизируют по-разному, speculative decoding не работает — токены несопоставимы.
- **Слишком большое K.** При K > 10 rejection почти гарантирован на ранних позициях, overhead assistant не окупается.
- **Думать, что speculative decoding даёт ускорение в batch-режиме.** Speculative decoding ускоряет latency одного запроса. Throughput в batch-режиме может не измениться или даже упасть (из-за overhead assistant).

---

## 6. Вопросы для самопроверки

1. Почему speculative decoding не меняет распределение генерации? Какое свойство rejection sampling это гарантирует?
2. От чего зависит acceptance rate и как его оценить для пары моделей?
3. Какой K оптимален, если assistant в 10 раз быстрее target, а acceptance rate = 0.8?
4. В чём разница между классическим speculative decoding, Stella и Medusa?
5. Почему speculative decoding не всегда даёт выигрыш в throughput?
6. Какая проблема возникает при комбинировании speculative decoding с continuous batching?