# Negatives Mining

## 1. In-Batch Negatives
### 1.1 Общая идея
В задачах обучения retrieval-моделей с использованием contrastive loss (например, InfoNCE), модель должна обучиться сопоставлять запрос (query) с релевантным документом (positive) и отдалять его от нерелевантных документов (negatives) в embedding-пространстве.
Для этого необходимы хорошие негативные примеры — документы, которые не соответствуют запросу, но при этом максимально похожи по структуре, лексике или теме, чтобы обучение было «сложным» (hard negatives).

### 1.2 Что такое In-Batch Negatives
In-batch negatives — это техника, при которой все остальные документы в текущем батче используются в качестве негативных примеров для каждого запроса.

#### Пример:

Пусть в батче 4 пары:

```
(q₁, d₁⁺), (q₂, d₂⁺), (q₃, d₃⁺), (q₄, d₄⁺)
```

Для `q₁`:

* **Positive:** `d₁⁺`
* **Negatives:** `d₂⁺`, `d₃⁺`, `d₄⁺`

И так далее для каждого запроса.

### 1.3 Формулировка в InfoNCE:

$$
L = -\log\left( \frac{\exp(\text{sim}(q, d^+))}{\sum_{j=1}^N \exp(\text{sim}(q, d_j))} \right)
$$

Где `d_j` — все документы из батча (включая позитивный)


### 1.4 Преимущества In-Batch Sampling

* **Бесплатные негативы**: не нужно извлекать или хранить дополнительные примеры
* **Эффективность**: позволяет обрабатывать десятки негативов за один проход
* **Обобщаемость**: в батчах часто оказываются релевантные, но нерелевантные по аннотации примеры, что повышает устойчивость модели
* **Совместимость с InfoNCE**: используется в CLIP, SimCLR, GTR, E5

### 1.5 Недостатки и ограничения

* Негативы могут быть **легкими** (easy negatives), т.е. очевидно нерелевантными
* В одном батче редко оказываются действительно **трудные негативы**
* При небольших размерах батча — ограниченное разнообразие негативов

---

## 2. Hard and Soft Negatives
### 2.1 Определения
#### 2.1.1 **Hard Negatives**
**Hard negatives** — это нерелевантные документы, которые **семантически близки к запросу**, но не являются правильным ответом.

#### Пример:

Запрос: "apple customer service phone"

* **Positive:** "Apple support hotline"
* **Hard negative:** "Apple store location", "Apple iPhone specifications"

Такие примеры сбивают модель с толку, потому что они лексически и тематически похожи, но не дают правильного ответа.

#### 2.1.2 **Soft Negatives**

**Soft negatives** — это нерелевантные примеры, которые ближе к запросу, чем случайные негативы, но менее опасны, чем hard.

#### Пример:

Запрос: "беспроводные наушники для спорта"

* **Soft negative:** "беспроводные наушники для офиса"
* **Hard negative:** "TWS наушники для бега (но не спортивные)"


### 2.2 Почему это важно?

Обучение с **только easy negatives** (например, случайных документов) быстро приводит к **снижению градиента** — модель уже умеет их различать. Hard и soft negatives обеспечивают более информативный loss и улучшают обобщающую способность.

### 2.3 Источники негативов

| Тип                | Как получают                         |
| ------------------ | ------------------------------------ |
| Easy negatives     | Случайная выборка нерелевантных пар  |
| In-batch negatives | Пары из других запросов внутри батча |
| Hard negatives     | Top-K из dense или BM25 индекса      |
| Soft negatives     | Отфильтрованные top-K (по семантике) |

---

### 2.4 Как использовать в обучении

#### 2.4.1 Explicit Hard Negatives

* Добавлять вручную или через BM25/dense retrieval
* Пример: `(q, d⁺, d⁻_hard)` в Triplet Loss

#### 2.4.2 Semi-Hard Mining

* Оставлять только те негативы, которые ближе к запросу, чем другие, но не ближе, чем позитив

#### 2.4.3 Curriculum Learning

* Постепенное добавление всё более трудных негативов по ходу обучения

#### 2.4.4 Hybrid Strategies

* Сочетание in-batch и hard negatives в одном батче
* Используется, например, в DPR и RocketQA

### 2.5 Риски и ограничения

* **False negatives:** выбранный hard negative может на самом деле быть релевантным, особенно при неточной аннотации
* **Mode collapse:** при чрезмерно агрессивных hard negatives модель может перестать различать классы
* **Compute cost:** требует внешнего индекса или модели для отбора негативов

### 2.6 Рекомендации

* Используйте hard negatives, если:

  * У вас есть внешний retriever (BM25 или dense)
  * Размер батча недостаточен для разнообразия in-batch
* Добавляйте soft negatives для баланса и стабильности
* В retrieval-задачах применяйте hard negatives с margin-based losses (Triplet, InfoNCE)

---

## 3. ACNE: Asymmetric Contrastive Negative Example Mining

**ACNE (Asymmetric Contrastive Negative Example mining)** — это стратегия негативного сэмплирования, при которой используется **асимметрия** между сторонами запроса и документа в contrastive learning.

Вместо использования пар (`q`, `d`) симметрично, как это делается в классическом InfoNCE, ACNE применяет одностороннее сэмплирование негативов.  
Здесь якорем выступает только эмбеддинг запроса (`q`), а негативы выбираются среди документов (`d⁻`), причём подбираются именно «трудные» — те, что находятся близко к `q` в embedding space

### 3.1 Мотивация

* В классическом InfoNCE все примеры обрабатываются симметрично: запрос и документ взаимозаменяемы.
* Однако в retrieval-сценариях, особенно при использовании **разных энкодеров для** `q`**и** `d` (асимметричный bi-encoder), такая симметрия может мешать: запрос и документ играют разные роли.

* Подход ACNE даёт три преимущества:

  * Убирает искусственную симметрию между `q` и `d`
  * Фокусируется на сложных негативных документах для конкретного запроса
  * Повысить устойчивость модели и улучшить градиенты

### 3.2 Как работает ACNE

#### 3.2.1 Алгоритм:

1. Выбирается якорь (обычно `query`)
2. Из внешнего индекса (BM25, dense index) подбираются top-K ближайших `d⁻`
3. Из них отфильтровываются `d⁺`, полученные по аннотации
4. Остальные — hard negatives

#### 3.2.2 Пример:

Для запроса `q`:

* **Positive:** `d⁺`
* **Negatives:** выбираются среди всех документов, **которые близки к** `q`, **но не являются** `d⁺`.

Для самого документа `d⁺` такие негативы не рассматриваются — они подбираются только относительно `q`, то есть **асимметрично**.

### 3.3 Где применяется

* Retrieval модели с раздельными encoder'ами (query/document)
* Сценарии, где запрос и документ имеют разные стили (например, short vs long text)
* Модели, в которых требуется контролируемое negatives mining, без риска "зеркальных" ошибок

### 3.4 Преимущества

* **Асимметрия** соответствует реальному workflow retrieval-систем
* **Более качественные негативы** — выбираются с учётом сложности запроса
* **Уменьшение ложных пар** — документ не используется как якорь, что снижает риск false positives

### 3.5 Ограничения

* Требуется внешний индекс (BM25 или dense) для предварительного поиска
* Не подходит для симметричных задач (например, paraphrase mining)
* Может пропустить трудные `query`-ориентированные негативы, если использовать только `d` в качестве якоря

---

## 4. MoCo: Momentum Contrast

**Идея.** MoCo поддерживает *динамический словарь* ключей (keys) и обучает энкодер запросов (queries) так, чтобы сближать представления позитивной пары и отталкивать от них представления негативных примеров. Словарь реализован как очередь (Memory Bank), а второй энкодер обновляется не градиентом, а *экспоненциальным сглаживанием* параметров (EMA, momentum update). Это позволяет иметь десятки тысяч негативов без гигантского batch size и сохранять согласованность между $q$ и $k$.

### 4.1 Компоненты

* **Два энкодера:** $f_q$ (query) и $f_k$ (key). Архитектуры совпадают, но лишь $f_q$ получает градиенты.
* **Projection head** $g(\cdot)$: обычно 2‑слойная MLP; итоговые векторы нормируются до единичной нормы: $q = \operatorname{norm}(g(f_q(x_q)))$, $k = \operatorname{norm}(g(f_k(x_k)))$.
* **Momentum‑обновление** параметров $\theta_k$ из $\theta_q$:

$$
\theta_k \leftarrow m\,\theta_k + (1 - m)\,\theta_q, \quad m \in [0,1)\;.
$$

* **Memory Bank (очередь)** фиксированного размера $K$: хранит последние $K$ ключей ${k_j^-}$.

### 4.2 Почему нужен momentum‑encoder?

Если ключи и запросы считаются одной и той же быстро меняющейся моделью, «негативы» теряют согласованность от шага к шагу → градиенты шумные. Медленно обновляемый $f_k$ даёт **стабильные ключи** на протяжении многих итераций, поэтому:

* *больше информации в каждом шаге* (тысячи негативов из очереди);
* *лучше условие оптимизации* (менее «дрожащая» цель).

### 4.3 Memory Bank (очередь)

Очередь $\mathcal{Q}$ хранит $K$ последних нормированных ключей. На каждом шаге **enqueue** новых $k_i$ и **dequeue** самых старых, сохраняя размер $K$. Благодаря этому каждый $q_i$ сравнивается не только с текущими $k_i^+$, но и с большим пулом $k_j^-$ из прошлых шагов.

### 4.4 Обучающий шаг

1. Сгенерировать две аугментации одного примера: $x_q, x_k$.

2. Посчитать представления:

* $q_i = \operatorname{norm}(g(f_q(x_{q,i})))$;
* $k_i = \operatorname{norm}(g(f_k(x_{k,i})))$ (без градиента).

3. Логиты сходства (скалярное произведение при $\lVert q\rVert=\lVert k\rVert=1$ эквивалентно косинусу) и температура $\tau$:

* $\ell_i^+ = (q_i\cdot k_i)/\tau$,
* $\ell_{ij}^- = (q_i\cdot k_j^-)/\tau$, где $k_j^- \in \mathcal{Q}$.

4. **InfoNCE‑лосс** для батча размера $B$:

$$
L = -\frac{1}{B}\sum_{i=1}^B \log
\frac{\exp(\ell_i^+)}{\exp(\ell_i^+) + \sum_{j=1}^{K} \exp(\ell_{ij}^-)}\;.
$$

5. Backprop только через $f_q$ и его head; $f_k$ — без градиента.

6. **EMA‑обновление** $\theta_k$ из $\theta_q$.

7. **Обновить очередь:** добавить ${k_i}_{i=1}^B$ и удалить $B$ самых старых ключей.

### 4.5 Гиперпараметры и «рабочие» значения

* Размер очереди $K$: 16k–65k. Слишком маленький $K$ → мало «разнообразных» негативов; слишком большой — рост памяти и устаревание ключей.
* Температура $\tau$: 0.07 — хороший старт. Меньше $\tau$ → жёстче штраф, но риск переобучения на «случайные» различия.
* Коэффициент импульса $m$: 0.999–0.9995. Слишком маленький $m$ разрушает «стабильность» ключей, слишком большой — делает $f_k$ инертным.
* Head: 2‑слойная MLP с нелинейностью; нормализация выходов обязательна.
* Аугментации: сильные цветовые/геометрические (для vision); для текста — дропаут токенов, перефраз, маскирование.

### 4.6 Реализация (минимальный каркас, PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    # last_hidden_state: (B, L, H); attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-6)                          # (B, 1)
    return summed / denom

class TextEncoder(nn.Module):
    """
    Обёртка над LLM/трансформером, возвращающая sentence embedding через pooling + MLP head.
    Ожидается backbone с сигнатурой forward(input_ids, attention_mask) -> last_hidden_state.
    Примеры: HF AutoModel (без LM Head), любой encoder-only/decoder-only с .config.hidden_size.
    """
    def __init__(self, backbone: nn.Module, out_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        hidden_size = (
            getattr(backbone, 'hidden_size', None)
            or getattr(getattr(backbone, 'config', None), 'hidden_size', None)
        )
        if hidden_size is None:
            raise ValueError("Не удалось определить hidden_size у backbone")
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_dim),
        )

    @torch.no_grad()
    def encode_tokens(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Вычисление эмбеддинга без proj (например, для отладки)
        last_hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        sent = mean_pooling(last_hidden, attention_mask)
        return sent

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        sent = mean_pooling(last_hidden, attention_mask)
        z = self.proj(sent)
        return F.normalize(z, dim=1)

class MoCo(nn.Module):
    def __init__(
        self, 
        backbone_q: nn.Module, 
        backbone_k: nn.Module, 
        dim: int = 128,
        K: int = 65536, 
        m: float = 0.999, 
        T: float = 0.07
    ):
        super().__init__()
        self.encoder_q = TextEncoder(backbone_q, out_dim=dim)
        self.encoder_k = TextEncoder(backbone_k, out_dim=dim)
        # Инициализация параметров ключевого энкодера
        for p_k, p_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False  # ключевой энкодер — без градиента
        self.m = m
        self.T = T
        self.K = K
        # Очередь ключей (dim, K) и указатель
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_k, p_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # keys: (B, dim)
        keys = keys.detach()
        B = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % B == 0, "K должно делиться на batch size для простоты"
        self.queue[:, ptr:ptr+B] = keys.T
        ptr = (ptr + B) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, batch_q: dict, batch_k: dict):
        """
        batch_q/batch_k: dict с тензорами 'input_ids' и 'attention_mask', размерности (B, L).
        Пример подготовки батча: токенизатор HF с padding/truncation, одинаковая L.
        """
        # 1) query-ветка (с градиентом)
        q = self.encoder_q(batch_q['input_ids'], batch_q['attention_mask'])  # (B, dim)

        # 2) EMA-обновление и ключи без градиента
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(batch_k['input_ids'], batch_k['attention_mask'])  # (B, dim)

        # 3) логиты: положительный и отрицательные
        l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(1)            # (B, 1)
        l_neg = torch.einsum('bd,dk->bk', q, self.queue.clone())       # (B, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T             # (B, 1+K)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # 4) обновить очередь новыми ключами
        self._dequeue_and_enqueue(k)
        return loss
```

> Заметки к коду: (i) нормализация выходов — критична; (ii) **нет** градиента через $k$ и очередь.

### 4.7 Практические нюансы

* **BatchNorm.** В ключевой ветке BN‑статистики не должны «подмешиваться» из текущего батча с градиентом. Безопаснее: SyncBN или «перемешивание» батча перед проходом ключевой ветки; либо заменить BN на GN/IN.
* **Multi‑GPU.** Очередь — *глобальная*: собирайте ключи со всех устройств, иначе каждый процесс будет видеть только локальные негативы.
* **Сходимость.** При слишком маленьком $K$ и/или большом $\tau$ модель может «смягчаться» и терять различающую способность; при слишком жёстких аугментациях и маленьком $\tau$ возможна нестабильность.
* **Домены.** Для текста/мультимодальности позитив — «парные» примеры (например, $(\text{text}, \text{image})$); аугментации должны сохранять семантику.

### 4.8 Эволюция вариантов

* **MoCo v1 → v2.** Усиленные аугментации и MLP‑голова заметно поднимают качество; базовые гиперпараметры: $m\approx0.999$, $\tau\approx0.07$, $K\in[16k, 65k]$.
* **MoCo с ViT (часто называют v3).** Те же принципы, но нюансы: чувствительность к $m$ и к стратегии обучения ViT (warm‑up, weight decay). Очередь может быть меньше из‑за более «сильной» архитектуры.
* **Супервизированный вариант.** Позитивами считаются все примеры одного класса (не только две аугментации одного образца), что особенно полезно при наличии меток.

### 4.10 Где MoCo особенно уместен

* Ограниченная память на устройстве (не потянем batch 8k–32k, как в SimCLR), но нужна высокая «энтропия» негативов.
* Потоковая/инкрементальная загрузка данных: очередь естественным образом «обновляет» негативы без перестройки батчей.

### 4.4 Пример применения

В retrieval или representation learning на больших коллекциях (веб-документы, изображения и т.д.), где один батч покрывает <0.01% пространства:

* batch size = 128
* memory bank = 65,536
* каждый `query` получает обучающий сигнал по 1 позитиву + 65,535 негативам