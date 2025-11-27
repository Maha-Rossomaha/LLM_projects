# Two-tower, LightGCN и user×item симметрия

## 0. User×item симметрия: общая картинка

Базовая постановка:

- множество пользователей $U$,
- множество объектов (items) $I$,
- матрица взаимодействий $R \in \mathbb R^{|U| \times |I|}$, где $r_{u,i}$ — сигнал интереса (клик, просмотр, покупка).

В современных рексистемах удобно думать не про матрицу, а про **эмбеддинги**:

- у каждого пользователя есть вектор $u_u \in \mathbb R^d$,
- у каждого объекта есть вектор $v_i \in \mathbb R^d$.

Скор (relevance) обычно задаётся скалярным произведением или косинусной похожестью:

$$
score(u, i) = \langle u_u, v_i \rangle \quad \text{или} \quad score(u, i) = \cos(u_u, v_i).
$$

Это сразу отражает **симметрию**:

- пользователи и объекты живут в одном пространстве,
- взаимодействие — просто геометрическая близость.

**Matrix Factorization (MF)** — минималистичный вариант: $u_u$ и $v_i$ — строки матриц $U$ и $V$, обучаемых напрямую. **Two-tower** и **LightGCN** — более гибкие и мощные обобщения этой идеи.

---

## 1. Two-tower (dual encoder) как обобщение MF

### 1.1. Архитектура

Two-tower модель состоит из двух независимых энкодеров:

- **User-tower**: строит embedding пользователя из его признаков,
- **Item-tower**: строит embedding объекта из его признаков.

Формально:

- признаки пользователя: $x_u = features_u(u)$,
- признаки объекта: $x_i = features_i(i)$.

Энкодеры:

$$
 u = f_u(x_u), \quad v = f_i(x_i),
$$

где $u, v \in \mathbb R^d$ — эмбеддинги.

Скор:

$$
score(u, i) = \langle u, v \rangle \quad \text{или} \quad score(u, i) = \cos(u, v).
$$

Типичные варианты $f_u, f_i$:

- простые embedding lookup по id (как в MF),
- embedding id + MLP поверх дополнительных фич,
- чисто MLP по признакам без id,
- комбинация (id + контент/мета-инфа).

### 1.2. Связь с MF

Если взять **очень простой** вариант:

- user-tower: $f_u(x_u) = E_u[u\_id]$ — просто lookup по id пользователя;
- item-tower: $f_i(x_i) = E_i[i\_id]$ — lookup по id объекта.

Тогда модель:

- имеет параметры $E_u \in \mathbb R^{|U| \times d}$ и $E_i \in \mathbb R^{|I| \times d}$,
- даёт $score(u,i) = \langle E_u[u], E_i[i] \rangle$.

Это ровно **matrix factorization** с $U = E_u$, $V = E_i$.

Дальше можно **обобщать MF**, добавляя фичи:

- $f_i(x_i)$ учитывает категорию, цену, текст описания, бренд,
- $f_u(x_u)$ учитывает регион, возраст, пол, persona embedding и т.д.

Тем самым two-tower даёт:

- MF на id-шниках как приватный случай,
- гибридную модель (id + контент) как общий случай.

### 1.3. Роль в персонализации

Two-tower модель — стандарт для **retrieval** (candidate generation):

1. В оффлайне считаем $v_i = f_i(x_i)$ для всех объектов и кладём в ANN-индекс.
2. В онлайне по запросу/сессии/пользователю считаем $u = f_u(x_u)$.
3. Ищем $K$ ближайших $v_i$ по $\langle u, v_i \rangle$ или косинусу.

Таким образом, персонализация реализуется через:

- user-tower, который учитывает историю, демографию, контекст,
- item-tower, который учитывает контент и мета-инфу.

Two-tower прекрасно сочетает:

- масштабируемость (ANN),
- персонализацию (user-эмбеддинг зависит от пользователя и контекста),
- гибкость по признакам.

### 1.4. Роль в cold-start (users и items)

**Cold-start item.**

Если $f_i(x_i)$ принимает **мета-фичи**, не только id:

- новый объект $i_{\text{new}}$ имеет признаки $x_{i_{\text{new}}}$ (категория, текст, цена),
- мы можем сразу посчитать $v_{i_{\text{new}}} = f_i(x_{i_{\text{new}}})$.

Даже без истории модель может:

- разместить новый объект в embedding-пространстве рядом с похожими,
- участвовать в retrieval через ANN.

**Cold-start user.**

Если $f_u(x_u)$ использует контекст/персону:

- демографию, канал, persona embedding, контекст сессии,
- возможно, cross-domain user embedding.

Тогда для нового пользователя $u_{\text{new}}$:

- известен $x_{u_{\text{new}}}$ (контекст, анкета, cross-domain профиль),
- можно посчитать $u_{\text{new}} = f_u(x_{u_{\text{new}}})$.

И сразу делать персонализированный retrieval, даже без истории в текущем домене.

Итого: two-tower — одна из главных точек, где **масштабная персонализация и cold-start** сходятся.

### 1.5. Обучение two-tower

Чаще всего обучаем на implicit feedback (клики):

- положительные примеры $(u, i^+)$ — реальные взаимодействия,
- отрицательные $(u, i^-)$ — sampling объектов, с которыми пользователь не взаимодействовал.

Типичные loss-ы:

1. **BCE over logits**:

   $$
   \mathcal L = -\sum_{(u,i,y)} \bigl[y \log \sigma(\langle u, v \rangle) + (1-y) \log(1-\sigma(\langle u, v \rangle))\bigr].
   $$

2. **Pairwise BPR-подобный loss**:

   для тройки $(u, i^+, i^-)$:

   $$
   \mathcal L_{BPR} = -\log \sigma(\langle u, v_{i^+} \rangle - \langle u, v_{i^-} \rangle).
   $$

3. **Sampled softmax / InfoNCE**: softmax по батчу положительных и отрицательных item-ов.

### 1.6. Простой пример two-tower на PyTorch

```python
import torch
import torch.nn as nn

class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

    def encode_user(self, user_ids: torch.LongTensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        # можно сделать L2-нормировку, если хотим косинус
        return u

    def encode_item(self, item_ids: torch.LongTensor) -> torch.Tensor:
        v = self.item_emb(item_ids)
        return v

    def forward(self, user_ids: torch.LongTensor, item_ids: torch.LongTensor) -> torch.Tensor:
        u = self.encode_user(user_ids)
        v = self.encode_item(item_ids)
        logits = (u * v).sum(dim=-1)  # dot-product
        return logits

# пример шага обучения с BCE
model = TwoTower(num_users=10_000, num_items=50_000, dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

batch_size = 1024
user_ids = torch.randint(0, 10_000, (batch_size,))
item_ids = torch.randint(0, 50_000, (batch_size,))
labels = torch.randint(0, 2, (batch_size,)).float()

logits = model(user_ids, item_ids)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

В реальной системе сюда добавляются:

- дополнительные признаки пользователя/объекта,
- более сложные negative sampling стратегии,
- отдельный экспорт item-эмбеддингов в ANN-индекс.

---

## 2. LightGCN и GCN-подобные модели на user×item графе

### 2.1. User×item граф

Рассматриваем двудольный граф $G = (V, E)$:

- $V = U \cup I$ — вершины: пользователи и объекты,
- $E$ — рёбра $(u, i)$, если у пользователя $u$ было взаимодействие с объектом $i$.

Связность в этом графе отражает **структуру вкусов**:

- пользователи, которые любят одни и те же объекты, оказываются близко через короткие пути,
- объекты, которые часто смотрят вместе, тоже оказываются близко.

### 2.2. Базовая идея LightGCN (на интуитивном уровне)

LightGCN можно рассматривать как **графовое обобщение MF**:

1. У каждого пользователя $u$ и объекта $i$ есть начальные эмбеддинги:

   $$
   e_u^{(0)}, \quad e_i^{(0)}.
   $$

   Это могут быть:

   - просто learnable embeddings по id (как в MF),
   - или уже контентные/гибридные эмбеддинги.

2. На каждом слое $k$ делаем **message passing** по графу:

   $$
   e_u^{(k+1)} = \sum_{i \in N(u)} w_{u,i} e_i^{(k)},
   $$

   $$
   e_i^{(k+1)} = \sum_{u \in N(i)} w_{u,i} e_u^{(k)},
   $$

   где $N(u)$ — соседи пользователя (его объекты), $N(i)$ — соседи объекта (его пользователи), $w_{u,i}$ — веса (обычно нормировки по степеням).

3. После $K$ слоёв итоговый эмбеддинг — сумма/среднее по слоям:

   $$
   e_u = \sum_{k=0}^K \alpha_k e_u^{(k)}, \quad e_i = \sum_{k=0}^K \alpha_k e_i^{(k)}.
   $$

4. Скор, как обычно:

   $$
   score(u, i) = \langle e_u, e_i \rangle.
   $$

Отличие от классического GCN:

- нет сложных нелинейностей и MLP на каждом слое,
- чистая линейная пропагация + суммирование слоёв,
- меньше переобучения, проще и быстрее.

### 2.3. Роль в персонализации

В MF пользователь/объект получают embedding только из своих взаимодействий, без явной структуры графа. LightGCN делает шаг дальше:

- эмбеддинг пользователя $e_u$ учитывает **не только его объекты**, но и объекты соседей, соседей соседей и т.д. (через несколько слоёв);
- эмбеддинг объекта $e_i$ учитывает **целые сообщества пользователей**, которые его смотрят, и соседние объекты.

Это позволяет лучше схватывать:

- **сообщества вкусов** (кластеры пользователей и объектов),
- связи в разреженных зонах (через несколько шагов по графу можно связать объекты, которые напрямую не пересекались по пользователям).

В итоге LightGCN обычно даёт **лучше качество персонализации** по сравнению с чистым MF, особенно в сценариях с сильно разреженными матрицами.

### 2.4. Роль в cold-start

Чистый графовый CF сам по себе страдает от полного cold-start (нет рёбер — нет сигналов). Но в момент, когда у cold-user или cold-item появляются **первые взаимодействия**, LightGCN начинает работать на нас:

- новый пользователь $u_{\text{new}}$ взаимодействует с объектами $i_1, i_2$;
- на следующих шагах пропагации $e_{u_{\text{new}}}$:
  - наследует информацию от эмбеддингов $e_{i_1}, e_{i_2}$,
  - через них — от их соседей и т.д.

Аналогично для нового объекта $i_{\text{new}}$:

- первые пользователи, которые его посмотрели, «подтягивают» его в своё сообщество;
- embedding объекта начинает отражать вкусовую структуру графа, а не только контент/мета-инфу.

Если начальные эмбеддинги $e_u^{(0)}, e_i^{(0)}$ содержат **контент** (см. cold-start items/users), то LightGCN фактически делает **гибридный CF**: контент + графовая коллаборативная информация.

### 2.5. Скелет LightGCN на PyTorch (упрощённо)

```python
import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int, edge_index: torch.LongTensor, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.num_layers = num_layers
        # инициализируем эмбеддинги по id
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        # edge_index: 2 x E (user_ids, item_ids)
        self.edge_index = edge_index

    def propagate(self, user_emb, item_emb):
        # один шаг пропагации (упрощённо, без нормировок)
        u_ids, i_ids = self.edge_index
        # собираем сообщения от items к users
        msg_item = torch.zeros_like(user_emb)
        msg_user = torch.zeros_like(item_emb)

        msg_item.index_add_(0, u_ids, item_emb[i_ids])
        msg_user.index_add_(0, i_ids, user_emb[u_ids])

        # усреднение по степеням можно добавить отдельно
        return msg_item, msg_user

    def forward(self):
        u = self.user_emb.weight
        v = self.item_emb.weight

        all_u = [u]
        all_v = [v]

        for _ in range(self.num_layers):
            msg_u, msg_v = self.propagate(u, v)
            u = msg_u
            v = msg_v
            all_u.append(u)
            all_v.append(v)

        # усредняем эмбеддинги по слоям
        u_final = torch.stack(all_u, dim=0).mean(dim=0)
        v_final = torch.stack(all_v, dim=0).mean(dim=0)
        return u_final, v_final

    def predict(self, user_ids: torch.LongTensor, item_ids: torch.LongTensor):
        u_final, v_final = self()
        u = u_final[user_ids]
        v = v_final[item_ids]
        return (u * v).sum(dim=-1)
```

Этот код иллюстративный, но отражает идею:

- есть начальные embedding'и по id,
- несколько раз прогоняем сообщения по user×item рёбрам,
- итоговый embedding — агрегация по слоям,
- скор — dot-product.

На практике добавляется нормировка, BPR-loss, batching и т.п.

---

## 3. User×item симметрия и роль моделей в персонализации/cold-start

### 3.1. Симметрия в MF, two-tower и LightGCN

Во всех трёх подходах идея одна и та же:

- есть пространство $\mathbb R^d$, где живут **и пользователи, и объекты**;
- взаимодействие $u \leftrightarrow i$ означает геометрическую близость $u_u$ и $v_i$;
- персонализация — это выбор объектов, близких к $u_u$;
- **симметрия** означает, что можно так же рассуждать и по объектам (кому показать этот объект?).

Различия:

- MF: эмбеддинги по id, обучаются напрямую;
- two-tower: эмбеддинги — выводы энкодеров из фичей (id + контент + контекст);
- LightGCN: эмбеддинги — результат графовой пропагации (с учётом структуры user×item графа).

### 3.2. Роли в персонализации

**Two-tower**:

- отвечает за **быстрое персонализированное retrieval**;
- умеет использовать богатые признаки пользователей и объектов;
- масштабируется через ANN.

**LightGCN**:

- уточняет и улучшает embedding'и, учитывая **сообщества вкусов**;
- часто даёт прирост качества в разреженных зонах;
- может использоваться как отдельная модель CF или как поставщик embedding'ов в two-tower/ранжировщик.

Обычно пайплайн выглядит так:

1. Two-tower генерирует кандидатов (до сотен/тысяч объектов).
2. Reranker (GBDT/NN) пересортировывает кандидатов по сложному скору.
3. LightGCN или другие графовые модели могут либо:
   - участвовать в candidate generation (ещё один retrieval-источник),
   - или давать embedding'и как признаки для reranker'а.

### 3.3. Роли в cold-start

**Cold-start items:**

- two-tower решает проблему на уровне item-tower: $v_i = f_i(features(i))$ даже без кликов;
- LightGCN вступает в игру, когда появляются первые рёбра: начинает **втягивать** cold-item в граф структурно.

**Cold-start users:**

- two-tower решает через user-tower: $u = f_u(context(u), persona(u), cross\text{-}domain\_emb)$;
- как только есть первые взаимодействия, LightGCN через графовое соседство обогащает embedding пользователя информацией о его сообществе.

Таким образом:

- two-tower — основной инструмент **до и во время раннего cold-start**, где главную роль играют признаки;
- LightGCN — усилитель, который постепенно превращает холодные id-шники в хорошо встроенные узлы графа, как только появляются рёбра.

---

## 4. Как two-tower и LightGCN живут вместе

В реальных системах их редко противопоставляют, чаще **комбинируют**:

1. **LightGCN как источник embedding'ов для two-tower**:
   - начальные id-эмбеддинги пользователей/объектов обучаются LightGCN;
   - two-tower использует их как одну из компонент векторов $u, v$ (concatenate или sum);
   - таким образом, two-tower получает уже "графово обогащённые" эмбеддинги.

2. **Two-tower + LightGCN ensemble**:
   - two-tower и LightGCN выступают как две отдельные модели candidate generation;
   - объединённый набор кандидатов передаётся в reranker;
   - diversity/coverage увеличиваются за счёт разных источников сигналов.

3. **LightGCN внутри item-/user-tower**:
   - можно рассматривать LightGCN как часть item-/user-encoder'а, который добавляет графовую информацию к контенту и id;
   - итоговый $v_i$ и $u_u$ уже включают информацию и о контенте, и о графе.

---

## 5. Резюме

1. **Two-tower (dual encoder)** — обобщение MF, где user- и item-эмбеддинги получаются из фичей, а не только из id. Он:
   - лежит в основе масштабируемого персонализированного retrieval через ANN,
   - естественно поддерживает cold-start items (через meta-features) и cold-start users (через контекст/персону).

2. **LightGCN / GCN-подобные модели на user×item графе** — графовое развитие CF/MF, которое:
   - учитывает структуру user×item графа через message passing,
   - лучше схватывает сообщества вкусов и помогает в разреженных зонах,
   - втягивает cold users/items в общую структуру, как только появляются первые рёбра.

3. **User×item симметрия** — общая идея: пользователи и объекты живут в одном embedding-пространстве, а релевантность — это геометрическая близость. MF, two-tower и LightGCN — разные уровни сложности реализации этой идеи.

4. В продовых системах two-tower и LightGCN обычно **работают вместе**:
   - two-tower решает задачу быстрой персонализации и cold-start через признаки,
   - LightGCN улучшает эмбеддинги с учётом графа взаимодействий.

Дальнейшие конспекты можно посвятить более глубокому разбору:

- обучения two-tower (sampled softmax, hard negatives, multi-task),
- архитектуры LightGCN и его вариаций,
- интеграции графовых и контентных моделей в единый стек рекомендаций.

