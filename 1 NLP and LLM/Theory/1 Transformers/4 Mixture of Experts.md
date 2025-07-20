# Конспект «Mixture of Experts (MoE)»  
URL:  
🔗 [Mixture of Experts Explained](https://huggingface.co/blog/moe)  
🔗 [A Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)  
🔗 [Understanding Mixture of Experts: Building a MoE Model with PyTorch](https://medium.com/@prateeksikdar/understanding-mixture-of-experts-building-a-moe-model-with-pytorch-dd373d9db81c)
> **Коротко:** MoE — это способ «разреженного» масштабирования трансформеров: каждый токен проходит через небольшой под‑набор *экспертов* (специализированных FFN‑веток), выбранных *роутером*. Благодаря этому активных параметров — всего 10–20 % от общего числа, что даёт почти линейную экономию FLOP без ухудшения качества.  

---

## 1. Суть идеи  
* Плотная LLM тратит вычисления на все параметры в каждом слое.  
* В MoE‑слое есть $E$ независимых **экспертов**; роутер оценкой $p_i$ выбирает топ‑$k$ (обычно 1–2) для каждого токена.   
* В итоге на токен активируется лишь $\frac{k}{E}$ части веса, но суммарная «память» модели растёт линейно с числом экспертов.  

---

## 2. Компоненты архитектуры  

### 2.1 Эксперты  
Каждый эксперт — это стандартный FFN со SwiGLU (или ReLU), но обучается на *подмножество* токенов, формируя «специализацию» (код, математика, диалоги).  

### 2.2 Роутер  
* Логистическая регрессия (softmax по $E$) оценивает вероятности выбора экспертов.  
* Применяется *Top‑k* обрезка (обычно $k=2$) + *capacity factor* чтобы ограничить очередь токенов на эксперта и избежать «переполнения».   

### 2.3 Вспомогательные лоссы  
* **Load‑balancing loss**: заставляет роутер равномерно распределять токены, минимизируя дисперсию использования.  
* **Z‑loss / Entropy loss**: сглаживает распределение, снижая «мёртвых» экспертов.   

---

## 3. Обучение  

| Этап | Что происходит | Сложность |  
|------|----------------|-----------|  
| Pre‑train | Стандартный next‑token loss + aux‑balance | Требует кластер NCCL‑friendly (много all‑to‑all)  |  
| Fine‑tune | Можно «замораживать» эксперты и обучать роутер, получая лёгкую адаптацию |  |  
| Distillation | Dense‑студент (например, 7B) поглощает знания MoE‑учителя | Снижает latency на inference  |  

---

## 4. Плюсы и минусы  

| Плюсы | Минусы |  
|-------|--------|  
| $\sim10\times$ больше параметров «в памяти» при том же FLOP | Сложный all‑to‑all → сетевой bottleneck |  
| Легко масштабировать, просто добавляя экспертов | «Игра» с capacity‑factor иначе токены дропаются  |  
| Специализация экспертов улучшает zero‑shot reasoning | Сложнее воспроизводить чекпойнты (ROUTER seed) |  
| Возможность гибридных задач (визуал/текст) через разные пулы экспертов | |  

---

## 5. Практические советы (из статей)  
1. **k=1 vs k=2**: один эксперт быстрее и проще, два — выше качество, но удваивает коммуникацию.   
2. **Capacity factor 1.25** — эмпирический баланс переполнений.   
3. Применение **FP8 / INT4** к активным весам даёт выигрыш в памяти без заметной деградации.   
4. На inference можно «закэшировать» роутер‑выбор для статичных промптов, снижая latency.   

---

## 6. Ключевые формулы  

**Роут‑логиты:** $z = W_r h$  

**Вероятности:** $p = \text{softmax}(z)$  

**Top‑k mask:**  
$$m_i = \begin{cases}1, & p_i \in \text{top‑k}\\0,&\text{иначе}\end{cases}$$  

**Load‑balancing loss:**  
$$L_{lb} = E \sum_j q_j \log q_j,\quad q_j = \frac{n_j}{\sum_k n_k}$$  

где $n_j$ — число токенов, направленных к эксперту $j$.   

---

## 7. Рецепт «Мини‑MoE» (по Medium)  

### 2.1 4 шага к рабочему блоку  
1. **Эксперты**: N линейных FFN‑веток с SwiGLU.   
2. **Гейт**: $p=\text{softmax}(W_r h)$, выбираем топ‑$k=2$, применяем capacity‑factor 1.25.  
3. **Сборка**: рассылаем токены по экспертам (scatter) → обрабатываем → собираем (gather).  
4. **Aux‑loss**: load‑balancing + энтропия, чтобы эксперты использовались равномерно.   

### 2.2 Пример кода   
```python
import torch, torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    """
    Классический FFN-блок Transformer-декодера (SwiGLU по умолчанию).
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1   = nn.Linear(d_model, d_ff * 2) # →2× для SwiGLU
        self.w2   = nn.Linear(d_ff, d_model)

    def forward(self, x):
        a, b = self.w1(x).chunk(2, dim=-1) # SwiGLU
        return self.w2(F.silu(a) * b)


class MoEBlock(nn.Module):
    """
    Mixture-of-Experts-слой (Top-k Gating) без распределённого all-to-all,
    пригодный для CPU / одной GPU. В DeepSpeed эта логика «зашита» в CUDA-ядра.
    """
    def __init__(
        self,
        d_model:   int  = 4096,
        d_ff:      int  = 14336,
        experts:   int  = 16,
        top_k:     int  = 2,
        capacity:  float = 1.25, # коэффициент «ёмкости» эксперта
    ):
        super().__init__()
        self.top_k      = top_k
        self.experts    = nn.ModuleList([FFN(d_model, d_ff) for _ in range(experts)])
        self.router     = nn.Linear(d_model, experts, bias=False)
        self.capacity   = capacity

        # метрики для auxiliary load-balance loss
        self.register_buffer("decay", torch.tensor(0.9))

    def forward(self, x):
        """
        x: [batch, seq, d_model]
        """
        B, S, D = x.shape
        E       = len(self.experts)
        tokens  = x.view(-1, D) # → [B·S, d_model]

        # 1) Gating: оценки важности экспертов
        logits   = self.router(tokens) # [B·S, E]
        scores   = F.softmax(logits, dim=-1)

        # 2) Top-k выбор и индексы экспертов
        topk_val, topk_idx = torch.topk(scores, self.top_k, dim=-1) # [B·S, k]

        # 3) Capacity – сколько токенов может принять эксперт
        cap = int(self.capacity * (B * S) / E)

        # 4) Формируем *dispatch*-маску (one-hot по топ-k) и считаем загрузку
        dispatch = torch.zeros(B * S, E, device=x.device)
        for i in range(self.top_k):
            idx = topk_idx[:, i]
            mask = (
                torch.arange(cap, device=x.device)[None, :] # [1, cap]
                < torch.bincount(idx, minlength=E)[idx][:, None]
            )
            dispatch.scatter_(1, idx.unsqueeze(-1), mask.float())

        # 5) Рассылаем токены к экспертам
        expert_inputs = [
            tokens[dispatch[:, e].bool()] for e in range(E) # variable length
        ]
        expert_outputs = [
            self.experts[e](inp) if len(inp) else torch.empty_like(inp)
            for e, inp in enumerate(expert_inputs)
        ]

        # 6) Обратная агрегация (gather) с весами топ-k-оценок
        output = torch.zeros_like(tokens)
        for e in range(E):
            out   = expert_outputs[e] # [n_e, D]
            sel   = dispatch[:, e].bool()
            output[sel] += out * scores[sel, e:e+1] # взвешиваем по softmax-оценке

        # 7) Доп-лосс балансировки нагрузки (Routers Auxiliary Loss)
        load = dispatch.mean(0) # доля токенов на эксперта
        prob = scores.mean(0) # среднее softmax-p
        aux_loss = (E * (load * prob).sum()) # Fedus et al., 2021

        # сохраняем EMA статистику, пригодится для мониторинга
        if self.training:
            if not hasattr(self, "aux_ema"):
                self.aux_ema = aux_loss.detach()
            self.aux_ema = self.decay * self.aux_ema + (1 - self.decay) * aux_loss.detach()

        return output.view(B, S, D), aux_loss
```

### 2.3 Объяснение кода
| Шаг                   | Код                                    | Объяснение                                                                                                                                          |
| --------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Router softmax** | `scores = F.softmax(logits, -1)`       | Линейный «гейт» оценивает вероятность $p_{i}$, что токен пойдёт к эксперту $i$.                                                               |
| **2. Top-k выбор**    | `topk_val, topk_idx = torch.topk(...)` | Отбираем $k$ экспертов с наибольшим $p$ для данного токена. Традиционные значения $k=1$ или $k=2$ — компромисс «качество vs коммуникации». |
| **3. Capacity**       | `cap = int(self.capacity * N / E)`     | Каждый эксперт обслуживает не более *capacity factor* × среднее число токенов. DeepSpeed-MoE по умолчанию 1.25.                                    |
| **4. Dispatch mask**  | `dispatch.scatter_`                    | One-hot-тензор `[N, E]` указывает, какой токен к какому эксперту пойдёт — аналог CUDA-kernel `dhs_dispatch` в DeepSpeed.                           |
| **5. Scatter**        | `expert_inputs = […]`                  | Разделяем входы по экспертам; в распределённой версии это *all-to-all* MPI-операция.                                                               |
| **6. Gather**         | `output[sel] += out * weight`          | Обратная агрегация результатов экспертов, взвешенная по исходным вероятностям (Switch Transformer trick).                                          |
| **7. Aux loss**       | `aux_loss = (E*(load*prob).sum())`     | **Load-balancing loss** стимулирует равномерное использование экспертов, предотвращая «голодание» .                                                 |


