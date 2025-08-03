# P-Tuning v2 

### Мотивация

Полный fine-tuning параметров крупномасштабных языковых моделей (LLM) сопряжён с высокими вычислительными затратами (в т.ч. по памяти), нестабильностью оптимизации (из-за чувствительности к lr и initialization), а также требует значительных объёмов аннотированных данных. 

> **P-Tuning v2** представляет собой метод адаптации LLM с замороженными весами за счёт внедрения обучаемых векторов в механизм внимания (attention) на **всех слоях модели**. Это обеспечивает параметрически эффективную настройку при сопоставимом с full fine-tuning качестве.

---

### Архитектурная концепция

В основе метода лежит генерация обучаемых векторов (soft prompts), которые не поступают на вход модели, а преобразуются через MLP в специальные ключи и значения attention-механизма (K/V), внедряемые на каждом слое трансформера.

1. Инициализируются виртуальные эмбеддинги $P \in \mathbb{R}^{n \times d}$
2. Эти векторы пропускаются через MLP, выдающий $K_i, V_i$ для каждого слоя $i \in \{1..L\}$
3. На каждом слое attention K/V дополняются этими обученными компонентами

Метод носит название **deep continuous prompt injection** — инъекция параметризованной информации непосредственно в механизм внимания на всех уровнях иерархии модели.

---

### Формализация

Пусть:
- $P \in \mathbb{R}^{n \times d}$ — soft prompt embedding
- $f(P) \rightarrow \{K_i^{\text{prefix}}, V_i^{\text{prefix}}\}_{i=1}^L$ — результат MLP
- $K_i^{\text{input}}, V_i^{\text{input}}, Q_i$ — стандартные входы attention

Тогда для каждого слоя:
$$
K_i = [K_i^{\text{prefix}}; K_i^{\text{input}}],\quad V_i = [V_i^{\text{prefix}}; V_i^{\text{input}}],\quad Q_i = Q_i^{\text{input}}
$$

А механизм внимания:
$$
\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i
$$

Таким образом, префикс влияет на attention через дополнение контекста (K/V), но не искажает запрос (Q).

---

### Реализация (PyTorch)

```python
class PTuningV2Prefix(nn.Module):
    def __init__(self, num_virtual_tokens, num_layers, d_model, d_kv):
        super().__init__()
        self.prompt_embed = nn.Embedding(num_virtual_tokens, d_model)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2 * num_layers * d_kv)
        )
        self.num_layers = num_layers
        self.d_kv = d_kv

    def forward(self, batch_size):
        n = self.prompt_embed.num_embeddings
        prompts = self.prompt_embed(torch.arange(n))
        prompts = self.prefix_mlp(prompts)
        prompts = prompts.view(n, self.num_layers, 2, self.d_kv)
        prompts = prompts.permute(1, 2, 0, 3).unsqueeze(1).repeat(1, batch_size, 1, 1)
        return prompts  # [L, 2 (K/V), B, n, d_kv]
```

На этапе attention нужно выполнить:
```python
K_full = torch.cat([K_prefix, K_input], dim=2)
V_full = torch.cat([V_prefix, V_input], dim=2)
```

---

### Преимущества

- **Замороженная модель**: не требует обновления весов LLM
- **Малая параметрическая нагрузка**: ~0.1–1% от модели
- **Лёгкая переносимость между задачами**: можно переиспользовать префиксы
- **Высокая эффективность**: достигает качества fine-tuning в большинстве NLP-задач

---

### Ограничения

- Требует доступ к **внутренностям модели**: нужно внедрение в attention
- Неинтерпретируемость prompt-векторов
- Для каждой задачи требуется отдельный prompt (не zero-shot)
- Сложно реализовать для **bidirectional** attention (например, BERT)
- Потенциально ограничен по длине контекста: число префиксов фиксировано

---

### Примеры применения

- **ChatGLM**, **GLM**, **ERNIE 4.0** используют P-Tuning v2 для instruction tuning
- Применим к задачам генерации, классификации, извлечения фактов и QA

---

### Заключение

P-Tuning v2 представляет собой высокоэффективную парадигму параметрической адаптации LLM, которая обеспечивает мощный компромисс между производительностью fine-tune и экономичностью soft prompting. Благодаря внедрению обучаемых префиксов во все уровни внимания, метод обеспечивает высокое качество генерации без обновления весов самой модели. Это делает его особенно привлекательным в условиях ограниченных ресурсов и в сценариях, где важна изоляция задачи от основной модели.