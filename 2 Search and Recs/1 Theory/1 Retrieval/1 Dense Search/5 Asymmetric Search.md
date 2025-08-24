# Asymmetric Search 

Asymmetric search (несимметричный поиск) — это ситуация, когда векторные представления запросов и документов формируются по-разному: разными энкодерами, разными нормализациями или даже разными признаковыми пространствами. Такой подход часто используется в dense retrieval (bi-encoder архитектура), но несёт риски ошибок в выборе метрики и падения качества.

---

## 1. Идея проблемы
- В **bi-encoder** запросы и документы кодируются разными сетями: $q = f_{query}(text)$, $d = f_{doc}(text)$.
- Пространства $q$ и $d$ могут быть **несимметричны**:
  - разные распределения норм,
  - разные статистики признаков,
  - разные масштабы.
- Если просто применить cosine/IP без выравнивания, поиск становится некорректным.

---

## 2. Причины
1. **Bi-encoder архитектура:** раздельные энкодеры без совместного fine-tune.
2. **Отсутствие L2-нормализации:** косинусное сходство не эквивалентно IP.
3. **Несогласованная метрика:** query обрабатывается как нормализованный вектор, а документ — нет.
4. **Domain adaptation:** энкодеры обучены на разных доменах.

---

## 3. Последствия
- Recall@K падает из-за несогласованности расстояний.
- Алгоритмы ANN (IVF/HNSW) работают хуже, так как распределение запросов и документов различается.
- Возможен bias: одни запросы всегда «тянут» к определённым документам.

---

## 4. Методы обнаружения
- Проверка средних норм: $E[\|q\|]$ vs $E[\|d\|]$.
- Распределение cosine similarities между случайным q и d.
- Валидация recall@K при разных метриках (L2 vs IP vs cosine).

---

## 5. Методы решения
1. **L2-нормализация** для query и doc.
2. **Совместное fine-tuning**: обучать bi-encoder на contrastive loss (InfoNCE, triplet).
3. **Метрика под задачу:**
   - cosine при нормализации,
   - IP при dot-product моделях,
   - L2 при автоэнкодерах.
4. **Alignment layer:** добавить линейное преобразование $W$, чтобы выровнять пространства ($q' = Wq$).
5. **Distillation:** учить query/doc энкодеры через общий teacher (например, cross-encoder).

---

## 6. Практические советы
- Всегда проверять распределение норм q и d.
- Если используете cosine — применяйте L2-нормализацию.
- Для новых эмбеддеров проверяйте recall@K на dev-сете.
- При мультиязычных данных — fine-tune bi-encoder на параллельных корпусах.
- Для production — хранить только нормализованные векторы.

---

## 7. Примеры 

### 7.1. Разные энкодеры для query и doc
```python
import torch
import torch.nn.functional as F

class QueryEncoder(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=-1)

class DocEncoder(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=-1)

q_encoder = QueryEncoder()
d_encoder = DocEncoder()

q = q_encoder(torch.randn(1, 768))
d = d_encoder(torch.randn(1, 768))
sim = (q @ d.T).item()
print(sim)
```

### 7.2. Alignment layer для выравнивания
```python
class AlignmentLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
    def forward(self, q):
        return F.normalize(self.linear(q), p=2, dim=-1)

align = AlignmentLayer(256)
q_aligned = align(q)
sim_aligned = (q_aligned @ d.T).item()
print(sim_aligned)
```

---

## 8. Чеклист тюнинга
- Проверить нормы q и d, привести к L2=1.
- Сравнить метрики (cosine vs IP vs L2).
- Провести fine-tune bi-encoder на contrastive loss.
- Добавить alignment layer при domain mismatch.
- Валидировать recall@K перед выкатыванием в прод.

