# Embedding Drift&#x20;

Embedding drift (дрейф эмбеддингов) — это ситуация, когда распределение векторов, генерируемых моделью, со временем изменяется. Это может происходить при обновлении модели, изменении корпуса или эволюции пользовательских данных. Дрейф приводит к падению качества поиска и требует мониторинга.

---

## 1. Идея проблемы

- Эмбеддинги $x = f_{\theta}(doc)$ зависят от модели $f_{\theta}$.
- При изменении $\theta$ (новая версия модели) или корпуса распределение $x$ может сместиться.
- Старый индекс ANN больше не отражает актуальную семантику.

---

## 2. Причины

1. **Обновление модели:**

   - Дообучение эмбеддера на новых данных.
   - Замена модели (например, SBERT → E5).

2. **Эволюция корпуса:**

   - Появление новых доменов (новости, сленг, мемы).
   - Изменение распределения языков.

3. **Пользовательский drift:**

   - Изменение интересов пользователей.
   - Устаревание старых объектов.

---

## 3. Последствия

- **Падение Recall\@K**: старый индекс ищет «по-старому», а новые векторы несовместимы.
- **Снижение nDCG**: релевантные документы уходят вниз в ранжировании.
- **Рост ошибок в кластеризации/персонализации.**

---

## 4. Методы обнаружения

1. **Сравнение распределений:**

   - PSI (Population Stability Index).
   - KL-divergence между старым и новым распределением норм.
   - Cosine norm shift (среднее смещение норм).

2. **Золотые примеры:**

   - Тестовый набор query→docs.
   - Сравнение recall/nDCG на старых vs новых эмбеддингах.

3. **Мониторинг в проде:**

   - CTR, dwell-time.
   - Увеличение числа жалоб/skip.

---

## 5. Методы решения

1. **Shadow index:**

   - Параллельно строим индекс на новых эмбеддингах.
   - Пускаем часть трафика и сравниваем качество/latency.

2. **Alias switch:**

   - Переключаем прод на новый индекс атомарно.
   - Возможен rollback.

3. **Versioning:**

   - Хранение нескольких версий эмбеддингов.
   - Переход на новую версию постепенно.

4. **Incremental re-embed:**

   - Пересчитываем только новые/часто используемые документы.
   - Старые остаются до ребилда.

---

## 6. Практические советы

- Всегда хранить версию модели вместе с эмбеддингами.
- Поддерживать мониторинг распределения норм и cosine similarities.
- Использовать shadow-index + A/B тест перед миграцией.
- Для больших корпусов делать batch re-embed (ночью или оффлайн).
- Для персонализации — обновлять эмбеддинги чаще, чем индекс.

---

## 7. Примеры 

### 7.1. Проверка drift через PSI

```python
import numpy as np

def psi(expected, actual, buckets=10):
    expected_perc, _ = np.histogram(expected, bins=buckets)
    actual_perc, _ = np.histogram(actual, bins=buckets)
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)
    psi_val = np.sum(
        (expected_perc - actual_perc) * np.log(
          (expected_perc+1e-6)/(actual_perc+1e-6)
        )
    )
    return psi_val

old_norms = np.random.normal(1, 0.1, 10000)
new_norms = np.random.normal(1.2, 0.15, 10000)
print("PSI:", psi(old_norms, new_norms))
```

### 7.2. Shadow index в FAISS

```python
import faiss
import numpy as np

# Старый индекс
index_old = faiss.IndexFlatIP(128)
old_emb = np.random.randn(10000, 128).astype('float32')
faiss.normalize_L2(old_emb)
index_old.add(old_emb)

# Новый индекс (shadow)
index_new = faiss.IndexFlatIP(128)
new_emb = np.random.randn(10000, 128).astype('float32')
faiss.normalize_L2(new_emb)
index_new.add(new_emb)

# Поиск на двух индексах
query = np.random.randn(5, 128).astype('float32')
faiss.normalize_L2(query)
D_old, I_old = index_old.search(query, 5)
D_new, I_new = index_new.search(query, 5)
print(I_old[0], I_new[0])
```

---

## 8. Чеклист тюнинга

- Сравнить распределение норм (старое vs новое).
- Считать PSI/KL-div при каждой смене модели.
- Держать shadow index перед выкатыванием.
- Поддерживать versioning индексов.
- Планировать batch re-embed для корпусов.

