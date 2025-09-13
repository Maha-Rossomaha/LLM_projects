# Сегментация длинных документов&#x20;

Длинные документы — одна из ключевых проблем retrieval-систем. Модели (особенно dense retrievers и rerankers) обычно работают на ограниченной длине входа (например, 512 или 1024 токена). Поэтому приходится делить документ на смысловые чанки.&#x20;

---

## 1. Fixed-size Sliding Window

### Идея:

* Разбиваем текст на фиксированные чанки (например, 512 токенов) с overlap (например, 128).
* Подходит для последовательных, однотипных текстов без разметки.

### Параметры:

* `chunk_size = 512`
* `stride = 384` (т.е. overlap = 128)

### Пример на Python (using `nltk`):

```python
from nltk.tokenize import word_tokenize

words = word_tokenize(document)
chunk_size, stride = 512, 384
chunks = [words[i:i+chunk_size] for i in range(0, len(words), stride)]
```

### Плюсы:

* Просто и эффективно.
* Хорошо работает для длинных потоков текста (википедия, статьи).

### Минусы:

* Не учитывается структура.
* Возможна обрезка смысловых границ.

---

## 2. Structure-aware splitting

### Идея:

* Делить по структуре: заголовкам, абзацам, HTML-блокам.
* Например, каждый `<h2> + абзацы` → один чанк.

### Реализация:

* Парсинг через BeautifulSoup / Markdown parser / PDF miner.
* Постобработка заголовков, абзацев.

### Пример:

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
chunks = []
for section in soup.find_all(['h1', 'h2', 'h3']):
    content = section.get_text() + ' '.join(p.get_text() for p in section.find_next_siblings('p'))
    chunks.append(content)
```

### Плюсы:

* Сохраняется логика документа.
* Повышает interpretability и downstream-качество.

### Минусы:

* Не все документы структурированы.
* Требует кастомного парсинга.

---

## 3. Semantic Chunking

### Идея:

* Разбивать документ на смысловые блоки по семантике, а не по длине.
* Часто используется кластеризация эмбеддингов абзацев или предложений.

### Пример (HuggingFace + sklearn):

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = document.split(".")
embeddings = model.encode(sentences)
labels = AgglomerativeClustering(n_clusters=10).fit_predict(embeddings)

clusters = [[] for _ in range(10)]
for i, label in enumerate(labels):
    clusters[label].append(sentences[i])
chunks = [" ".join(cluster) for cluster in clusters]
```

### Плюсы:

* Автоматическая адаптация к содержанию.
* Подходит для технических или неструктурированных текстов.

### Минусы:

* Высокая сложность.
* Не всегда устойчиво при noise / коротких текстах.

---

## 4. Контекст и метаданные

При любом способе сегментации важно:

* \*\*Сохранять \*\*\`\`, чтобы потом агрегировать результаты.
* \*\*Добавлять \*\*\`\`, чтобы учитывать порядок.
* **section\_title** (если есть) помогает при генерации.
* **parent/child context:** иногда полезно добавлять заголовок предыдущего блока или summary.

### Пример структуры чанка:

```json
{
  "document_id": "doc123",
  "chunk_id": 4,
  "text": "...",
  "section_title": "Результаты эксперимента"
}
```

---

## 5. Как использовать чанки в retrieval

### При поиске:

* Dense retriever возвращает отдельные чанки.
* Reranker (например, cross-encoder) оценивает query+chunk.

### При генерации (RAG):

* Выбирается top-K чанков по сходству.
* Передаются в генератор.
* **Fusion-in-decoder**: модель объединяет куски при генерации ответа.

---

## 6. Как выбирать стратегию

| Метод             | Когда использовать              |
| ----------------- | ------------------------------- |
| Sliding window    | Потоковый текст, нет структуры  |
| Structure-aware   | HTML/PDF/Markdown с заголовками |
| Semantic chunking | Технические/научные документы   |
