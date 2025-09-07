# RAG&#x20;

## 1. Типовой пайплайн RAG

**RAG (Retrieval-Augmented Generation)** — это схема, в которой LLM получает дополнительные знания через поиск в векторной базе.

Базовый процесс:

1. **Эмбеддинг модели** (например, OpenAI, Sentence-BERT, Cohere) преобразует текст или изображение в вектор.
2. **Qdrant** хранит векторы и метаданные (payload).
3. При запросе:
   - генерируется эмбеддинг запроса,
   - Qdrant ищет топ-N ближайших документов,
   - (опционально) применяется переранжировка (cross-encoder, reranker),
   - результаты передаются в LLM для ответа.

---

## 2. Хранение нескольких векторов

В одной коллекции можно хранить разные представления одного объекта:

- `text_vector` — эмбеддинг текста,
- `image_vector` — эмбеддинг изображения,
- `sparse_vector` — BM25/SPLADE.

Пример коллекции с несколькими векторами:

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="multimodal_docs",
    vectors_config={
        "text_vector": rest.VectorParams(size=768, distance=rest.Distance.COSINE),
        "image_vector": rest.VectorParams(size=512, distance=rest.Distance.COSINE),
    }
)
```

---

## 3. Метаданные (payload)

Payload хранит служебную информацию:

- источник документа (url, id),
- язык,
- дата публикации,
- права доступа,
- тип данных (текст/изображение/таблица).

Пример записи с метаданными:

```python
point = rest.PointStruct(
    id=1,
    vector={
        "text_vector": [0.1, 0.2, 0.3],
        "image_vector": [0.05, 0.07, 0.1]
    },
    payload={
        "lang": "en",
        "url": "https://example.com/article",
        "tags": ["ai", "rag"]
    }
)
client.upsert(collection_name="multimodal_docs", points=[point])
```

---

## 4. Гибридный поиск

Чтобы повысить качество поиска, можно комбинировать dense и sparse-вектора.

Пример запроса с гибридом:

```python
from qdrant_client.http import models as rest

query = rest.Query(
    vector={
        "text_vector": [0.5, 0.1, 0.3],
        "sparse_vector": rest.SparseVector(
            indices=[7, 13],
            values=[0.8, 1.0]
        )
    },
    limit=5
)

results = client.query_points(
    collection_name="multimodal_docs",
    query_points=[query]
)
```

Гибридный подход позволяет учитывать и семантику (dense), и точные совпадения (sparse).

---

## 5. Переранжировка

После поиска в Qdrant можно прогонять кандидаты через более точные модели:

- **Cross-encoder** — оценивает релевантность пары (запрос, документ).
- **Late interaction (ColBERT)** — работает по токенам.

Обычно: Qdrant выдаёт top-100 кандидатов → reranker выбирает лучшие 10–20.

---

## 6. Готовые интеграции

Qdrant уже интегрирован с популярными фреймворками:

- **LangChain** — `Qdrant` как retriever для RAG.
- **LlamaIndex** — нативная поддержка.
- **Haystack** — backend для векторного поиска.
- **OpenAI API + Qdrant** — хранение эмбеддингов из `text-embedding-ada-002`.

Пример интеграции с LangChain:

```python
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Qdrant(
    client=client,
    collection_name="multimodal_docs",
    embeddings=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---
