# System Design, Domain Modeling и Data Architecture

## 1. Интуиция

При проектировании системы легко сразу начать обсуждать классы, базы данных или технологии:

```text
FastAPI
PostgreSQL
Kafka
HDFS
Kubernetes
```

Но список технологий ещё не является архитектурой. Сначала нужно понять:

- что система должна делать;
- какие понятия существуют в предметной области;
- какие данные нужно хранить;
- какие компоненты выполняют работу;
- как данные проходят через систему;
- что произойдёт при ошибке;
- когда результат считается готовым.

Для этого полезно разделять три близких, но разных уровня:

```text
Domain Model
Data Architecture
System Architecture
```

Короткая формула:

```text
Domain Model       = что существует
Data Architecture  = как информация об этом хранится
System Architecture = кто этим управляет и как выполняется процесс
```

---

## 2. Что такое System Design

System design — это проектирование системы на уровне компонентов, данных, взаимодействий и эксплуатационных гарантий.

Он отвечает на вопросы:

- из каких частей состоит система;
- где проходят границы ответственности;
- какие компоненты вызывают друг друга;
- где хранится состояние;
- какие операции синхронные, а какие фоновые;
- что делать при частичном сбое;
- как повторять операции безопасно;
- как система масштабируется;
- какие компромиссы принимаются.

Хорошее архитектурное описание включает не только схему из прямоугольников, но и:

```text
requirements
contracts
state transitions
failure modes
consistency guarantees
operational constraints
trade-offs
```

### Пример верхнеуровневой системы

Представим нейтральную платформу обработки документов:

```text
External Source
      │
      ▼
Ingestion API / Job
      │
      ▼
Ingestion Orchestrator
      ├──────────► Metadata Repository
      │
      ├──────────► Parser Registry
      │                 │
      │                 ▼
      │          Canonical Document
      │                 │
      ├──────────► Artifact Builders
      │                 │
      └──────────► Artifact Storage
                        │
                  staging → validation
                        │
                        ▼
                  published revision
                        │
                        ▼
                 Search / QA / Wiki
```

Эта схема показывает ответственность компонентов, но ещё не определяет конкретный язык программирования, тип БД или формат очереди.

---

## 3. Functional и Non-functional Requirements

Перед архитектурой нужно разделить требования на две группы.

### Functional requirements

Что система должна уметь делать:

- принимать один файл или пакет файлов;
- обнаруживать вложения;
- парсить разные форматы;
- создавать производные артефакты;
- хранить версии;
- публиковать готовые результаты;
- предоставлять их поиску или QA.

### Non-functional requirements

Какими свойствами должна обладать система:

- допустимый размер файла;
- количество загрузок в сутки;
- время обработки;
- требования к доступности;
- срок хранения истории;
- допустимость eventual consistency;
- необходимость воспроизводимости;
- требования к безопасности;
- допустимая потеря данных;
- требования к аудиту.

Одна и та же функциональность может иметь совершенно разную архитектуру при разных non-functional requirements.

Например:

```text
10 документов в день
```

можно обрабатывать одной локальной job.

А для:

```text
10 миллионов документов в день,
обработка не более 5 минут,
несколько регионов,
строгий аудит
```

потребуются очереди, горизонтальное масштабирование, partitioning, retry policies и наблюдаемость.

---

## 4. Domain Modeling

Domain modeling отвечает на вопрос:

> Какие разные понятия существуют в предметной области и какие правила для них действуют?

Для платформы обработки документов это могут быть:

```text
IngestionPackage
SourceFile
ParsedDocument
LogicalDocument
DocumentRevision
ExternalEntityRef
Artifact
Manifest
```

Это не обязательно таблицы и не обязательно сервисы.

Например, `DocumentRevision` — предметное понятие: конкретное зафиксированное состояние логического документа.

Оно может храниться:

- строкой в PostgreSQL;
- документом в MongoDB;
- JSON-файлом;
- только в памяти в тестовой реализации.

Способ хранения не меняет смысл доменного понятия.

### Почему важно разделять понятия

Плохая модель часто начинается с универсального класса:

```python
class Document:
    file_path: str
    parsed_text: str
    external_id: str
    revision: int
    status: str
    markdown: str
    entities: list
```

Здесь смешаны:

- физический файл;
- результат парсинга;
- логический документ;
- его версия;
- внешняя ссылка;
- артефакты;
- workflow state.

Такой объект трудно тестировать, версионировать и безопасно изменять.

---

## 5. Data Architecture

Data architecture описывает:

- какие данные сохраняются;
- где они хранятся;
- как связаны;
- какие ключи и индексы используются;
- как хранится история;
- как поддерживается lineage;
- какие данные являются активными, а какие историческими.

Пример:

```text
LogicalDocument
├── Revision 1
├── Revision 2
└── Revision 3
```

Одна реализация может выглядеть так:

```text
PostgreSQL
├── logical_documents
├── document_revisions
├── source_files
├── manifests
└── artifact_metadata

Object Storage
├── original files
├── parsed documents
├── markdown
├── indexes
└── summaries
```

### Важное различие

```text
DocumentRevision
```

— это domain model.

```text
document_revisions
```

— таблица, то есть элемент data architecture.

Таблица не становится доменной сущностью только потому, что хранит её поля.

---

## 6. System Architecture

System architecture определяет работающие компоненты и их взаимодействия.

Например:

```text
IngestionOrchestrator
ParserRegistry
ArtifactBuilder
ArtifactStorage
DocumentRepository
ManifestRepository
PublicationService
ServingService
```

Компонент может управлять доменной сущностью, но не совпадает с ней.

Пример:

```text
DocumentRevision              — domain entity
DocumentRevisionRepository    — system component
 document_revisions           — data storage structure
```

Одна сущность рассматривается на трёх разных уровнях.

---

## 7. Data Flow и Control Flow

Эти понятия часто путают.

### Data Flow

Показывает, как данные преобразуются:

```text
DOCX bytes
→ SourceFile
→ ParsedDocument
→ Markdown
→ Chunks
→ Search Index
```

Промежуточное представление не обязательно сохраняется на диск. Оно может существовать только в памяти.

### Control Flow

Показывает, кто управляет выполнением:

```text
IngestionOrchestrator
→ выбирает Parser
→ запускает parse
→ проверяет результат
→ запускает ArtifactBuilder
→ обрабатывает ошибку
→ делает retry или переводит процесс в FAILED
```

Коротко:

```text
Data Flow    = что происходит с данными
Control Flow = кто, когда и при каких условиях запускает действия
```

### Почему различие важно

Правильный data flow не гарантирует хороший control flow.

Например, цепочка преобразований может быть разумной:

```text
SourceFile → ParsedDocument → Markdown
```

Но управление может быть размазано по API-контроллеру, парсеру и storage-классу. Тогда невозможно централизованно реализовать retries, статусы и восстановление.

---

## 8. Pipeline и State Machine

Ingestion часто изображают только как pipeline:

```text
receive
→ parse
→ build
→ publish
```

Pipeline показывает последовательность обработки.

Но надёжная система также нуждается в state machine:

```text
RECEIVED
→ PARSING
→ BUILDING
→ VALIDATING
→ PUBLISHING
→ READY
```

При ошибке:

```text
PARSING   → FAILED
BUILDING  → FAILED
PUBLISHING → FAILED
```

State machine отвечает на вопросы:

- где процесс находится сейчас;
- разрешён ли следующий переход;
- можно ли повторить шаг;
- является ли состояние терминальным;
- доступен ли результат потребителям.

Pipeline и state machine дополняют друг друга:

```text
pipeline      = этапы работы
state machine = подтверждённое состояние процесса
```

---

## 9. Границы ответственности

Хорошая граница отвечает на вопрос:

> Что компонент должен знать, а чего знать не должен?

### Parser

Должен знать:

- формат файла;
- как извлечь текст, таблицы и изображения;
- как построить canonical representation.

Не должен знать:

- какой документ является активным;
- когда вся ревизия готова;
- где находится production storage;
- как обновить внешнюю бизнес-систему.

### ArtifactStorage

Должен знать:

- как записать и прочитать байты;
- как проверить существование объекта;
- как выполнить atomic write, если backend это поддерживает;
- как удалить key или prefix.

Не должен знать:

- бизнес-тип документа;
- lifecycle ревизии;
- правила выбора активной версии.

### Orchestrator

Должен знать:

- порядок шагов;
- условия переходов;
- какие компоненты вызвать;
- как обработать частичный сбой;
- когда процесс можно считать завершённым.

Не должен знать детали структуры DOCX или протокола конкретного object storage.

---

## 10. Разбор анти-паттерна

```python
class DocumentService:
    def process(self, path: str) -> None:
        document = DocxParser().parse(path)
        markdown = MarkdownBuilder().build(document)

        with open(f"data/{document.id}/document.md", "w") as file:
            file.write(markdown)

        self.external_client.mark_ready(document.id)
```

Проблемы:

1. Жёсткая зависимость от одного парсера.
2. Сервис сам выбирает формат обработки.
3. Смешаны parsing, artifact building и storage.
4. Код привязан к локальному filesystem.
5. Вызывающий код сам строит physical path.
6. Нет staging.
7. Нет validation gate.
8. Нет manifest.
9. Нет обработки частичного сбоя.
10. `READY` выставляется без знания о полноте всей ревизии.

Более безопасная схема:

```text
Orchestrator
├── ParserRegistry.get(source)
├── parser.parse(source)
├── builders.build(parsed_document)
├── repositories.write_to_staging(...)
├── validator.validate(...)
├── publication.publish(...)
└── revision_repository.mark_ready(...)
```

Фабрика или registry исправляет только выбор parser-а. Она не решает автоматически проблемы хранения, статусов и публикации.

---

## 11. Частые ошибки и заблуждения

### Ошибка 1. Таблица считается domain entity

Неверно:

```text
Domain Model: document_revisions table
```

Правильно:

```text
Domain Model: DocumentRevision
Data Architecture: document_revisions table
```

### Ошибка 2. Data flow — это список сохраняемых форматов

Data flow описывает преобразование данных, даже если промежуточные объекты не сохраняются.

### Ошибка 3. Parser может выставить READY

Parser видит только свой локальный результат. Он не знает:

- обработаны ли остальные источники;
- построены ли все обязательные артефакты;
- прошла ли валидация;
- завершилась ли публикация.

Поэтому parser не обладает достаточной информацией для глобального решения.

### Ошибка 4. Файлы существуют — значит ревизия готова

Наличие файлов не доказывает, что набор:

- полный;
- непротиворечивый;
- прошёл validation;
- опубликован атомарно;
- соответствует manifest.

Готовность должна подтверждаться явным переходом состояния.

### Ошибка 5. Чем больше сервисов, тем лучше архитектура

Разделение ответственности можно реализовать в modular monolith. Сетевые границы добавляют latency, deployment complexity и distributed failure modes.

---

## 12. Вопросы на собеседованиях

1. Чем domain model отличается от database schema?
2. Чем data flow отличается от control flow?
3. Почему наличие артефактов не гарантирует READY?
4. Кто должен принимать решение о публикации ревизии?
5. Когда modular monolith предпочтительнее микросервисов?
6. Какие non-functional requirements сильнее всего меняют архитектуру ingestion-системы?

---

## 13. Практические задачи

### Задача 1

Разделить обязанности следующего класса между компонентами:

```python
class ProcessingService:
    def upload(self): ...
    def unpack(self): ...
    def parse_pdf(self): ...
    def save_to_storage(self): ...
    def update_status(self): ...
    def answer_question(self): ...
```

### Задача 2

Для pipeline:

```text
PDF → ParsedDocument → Chunks → DenseIndex
```

отдельно описать:

- data flow;
- control flow;
- state machine;
- failure transitions.

### Задача 3

Объяснить, почему событие `FileWritten` не эквивалентно событию `RevisionReady`.

---

## 14. Краткое резюме

```text
Domain Model
= понятия и правила предметной области

Data Architecture
= хранение, связи, история и размещение данных

System Architecture
= компоненты, взаимодействия и эксплуатационные гарантии

Data Flow
= преобразование данных

Control Flow
= управление выполнением

Pipeline
= этапы обработки

State Machine
= подтверждённое состояние и разрешённые переходы
```

Главное правило: начинать проектирование нужно не с технологий, а с требований, сущностей, потоков данных, состояний и границ ответственности.
