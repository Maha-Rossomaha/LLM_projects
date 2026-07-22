# Domain Modeling и границы сущностей

## 1. Интуиция

Domain modeling начинается не с вопроса «какие классы написать», а с вопроса:

> Какие разные вещи существуют в процессе и какие правила нельзя смешивать?

Для платформы обработки документов важно различать:

```text
поступление данных
≠ физический файл
≠ распарсенное содержимое
≠ логический документ
≠ конкретная версия документа
≠ внешняя бизнес-сущность
≠ производный артефакт
```

Если всё назвать `Document`, объект быстро превращается в контейнер для несвязанных полей и обязанностей.

---

## 2. Основные понятия

В нейтральной модели будем использовать:

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

### IngestionPackage

Единица одного поступления данных.

Пример:

```text
package.zip
├── report.docx
├── metrics.xlsx
└── appendix.pdf
```

Пакет отвечает на вопросы:

- что пришло вместе;
- когда и откуда пришло;
- какой внешний запрос инициировал поступление;
- какие файлы были обнаружены;
- завершилась ли обработка всего поступления.

Пакет не обязан соответствовать одному логическому документу. В одном пакете могут находиться несколько документов, служебные файлы и вложения.

### SourceFile

Конкретный зарегистрированный факт получения физического файла.

```python
SourceFile(
    source_file_id="src-201",
    package_id="pkg-100",
    original_name="metrics.xlsx",
    media_type="application/vnd...",
    size_bytes=148230,
    checksum_sha256="...",
)
```

`SourceFile` отвечает на вопрос:

> Какой файл был обнаружен в конкретном поступлении?

Это не то же самое, что content hash. Один и тот же набор байтов может поступить несколько раз и создать несколько `SourceFile`.

### ParsedDocument

Результат интерпретации `SourceFile` конкретным parser pipeline.

```text
SourceFile
→ Parser v2.1
→ ParsedDocument
```

Пример canonical representation:

```python
ParsedDocument(
    blocks=[
        Heading(...),
        Paragraph(...),
        Table(...),
    ],
    parser_name="xlsx_parser",
    parser_version="2.1.0",
)
```

`ParsedDocument` зависит не только от исходных байтов, но и от:

- parser name;
- parser version;
- configuration;
- canonical schema version.

В данной модели его удобнее считать версионируемым артефактом, а не самостоятельной бизнес-сущностью.

### LogicalDocument

Устойчивое понятие документа во времени.

Например:

```text
«Годовой аналитический отчёт по объекту ENTITY-123»
```

Логический документ может состоять из нескольких физических файлов:

```text
LogicalDocument
├── main.docx
├── metrics.xlsx
└── appendix.pdf
```

Он отвечает на вопрос:

> Какой документ существует независимо от конкретной версии его содержимого?

### DocumentRevision

Конкретное зафиксированное состояние `LogicalDocument`.

```text
LogicalDocument document-42
├── Revision 1
├── Revision 2
└── Revision 3
```

Короткая аналогия:

```text
Git repository ≈ LogicalDocument
Git commit     ≈ DocumentRevision
```

После публикации revision обычно становится immutable.

### ExternalEntityRef

Ссылка на объект внешней системы.

```python
ExternalEntityRef(
    system="EXTERNAL_REGISTRY",
    entity_type="BUSINESS_OBJECT",
    external_id="ENTITY-123",
)
```

Внешняя сущность не должна автоматически становиться внутренним документом.

Одна внешняя сущность может быть связана с несколькими документами:

```text
External Entity ENTITY-123
├── development document
├── validation document
└── monitoring document
```

И один документ может относиться к нескольким внешним сущностям.

### Artifact

Результат обработки:

```text
parsed_document.json
document.md
table_index.json
entities.json
summary.json
chunks.jsonl
dense_index
manifest.json
```

Артефакт принадлежит некоторому scope:

```text
package
source file
document revision
logical document
```

Его смысл определяется происхождением и версией builder-а.

---

## 3. Entity, Value Object и Artifact

### Entity

Entity имеет устойчивую identity.

Примеры:

```text
IngestionPackage
SourceFile
LogicalDocument
DocumentRevision
```

Два entity могут иметь одинаковые значения полей, но оставаться разными объектами.

Например, одинаковый файл пришёл в двух пакетах:

```text
pkg-A / appendix.pdf → src-1
pkg-B / appendix.pdf → src-2
```

Checksum совпадает, но факты поступления разные.

### Value Object

Value Object определяется значениями.

Примеры:

```text
Checksum
StorageKey
MediaType
ParserVersion
ExternalEntityRef
```

Два одинаковых `ExternalEntityRef` описывают одну и ту же внешнюю ссылку, если совпадают:

```text
system
entity_type
external_id
```

### Artifact

Artifact — сохранённый или производный результат процесса.

Примеры:

```text
ParsedDocument
Markdown
TableIndex
DenseIndex
Manifest
```

У артефакта может быть технический ID, но его identity обычно выводится из:

- source или revision;
- artifact type;
- версии pipeline;
- configuration;
- schema version.

---

## 4. Почему файл, документ и revision различаются

Рассмотрим два поступления.

### Первое

```text
package-A
├── report.docx
├── metrics.xlsx
└── appendix.pdf
```

### Второе

```text
package-B
├── report.docx       — изменён
├── metrics.xlsx      — изменён
└── appendix.pdf      — не изменён
```

Оба относятся к одному логическому документу.

Получаем:

```text
IngestionPackage: 2
SourceFile records: 6
LogicalDocument: 1
DocumentRevision: 2
Unique binary blobs: возможно 5
```

Почему `LogicalDocument` один?

Потому что второе поступление не создаёт новый смысловой документ. Оно создаёт новое состояние существующего документа.

Неправильно говорить:

```text
второй пакет заменил LogicalDocument
```

Правильно:

```text
LogicalDocument остался прежним
active_revision_id переключился с revision-1 на revision-2
```

Историческая revision-1 продолжает существовать.

---

## 5. Active pointer и immutable revisions

Полезно разделять:

```text
revision data             — immutable
active_revision_id        — mutable pointer
```

При новом поступлении:

```text
revision-1 остаётся неизменной
revision-2 создаётся отдельно
active_revision_id → revision-2
```

Это даёт:

- воспроизводимость старых ответов;
- аудит;
- сравнение версий;
- безопасный rollback;
- возможность расследовать ошибки;
- отсутствие смешивания старых и новых артефактов.

После публикации нельзя менять:

```text
member_source_file_ids
manifest
artifact inventory
pipeline metadata
```

Если эти данные изменились, должна появиться новая revision.

---

## 6. ExternalEntityRef как отдельная связь

Плохая модель:

```python
LogicalDocument(
    id="ENTITY-123",
    external_type="BUSINESS_OBJECT",
)
```

Проблемы:

1. Внутренняя identity зависит от внешней системы.
2. Одна external entity может иметь много документов.
3. Один документ может относиться к нескольким external entities.
4. При смене внешнего идентификатора придётся менять внутренний объект.
5. Появление второй внешней системы ломает модель.

Лучше:

```text
LogicalDocument
      │
      ├── relation: DESCRIBES
      ▼
ExternalEntityRef
```

Отношение может хранить дополнительную семантику:

```text
DESCRIBES
VALIDATES
MONITORS
SUPPORTS
```

Важно не угадывать подтверждённую связь только по имени файла. Автоматическое сопоставление лучше хранить как candidate relation с confidence и последующей валидацией.

---

## 7. Aggregate boundaries

Aggregate — группа объектов, внутри которой поддерживаются согласованные правила.

### IngestionPackage aggregate

Может отвечать за:

- принадлежность source files одному поступлению;
- уникальность внешнего request key;
- завершённость регистрации входов;
- package-level status.

Не нужно помещать внутрь объекта байты всех файлов и все parsed artifacts.

### LogicalDocument aggregate

Может отвечать за:

- identity документа;
- document type;
- lifecycle;
- active revision pointer.

Пример правила:

> Активной может быть только READY revision этого LogicalDocument.

### DocumentRevision aggregate

Может отвечать за:

- membership source files;
- publication state;
- manifest;
- immutability после READY.

Пример правила:

> После READY состав source files и artifact inventory не изменяется.

### Почему не нужен один гигантский aggregate

Плохая вложенность:

```text
LogicalDocument
└── all revisions
    └── all source files
        └── all parsed documents
            └── all artifacts
```

Тогда для переключения `active_revision_id` пришлось бы загружать всю историю и все артефакты.

Aggregate boundary нужна для:

- локальной consistency;
- размера транзакции;
- конкурентных изменений;
- производительности;
- ясной ответственности.

---

## 8. Повторное использование ParsedDocument

Если файл не изменился, можно не парсить его заново, но checksum недостаточно.

Безопасный cache key должен учитывать:

```text
source checksum
parser name
parser version
parser configuration
canonical schema version
```

Пример:

```text
одинаковый файл
+ тот же parser
+ та же config
+ та же schema
→ ParsedDocument можно переиспользовать
```

Но:

```text
одинаковый файл
+ parser v1 → v2
→ результат может измениться
```

Новая версия parser-а может исправить таблицы, порядок блоков или embedded attachments.

---

## 9. Частые ошибки и заблуждения

### Ошибка 1. Новое поступление заменяет LogicalDocument

Новое поступление создаёт новую revision. Сам LogicalDocument сохраняет identity.

### Ошибка 2. ParsedDocument всегда является entity

Технический `document_id` не делает объект бизнес-сущностью. В этой модели ParsedDocument — производный artifact, identity которого зависит от source и pipeline.

### Ошибка 3. ExternalEntityRef — entity внутри нашей системы

Обычно это Value Object, описывающий внешнюю ссылку. Отдельной entity может быть уже relation record, если у связи есть собственный lifecycle, статус подтверждения и audit history.

### Ошибка 4. Один checksum означает один SourceFile

Checksum означает одинаковые байты, а не один факт поступления.

### Ошибка 5. Дедупликация blob-ов должна удалять domain records

Можно хранить один physical blob и несколько `SourceFile`, ссылающихся на него. Physical deduplication не должна уничтожать provenance.

### Ошибка 6. Один package всегда равен одной revision

Это возможно в простом сценарии, но не является универсальным правилом:

- пакет может содержать несколько документов;
- пакет может быть отклонён;
- один документ может собираться из нескольких поступлений;
- часть файлов может быть служебной.

### Ошибка 7. Relationship можно вывести из имени файла как факт

Имя файла может служить сигналом для matching, но не должно автоматически становиться подтверждённой бизнес-связью.

---

## 10. Вопросы на собеседованиях

1. Почему `SourceFile` и binary blob — разные объекты?
2. Чем `LogicalDocument` отличается от `DocumentRevision`?
3. Почему active pointer должен быть отделён от revision contents?
4. Когда ParsedDocument является artifact, а когда его можно моделировать как entity?
5. Как смоделировать many-to-many между документами и внешними сущностями?
6. Что должно входить в aggregate `DocumentRevision`?
7. Какие поля нельзя изменять после публикации?

---

## 11. Практические задачи

### Задача 1

Смоделировать поступление двух пакетов, где один appendix повторяется байт-в-байт. Посчитать:

- package records;
- source records;
- logical documents;
- revisions;
- unique blobs.

### Задача 2

Предложить модель relation между:

```text
LogicalDocument
ExternalEntityRef
```

если связь может быть:

```text
PROPOSED
CONFIRMED
REJECTED
```

### Задача 3

Определить aggregate boundaries для document platform и объяснить, какие операции требуют одной транзакции.

---

## 12. Краткое резюме

```text
IngestionPackage
= одно поступление

SourceFile
= конкретный факт обнаружения физического файла

ParsedDocument
= результат parser pipeline

LogicalDocument
= устойчивый документ во времени

DocumentRevision
= конкретное immutable состояние документа

ExternalEntityRef
= ссылка на объект внешней системы

Artifact
= производный результат обработки
```

Ключевой принцип:

```text
identity объекта
≠ его содержимое
≠ место хранения
≠ внешний идентификатор
≠ текущая активная версия
```
