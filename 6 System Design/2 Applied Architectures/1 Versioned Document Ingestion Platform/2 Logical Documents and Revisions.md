# LogicalDocument и DocumentRevision

## 1. Зачем разделять документ и его версию

В системе обработки документов нужно различать:

```text
документ как устойчивое бизнес-понятие
≠
конкретное зафиксированное состояние его содержимого
```

Например, существует логический документ:

> Ежегодный аналитический отчёт по объекту `ENTITY-123`.

Он может обновляться несколько раз:

```text
LogicalDocument doc-42
├── Revision 1
├── Revision 2
└── Revision 3
```

Смысловой документ остаётся тем же, но его содержимое, состав файлов и опубликованные артефакты меняются.

Для этого используются две сущности:

```text
LogicalDocument
DocumentRevision
```

---

## 2. LogicalDocument

`LogicalDocument` представляет документ, существующий во времени независимо от конкретной версии.

Пример:

```python
LogicalDocument(
    logical_document_id="doc-42",
    document_type="ANALYTICAL_REPORT",
    title="Annual report for ENTITY-123",
    active_revision_id="rev-3",
)
```

Он отвечает на вопрос:

> О каком документе идёт речь независимо от того, какая версия сейчас активна?

### Что обычно хранит LogicalDocument

```text
logical_document_id
business type
human-readable title
external relations
business lifecycle
active_revision_id
created_at
```

### Что не следует хранить прямо в LogicalDocument

```text
конкретный набор SourceFile
checksum текущего Markdown
parser version
artifact inventory
manifest
```

Эти значения меняются между ревизиями и должны принадлежать конкретной `DocumentRevision`.

### LogicalDocument не равен внешнему процессу

Логический документ может быть связан с внешним процессом, проектом или бизнес-сущностью, но не должен автоматически совпадать с ними.

```text
External Business Process
        │
        ├── development report
        ├── validation report
        └── monitoring report
```

Один процесс может иметь несколько документов. Один документ также может относиться к нескольким внешним объектам.

---

## 3. DocumentRevision

`DocumentRevision` — конкретное зафиксированное состояние `LogicalDocument`.

```python
DocumentRevision(
    revision_id="rev-3",
    logical_document_id="doc-42",
    revision_number=3,
    member_source_file_ids=["src-10", "src-11", "src-12"],
    readiness="READY",
    publication_status="PUBLISHED",
)
```

Она отвечает на вопрос:

> Как именно выглядел документ в конкретный момент и из чего был построен?

### Revision может включать несколько SourceFile

```text
DocumentRevision rev-3
├── main.docx
├── metrics.xlsx
└── appendix.pdf
```

Поэтому revision относится не к одному физическому файлу, а к версии всего логического документа.

### Аналогия с Git

```text
Git repository ≈ LogicalDocument
Git commit     ≈ DocumentRevision
```

Репозиторий остаётся тем же, а каждый commit фиксирует отдельное состояние.

---

## 4. Новое поступление не заменяет LogicalDocument

Представим два поступления.

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
└── appendix.pdf      — прежний
```

Оба пакета относятся к одному смысловому документу.

Получаем:

```text
IngestionPackage: 2
SourceFile records: 6
LogicalDocument: 1
DocumentRevision: 2
Unique blobs: возможно 5
```

Правильная модель:

```text
LogicalDocument doc-42
├── rev-1
└── rev-2
```

При публикации второй версии изменяется только указатель:

```text
active_revision_id:
rev-1 → rev-2
```

`LogicalDocument` не удаляется и не заменяется новым объектом.

---

## 5. Revision и ProcessingAttempt

Нужно отдельно различать версию документа и попытку её построения.

```text
DocumentRevision
= версия входных данных и опубликованного результата

ProcessingAttempt
= одна техническая попытка построить эту revision
```

Пример:

```text
rev-3
├── attempt-1 FAILED
└── attempt-2 SUCCEEDED
```

Первая попытка могла упасть из-за timeout или временной ошибки storage. Это не означает, что нужно создавать новую бизнес-ревизию.

### Когда достаточно новой attempt

```text
исходные SourceFile не изменились
pipeline version не изменилась
configuration не изменилась
ожидаемый результат тот же
ошибка была технической
```

Тогда:

```text
rev-3
├── attempt-1 FAILED
└── attempt-2 SUCCEEDED
```

### Когда нужна новая revision

В выбранной строгой модели новая revision создаётся, если изменилось хотя бы одно из следующего:

```text
состав SourceFile
байты SourceFile
роли файлов в документе
parser version
builder version
pipeline configuration
canonical schema version
published artifact inventory
```

Так один `revision_id` однозначно фиксирует и вход, и способ обработки, и опубликованный результат.

---

## 6. Создание и активация — разные операции

Создание revision ещё не делает её доступной потребителям.

```text
create revision
→ register sources
→ parse
→ build artifacts
→ validate
→ publish
→ mark READY
→ activate
```

Пока новая revision строится, старая остаётся активной:

```text
LogicalDocument doc-42
active_revision_id = rev-2

rev-3 readiness = DRAFT
```

Только после успешной публикации:

```text
rev-3 publication_status = PUBLISHED
rev-3 readiness = READY
active_revision_id = rev-3
```

### Главная гарантия

> Serving не должен увидеть частично построенную revision.

Если переключить указатель слишком рано, возможна ситуация:

```text
active_revision_id = rev-3

document.md существует
chunks.jsonl отсутствует
manifest не опубликован
```

Search и QA получат неполный набор данных.

---

## 7. Неуспешная новая revision

Исходное состояние:

```text
LogicalDocument doc-42
active_revision_id = rev-2

rev-1 READY
rev-2 READY
```

Создаётся `rev-3`, но validation завершается ошибкой.

Корректное состояние:

```text
active_revision_id = rev-2

rev-3 readiness = DRAFT
attempt-1 status = FAILED
attempt-1 stage = VALIDATING
```

Search и QA продолжают читать `rev-2`.

### Нужно ли удалять rev-3

Обычно metadata неуспешной revision сохраняется для:

```text
аудита
диагностики
истории attempts
анализа причины ошибки
повторной обработки
```

При этом временные staging-файлы можно удалить по retention policy.

### Повтор после технической ошибки

Если вход и pipeline не изменились:

```text
rev-3
├── attempt-1 FAILED
└── attempt-2 SUCCEEDED
```

После успешной второй попытки `rev-3` может стать `READY` и активной.

Если вход или pipeline изменились, создаётся `rev-4`.

---

## 8. Immutable revision

После публикации revision становится неизменяемым snapshot.

Нельзя менять:

```text
member_source_file_ids
source roles
source checksums
parser and builder versions
configuration fingerprints
artifact inventory
manifest
artifact checksums
publication metadata
```

Причина:

> Один `revision_id` должен всегда обозначать один и тот же подтверждённый набор данных.

Если опубликованный `document.md` был перезаписан, а `revision_id` остался прежним, система теряет:

```text
воспроизводимость
аудит
checksum consistency
возможность объяснить старый ответ QA
безопасный rollback
```

Любое содержательное изменение опубликованного snapshot создаёт новую revision.

---

## 9. Mutable pointer поверх immutable revisions

Изменяемым остаётся указатель:

```text
LogicalDocument.active_revision_id
```

Это позволяет быстро активировать новую версию:

```text
rev-2 → rev-3
```

и делать rollback:

```text
rev-3 → rev-2
```

Сами `rev-2` и `rev-3` при этом не переписываются.

### Атомарность переключения

Изменение `active_revision_id` должно быть атомарным. Потребитель должен увидеть либо старую, либо новую готовую revision, но не промежуточное состояние.

---

## 10. Revision ID и revision number

```python
DocumentRevision(
    revision_id="f71f...",
    logical_document_id="doc-42",
    revision_number=3,
)
```

```text
revision_id
= глобальный технический идентификатор

revision_number
= порядковый номер внутри одного LogicalDocument
```

Номер `3` не уникален во всей системе:

```text
doc-42 / revision 3
doc-91 / revision 3
```

Поэтому полезен составной constraint:

```sql
UNIQUE(logical_document_id, revision_number)
```

Добавление `revision_id` в этот unique key не защищает от двух revision с номером `4`, потому что UUID у них и так разные.

---

## 11. Конкурентное создание revision

Два worker одновременно читают:

```text
last revision number = 3
```

Оба пытаются создать `revision_number = 4`.

Защита может включать:

```text
UNIQUE(logical_document_id, revision_number)
database transaction
SELECT ... FOR UPDATE
optimistic locking
atomic counter per LogicalDocument
unique constraint + retry
```

Минимальная надёжная схема:

```text
1. Прочитать текущий номер в транзакции
2. Попытаться создать следующий
3. Иметь UNIQUE constraint
4. При конфликте перечитать состояние и повторить
```

При конкурентной активации revision также полезен optimistic lock на `LogicalDocument`.

---

## 12. Повторное поступление и дедупликация revisions

Повторная загрузка одних и тех же байтов не всегда обязана создавать новую revision.

### Не создавать новую revision

Если совпадают:

```text
LogicalDocument
ordered source membership
source roles
source checksums
processing fingerprint
```

система может вернуть уже существующую revision.

### Создавать новую revision

Это допустимо, если бизнес-требование требует фиксировать каждый подтверждённый факт публикации, даже при одинаковом содержимом.

Главное — определить семантику заранее и не путать:

```text
повторное поступление
повторную attempt
новую document revision
```

---

## 13. Lifecycle LogicalDocument и состояние revision

Business lifecycle документа:

```text
ACTIVE
ARCHIVED
RETIRED
DELETED
```

Readiness revision:

```text
DRAFT
READY
INVALID
```

Processing attempt:

```text
PENDING
RUNNING
SUCCEEDED
FAILED
CANCELLED
```

Они могут сочетаться:

```text
LogicalDocument lifecycle = ACTIVE
active_revision_id = rev-2
rev-3 readiness = DRAFT
attempt-1 status = FAILED
```

Здесь нет противоречия: документ продолжает использовать корректную `rev-2`, пока новая версия не прошла обработку.

---

## 14. Важные различия

```text
Package
= граница внешнего поступления

SourceFile
= факт получения физического файла

LogicalDocument
= устойчивый документ во времени

DocumentRevision
= конкретная версия LogicalDocument

ProcessingAttempt
= техническая попытка построить revision
```

Также важно:

```text
revision READY
≠
revision active
```

Revision может быть корректной и опубликованной, но не выбранной текущей.

---

## 15. Вопросы для самопроверки

1. Почему новое поступление обычно создаёт revision, а не новый LogicalDocument?
2. Может ли одна revision включать несколько SourceFile?
3. В каком случае повторная обработка создаёт новую attempt, а не новую revision?
4. Почему активация выполняется только после publication и validation?
5. Почему `UNIQUE(revision_id, revision_number)` не защищает от повторяющихся номеров внутри документа?
6. Какие поля опубликованной revision должны быть immutable?
7. Как mutable pointer упрощает rollback?

## 16. Краткое резюме

```text
LogicalDocument = документ как устойчивое понятие
DocumentRevision = зафиксированная версия документа
ProcessingAttempt = одна попытка обработки revision
```

```text
build revision
→ validate
→ publish
→ READY
→ atomic activation
```

```text
immutable revisions
+
mutable active_revision_id
=
безопасное обновление и rollback
```
