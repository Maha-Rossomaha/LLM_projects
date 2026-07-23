# Versioning, Lifecycle и Immutable Revisions

## 1. Три разных вида версий

В системах обработки данных часто смешивают:

```text
DocumentRevision
ProcessingAttempt
Artifact or Pipeline Version
```

Они отвечают на разные вопросы.

```text
DocumentRevision
= какое состояние документа зафиксировано

ProcessingAttempt
= сколько раз система пыталась построить это состояние

Pipeline Version
= каким кодом и конфигурацией был получен результат
```

Пример:

```text
LogicalDocument doc-42
└── Revision 3
    ├── attempt-1 FAILED
    ├── attempt-2 SUCCEEDED
    ├── Parser 2.1.0
    └── MarkdownBuilder 1.4.0
```

Здесь одна revision, две технические попытки и несколько версий компонентов обработки.

---

## 2. Семантика DocumentRevision

Прежде чем проектировать versioning, нужно определить:

> Что именно идентифицирует одна revision?

Возможны разные модели.

### Content-only revision

Revision фиксирует только бизнес-содержимое и состав исходных документов.

```text
rev-3
├── source fingerprint A
├── parsed artifact v1
└── parsed artifact v2
```

Новый parser может пересобрать артефакты внутри той же бизнес-revision, если публикации артефактов тоже версионируются отдельно.

### Full publication revision

Revision фиксирует:

```text
sources
source roles
pipeline versions
configuration
artifact inventory
manifest
```

В этой модели изменение parser-а или builder-а создаёт новую revision, даже если исходные файлы не изменились.

### Выбранная строгая модель

Для учебной versioned document platform используется второй вариант:

> Published DocumentRevision — immutable snapshot входов, способа обработки и опубликованных артефактов.

Это упрощает аудит и serving: один `revision_id` всегда ведёт к одному полностью определённому результату.

---

## 3. Когда создаётся новая revision

Новая revision требуется, если изменилось бизнес-содержимое:

```text
изменился main document
обновилась таблица
добавилось приложение
изменились роли файлов
изменился подтверждённый состав документа
```

В строгой модели новая revision также требуется, если изменилось производство результата:

```text
parser version
builder version
configuration
canonical schema
required artifact set
publication schema
```

Пример:

```text
same sources + Parser 2.0 → rev-3
same sources + Parser 2.1 → rev-4
```

---

## 4. Когда достаточно новой ProcessingAttempt

Новая attempt используется, если логически должен получиться тот же snapshot.

Совпадают:

```text
SourceFile membership
source checksums
source roles
parser versions
builder versions
configurations
schema versions
required artifacts
```

Но первая попытка упала по технической причине:

```text
worker crash
network timeout
temporary storage failure
lease expiration
transient dependency error
```

Тогда:

```text
rev-4
├── attempt-1 FAILED
└── attempt-2 SUCCEEDED
```

Attempt — это история выполнения, а не новая версия данных.

---

## 5. Source fingerprint

Source fingerprint характеризует входное состояние revision.

Упрощённо:

```text
hash(
  ordered source checksums,
  source roles,
  document membership,
  relevant source metadata
)
```

Почему недостаточно просто отсортировать checksums:

```text
main.docx checksum A
appendix.pdf checksum B
```

и:

```text
main.docx checksum B
appendix.pdf checksum A
```

имеют одинаковый набор hashes, но разную семантику ролей.

Fingerprint должен учитывать не только байты, но и структуру документа.

---

## 6. Processing fingerprint

Processing fingerprint характеризует способ обработки:

```text
hash(
  parser names and versions,
  builder names and versions,
  configurations,
  canonical schema version,
  output schema versions
)
```

Пример:

```text
source fingerprint = SRC-A
processing fingerprint = PIPELINE-2.1
```

Два запуска могут иметь одинаковые sources, но разные processing fingerprints.

```text
SRC-A + PIPELINE-2.0 → result A
SRC-A + PIPELINE-2.1 → result B
```

---

## 7. Почему нужно сохранять parser и builder versions

Одинаковые байты не гарантируют одинаковый результат.

```text
same PDF + Parser 1.0 → blocks A
same PDF + Parser 2.0 → blocks B
```

Могут измениться:

```text
извлечение таблиц
порядок блоков
распознавание заголовков
обработка merged cells
normalization
image extraction
metadata mapping
```

Даже одна версия parser-а с разными конфигурациями может дать разные результаты.

Поэтому revision или artifact provenance хранит:

```text
component name
component version
configuration fingerprint
schema version
input references
output checksum
```

---

## 8. Immutable revision

Immutable revision означает, что после publication её содержание нельзя менять in-place.

Фиксируются:

```text
member_source_file_ids
source roles
source checksums
source fingerprint
processing fingerprint
parser and builder versions
configuration fingerprints
artifact inventory
artifact checksums
manifest
publication timestamp
```

Если любое из этих значений изменилось, появляется новый snapshot.

### Почему immutable важнее удобства редактирования

In-place исправление кажется проще:

```text
replace document.md
update one JSON
fix one table
```

Но после этого прежний `revision_id` обозначает уже другой набор данных.

Ломаются:

```text
reproducibility
audit
rollback
cache correctness
old answer investigation
checksum validation
lineage
```

---

## 9. Mutable pointer поверх immutable data

Обычно меняется только:

```text
LogicalDocument.active_revision_id
```

Пример активации:

```text
active_revision_id:
rev-2 → rev-3
```

Rollback:

```text
active_revision_id:
rev-3 → rev-2
```

Сами ревизии не переписываются.

Преимущества:

```text
быстрый rollback
атомарное переключение
историческая воспроизводимость
простая модель serving
отсутствие смешивания артефактов
```

---

## 10. Revision readiness и lifecycle

Возможные состояния revision:

```text
DRAFT
READY
READY_WITH_WARNINGS
INVALID
```

Отдельный lifecycle может включать:

```text
CURRENT
SUPERSEDED
RETIRED
```

Но не всегда нужно хранить все эти значения.

### SUPERSEDED

```text
Revision корректна и потенциально пригодна,
но активной является другая revision.
```

Например:

```text
rev-2 READY, not active
rev-3 READY, active
```

Состояние superseded часто можно вычислять:

```text
revision READY and revision_id != active_revision_id
```

### RETIRED

```text
Revision больше нельзя активировать
по бизнес- или эксплуатационным правилам.
```

Причины:

```text
устаревший формат
истёкший нормативный период
неподдерживаемая версия pipeline
решение владельца данных
```

### INVALID

```text
В revision обнаружена ошибка,
поэтому данным нельзя доверять.
```

Например:

```text
ошибочный состав sources
сломанная normalization
неверные таблицы
несоответствие manifest
```

---

## 11. Rollback с ошибочной revision

Исходное состояние:

```text
active_revision_id = rev-3
rev-2 READY
rev-3 READY
```

Позже `rev-3` признана ошибочной.

Действия:

```text
1. atomically set active_revision_id = rev-2
2. mark rev-3 INVALID
3. invalidate serving caches
4. notify dependent compilers or indexes
5. preserve rev-3 for investigation
```

`rev-2` по содержимому менять нельзя и не нужно.

Если superseded вычисляется, после rollback:

```text
rev-2 становится active
rev-3 становится INVALID
```

---

## 12. Rollback на корректную старую revision

Иногда новая revision технически корректна, но показывает плохие эксплуатационные результаты.

Пример:

```text
rev-3 valid
но search relevance ухудшилась
```

Можно временно переключиться:

```text
active_revision_id = rev-2
```

При этом `rev-3` не обязательно `INVALID`.

Она может остаться:

```text
READY, not active
```

или стать `RETIRED`, если принято решение больше её не использовать.

Rollback сам по себе не доказывает ошибочность revision.

---

## 13. Lifecycle LogicalDocument

У LogicalDocument отдельные состояния:

```text
ACTIVE
ARCHIVED
RETIRED
DELETED
```

Пример:

```text
LogicalDocument lifecycle = ARCHIVED
active_revision_id = rev-5
rev-5 readiness = READY
```

Revision остаётся корректной, но документ скрыт из обычных процессов.

```text
document lifecycle
≠
revision readiness
```

---

## 14. Logical deletion

Logical deletion означает:

```text
объект помечен удалённым
serving перестаёт его показывать
новые ссылки запрещаются
создаётся tombstone или deleted_at
```

Данные при этом ещё физически существуют.

Пример:

```python
LogicalDocument(
    lifecycle="DELETED",
    deleted_at="...",
)
```

Это позволяет:

```text
быстро скрыть объект
сохранить аудит
отложить дорогое удаление
восстановиться в grace period
координировать очистку зависимостей
```

---

## 15. Physical deletion

Physical deletion — фактическое удаление из:

```text
metadata database
blob or object storage
search indexes
caches
compiled knowledge layer
feature or vector stores
backups according to policy
```

Оно обычно выполняется отдельно и асинхронно.

Порядок:

```text
logical delete
→ tombstone
→ stop serving
→ retention or grace period
→ dependency check
→ physical cleanup
→ reconciliation
```

---

## 16. Retention policies

Разные данные могут иметь разные сроки хранения.

Пример:

```text
READY revisions       — несколько лет или бессрочно
FAILED attempts       — 90 дней
staging artifacts     — 7 дней
technical logs        — 30 дней
audit events          — 5 лет
```

Это иллюстрация, а не универсальные значения.

Retention определяется:

```text
бизнес-требованиями
стоимостью хранения
аудитом
безопасностью
правовыми ограничениями
возможностью повторной обработки
```

---

## 17. Shared blobs и reference safety

Две revisions могут использовать один blob:

```text
rev-2 → appendix blob A
rev-3 → appendix blob A
```

После удаления `rev-3` blob A всё ещё нужен `rev-2`.

Поэтому нельзя удалять blob только потому, что удаляется одна revision.

Перед physical deletion проверяется:

```text
живые SourceFile references
живые Artifact references
revision manifests
active indexes
knowledge snapshots
legal holds
retention locks
```

---

## 18. Garbage collection

Подходы:

### Reference counting

```text
blob.ref_count
```

Blob удаляется при `ref_count = 0`.

Проблемы:

```text
счётчик может рассинхронизироваться
сложны транзакции между storage и metadata
```

### Live reference scan

Перед удалением система ищет все живые ссылки.

Плюсы:

```text
прямая проверка истины
```

Минусы:

```text
дорого на больших объёмах
```

### Mark-and-sweep

```text
mark all reachable blobs
sweep unreferenced blobs after grace period
```

Подходит для периодической reconciliation.

---

## 19. Tombstones

Tombstone — запись, подтверждающая, что объект был удалён логически.

Она полезна в распределённых системах:

```text
предотвращает повторное появление старых данных
передаёт deletion в downstream indexes
помогает eventual consistency
сохраняет audit identity
```

Если просто удалить строку из metadata, отстающий consumer может снова восстановить объект из старого события.

---

## 20. Rebuild новой версией builder-а

Есть опубликованная `rev-5`:

```text
document.md built by MarkdownBuilder 1.4
```

Появляется Builder 2.0, который меняет результат.

В строгой модели:

```text
create rev-6
reuse same source blobs
build new artifacts
validate
publish new manifest
activate rev-6
```

`rev-5/document.md` не перезаписывается.

Это сохраняет обе версии и позволяет сравнить результаты.

---

## 21. Важные различия

```text
New attempt
= тот же ожидаемый snapshot, новый технический запуск

New revision
= новый immutable published snapshot
```

```text
SUPERSEDED
= корректна, но не активна

RETIRED
= использование запрещено правилами

INVALID
= данным нельзя доверять
```

```text
Logical deletion
= скрыть и пометить

Physical deletion
= удалить реальные данные из всех хранилищ
```

---

## 22. Вопросы для самопроверки

1. Чем revision отличается от attempt?
2. Почему изменение parser config создаёт новую revision в строгой модели?
3. Что должно входить в source fingerprint?
4. Чем processing fingerprint отличается от source fingerprint?
5. Когда rollback не означает, что новая revision invalid?
6. Почему нельзя удалить shared blob вместе с одной revision?
7. Зачем tombstone после logical deletion?
8. Какие проверки нужны перед physical deletion?

## 23. Краткое резюме

```text
Published revision
= immutable sources + pipeline + artifacts + manifest
```

```text
technical retry
→ new ProcessingAttempt

content or pipeline change
→ new DocumentRevision
```

```text
immutable revisions
+
mutable active pointer
=
atomic activation and rollback
```

```text
logical delete first
physical cleanup later
```
