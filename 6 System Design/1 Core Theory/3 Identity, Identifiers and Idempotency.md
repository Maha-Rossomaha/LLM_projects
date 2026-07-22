# Identity, Identifiers и Idempotency

## 1. Интуиция

В одной системе может одновременно существовать много значений, похожих на «ID»:

```text
package_id
source_file_id
checksum
logical_document_id
revision_id
revision_number
external_id
storage_key
idempotency_key
artifact_fingerprint
```

Это не дублирование. Они отвечают на разные вопросы.

Главная идея:

```text
Entity ID             = какой объект или факт
Checksum              = какое содержимое
Storage Key           = где лежат данные
External ID           = как объект называется в другой системе
Idempotency Key       = какую операцию нельзя выполнить повторно
Artifact Fingerprint  = строился ли уже такой результат обработки
```

Ошибки начинаются, когда одно значение пытаются использовать сразу для нескольких ролей.

---

## 2. Identity и Identifier

### Identity

Identity — смысловая уникальность объекта во времени.

Например, конкретный `SourceFile` остаётся тем же фактом поступления, даже если:

- файл переместили в другое хранилище;
- изменился storage key;
- добавились metadata;
- запись была прочитана из другой БД.

### Identifier

Identifier — техническое представление identity.

Например:

```text
src-201
8c97f36a-20a6-4a2d-b640-c851a40870ec
```

Один объект может иметь несколько идентификаторов:

```text
internal UUID
external business key
human-readable number
composite natural key
```

Но нужно понимать, какой из них является основным внутри системы.

---

## 3. Технический идентификатор

Часто используется UUID.

```python
SourceFile(
    source_file_id=UUID("8c97f36a-20a6-4a2d-b640-c851a40870ec")
)
```

Преимущества:

- можно создать без центрального счётчика;
- практически нет случайных совпадений;
- не зависит от имени файла;
- не зависит от storage backend;
- не раскрывает бизнес-смысл;
- удобно использовать между сервисами.

Недостаток — UUID плохо читается человеком. Но технический ID не обязан быть описательным.

### Неинформативность — не дефект ID

Иногда checksum или UUID называют плохими идентификаторами, потому что они «ничего не говорят человеку».

Это неверный критерий.

У идентификатора другая задача:

> Стабильно и однозначно ссылаться на объект.

Читаемость обеспечивается отдельными полями:

```text
title
original_name
revision_number
external_number
```

---

## 4. Business Key

Business key имеет смысл в предметной области.

Примеры:

```text
ORDER-2026-00125
DOCUMENT-42
ENTITY-123
```

Плюсы:

- понимается пользователями;
- может использоваться в отчётности;
- упрощает интеграции.

Минусы:

- правила могут измениться;
- значение может быть присвоено поздно;
- возможны повторы между системами;
- иногда требуется переименование;
- формат может нести нежелательную бизнес-информацию.

Обычно полезно хранить оба:

```python
LogicalDocument(
    logical_document_id=UUID(...),
    document_number="DOCUMENT-42",
)
```

Внутренняя identity остаётся стабильной, а business key используется как атрибут.

---

## 5. External ID

External ID принадлежит другой системе.

Одного значения недостаточно:

```text
12345
```

Нужно знать контекст:

```text
external system
entity type
external id
```

Пример:

```python
ExternalEntityRef(
    system="EXTERNAL_REGISTRY",
    entity_type="BUSINESS_OBJECT",
    external_id="12345",
)
```

Почему нельзя использовать external ID как внутренний ID:

- внешний объект может изменить ключ;
- одна сущность может иметь несколько external references;
- несколько систем могут использовать одинаковое значение;
- внутренняя модель становится зависимой от внешней;
- внешний объект и внутренний документ могут иметь разные lifecycles.

---

## 6. Checksum

Checksum вычисляется из содержимого:

```text
checksum = SHA-256(file bytes)
```

Он отвечает на вопрос:

> Совпадают ли байты?

Checksum полезен для:

- integrity checks;
- physical deduplication;
- cache lookup;
- обнаружения повторного содержимого;
- проверки публикации;
- content-addressed storage.

### Checksum не равен SourceFile ID

Пример:

```text
package-A / appendix.pdf → source-file-1
package-B / appendix.pdf → source-file-2
```

Файлы одинаковы:

```text
checksum(source-file-1) == checksum(source-file-2)
```

Но это разные факты поступления.

Поэтому:

```text
source_file_id = какой факт поступления
checksum       = какие байты
```

### Коллизии

Для современного cryptographic hash вроде SHA-256 случайная коллизия практически не является главным архитектурным риском в обычной системе.

Основная причина не использовать checksum как entity ID — семантическая:

> одинаковое содержимое не означает один и тот же объект или событие.

---

## 7. Storage Key

Storage key отвечает на вопрос:

> Где находится объект в конкретном backend?

Пример:

```text
packages/pkg-100/sources/src-201/original.bin
```

Storage key не является identity, потому что location может измениться:

```text
local filesystem
→ HDFS
→ object storage
```

Объект остаётся тем же, а путь меняется.

Полезная формула:

```text
ID          = кто это
Checksum    = что внутри
Storage Key = где лежит
```

Не стоит строить доменные связи через физические пути.

---

## 8. Revision ID и Revision Number

`revision_number = 3` не уникален во всей системе.

```text
document-42 / revision 3
document-99 / revision 3
```

Поэтому полезны два поля:

```python
DocumentRevision(
    revision_id=UUID(...),
    logical_document_id=UUID(...),
    revision_number=3,
)
```

Где:

```text
revision_id
= глобальный технический идентификатор

revision_number
= понятный порядковый номер внутри одного LogicalDocument
```

Объект можно найти:

```text
по revision_id
```

или по составному ключу:

```text
(logical_document_id, revision_number)
```

Важно не говорить «revision выбранного файла»: revision относится к логическому документу и может включать несколько файлов.

---

## 9. Случайные и детерминированные ID

Главный критерий — не количество объектов.

Нужно спросить:

> Повторное выполнение должно создать новый факт или обратиться к тому же логическому результату?

### Случайные ID

Обычно подходят entity, представляющим отдельные факты:

```text
IngestionPackage ID
SourceFile ID
DocumentRevision ID
```

Два одинаковых поступления всё равно могут быть разными событиями.

```text
same bytes
→ package-1, source-1
→ package-2, source-2
```

### Детерминированные идентификаторы и fingerprints

Полезны для производных артефактов и кэшей.

Например, `ParsedDocument` зависит от:

```text
source checksum
parser name
parser version
parser config
canonical schema version
```

Fingerprint:

```text
hash(
  source_checksum,
  parser_name,
  parser_version,
  parser_config,
  schema_version
)
```

Повторный запуск с теми же входами получает тот же fingerprint и может переиспользовать результат.

Markdown artifact может зависеть от:

```text
parsed_document_fingerprint
markdown_builder_version
builder_config
output_schema_version
```

### Комбинация двух подходов

В реальной модели artifact может иметь:

```text
artifact_id          — случайный технический ID записи
artifact_fingerprint — детерминированный ключ содержимого процесса
```

Это часто удобнее, чем заставлять один ID выполнять обе роли.

---

## 10. Ошибка: fingerprint только из checksum

Неверно:

```text
parsed_document_fingerprint = hash(source_checksum)
```

Почему?

Один и тот же файл может дать разные результаты:

```text
same PDF + Parser v1 → ParsedDocument A
same PDF + Parser v2 → ParsedDocument B
```

Могут измениться:

- таблицы;
- порядок блоков;
- изображения;
- обработка OCR;
- embedded attachments;
- canonical schema.

Если fingerprint одинаковый, система ошибочно решит, что артефакты эквивалентны, и может:

- вернуть старый результат;
- перезаписать новый;
- удалить одну версию при deduplication;
- нарушить воспроизводимость.

---

## 11. Idempotency

Идемпотентная операция при повторном вызове не создаёт новые нежелательные побочные эффекты.

Пример: клиент отправил запрос, сервер создал package, но ответ потерялся. Клиент повторяет запрос.

Без idempotency:

```text
request 1 → package-1
request 2 → package-2
```

Хотя пользователь хотел одну операцию.

### Idempotency Key

Клиент передаёт стабильный ключ:

```text
idempotency_key = request-500
```

Система сохраняет соответствие:

```text
request-500 → package-1
```

### Тот же ключ и тот же payload

Правильное поведение:

```text
найти существующую операцию
→ проверить совпадение payload fingerprint
→ не создавать новый package
→ вернуть или продолжить существующую операцию
```

В зависимости от состояния:

```text
READY
→ вернуть готовый результат

PROCESSING
→ вернуть текущий operation ID и status

FAILED + retry allowed
→ продолжить или повторить допустимый шаг
```

Удалять уже созданные `Package` и `SourceFile` обычно не нужно и опасно.

### Тот же ключ и другой payload

Правильное поведение:

```text
409 Conflict
```

Один idempotency key должен обозначать один неизменный логический запрос.

Если система молча удалит старую операцию и создаст новую, ключ перестаёт выполнять свою функцию.

### Idempotency Key не равен Entity ID

```text
idempotency_key
= identity клиентской операции

package_id
= identity созданного package
```

Один ключ может ссылаться на результат одной созданной entity, но эти значения выполняют разные роли.

---

## 12. Как сравнивать payload

Нельзя полагаться только на raw JSON string: порядок полей или незначащие различия могут дать другой байтовый вид.

Полезно вычислять canonical request fingerprint:

```text
normalize request
→ stable serialization
→ hash
```

Для file upload fingerprint может включать:

```text
file checksum
operation type
external entity refs
relevant options
schema version
```

Если idempotency key уже существует, сравнивается сохранённый request fingerprint.

---

## 13. Scope и срок жизни Idempotency Key

Нужно определить:

- уникален ли ключ глобально или на клиента;
- сколько хранится запись;
- можно ли повторить запрос через год;
- что происходит после expiration;
- входит ли endpoint или operation type в namespace.

Пример composite uniqueness:

```text
(client_id, operation_type, idempotency_key)
```

Без namespace одинаковый ключ разных клиентов может ошибочно конфликтовать.

---

## 14. Частые ошибки и заблуждения

### Ошибка 1. SourceFile ID строится из checksum

Тогда одинаковые файлы из разных поступлений схлопнутся в один entity record, и provenance потеряется.

### Ошибка 2. UUID плох, потому что неинформативен

Читаемость и identity — разные задачи.

### Ошибка 3. Очень много объектов означает, что ID должны быть случайными

Количество не определяет выбор. Важно, создаётся новый факт или повторяется тот же логический результат.

### Ошибка 4. Derived artifacts лучше всегда делать со случайным ID

Тогда retries будут создавать дубликаты, если нет отдельного deterministic fingerprint.

### Ошибка 5. При повторном idempotent request нужно удалить старое и начать заново

Это разрушает гарантию «один ключ — одна операция», создаёт race conditions и теряет историю.

### Ошибка 6. Тот же key с другим payload можно принять как новую версию запроса

Нет. Это конфликт. Для новой операции нужен новый key.

### Ошибка 7. `revision_number` достаточно как ID

Он уникален только внутри parent document.

### Ошибка 8. Storage path можно использовать как primary ID

Миграция хранилища изменит location, но не identity.

---

## 15. Полный пример

Один запрос может создать набор идентификаторов:

```text
idempotency_key        = request-500
package_id             = pkg-100
source_file_id         = src-201
source_checksum        = a8b54...
logical_document_id    = doc-42
revision_id            = rev-301
revision_number        = 3
external_ref           = EXTERNAL_REGISTRY:BUSINESS_OBJECT:ENTITY-123
storage_key            = documents/doc-42/revisions/rev-301/...
parsed_fingerprint     = f91c...
```

Вопросы:

```text
Какой клиентский запрос?         → idempotency_key
Какое поступление?               → package_id
Какой факт получения файла?      → source_file_id
Какие байты?                     → checksum
Какой документ во времени?       → logical_document_id
Какое конкретное состояние?      → revision_id
Какая по счёту версия?           → revision_number
Какая внешняя связь?             → external_ref
Где лежат данные?                → storage_key
Строился ли такой parse result?  → parsed_fingerprint
```

---

## 16. Вопросы на собеседованиях

1. Почему checksum не является хорошим `SourceFile ID`?
2. Когда использовать UUID, а когда deterministic hash?
3. Чем artifact ID отличается от artifact fingerprint?
4. Как реализовать idempotent file upload?
5. Что делать, если key совпадает, а payload отличается?
6. Как выбрать retention для idempotency records?
7. Почему storage path не должен быть domain identity?
8. Какие параметры должны входить в parser cache key?

---

## 17. Практические задачи

### Задача 1

Спроектировать таблицу idempotency records:

```text
client_id
operation_type
idempotency_key
request_fingerprint
resource_id
status
created_at
expires_at
```

Описать unique constraint и поведение конкурентных запросов.

### Задача 2

Составить fingerprint для:

```text
PDF → ParsedDocument → Markdown
```

так, чтобы смена parser-а или builder-а корректно инвалидировала cache.

### Задача 3

Объяснить, какие ID должны измениться, если тот же файл повторно загружен в новом package.

---

## 18. Краткое резюме

```text
Entity ID
= объект или отдельный факт

Checksum
= содержимое

Storage Key
= location

Business Key
= понятный предметной области номер

External ID
= identity в другой системе

Revision Number
= локальный порядковый номер

Idempotency Key
= логическая клиентская операция

Artifact Fingerprint
= функция от всех значимых входов pipeline
```

Самая важная проверка при проектировании идентификатора:

> Что именно должно считаться тем же самым, а что должно оставаться отдельным фактом?
