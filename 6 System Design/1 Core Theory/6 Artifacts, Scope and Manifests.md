# Artifacts, Scope и Manifests

## 1. Что такое Artifact

`Artifact` — производный результат обработки входных данных.

Примеры:

```text
parsed_document.json
document.md
tables.json
chunks.jsonl
summary.json
entities.json
search index
vector snapshot
manifest.json
```

Artifact не обязательно является одним обычным файлом. Это может быть:

```text
JSON document
binary object
directory
set of files
index snapshot
serialized model
record in specialized storage
```

Главная идея:

```text
SourceFile
= вход системы

Artifact
= результат преобразования
```

---

## 2. SourceFile и Artifact

Пример pipeline:

```text
report.docx                SourceFile
    │
    ▼
parsed_document.json       Artifact
    │
    ▼
document.md                Artifact
    │
    ▼
chunks.jsonl               Artifact
    │
    ▼
search_index/              Artifact
```

Файл, извлечённый из ZIP или embedded attachment, остаётся `SourceFile`, потому что он является частью входного материала.

```text
package.zip
└── report.docx
    └── embedded.xlsx
```

Здесь `embedded.xlsx` — `SourceFile`, а построенный из него `parsed_document.json` — `Artifact`.

---

## 3. Artifact scope

Scope отвечает на вопрос:

> Какой доменной сущности принадлежит artifact?

Возможные scopes:

```text
INGESTION_PACKAGE
SOURCE_FILE
DOCUMENT_REVISION
LOGICAL_DOCUMENT
```

Scope — это не просто имя процесса и не storage path. Это ссылка на владельца результата.

Пример модели:

```python
Artifact(
    artifact_id="art-501",
    artifact_type="PARSED_DOCUMENT",
    scope_type="SOURCE_FILE",
    scope_id="src-10",
)
```

### SourceFile-scoped artifacts

Артефакт строится из одного физического source:

```text
src-1 main.docx
→ parsed_document/main.json

src-2 metrics.xlsx
→ parsed_document/metrics.json
```

Оба `ParsedDocument` имеют scope `SOURCE_FILE`.

### DocumentRevision-scoped artifacts

Артефакт агрегирует данные всей версии документа:

```text
main ParsedDocument
metrics ParsedDocument
appendix ParsedDocument
        │
        ▼
    document.md
```

Тогда:

```python
Artifact(
    artifact_type="DOCUMENT_MARKDOWN",
    scope_type="DOCUMENT_REVISION",
    scope_id="rev-5",
)
```

К revision-scoped artifacts часто относятся:

```text
document.md
tables.json
chunks.jsonl
summary.json
revision manifest
search snapshot
```

---

## 4. Почему scope важен

Без scope невозможно надёжно ответить:

```text
какому source или revision принадлежит artifact
можно ли его переиспользовать
что нужно пересобрать
какие данные удалить
что включить в manifest
какие downstream outputs устарели
```

Плохая связь:

```text
artifact belongs to path /some/folder/rev-5/
```

Путь может измениться при миграции storage.

Правильная связь:

```text
scope_type = DOCUMENT_REVISION
scope_id = rev-5
```

Storage key хранится отдельно.

---

## 5. Metadata артефакта

Пример:

```python
Artifact(
    artifact_id="art-501",
    artifact_type="DOCUMENT_MARKDOWN",

    scope_type="DOCUMENT_REVISION",
    scope_id="rev-5",

    storage_key="documents/doc-42/revisions/rev-5/document.md",
    checksum_sha256="...",
    size_bytes=185420,
    media_type="text/markdown",

    builder_name="markdown_builder",
    builder_version="2.1.0",
    builder_config_fingerprint="...",

    input_artifact_ids=["art-101", "art-102"],
    schema_version="1.3",
    created_at="...",
)
```

Полезные группы metadata:

```text
Identity
Scope
Storage location
Content integrity
Builder provenance
Input lineage
Schema
Timestamps
```

---

## 6. Identity и fingerprint артефакта

Можно разделять:

```text
artifact_id
= identity metadata-записи

artifact_fingerprint
= deterministic identity результата вычисления
```

Fingerprint должен учитывать всё, что влияет на output:

```text
input checksums or fingerprints
builder name
builder version
builder configuration
output schema version
artifact type
```

Пример:

```text
hash(
  parsed_document_fingerprint,
  markdown_builder_version,
  builder_config,
  output_schema_version
)
```

Fingerprint только из source checksum недостаточен: новый parser или builder может создать другой output из тех же байтов.

---

## 7. Builder provenance

Для воспроизводимости сохраняются:

```text
builder name
builder version
configuration fingerprint
input references
output checksum
execution metadata
```

Одинаковые inputs могут дать разные результаты:

```text
ParsedDocument
+ MarkdownBuilder 1.0
→ document.md A
```

```text
ParsedDocument
+ MarkdownBuilder 2.0
→ document.md B
```

Также одна версия builder-а может работать с разными параметрами:

```text
include_tables = true
include_images = false
heading_style = compact
```

Без конфигурации версия компонента не обеспечивает воспроизводимость.

---

## 8. Artifact inventory

Artifact inventory — рабочий список артефактов, связанных со scope.

Во время обработки он может изменяться:

```text
rev-5 inventory
├── parsed/main.json
├── parsed/metrics.json
├── document.md
└── tables.json
```

После следующего builder-а:

```text
rev-5 inventory
├── parsed/main.json
├── parsed/metrics.json
├── document.md
├── tables.json
└── chunks.jsonl
```

Inventory обычно используется во время building и staging.

Он отвечает:

> Какие artifact records сейчас зарегистрированы для этого scope?

---

## 9. Manifest

Manifest — immutable snapshot опубликованного набора.

Он фиксирует:

```text
revision identity
source membership
source checksums
artifact inventory
artifact checksums
artifact sizes
storage references
pipeline versions
configuration fingerprints
schema versions
publication metadata
```

Упрощённый пример:

```json
{
  "revision_id": "rev-5",
  "source_files": [
    {
      "source_file_id": "src-1",
      "checksum": "AAA"
    },
    {
      "source_file_id": "src-2",
      "checksum": "BBB"
    }
  ],
  "artifacts": [
    {
      "artifact_type": "DOCUMENT_MARKDOWN",
      "storage_key": "documents/doc-42/revisions/rev-5/document.md",
      "checksum": "CCC",
      "size_bytes": 185420,
      "required": true,
      "status": "AVAILABLE"
    }
  ],
  "pipeline": {
    "parser_version": "2.1.0",
    "builder_version": "1.4.0"
  }
}
```

---

## 10. Inventory и manifest

```text
Artifact inventory
= mutable рабочий список зарегистрированных outputs
```

```text
Manifest
= immutable опубликованный snapshot
```

Типичный lifecycle:

```text
mutable staging inventory
→ validation
→ manifest generation
→ immutable publication
```

Просмотр каталога storage не заменяет manifest, потому что там могут быть:

```text
temporary files
outputs старой attempt
частично записанные files
лишние debug artifacts
невалидированные outputs
```

---

## 11. Required и optional artifacts

Artifacts могут иметь разную обязательность.

Пример:

```text
REQUIRED
├── parsed_document for every required source
├── document.md
├── tables.json
└── manifest.json

OPTIONAL
└── summary.json
```

Если отсутствует required artifact:

```text
revision cannot become READY
```

Если отсутствует optional artifact, политика может разрешать:

```text
READY
```

или:

```text
READY_WITH_WARNINGS
```

Выбор зависит от product contract.

---

## 12. Как manifest описывает optional failure

Представим, все required artifacts построены, но `summary.json` не создан.

Manifest должен явно фиксировать состояние, а не просто молча исключать artifact.

Пример:

```json
{
  "artifact_type": "SUMMARY",
  "required": false,
  "status": "NOT_BUILT",
  "reason_code": "OPTIONAL_BUILDER_FAILED"
}
```

или:

```json
{
  "optional_artifacts": {
    "summary": {
      "status": "UNAVAILABLE",
      "reason": "builder timeout"
    }
  }
}
```

Это позволяет отличить:

```text
artifact не предусмотрен
artifact отключён конфигурацией
artifact не построился
artifact построился и опубликован
```

Revision может стать `READY_WITH_WARNINGS`, если contract разрешает отсутствие summary.

---

## 13. READY gate

Переход в `READY` или `READY_WITH_WARNINGS` разрешён только после проверки:

```text
все required SourceFile зарегистрированы
все required parsed artifacts построены
все required revision artifacts построены
validation passed
checksums match
manifest valid
publication_status = PUBLISHED
optional failures описаны явно
```

Псевдокод:

```python
def evaluate_readiness(revision):
    if not revision.required_artifacts_present:
        return "DRAFT"
    if not revision.validation_passed:
        return "DRAFT"
    if revision.publication_status != "PUBLISHED":
        return "DRAFT"
    if revision.optional_artifacts_failed:
        return "READY_WITH_WARNINGS"
    return "READY"
```

---

## 14. Почему существования файла недостаточно

Проверка:

```python
path.exists()
```

не доказывает, что artifact готов.

Файл может быть:

```text
нулевого размера
частично записан
повреждён
не соответствовать schema
иметь неверный checksum
принадлежать старой attempt
быть создан другим builder version
лежать в staging
не входить в manifest
```

Минимальная проверка включает:

```text
metadata record exists
size is valid
checksum matches
schema validation passes
scope matches
builder provenance matches
artifact included in manifest
publication completed
```

---

## 15. Staging и published area

Артефакты сначала создаются во временной области:

```text
staging/rev-5/attempt-1/
```

Затем:

```text
build
→ validate
→ generate manifest
→ publish
```

Serving читает только:

```text
published/doc-42/rev-5/
```

Search и QA не должны читать staging, потому что данные там могут быть неполными и изменяемыми.

---

## 16. Immutable artifacts после publication

После publication нельзя перезаписывать `document.md` по тому же ключу.

Представим manifest:

```text
document.md checksum = AAA
```

Кто-то заменил файл, и новый checksum стал `BBB`.

При чтении или periodic validation система должна обнаружить mismatch:

```text
expected checksum AAA
actual checksum BBB
→ integrity error
```

В зависимости от политики:

```text
artifact marked corrupted
revision marked INVALID
serving blocked
alert created
rollback initiated
```

Fingerprint помогает определить ожидаемую версию вычисления, а checksum — фактическое содержимое сохранённого output.

---

## 17. Rebuild новым builder version

Опубликованная `rev-5` содержит:

```text
document.md built by MarkdownBuilder 1.4
```

Появился Builder 2.0.

В строгой revision model:

```text
1. create rev-6
2. reuse immutable source blobs
3. build new artifacts with Builder 2.0
4. validate outputs
5. publish new manifest
6. activate rev-6
```

`rev-5/document.md` остаётся неизменным.

Это позволяет:

```text
сравнивать outputs
делать rollback
объяснять старые QA answers
сохранять audit trail
```

---

## 18. Artifact lineage

Для каждого output полезно знать inputs.

Пример:

```text
src-1 main.docx
→ parsed-main.json

src-2 metrics.xlsx
→ parsed-metrics.json

parsed-main.json + parsed-metrics.json
→ document.md

document.md
→ chunks.jsonl
```

Lineage record может содержать:

```text
input_artifact_id
output_artifact_id
operation_type
component_name
component_version
configuration_fingerprint
attempt_id
created_at
```

Это позволяет понять:

```text
что пересобрать после parser update
какие outputs зависят от corrupted artifact
какой input породил ошибочную таблицу
какие indexes используют старую revision
```

---

## 19. Artifact status

Отдельный artifact может иметь собственное техническое состояние:

```text
PLANNED
BUILDING
BUILT
VALIDATED
PUBLISHED
FAILED
CORRUPTED
```

Но нужно избегать дублирования state machines без необходимости.

Например, если inventory record появляется только после успешного build, отдельный `BUILT` может быть не нужен.

Состояния проектируются под реальные операции и recovery, а не для максимального количества enum.

---

## 20. Artifact keys и identity

Storage key отвечает:

```text
где лежит artifact
```

Scope отвечает:

```text
кому принадлежит artifact
```

Fingerprint отвечает:

```text
какой вычислительный результат это представляет
```

Checksum отвечает:

```text
какие байты фактически сохранены
```

```text
artifact_id     = какая metadata entity
scope           = кому принадлежит
storage_key     = где лежит
fingerprint     = каким вычислением получен
checksum        = что записано
```

---

## 21. Пример rev-5

Исходные файлы:

```text
src-1 main.docx
src-2 metrics.xlsx
```

Требуемые outputs:

```text
parsed_document for src-1      REQUIRED
parsed_document for src-2      REQUIRED
document.md for rev-5          REQUIRED
tables.json for rev-5          REQUIRED
summary.json for rev-5         OPTIONAL
manifest.json for rev-5        REQUIRED
```

Scopes:

```text
parsed_document src-1 → SOURCE_FILE/src-1
parsed_document src-2 → SOURCE_FILE/src-2

document.md            → DOCUMENT_REVISION/rev-5
tables.json             → DOCUMENT_REVISION/rev-5
summary.json            → DOCUMENT_REVISION/rev-5
manifest.json           → DOCUMENT_REVISION/rev-5
```

Если summary не построился:

```text
required outputs complete
summary optional failure recorded
publication successful
→ READY_WITH_WARNINGS
```

Допустим и `READY`, если внешний contract не различает warnings, но информация об optional failure всё равно должна сохраниться.

---

## 22. Важные различия

```text
SourceFile
= входная физическая единица

Artifact
= производный output
```

```text
Scope
= доменный владелец artifact

Storage key
= физическое расположение
```

```text
Inventory
= рабочий mutable список

Manifest
= immutable published snapshot
```

```text
Fingerprint
= identity вычисления

Checksum
= integrity фактических bytes
```

---

## 23. Вопросы для самопроверки

1. Почему embedded XLSX является SourceFile, а ParsedDocument — Artifact?
2. Какие outputs должны иметь scope SOURCE_FILE?
3. Какие outputs обычно принадлежат DOCUMENT_REVISION?
4. Почему scope нельзя выводить только из storage path?
5. Чем inventory отличается от manifest?
6. Как manifest должен описывать optional artifact, который не построился?
7. Почему `exists()` недостаточно для readiness?
8. Что происходит при checksum mismatch опубликованного artifact?
9. Как пересобрать Markdown новой версией builder-а в strict revision model?

## 24. Краткое резюме

```text
Artifact
= versioned, scoped and reproducible output
```

```text
SOURCE_FILE scope
→ per-file ParsedDocument

DOCUMENT_REVISION scope
→ combined Markdown, tables, chunks, summary, manifest
```

```text
inventory
→ validate
→ immutable manifest
→ publish
→ READY or READY_WITH_WARNINGS
```

```text
published artifacts are immutable
new builder output creates a new revision
```
