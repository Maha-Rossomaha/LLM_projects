# IngestionPackage, SourceFile, вложения и provenance

## 1. Интуиция

Система обработки документов редко получает только один простой файл.

Вход может выглядеть так:

```text
package.zip
├── report.docx
│   ├── embedded.xlsx
│   └── embedded.pdf
├── appendix.pdf
└── nested.zip
    └── old_report.docx
```

Недостаточно просто распаковать архив и передать найденные файлы parser-ам. Надёжная система должна сохранять:

- что поступило вместе;
- точные исходные байты;
- где был обнаружен каждый файл;
- кто является его родителем;
- каким способом он извлечён;
- сколько уровней вложенности пройдено;
- не нарушены ли security limits;
- как из source-файла появились производные артефакты.

Главные понятия:

```text
IngestionPackage
SourceFile
Source Provenance
Lineage
Binary Blob
Artifact
```

---

## 2. IngestionPackage как граница поступления

`IngestionPackage` представляет одну внешнюю операцию передачи данных.

Например:

```text
External Request request-500
          │
          ▼
IngestionPackage pkg-100
```

Пакет отвечает на вопросы:

- какой запрос инициировал поступление;
- когда данные были приняты;
- от какого клиента или канала они пришли;
- какие корневые и вложенные файлы обнаружены;
- завершилась ли обработка всего поступления;
- какие ошибки относятся к этому поступлению.

### Пакет не равен логическому документу

Один package может содержать:

- несколько документов;
- приложения;
- изображения;
- служебные metadata-файлы;
- неподдерживаемые вложения;
- старые версии документов.

И наоборот, один логический документ может собираться из нескольких поступлений.

Поэтому:

```text
IngestionPackage
= техническая и операционная граница поступления

LogicalDocument
= бизнес-identity документа во времени
```

---

## 3. SourceFile как факт обнаружения файла

`SourceFile` — конкретный зарегистрированный файл внутри package.

Пример:

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

SourceFile хранит:

- internal ID;
- package ID;
- original name;
- media type;
- size;
- checksum;
- original artifact key;
- source kind;
- parent source ID;
- discovery method;
- discovery path;
- depth.

### Один blob и несколько SourceFile

Представим два одинаковых файла:

```text
report.docx / embedded appendix.pdf
package root / copy_of_appendix.pdf
```

Их checksum совпадает. Physical storage может хранить один deduplicated blob.

Но записи остаются разными:

```text
src-10 parent = report.docx
src-11 parent = package.zip
```

Почему?

Потому что provenance различается. Один и тот же контент встретился в двух разных местах и может иметь разную роль.

---

## 4. Корневые и вложенные файлы

Рассмотрим:

```text
upload.zip
├── report.docx
│   └── embedded.xlsx
└── appendix.pdf
```

Если сам архив тоже сохраняется как SourceFile:

```text
src-1 upload.zip
├── src-2 report.docx
│   └── src-3 embedded.xlsx
└── src-4 appendix.pdf
```

Пример metadata для вложения:

```python
SourceFile(
    source_file_id="src-3",
    package_id="pkg-1",
    parent_source_file_id="src-2",
    source_kind="EMBEDDED_ATTACHMENT",
    discovery_method="DOCX_EMBEDDED_ATTACHMENT",
    discovery_path="word/embeddings/object1.xlsx",
    original_name="embedded.xlsx",
    depth=2,
)
```

### Source kinds

Полезная классификация:

```text
UPLOADED
ARCHIVE_MEMBER
EMBEDDED_ATTACHMENT
```

`GENERATED` обычно не должен быть SourceFile: результат работы системы является Artifact.

Пример:

```text
document.md
```

— это не вход, а производный результат.

---

## 5. Нужно ли считать архив SourceFile

Есть два допустимых подхода.

### Подход A. Архив — SourceFile

```text
src-1 package.zip
├── src-2 report.docx
└── src-3 appendix.pdf
```

Преимущества:

- сохраняются точные исходные байты;
- можно повторить extraction другой версией extractor-а;
- сохраняется checksum транспорта;
- проще аудит;
- легко доказать, что именно загрузил клиент.

### Подход B. Архив — transport envelope

В package metadata хранится оригинальный upload artifact, а SourceFile создаются только для содержимого.

Это тоже возможно, если оригинальный архив не теряется.

Ключевое правило:

> Независимо от модели исходные байты поступления должны сохраняться до необратимой обработки.

---

## 6. Provenance

Provenance отвечает на вопрос:

> Откуда появился этот объект?

Для embedded file:

```text
package pkg-1
→ source report.docx
→ embedded attachment
→ internal path word/embeddings/object1.xlsx
```

Пример value object:

```python
SourceProvenance(
    package_id="pkg-1",
    parent_source_file_id="src-2",
    discovery_method="EMBEDDED_ATTACHMENT",
    discovery_path="word/embeddings/object1.xlsx",
    extractor_name="docx_attachment_extractor",
    extractor_version="1.3.0",
)
```

Provenance нужна для:

- воспроизводимости;
- аудита;
- объяснения происхождения данных;
- повторной обработки;
- расследования ошибок;
- корректного deduplication;
- связи артефактов с evidence.

Без provenance файл `embedded.xlsx` существует, но система не может объяснить, из какого документа и каким процессом он получен.

---

## 7. Lineage

Lineage описывает более широкую цепочку преобразований:

```text
SourceFile report.docx
→ ParsedDocument
→ Markdown
→ Chunks
→ Dense Index
→ Search Result
→ Generated Answer
```

Короткое различие:

```text
Provenance
= откуда появился конкретный объект

Lineage
= через какие преобразования прошли данные
```

Они пересекаются, но используются на разных уровнях.

### Lineage record

Можно хранить relation:

```text
input_artifact_id
output_artifact_id
operation_type
component_name
component_version
configuration_fingerprint
run_id
created_at
```

Тогда можно ответить:

- из каких sources построен index;
- какой parser создал canonical document;
- какие ответы зависят от устаревшей revision;
- что нужно пересобрать после обновления builder-а.

---

## 8. Порядок надёжной обработки

Безопасный порядок:

```text
1. Получить входной stream
2. Применить входные size limits
3. Вычислить checksum
4. Сохранить оригинальные байты
5. Зарегистрировать package/source metadata
6. Определить тип и extractor
7. Безопасно извлечь дочерние файлы
8. Сохранить каждого ребёнка как SourceFile
9. Провести format/security validation
10. Передать допустимые sources parser-ам
```

Почему оригинал сохраняется до parsing?

Если parser падает, система всё равно имеет точный вход и может:

- повторить обработку;
- обновить parser;
- провести forensic analysis;
- доказать неизменность источника.

---

## 9. Безопасное извлечение архивов

Наивный код:

```python
zipfile.extractall(target_dir)
```

опасен без проверки путей и лимитов.

### Path Traversal / Zip Slip

Архив может содержать entry:

```text
../../outside/config
```

После нормализации target path должен оставаться внутри staging directory.

Общая проверка:

```text
resolved_output_path starts with resolved_staging_root
```

Нельзя доверять имени entry до canonical path validation.

### Symbolic links

Архив может содержать symlink, ведущий за пределы staging. Нужно либо запрещать symlinks, либо строго проверять final target.

### Absolute paths

Entries с абсолютными путями должны отклоняться.

---

## 10. Zip Bomb и ресурсные ограничения

Маленький архив может распаковаться в огромный объём.

Пример:

```text
compressed size   = 10 MB
uncompressed size = 500 GB
```

Нужны ограничения:

```text
max compressed package size
max uncompressed total size
max single extracted file size
max file count
max nesting depth
max compression ratio
max extraction time
allowed formats
```

Дополнительно полезны:

- CPU quota;
- memory limit;
- disk quota;
- cancellation;
- sandbox;
- antivirus или content scanning;
- timeout на extractor.

Конкретные значения зависят от требований, но сами ограничения должны существовать явно.

---

## 11. Рекурсивное обнаружение и циклы

Возможны ситуации:

```text
A.zip содержит B.zip
B.zip содержит копию A.zip
```

Или один и тот же blob встречается во многих местах.

Защита:

- max depth;
- max total source count;
- max total extracted bytes;
- visited extraction contexts;
- timeout;
- cancellation token.

Checksum помогает обнаруживать повторный контент, но не должен автоматически запрещать новую запись SourceFile.

Почему?

```text
same checksum + different parent/discovery path
= тот же контент, но другой факт обнаружения
```

Можно пропустить повторное физическое extraction или reuse blob, но provenance record нужно сохранить.

---

## 12. Required, Optional, Ignored и Quarantined inputs

Не все обнаруженные файлы имеют одинаковую роль.

Пример:

```text
main.docx       REQUIRED
metrics.xlsx    REQUIRED
logo.png        OPTIONAL
notes.txt       IGNORED
unknown.exe     QUARANTINED
```

Возможные категории:

```text
REQUIRED
OPTIONAL
UNSUPPORTED
IGNORED
QUARANTINED
```

Package readiness не обязательно означает успешный parse каждого байта.

Правило может быть таким:

> Все required sources успешно обработаны, а optional, unsupported и quarantined sources явно учтены в manifest.

Это позволяет отличить:

```text
READY
READY_WITH_WARNINGS
FAILED
QUARANTINED
```

Точная status model рассматривается отдельно.

---

## 13. Deduplication: три разных уровня

### 1. Request deduplication

Одинаковый idempotency key не должен создавать второй package.

### 2. Source content deduplication

Одинаковые file bytes могут храниться как один binary blob.

### 3. Processing deduplication

Одинаковый source content с тем же parser pipeline может переиспользовать ParsedDocument.

Эти уровни нельзя смешивать.

Пример:

```text
2 SourceFile records
1 binary blob
1 ParsedDocument cache entry
```

Это корректная модель.

---

## 14. Полный пример

Вход:

```text
package.zip
├── report.docx
│   ├── metrics.xlsx
│   └── appendix.pdf
├── copy_of_appendix.pdf
└── nested.zip
    └── old_report.docx
```

Если архивы также регистрируются как sources:

```text
src-1 package.zip
├── src-2 report.docx
│   ├── src-3 metrics.xlsx
│   └── src-4 appendix.pdf
├── src-5 copy_of_appendix.pdf
└── src-6 nested.zip
    └── src-7 old_report.docx
```

Всего:

```text
7 SourceFile records
```

Если `appendix.pdf` и `copy_of_appendix.pdf` одинаковы, unique blobs:

```text
6
```

при условии, что остальные файлы различаются.

Но `src-4` и `src-5` не объединяются в одну domain record, потому что имеют разных родителей и discovery paths.

---

## 15. Частые ошибки и заблуждения

### Ошибка 1. После распаковки архив можно удалить

Теряются точные исходные байты и возможность воспроизвести extraction.

### Ошибка 2. Одинаковый checksum означает один SourceFile

Checksum описывает content, SourceFile — факт обнаружения.

### Ошибка 3. Embedded attachment является artifact

Если вложение было частью входного документа, это source data. Artifact создаёт наша система.

### Ошибка 4. Generated Markdown нужно хранить как SourceFile

Markdown — output builder-а, поэтому это Artifact.

### Ошибка 5. Можно сначала parse, а потом сохранять оригинал

При падении parser-а точный input может потеряться.

### Ошибка 6. Достаточно ограничить compressed size

Zip bomb определяется объёмом после распаковки, file count, ratio и nesting.

### Ошибка 7. Дедупликация должна схлопывать parent relations

Physical deduplication не должна уничтожать provenance.

### Ошибка 8. Один package всегда соответствует одному document revision

Это упрощение, а не универсальное правило.

---

## 16. Вопросы на собеседованиях

1. Почему SourceFile и binary blob нужно моделировать отдельно?
2. Как безопасно распаковать пользовательский ZIP?
3. Какие limits защищают от Zip Bomb?
4. Почему same checksum не позволяет удалить вторую source record?
5. Чем provenance отличается от lineage?
6. Нужно ли хранить оригинальный архив?
7. Как избежать повторного parsing неизменившегося embedded file?
8. Как моделировать nested attachments?
9. Когда package может быть READY_WITH_WARNINGS?

---

## 17. Практические задачи

### Задача 1

Для дерева вложений определить:

- количество SourceFile records;
- unique blobs;
- parent relations;
- discovery methods;
- depth каждого source.

### Задача 2

Спроектировать безопасный archive extractor contract:

```python
extract(
    source,
    limits,
    staging,
) -> list[DiscoveredSource]
```

Описать ошибки и partial results.

### Задача 3

Предложить schema для provenance и lineage, позволяющую найти все артефакты, зависящие от конкретного source checksum.

---

## 18. Краткое резюме

```text
IngestionPackage
= одно внешнее поступление

SourceFile
= отдельный факт обнаружения файла

Binary Blob
= физически сохранённые байты

Provenance
= происхождение конкретного source/artifact

Lineage
= цепочка преобразований данных

Artifact
= производный результат обработки
```

Надёжный ingestion сохраняет не только контент, но и контекст его появления. Deduplication должна экономить storage и compute, но не уничтожать identity, provenance и audit history.
