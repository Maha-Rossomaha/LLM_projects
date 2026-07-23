# Versioned Document Ingestion Platform

Этот прикладной трек показывает, как фундаментальные темы System Design соединяются в одной системе обработки документов.

## Синтетическая постановка

Компания получает аналитические документы в форматах DOCX, XLSX, HTML и PDF.

Система должна:

- принимать один файл или пакет;
- обнаруживать архивные и embedded attachments;
- сохранять оригинальные байты;
- строить canonical representation;
- объединять несколько файлов в один LogicalDocument;
- создавать immutable DocumentRevision;
- строить производные artifacts;
- публиковать их через staging и validation gate;
- разрешать Serving только по опубликованным READY revisions;
- связывать документы с объектами External Registry;
- в дальнейшем строить Search, RAG и LLM Wiki.

## Карта архитектуры

```text
External Source
→ Ingestion Entry Point
→ Package and Source Registration
→ Discovery and Security Validation
→ Parsers
→ Canonical Documents
→ Artifact Builders
→ Staging
→ Validation
→ Manifest
→ Atomic Publication
→ READY Revision
→ Search / QA / Knowledge Compilation
```

## Уже разобранные темы

### 1. Общая архитектура

[System Design, Domain Modeling и Data Architecture](../../1%20Core%20Theory/1%20System%20Design,%20Domain%20Modeling%20and%20Data%20Architecture.md)

Разделяет:

```text
Domain Model
Data Architecture
System Architecture
Data Flow
Control Flow
Pipeline
State Machine
```

### 2. Domain model и границы сущностей

[Domain Modeling и границы сущностей](../../1%20Core%20Theory/2%20Domain%20Modeling%20and%20Entity%20Boundaries.md)

Основная формула:

```text
IngestionPackage
≠ SourceFile
≠ ParsedDocument
≠ LogicalDocument
≠ DocumentRevision
≠ ExternalEntityRef
≠ Artifact
```

### 3. Identity и идентификаторы

[Identity, Identifiers и Idempotency](../../1%20Core%20Theory/3%20Identity,%20Identifiers%20and%20Idempotency.md)

Разбираются:

```text
entity ID
business key
external ID
checksum
storage key
revision number
artifact fingerprint
idempotency key
```

### 4. Package, SourceFile и вложения

[IngestionPackage, SourceFile, вложения и provenance](1%20Packages,%20Source%20Files%20and%20Provenance.md)

Разбираются:

```text
archive members
embedded attachments
parent relations
provenance
lineage
blob deduplication
safe extraction
zip bomb limits
```

### 5. LogicalDocument и DocumentRevision

[LogicalDocument и DocumentRevision](2%20Logical%20Documents%20and%20Revisions.md)

Разбираются:

```text
stable document identity
immutable revisions
ProcessingAttempt
atomic activation
active_revision_id
concurrent revision numbering
```

### 6. Status Models и State Machines

[Status Models и State Machines](../../1%20Core%20Theory/4%20Status%20Models%20and%20State%20Machines.md)

Раздельно моделируются:

```text
attempt status
current stage
publication status
revision readiness
document lifecycle
active revision
```

### 7. Versioning и lifecycle

[Versioning, Lifecycle и Immutable Revisions](../../1%20Core%20Theory/5%20Versioning,%20Lifecycle%20and%20Immutable%20Revisions.md)

Разбираются:

```text
new attempt vs new revision
source fingerprint
processing fingerprint
SUPERSEDED / RETIRED / INVALID
rollback
logical and physical deletion
shared blob garbage collection
```

### 8. Artifacts, scope и manifest

[Artifacts, Scope и Manifests](../../1%20Core%20Theory/6%20Artifacts,%20Scope%20and%20Manifests.md)

Разбираются:

```text
SOURCE_FILE and DOCUMENT_REVISION scopes
artifact metadata
builder provenance
inventory vs manifest
required and optional artifacts
READY_WITH_WARNINGS
checksum and fingerprint validation
immutable publication
```

## Полный план трека

1. Требования и общая архитектура ingestion-системы
2. Domain model и границы сущностей
3. Identity и идентификаторы
4. IngestionPackage, SourceFile и embedded attachments
5. LogicalDocument и DocumentRevision
6. Status model и state machines
7. Versioning и lifecycle
8. Artifacts, scope, provenance и manifest
9. Storage abstraction
10. Storage keys и artifact layout
11. Atomic writes, staging и publication
12. Manifests и READY gate
13. Repository и artifact-service patterns
14. Idempotency, retries и failure recovery
15. SAGA, compensation, Outbox и reconciliation
16. Automatic ingestion orchestrator
17. Serving и QA по опубликованным данным
18. Knowledge layer, LLM Wiki и incremental compilation

## Связь с Core Theory

Каждый прикладной конспект должен ссылаться на общую теорию, а не дублировать её полностью.

Пример:

```text
Core Theory: Identity, Identifiers and Idempotency
        │
        ▼
Applied Architecture: Package and Source identity
```

## Формат

Каждая тема содержит:

- простую интуицию;
- определения и основную модель;
- применение в document platform;
- альтернативы и компромиссы;
- failure modes;
- важные различия;
- вопросы для самопроверки;
- краткое резюме.

## Публичность и анонимизация

Все примеры являются синтетическими. Используются нейтральные названия:

```text
External Registry
ExternalEntityRef
LogicalDocument
DocumentRevision
ENTITY-123
```

В трек не включаются реальные внутренние системы, API, роли, storage paths, идентификаторы и бизнес-процессы.
