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

## План трека

1. Требования и общая архитектура ingestion-системы
2. Domain model и границы сущностей
3. Identity и идентификаторы
4. IngestionPackage, SourceFile и embedded attachments
5. LogicalDocument и DocumentRevision
6. Status model и state machines
7. Versioning и lifecycle
8. Artifacts, provenance и lineage
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
- общую теорию;
- применение в document platform;
- альтернативы и компромиссы;
- failure modes;
- частые ошибки;
- вопросы на собеседованиях;
- практические задачи.

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
