# Status Models и State Machines

## 1. Почему одного поля status недостаточно

В сложной системе часто пытаются хранить всё в одном поле:

```python
status: str
```

Туда начинают попадать значения:

```text
PARSING
READY
FAILED
ACTIVE
PUBLISHED
ARCHIVED
```

Проблема в том, что они отвечают на разные вопросы.

```text
PARSING
= какой этап выполняется

FAILED
= чем завершилась попытка

PUBLISHED
= завершилась ли публикация

READY
= можно ли использовать revision

ACTIVE
= каково бизнес-состояние документа
```

Одна система одновременно имеет несколько независимых измерений состояния.

---

## 2. Основные измерения состояния

Для versioned ingestion platform полезно разделить как минимум:

```text
ProcessingAttempt status
Current processing stage
Publication status
Revision readiness
LogicalDocument lifecycle
Active revision pointer
```

Пример полного состояния:

```text
LogicalDocument doc-42
├── lifecycle: ACTIVE
├── active_revision_id: rev-2
│
├── rev-2
│   ├── readiness: READY
│   └── publication: PUBLISHED
│
└── rev-3
    ├── readiness: DRAFT
    ├── publication: PUBLICATION_FAILED
    └── attempt-1
        ├── status: FAILED
        └── stage: PUBLISHING
```

Здесь нет противоречия. Документ активен, старая revision доступна, новая revision ещё не готова, а первая попытка её публикации завершилась ошибкой.

---

## 3. ProcessingAttempt status

Processing attempt отвечает на вопрос:

> Что происходит с конкретным запуском обработки?

Пример набора состояний:

```text
PENDING
RUNNING
RETRYING
SUCCEEDED
FAILED
CANCELLED
```

### Семантика

```text
PENDING
= attempt создана, но ещё не запущена

RUNNING
= worker выполняет работу

RETRYING
= система готовит повтор внутри той же orchestration policy

SUCCEEDED
= attempt завершилась успешно

FAILED
= attempt завершилась ошибкой

CANCELLED
= выполнение остановлено намеренно
```

В простой модели можно не хранить `RETRYING`, а создавать новую attempt со статусом `PENDING`.

---

## 4. Current processing stage

Stage отвечает на вопрос:

> Какую часть pipeline сейчас выполняет attempt?

Пример:

```text
RECEIVING
DISCOVERING
VALIDATING_INPUT
PARSING
NORMALIZING
BUILDING
VALIDATING_ARTIFACTS
PUBLISHING
```

Полное описание attempt:

```python
ProcessingAttempt(
    attempt_id="attempt-2",
    revision_id="rev-3",
    status="RUNNING",
    current_stage="BUILDING",
)
```

Важно:

```text
RUNNING
= состояние выполнения

BUILDING
= текущий этап
```

Нельзя заменять одно другим.

---

## 5. Publication status

Publication status отвечает:

> В каком состоянии находится публикация immutable snapshot?

Возможная модель:

```text
NOT_PUBLISHED
STAGED
PUBLISHED
PUBLICATION_FAILED
```

### Семантика

```text
NOT_PUBLISHED
= публикация ещё не началась

STAGED
= артефакты находятся во временной области

PUBLISHED
= опубликованный snapshot полностью зафиксирован

PUBLICATION_FAILED
= операция публикации завершилась ошибкой
```

`PUBLISHING` и `PUBLISHED` — разные понятия:

```text
PUBLISHING
= processing stage

PUBLISHED
= итог publication state
```

---

## 6. Revision readiness

Readiness отвечает:

> Можно ли использовать revision в serving?

Простая модель:

```text
DRAFT
READY
INVALID
```

Расширенная модель может включать:

```text
READY_WITH_WARNINGS
```

### Семантика

```text
DRAFT
= revision ещё строится или не прошла все проверки

READY
= все обязательные условия выполнены

READY_WITH_WARNINGS
= обязательные условия выполнены, но отсутствуют optional outputs

INVALID
= обнаружена ошибка, из-за которой revision нельзя доверять
```

Технический сбой одной attempt не обязан сразу делать revision `INVALID`.

```text
attempt-1 FAILED
revision DRAFT
```

После успешной attempt та же revision может стать `READY`.

---

## 7. LogicalDocument lifecycle

Business lifecycle отвечает:

> Можно ли использовать сам документ как бизнес-объект?

Пример:

```text
ACTIVE
ARCHIVED
RETIRED
DELETED
```

### Семантика

```text
ACTIVE
= документ используется в обычных процессах

ARCHIVED
= документ сохранён для истории, но не участвует в обычной работе

RETIRED
= использование запрещено по бизнес- или эксплуатационным правилам

DELETED
= документ логически удалён и скрыт от serving
```

Один документ может быть `ACTIVE`, даже если новая revision упала:

```text
LogicalDocument lifecycle = ACTIVE
active_revision_id = rev-2
rev-3 readiness = DRAFT
attempt-1 = FAILED
```

---

## 8. Active revision как вычисляемое состояние

Если `LogicalDocument` хранит:

```text
active_revision_id
```

то `revision.is_active` можно вычислить:

```text
revision.is_active =
logical_document.active_revision_id == revision.revision_id
```

Хранить отдельный mutable flag `is_active` опасно:

```text
active_revision_id = rev-3
rev-2.is_active = true
```

Появляются два источника истины.

Поэтому лучше:

```text
хранить active_revision_id
вычислять is_active
```

---

## 9. State machine

State machine явно определяет:

```text
возможные состояния
разрешённые переходы
запрещённые переходы
guards
side effects
terminal states
```

Пример для attempt:

```text
PENDING
   │
   ▼
RUNNING
 ├──────► SUCCEEDED
 ├──────► FAILED
 └──────► CANCELLED
```

Допустимые переходы можно описать таблицей:

| From | To | Условие |
|---|---|---|
| PENDING | RUNNING | worker получил lease |
| RUNNING | SUCCEEDED | весь запуск завершён |
| RUNNING | FAILED | возникла необработанная ошибка |
| RUNNING | CANCELLED | получен корректный cancel request |

---

## 10. Запрещённые переходы

Например:

```text
SUCCEEDED → RUNNING
```

обычно запрещён.

Успешно завершённая attempt — исторический факт. Для повторной обработки создаётся новая attempt:

```text
attempt-1 SUCCEEDED
attempt-2 PENDING
```

Также нельзя делать:

```text
attempt-1 FAILED → SUCCEEDED
```

после повторного запуска. Это уничтожает историю первой ошибки.

Правильно:

```text
attempt-1 FAILED
attempt-2 SUCCEEDED
```

---

## 11. Guard condition

Guard — условие, которое должно быть истинным для перехода.

Например:

```text
Revision DRAFT → READY
```

может быть разрешён только при выполнении всех условий:

```text
последняя attempt SUCCEEDED
validation passed
все required artifacts существуют
manifest соответствует схеме
checksums совпадают
publication_status = PUBLISHED
```

Псевдокод:

```python
def can_mark_ready(revision) -> bool:
    return (
        revision.latest_attempt.status == "SUCCEEDED"
        and revision.validation_passed
        and revision.required_artifacts_present
        and revision.manifest_valid
        and revision.checksums_valid
        and revision.publication_status == "PUBLISHED"
    )
```

Pipeline мог пройти все функции без exception, но это ещё не означает, что guard выполнен.

---

## 12. Side effects переходов

Переход может сопровождаться действиями.

Например:

```text
DRAFT → READY
```

может требовать:

```text
зафиксировать ready_at
заморозить manifest
запретить изменение artifact inventory
создать событие RevisionReady
```

А переключение active revision:

```text
active_revision_id: rev-2 → rev-3
```

может инициировать:

```text
cache invalidation
search refresh
knowledge compilation
notification
```

Надёжный принцип:

> Сначала зафиксировать state transition, затем доставить side effects через надёжный механизм.

Для этого позже используется Transactional Outbox.

---

## 13. Terminal states

Для attempt терминальными обычно являются:

```text
SUCCEEDED
FAILED
CANCELLED
```

Но `FAILED` attempt не означает окончательный провал всей revision.

```text
rev-3
├── attempt-1 FAILED
└── attempt-2 SUCCEEDED
```

Terminal state локален своей state machine.

---

## 14. Пример ошибки публикации

Исходное состояние:

```text
LogicalDocument doc-42
lifecycle = ACTIVE
active_revision_id = rev-2
```

Новая `rev-3`:

```text
attempt-1.status = RUNNING
attempt-1.stage = PUBLISHING
```

Артефакты находятся в staging, но копирование manifest в published area падает.

Корректное состояние:

```text
attempt-1.status = FAILED
attempt-1.stage = PUBLISHING

rev-3.publication_status = PUBLICATION_FAILED
rev-3.readiness = DRAFT

active_revision_id = rev-2
```

Search и QA продолжают читать `rev-2`.

### Успешный retry

```text
attempt-1 FAILED
attempt-2 PENDING
→ attempt-2 RUNNING
→ attempt-2 SUCCEEDED
```

После успешной публикации:

```text
rev-3.publication_status = PUBLISHED
rev-3.readiness = READY
active_revision_id = rev-3
```

`attempt-2.stage` при завершении может остаться `PUBLISHING` как последний выполненный этап. `PUBLISHED` хранится отдельно как publication status.

---

## 15. Что хранить, а что вычислять

Обычно хранят:

```text
attempt status
current processing stage
revision readiness
publication status
LogicalDocument lifecycle
active_revision_id
```

Обычно вычисляют:

```text
revision is_active
```

Некоторые derived states также можно вычислять:

```text
is_superseded = revision READY and not active
```

Но если `SUPERSEDED` имеет отдельные бизнес-правила, его можно хранить как lifecycle revision.

---

## 16. Отдельные timestamps

Одного `updated_at` недостаточно.

Полезны:

```text
created_at
started_at
last_heartbeat_at
finished_at
failed_at
published_at
ready_at
activated_at
cancelled_at
```

Они помогают:

```text
вычислять длительности
находить зависшие jobs
строить SLA/SLO
проводить аудит
различать retries
```

---

## 17. Heartbeat и зависшие attempts

Состояние `RUNNING` не гарантирует, что worker жив.

Поэтому attempt может хранить:

```text
worker_id
lease_until
last_heartbeat_at
```

Если heartbeat просрочен:

```text
RUNNING
→ FAILED или TIMED_OUT
```

либо создаётся новая attempt после освобождения lease.

Нельзя просто запускать второй worker, не проверив ownership, иначе оба могут одновременно писать в один staging scope.

---

## 18. Ошибки как структурированные данные

Вместо одного текстового поля полезно хранить:

```text
error_code
error_category
error_message
failed_stage
retryable
technical_details_ref
```

Пример:

```python
AttemptError(
    error_code="STORAGE_TIMEOUT",
    category="TRANSIENT",
    failed_stage="PUBLISHING",
    retryable=True,
)
```

Это позволяет автоматически решать:

```text
повторять ли attempt
нужен ли human review
нужно ли quarantine
какую метрику увеличить
```

---

## 19. Универсальный status против нескольких моделей

Плохая модель:

```python
revision.status = "PUBLISHING"
revision.status = "FAILED"
revision.status = "ACTIVE"
```

Хорошая модель:

```python
revision.readiness = "DRAFT"
revision.publication_status = "PUBLICATION_FAILED"

attempt.status = "FAILED"
attempt.current_stage = "PUBLISHING"

document.lifecycle = "ACTIVE"
document.active_revision_id = "rev-2"
```

Каждое поле имеет одну ясную семантику.

---

## 20. Вопросы для самопроверки

1. Почему `BUILDING`, `FAILED`, `READY` и `ACTIVE` нельзя хранить в одной enum?
2. Чем status attempt отличается от current stage?
3. Почему `PUBLISHING` и `PUBLISHED` относятся к разным измерениям?
4. Какие guards нужны перед `DRAFT → READY`?
5. Почему retry не должен переписывать `FAILED` attempt?
6. Какие состояния лучше вычислять, а не хранить?
7. Как определить зависшую attempt?

## 21. Краткое резюме

```text
attempt status
= как завершился конкретный запуск

current stage
= что запуск делает сейчас

publication status
= опубликован ли snapshot

revision readiness
= можно ли использовать revision

document lifecycle
= каково бизнес-состояние документа
```

```text
state machine
= states + transitions + guards + side effects
```

```text
attempt-1 FAILED
attempt-2 SUCCEEDED
revision READY
```

История attempts сохраняется, а serving переключается только на полностью опубликованную revision.
