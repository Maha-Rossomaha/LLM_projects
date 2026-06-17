# Pod, Deployment and Lifecycle

## 1. Pod — углублённо

### 1.1 Анатомия Pod

**Pod** — минимальная единица планирования в Kubernetes. Это группа из одного или нескольких контейнеров, которые:

* запускаются **вместе** на одной Node,
* разделяют **network namespace** (один IP, общий `localhost`),
* могут разделять **IPC namespace** (shared memory),
* могут разделять **volumes**,
* имеют **общий lifecycle** (создаются и умирают вместе).

```
Pod
├── infra-контейнер (pause) — создаёт network namespace, держит его,
│                            пока живы остальные
├── app-контейнер 1
├── app-контейнер 2
├── ...
├── init-контейнеры (опционально)
└── ephemeral-контейнеры (отладка)
```

**infra-контейнер** (`pause`) — технический контейнер, который запускается первым и держит network namespace. Прикладные контейнеры подключаются к нему. Появляется и исчезает прозрачно.

### 1.2 Pod vs Container: что где задавать

| Что | Где задаётся |
|-----|-------------|
| `image`, `command`, `args`, `env` | На уровне контейнера |
| `resources.requests/limits` | На уровне контейнера |
| `ports` | На уровне контейнера |
| `volumes` | На уровне Pod (монтируются в контейнеры через `volumeMounts`) |
| `restartPolicy` | На уровне Pod |
| `serviceAccountName` | На уровне Pod |
| `affinity`, `tolerations`, `nodeSelector` | На уровне Pod |

### 1.3 Multi-container паттерны

Когда в одном Pod несколько контейнеров, это всегда **тесная связка** — они не могут жить на разных нодах.

**Sidecar:**
* Основной контейнер делает работу.
* Sidecar добавляет функциональность: прокси (Envoy), сбор логов (Fluent Bit), обновление конфигов.
* Пример: `app` + `log-shipper`, которые разделяют `emptyDir` volume.

```yaml
spec:
  containers:
  - name: app
    image: my-app:v1
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
  - name: log-shipper
    image: fluent-bit:latest
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
      readOnly: true
  volumes:
  - name: logs
    emptyDir: {}
```

**Ambassador:**
* Прокси-контейнер, который «представляет» внешние сервисы локально.
* Пример: основной контейнер ходит на `localhost:6379`, ambassador проксирует на внешний Redis c TLS/аутентификацией.

**Adapter:**
* Нормализует вывод основного контейнера под общий стандарт.
* Пример: адаптер читает логи приложения и преобразует в структурированный JSON для централизованного мониторинга.

---

## 2. Pod Lifecycle

### 2.1 Фазы (phase) Pod

Pod всегда находится в одной из фаз:

| Фаза | Значение |
|------|----------|
| `Pending` | Pod принят API, но контейнеры ещё не созданы (скачивается образ, ожидает планирования на Node) |
| `Running` | Pod привязан к Node, все контейнеры созданы, хотя бы один ещё работает |
| `Succeeded` | Все контейнеры завершились с exit code 0 (характерно для Job) |
| `Failed` | Хотя бы один контейнер завершился с ненулевым exit code |
| `Unknown` | Kubelet потерял связь с Pod (например, нода недоступна) |

Фаза — свойство **Pod** (`.status.phase`), не контейнера.

### 2.2 Состояния контейнера (Container Status)

Внутри `.status.containerStatuses[]` каждый контейнер имеет:

* `state`: `Waiting` / `Running` / `Terminated`
* `reason`: при `Waiting` — `CrashLoopBackOff`, `ImagePullBackOff`, `ErrImagePull`, `ContainerCreating` и т.д.
* `lastState`: предыдущее состояние (нужно для диагностики перезапусков)

### 2.3 Перезапуск контейнеров и `restartPolicy`

`restartPolicy` задаётся на уровне Pod и может быть:

* `Always` (по умолчанию) — перезапускать всегда (Deployment, StatefulSet, DaemonSet)
* `OnFailure` — перезапускать только при ошибке (exit code ≠ 0)
* `Never` — не перезапускать (Job с одним запуском)

**Backoff при перезапусках:**
При повторных падениях kubelet делает нарастающие задержки: 10s, 20s, 40s, ..., макс 5 минут. Сбрасывается после 10 минут успешной работы.

**CrashLoopBackOff** — состояние контейнера, когда он падает, перезапускается и снова падает. Причина: ошибка в приложении, неверная команда запуска, недостаток ресурсов (OOMKilled).

### 2.4 PreStop hook

Перед остановкой контейнера, если задан `preStop` hook, Kubernetes:
1. отправляет `SIGTERM`,
2. вызывает `preStop` hook,
3. ждёт `terminationGracePeriodSeconds` (по умолчанию 30s),
4. если контейнер всё ещё жив — `SIGKILL`.

```yaml
containers:
- name: app
  image: my-app:v1
  lifecycle:
    preStop:
      exec:
        command: ["/bin/sh", "-c", "sleep 5 && kill -SIGTERM 1"]
```

Зачем: дать приложению время завершить текущие запросы, закрыть соединения с БД, сбросить буферы.

### 2.5 PostStart hook

Выполняется сразу после создания контейнера, **асинхронно** с `ENTRYPOINT`. Не гарантирует, что hook завершится до старта приложения. Используется реже, в основном для инициализации (например, записать конфиг).

---

## 3. Ресурсы: requests и limits

### 3.1 Определения

* **`resources.requests`** — **гарантированные** ресурсы. Scheduler использует их для выбора Node. Контейнер получит не меньше этого.
* **`resources.limits`** — **верхний потолок**. Контейнер не может превысить лимит.

Поддерживаемые ресурсы:
* `cpu` — в милли-ядрах (m). `1000m` = 1 полное ядро. `250m` = 0.25 ядра.
* `memory` — в байтах: `Mi`, `Gi`, `M`, `G`. `512Mi` = 512 мебибайт.
* `ephemeral-storage` — временное дисковое пространство.

```yaml
containers:
- name: app
  image: my-app:v1
  resources:
    requests:
      cpu: "500m"
      memory: "256Mi"
    limits:
      cpu: "2000m"
      memory: "1Gi"
```

### 3.2 Механика CPU

* Если контейнер хочет **больше CPU**, чем `limits` — ядро **throttle'ит** его (контейнер ждёт). Throttle метрика: `container_cpu_cfs_throttled_seconds_total`.
* Если Node простаивает, контейнер может использовать больше `requests`, но не больше `limits`.

### 3.3 Механика memory

* Если контейнер превышает `limits` по памяти — **OOMKill** (Out Of Memory Kill). Контейнер убивается, срабатывает restart policy.
* При нехватке памяти на Node kubelet **evicts** Pod'ы по приоритетам:
  1. Pod'ы без requests (BestEffort)
  2. Pod'ы, превысившие requests (Burstable)
  3. Guaranteed (requests = limits) — в последнюю очередь

### 3.4 QoS-классы

Kubernetes автоматически назначает Pod один из трёх классов:

| Класс | Условие | Приоритет при вытеснении |
|-------|---------|--------------------------|
| **Guaranteed** | `requests = limits` для CPU и memory **для всех контейнеров** | Самый высокий |
| **Burstable** | Хотя бы один контейнер имеет `requests ≠ limits` | Средний |
| **BestEffort** | Ни один контейнер не имеет `requests/limits` | Самый низкий |

---

## 4. Probes

**Probes** — проверки, которые kubelet выполняет над контейнерами. Три типа:

### 4.1 livenessProbe

Проверяет, **жив ли** контейнер. Если probe проваливается — контейнер перезапускается.

* Когда применять: приложение может «зависнуть» (deadlock, бесконечный цикл) без падения процесса.
* Когда НЕ применять: если приложение падает само (достаточно restartPolicy), если probe сам вызывает нагрузку.

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 15   # ждём после старта
  periodSeconds: 10          # проверяем каждые 10s
  failureThreshold: 3        # после 3 провалов → перезапуск
  timeoutSeconds: 5          # таймаут одного запроса
```

### 4.2 readinessProbe

Проверяет, **готов ли** контейнер принимать трафик. Если probe проваливается — Pod удаляется из endpoints Service'а (трафик не идёт), но **не перезапускается**.

* Когда применять: приложение стартует медленно (прогрев кэша, загрузка модели), временная деградация.
* Без readinessProbe Pod считается готовым сразу после старта контейнера.

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3
```

### 4.3 startupProbe

Проверяет, **успешно ли стартовал** контейнер. Пока startupProbe не пройдёт — liveness и readiness **не выполняются**.

* Когда применять: приложение с долгим стартом (> `initialDelaySeconds`). Без startupProbe можно получить ложный restart от liveness при медленной загрузке модели.
* После успешного прохождения — больше не вызывается.

```yaml
startupProbe:
  httpGet:
    path: /startup
    port: 8080
  failureThreshold: 30       # до 30 попыток
  periodSeconds: 10          # каждые 10s → до 5 минут на старт
```

### 4.4 Типы проверок

Все три probe поддерживают одинаковые механизмы:

| Механизм | Описание | Пример |
|----------|----------|--------|
| `httpGet` | HTTP GET на указанный path/port | `/healthz:8080` |
| `tcpSocket` | TCP-соединение на порт | порт 5432 для PostgreSQL |
| `exec` | Выполнение команды внутри контейнера | `pg_isready` |
| `gRPC` | gRPC health check (с Kubernetes 1.24+) | `grpc.health.v1.Health/Check` |

### 4.5 Параметры probe

| Параметр | Значение по умолчанию | Смысл |
|----------|----------------------|-------|
| `initialDelaySeconds` | 0 | Задержка перед первым запуском |
| `periodSeconds` | 10 | Интервал между проверками |
| `timeoutSeconds` | 1 | Таймаут одного запроса |
| `successThreshold` | 1 | Число успехов для признания «здоровым» |
| `failureThreshold` | 3 | Число провалов для признания «сбойным» |

---

## 5. Deployment

### 5.1 Что такое Deployment

**Deployment** — контроллер, который управляет ReplicaSet'ами и Pod'ами, обеспечивая:

* декларативное обновление (RollingUpdate / Recreate),
* масштабирование (`replicas`),
* откат на предыдущую версию (`rollback`),
* паузу/возобновление обновления.

Иерархия:
```
Deployment → ReplicaSet → Pod
```

Deployment **не управляет Pod'ами напрямую** — он создаёт ReplicaSet, который уже создаёт Pod'ы.

### 5.2 Стратегии обновления

#### 5.2.1 RollingUpdate (по умолчанию)

Постепенная замена старых Pod'ов новыми. Параметры:

* `maxUnavailable` — сколько Pod'ов может быть недоступно в процессе обновления (число или процент). По умолчанию 25%.
* `maxSurge` — на сколько Pod'ов можно превысить `replicas` временно (число или процент). По умолчанию 25%.

```
replicas: 4
maxUnavailable: 1
maxSurge: 1

Состояние: 4 старых Pod
Шаг 1:    создан 1 новый Pod        → 4 старых + 1 новый  = 5 всего
Шаг 2:    удалён 1 старый Pod       → 3 старых + 1 новый  = 4 всего
Шаг 3:    создан 1 новый Pod        → 3 старых + 2 новых  = 5 всего
Шаг 4:    удалён 1 старый Pod       → 2 старых + 2 новых  = 4 всего
...
```

**Как понять, что новый Pod «готов»:** проверяется `readinessProbe`. Если readiness не проходит — новый Pod не считается готовым, старый не удаляется.

**`minReadySeconds`** — минимальное время, которое Pod должен проработать после readiness, прежде чем считаться «доступным». Защита от ложной готовности (приложение ответило на readiness, но через секунду упало).

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  minReadySeconds: 10
```

#### 5.2.2 Recreate

Удалить **все** старые Pod'ы, потом создать новые. Даунтайм равен времени старта новых Pod'ов.

Применяется когда:
* нельзя иметь две версии одновременно (например, миграция схемы БД),
* нет механизма graceful degradation,
* экономия ресурсов (не нужны лишние Pod'ы на время обновления).

```yaml
spec:
  strategy:
    type: Recreate
```

### 5.3 Rollout и rollback

**Просмотр истории:**
```
kubectl rollout history deployment/my-app
```

**Откат на предыдущую версию:**
```
kubectl rollout undo deployment/my-app
```

**Откат на конкретную ревизию:**
```
kubectl rollout undo deployment/my-app --to-revision=3
```

**Механика отката:** старый ReplicaSet не удаляется сразу после обновления (по умолчанию хранится `revisionHistoryLimit: 10` последних). При откате Kubernetes масштабирует старый ReplicaSet обратно, новый сжимает до 0.

**`progressDeadlineSeconds`** — если обновление не завершилось за это время (по умолчанию 600s), Deployment помечается как `ProgressDeadlineExceeded`. Причины: нехватка ресурсов, ошибка в образе (ImagePullBackOff), провал readiness.

### 5.4 Диагностика проблем

**`kubectl describe deployment <name>`** — показывает Conditions:
* `Available`: Pod доступен (minReadySeconds прошёл)
* `Progressing`: обновление в процессе / завершено / застряло

**`kubectl rollout status deployment/<name>`** — следит за ходом обновления в реальном времени.

**Типовые причины ProgressDeadlineExceeded:**
* Образ не существует или нет прав на pull (`ErrImagePull`, `ImagePullBackOff`)
* Новый Pod не проходит readiness (ошибка в коде, неверный порт)
* Недостаток ресурсов (Pod не может запланироваться — `Pending`)
* CrashLoopBackOff на новом Pod

---

## 6. Init Containers

**Init-контейнеры** — контейнеры, которые выполняются **до** основных и **строго последовательно**.

```yaml
spec:
  initContainers:
  - name: wait-for-db
    image: busybox:1.36
    command: ['sh', '-c', 'until nc -z postgres 5432; do sleep 2; done']
  - name: migrate-db
    image: my-app-migrations:v1
  containers:
  - name: app
    image: my-app:v1
```

**Свойства:**
* Выполняются по порядку (следующий — после успешного завершения предыдущего).
* Если init-контейнер падает — kubelet перезапускает его (согласно `restartPolicy` Pod).
* Каждый имеет свой образ, свой набор команд.
* Должны завершиться до старта основных контейнеров.

**Типовые сценарии:**
* Ожидание внешней зависимости (БД, кэш, API)
* Миграции схемы БД
* Генерация конфигов из шаблонов
* Установка прав на shared volumes

---

## 7. Чек-лист: Pod to Deployment

1. **Контейнер**: образ, порты, `command`/`args`, `env` — определены.
2. **Ресурсы**: `requests` и `limits` заданы для каждого контейнера (CPU + memory).
3. **Probes**: `readiness` (для трафика), `liveness` (для восстановления), `startup` (если долгий старт).
4. **`terminationGracePeriodSeconds`** и `preStop` hook — дают приложению корректно завершиться.
5. **Replicas**: ≥2 для отказоустойчивости (если не StatefulSet с особыми требованиями).
6. **Стратегия обновления**: RollingUpdate с разумными `maxUnavailable`/`maxSurge`, или Recreate когда нужно.
7. **`revisionHistoryLimit`**: оставить последние 3-5 ревизий для отката.
8. **Init-контейнеры**: миграции, ожидания зависимостей — до запуска основных контейнеров.

---

## 8. Типичные ошибки

1. **Путать Pod и контейнер**: `env` и `resources` задаются на контейнер, `volumes` и `restartPolicy` — на Pod.
2. **Не заданы resources**: Pod получает QoS BestEffort → первый кандидат на вытеснение при давлении памяти.
3. **Liveness без readiness**: падающий Pod перезапускается, но трафик идёт на ещё не готовый → каскад ошибок.
4. **Liveness слишком агрессивный** (`failureThreshold=1`, `periodSeconds=1`): кратковременная задержка → убийство контейнера.
5. **Не задан `terminationGracePeriodSeconds`**: приложение получает SIGTERM и сразу SIGKILL, не успевая завершить обработку.
6. **`maxUnavailable=0` + один релевантный Pod**: обновление зависает — нельзя удалить старый Pod, пока новый не готов.
7. **Нет readinessProbe**: Pod сразу в endpoints → получает трафик до полной инициализации → ошибки.
8. **Init-контейнер в бесконечном цикле ожидания**: нет таймаута на ожидание зависимости → Pod висит в `Init:0/1` вечно.