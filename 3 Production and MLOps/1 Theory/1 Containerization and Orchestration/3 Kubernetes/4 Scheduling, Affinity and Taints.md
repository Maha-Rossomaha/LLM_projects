# Scheduling, Affinity and Taints

## 1. Планирование Pod'ов: обзор

kube-scheduler решает **на какой Node запустить Pod**. Pod создаётся контроллером в состоянии `Pending`, scheduler видит его и выполняет:

```
Pod (Pending) → Filtering → Scoring → Bind → Pod (Node назначена)
```

### 1.1 Filtering (Feasibility)

Отбрасываются Node, которые **не могут** запустить Pod. Критерии:

* **Ресурсы:** достаточно ли CPU/memory для `requests` Pod, плюс сумма requests всех Pod'ов на Node не превышает capacity (с учётом kube-reserved и system-reserved).
* **Node Selector:** `.spec.nodeSelector` — Node должна иметь указанные labels.
* **Taints & Tolerations:** Pod должен толерировать taint Node.
* **Affinity/Anti-affinity:** Node должна удовлетворять правилам.
* **Node name:** если Pod уже привязан к Node (`.spec.nodeName`), фильтр пропускает только её.
* **Node conditions:** Node должна быть `Ready`, не иметь `MemoryPressure`, `DiskPressure`, `PIDPressure`.

### 1.2 Scoring (Prioritization)

Оставшиеся Node ранжируются по предпочтениям:

* **LeastRequestedPriority / LeastAllocated:** набирает баллы Node с наибольшим свободным остатком ресурсов (балансировка нагрузки).
* **MostRequestedPriority / MostAllocated:** набирает баллы Node с наименьшим свободным остатком (плотная упаковка).
* **SelectorSpreadPriority:** раскидывает Pod одного Service/RS по разным Node (по умолчанию в 1.24+).
* **NodeAffinityPriority:** бонус за preferred affinity.
* **ImageLocalityPriority:** бонус за Node, где образ уже скачан.
* **PodTopologySpread:** бонус за равномерное распределение по топологии.

Можно писать свои **scheduler plugins** (Kubernetes Scheduling Framework).

### 1.3 Bind

Scheduler отправляет API-server запрос на привязку Pod к выбранной Node. Дальше kubelet на той Node запускает Pod.

---

## 2. nodeSelector и nodeName

### 2.1 nodeSelector

Простейший способ привязать Pod к Node с определёнными labels.

**Добавить label Node:**
```
kubectl label node worker-1 gpu=true
```

**Использовать в Pod:**
```yaml
spec:
  nodeSelector:
    gpu: "true"
```

Pod запустится только на Node с label `gpu=true`. Если таких нет — Pod висит `Pending`.

### 2.2 nodeName

Жёсткая привязка к конкретной Node по имени. Обходит scheduler полностью.

```yaml
spec:
  nodeName: worker-3
```

Применение редко: отладка, ручное управление. Риск: если Node недоступна — Pod не запустится нигде.

---

## 3. Affinity и Anti-Affinity

### 3.1 Мотивация

`nodeSelector` решает только «где НЕ запускать» / «где точно запускать» по equality-условиям. Affinity даёт:

* **Выражения** (In, NotIn, Exists, DoesNotExist, Gt, Lt).
* **Preferred** (soft) правила — желательно, но не обязательно.
* **Pod Affinity** — размещать Pod'ы **рядом** (colocate).
* **Pod Anti-Affinity** — размещать Pod'ы **врозь** (spread).

### 3.2 Node Affinity

**requiredDuringSchedulingIgnoredDuringExecution** (hard):
```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: node-type
            operator: In
            values:
            - gpu
            - high-memory
```

Pod **не будет** запланирован на Node без `node-type=gpu` или `node-type=high-memory`.

**preferredDuringSchedulingIgnoredDuringExecution** (soft):
```yaml
spec:
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 80
        preference:
          matchExpressions:
          - key: disk-type
            operator: In
            values:
            - ssd
      - weight: 20
        preference:
          matchExpressions:
          - key: zone
            operator: In
            values:
            - zone-a
```

* Scheduler старается разместить на Node с `disk-type=ssd` (weight 80), затем в `zone-a` (weight 20), но если таких нет — запустит на любой подходящей.

**Семантика `IgnoredDuringExecution`:**
* Правило действует **только при планировании**.
* Если label Node изменилась позже — Pod **не выселяется**.

### 3.3 Pod Affinity

Размещает Pod **рядом** с другими Pod'ами (по label).

```yaml
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: cache
        topologyKey: "kubernetes.io/hostname"
```

* `topologyKey: kubernetes.io/hostname` — «одна Node».
* Pod размещается **на той же Node**, где уже есть Pod с `app=cache`.
* Если таких Node нет — `Pending`.

**Типовые topologyKey:**
| Ключ | Значение |
|------|----------|
| `kubernetes.io/hostname` | Одна Node |
| `topology.kubernetes.io/zone` | Одна зона доступности |
| `topology.kubernetes.io/region` | Один регион |

### 3.4 Pod Anti-Affinity

Размещает Pod **врозь** от других Pod'ов.

```yaml
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: my-app
        topologyKey: "kubernetes.io/hostname"
```

* Гарантирует: **не более одного** Pod с `app=my-app` на одной Node.
* Если `replicas=3` и Node'ы всего 2 — третий Pod будет `Pending`.

**Preferred (soft) anti-affinity:**
```yaml
spec:
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: my-app
          topologyKey: "kubernetes.io/hostname"
```

* Scheduler старается разносить Pod'ы, но если не хватает Node — запускает вместе.

### 3.5 Правила работы с topologyKey

* `topologyKey` определяет **домен**, в котором действует правило.
* Для podAntiAffinity: **нельзя** размещать два Pod'а с совпадающими labels в одном домене.
* Для podAffinity: **можно** размещать только в домене, где уже есть совпадающий Pod.

**Пример: anti-affinity по зонам**
```yaml
podAntiAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
  - labelSelector:
      matchLabels:
        app: my-app
    topologyKey: "topology.kubernetes.io/zone"
```
* В каждой зоне — не более одного Pod `app=my-app`.

---

## 4. Taints и Tolerations

### 4.1 Концепция

* **Taint** — «пятно» на Node: «не запускай Pod'ы, которые не толерируют это».
* **Toleration** — «разрешение» Pod'а: «я могу терпеть этот taint».

В отличие от affinity, который **притягивает** Pod к Node, taints **отталкивают** Pod'ы от Node.

**Формат taint:**
```
key=value:Effect
```

Где Effect:
| Effect | Поведение |
|--------|-----------|
| `NoSchedule` | Pod без toleration **не планируется** на Node |
| `PreferNoSchedule` | Scheduler старается не планировать, но если некуда — может |
| `NoExecute` | Pod без toleration **выселяется** (eviction), если уже запущен |

### 4.2 Добавление taint на Node

```
kubectl taint node worker-1 gpu=true:NoSchedule
```

* Pod без toleration к `gpu=true:NoSchedule` на `worker-1` не запустится.

**Удаление taint:**
```
kubectl taint node worker-1 gpu=true:NoSchedule-
```

### 4.3 Toleration в Pod

```yaml
spec:
  tolerations:
  - key: "gpu"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
```

* Pod с такой toleration **может** запускаться на Node с taint `gpu=true:NoSchedule`.

**Toleration на всё (полезно для системных DaemonSet):**
```yaml
spec:
  tolerations:
  - operator: "Exists"
```

**Toleration с `NoExecute` и `tolerationSeconds`:**
```yaml
spec:
  tolerations:
  - key: "node.kubernetes.io/not-ready"
    operator: "Exists"
    effect: "NoExecute"
    tolerationSeconds: 300
```

* Если Node становится NotReady, Pod **остаётся** на ней 300 секунд, затем выселяется.

### 4.4 Встроенные taint'ы

Kubernetes автоматически ставит taint'ы на Node при проблемах:

| Taint | Effect | Причина |
|-------|--------|---------|
| `node.kubernetes.io/not-ready` | NoExecute | Node не готова |
| `node.kubernetes.io/unreachable` | NoExecute | API-server потерял связь с Node |
| `node.kubernetes.io/memory-pressure` | NoSchedule | Нехватка памяти на Node |
| `node.kubernetes.io/disk-pressure` | NoSchedule | Заканчивается диск на Node |
| `node.kubernetes.io/pid-pressure` | NoSchedule | Слишком много процессов |
| `node.kubernetes.io/unschedulable` | NoSchedule | Node помечена как unschedulable (`kubectl cordon`) |

Kubernetes автоматически добавляет дефолтные tolerations с `tolerationSeconds: 300` для `not-ready` и `unreachable`.

### 4.5 Типовые сценарии

**Выделенные GPU-ноды:**
```
kubectl taint node gpu-1 nvidia.com/gpu=true:NoSchedule
kubectl taint node gpu-2 nvidia.com/gpu=true:NoSchedule
```

Только Pod'ы с toleration запускаются на GPU-нодах.

**Изоляция control-plane:**
Control-plane ноды имеют taint:
```
node-role.kubernetes.io/control-plane:NoSchedule
```

Обычные Pod'ы на них не планируются.

**Нода для экспериментов (PreferNoSchedule):**
```
kubectl taint node test-node testing=true:PreferNoSchedule
```

Scheduler избегает эту Node, но при нехватке ресурсов может использовать.

---

## 5. Topology Spread Constraints

### 5.1 Мотивация

Pod Anti-Affinity гарантирует «не более одного в домене». А если нужно «примерно равное количество Pod'ов в каждом домене» — нужны **Topology Spread Constraints**.

### 5.2 Синтаксис

```yaml
spec:
  topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: "topology.kubernetes.io/zone"
    whenUnsatisfiable: "DoNotSchedule"
    labelSelector:
      matchLabels:
        app: my-app
```

**Параметры:**
| Параметр | Значение |
|----------|----------|
| `maxSkew` | Максимальная разница между числом Pod'ов в любых двух доменах |
| `topologyKey` | Ключ, по которому Node группируются в домены |
| `whenUnsatisfiable` | `DoNotSchedule` (hard) или `ScheduleAnyway` (soft) |
| `labelSelector` | Какие Pod'ы считать |

**Пример:**
* 3 зоны: zone-a, zone-b, zone-c
* `replicas: 5`
* `maxSkew: 1`

```
zone-a: 2 Pod
zone-b: 2 Pod
zone-c: 1 Pod
```

Разница 2 − 1 = 1 ≤ maxSkew ✓

Если `maxSkew: 1`, нельзя 3-0-2 (разница 3).

### 5.3 `minDomains` (Kubernetes 1.25+)

```yaml
spec:
  topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: "topology.kubernetes.io/zone"
    minDomains: 3
    whenUnsatisfiable: "DoNotSchedule"
    labelSelector:
      matchLabels:
        app: my-app
```

* Если зон меньше, чем `minDomains`, scheduler считает, что их всё равно `minDomains` (виртуально), и применяет ограничения.

### 5.4 `nodeAffinityPolicy` и `nodeTaintsPolicy` (Kubernetes 1.26+)

Определяют, учитывать ли Node Affinity и taints при подсчёте skew:

* `nodeAffinityPolicy: Honor` / `Ignore`
* `nodeTaintsPolicy: Honor` / `Ignore`

По умолчанию `Honor` — Node, не соответствующие affinity или отфильтрованные taints, не учитываются в расчёте.

---

## 6. Priority и Preemption

### 6.1 PriorityClass

**PriorityClass** — объект, задающий приоритет Pod'а. Pod с более высоким приоритетом может **вытеснить** (preempt) Pod с низким.

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 1000000
globalDefault: false
description: "For production workloads"
```

Использование в Pod:
```yaml
spec:
  priorityClassName: high-priority
```

### 6.2 Как работает Preemption

1. Scheduler пытается запланировать Pod — нет подходящих Node.
2. Scheduler ищет Pod'ы с **более низким** приоритетом, удаление которых освободит ресурсы.
3. Найденные Pod'ы выселяются (graceful termination).
4. Высокоприоритетный Pod планируется на освободившееся место.

### 6.3 Системные PriorityClass

| Имя | Value | Назначение |
|-----|-------|------------|
| `system-cluster-critical` | 2000000000 | etcd, API-server, scheduler |
| `system-node-critical` | 2000001000 | kubelet, kube-proxy |
| `default` | 0 | Если PriorityClass не задан |

---

## 7. Чек-лист

1. **Node labels**: размечены GPU, SSD, зоны, регионы — есть что использовать в affinity.
2. **nodeSelector** / **nodeAffinity**: жёстко для обязательного оборудования (GPU), мягко для предпочтений.
3. **Pod anti-affinity**: реплики одного сервиса разнесены по нодам (`topologyKey: kubernetes.io/hostname`).
4. **Taints**: GPU-ноды изолированы taint'ами, tolerations — только у GPU-Pod'ов.
5. **TopologySpreadConstraints**: Pod'ы равномерно по зонам, `maxSkew` задан.
6. **PriorityClass**: критичные сервисы имеют `priorityClassName`, некритичные — нет (preemption для важных).
7. **PreferNoSchedule**: для экспериментальных/второстепенных нод.

---

## 8. Типичные ошибки

1. **Pod висит Pending из-за taint без toleration**: `kubectl describe pod` показывает предупреждение: «1 node(s) had taint that the pod didn't tolerate».
2. **Anti-affinity жёсткая + мало Node**: Pod не планируется, потому что «no nodes available that match pod anti-affinity rules».
3. **Node label изменилась, а Pod остался**: affinity работает только при планировании (`IgnoredDuringExecution`).
4. **Забыт `topologyKey` в podAffinity/podAntiAffinity**: ошибка валидации.
5. **Taint с `NoExecute` без `tolerationSeconds`**: Pod выселяется мгновенно, даже без graceful shutdown.
6. **TopologySpreadConstraint без `labelSelector`**: ограничение применяется ко всем Pod'ам в namespace, а не только к нужным.
7. **Все Pod'ы с высоким PriorityClass**: нет кандидатов на вытеснение, scheduler не может решить задачу планирования.
8. **`nodeSelector` + `nodeAffinity` конфликтуют**: должно выполняться и то и другое (AND). Если `nodeSelector: gpu=true`, а `nodeAffinity: node-type In [cpu]` — Pod не запустится нигде.