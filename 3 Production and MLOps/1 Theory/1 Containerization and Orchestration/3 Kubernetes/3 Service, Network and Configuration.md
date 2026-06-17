# Service, Network and Configuration

## 1. Service — стабильный доступ к Pod'ам

### 1.1 Проблема: Pod'ы эфемерны

Pod'ы постоянно:
* пересоздаются (обновление Deployment, падения),
* меняют IP (каждый Pod получает новый IP из диапазона кластера),
* масштабируются (replicas меняется).

Прямой доступ по Pod IP неработоспособен.

**Service** решает это:
* даёт **стабильный IP** (ClusterIP) и **DNS-имя**,
* обеспечивает **балансировку** трафика между Pod'ами,
* автоматически обновляет список endpoints при изменениях.

### 1.2 Как Service находит Pod'ы

Service использует **labels** и **selectors**:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app          # все Pod'ы с label app=my-app
  ports:
  - port: 80             # порт Service
    targetPort: 8080     # порт внутри Pod
    protocol: TCP
```

**Механика:**
1. **Endpoints controller** следит за Pod'ами с `app=my-app`.
2. Формирует объект `Endpoints` (или `EndpointSlice`) со списком `PodIP:targetPort`.
3. kube-proxy на каждой Node настраивает правила (iptables/ipvs) для маршрутизации `ServiceIP:port → PodIP:targetPort`.

### 1.3 Типы Service

| Тип | Доступ | Когда использовать |
|-----|--------|--------------------|
| **ClusterIP** | Только внутри кластера | Внутренние сервисы, микросервисы |
| **NodePort** | `<NodeIP>:<порт>` (30000–32767) | Отладка, простой внешний доступ, bare-metal |
| **LoadBalancer** | Внешний IP через облачный LB | Продакшен, внешний трафик |
| **ExternalName** | DNS CNAME (без прокси) | Прокси на внешний сервис по имени |

#### ClusterIP (по умолчанию)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  type: ClusterIP
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
```

* Получает внутренний IP из диапазона `--service-cluster-ip-range` (например, `10.96.0.0/12`).
* Доступен из любого Pod'а внутри кластера по `my-app.<namespace>.svc.cluster.local`.

#### NodePort

```yaml
spec:
  type: NodePort
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080      # опционально, иначе выбирается случайный
```

* Открывает порт на **каждой** Node кластера.
* Запрос на `<любая Node>:30080` → Service → Pod.
* Работает даже если Pod только на одной Node: kube-proxy проксирует между нодами.

#### LoadBalancer

```yaml
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
```

* Cloud-controller-manager создаёт внешний LB (AWS NLB/ALB, GCP LB, Azure LB).
* LB получает внешний IP, трафик идёт: `external-IP:80 → Node:NodePort → Pod`.
* На bare-metal требует MetalLB или аналог.

### 1.4 Headless Service

Если `clusterIP: None` — Service **не получает** ClusterIP. DNS-запрос возвращает IP всех Pod'ов напрямую (A-записи).

```yaml
spec:
  clusterIP: None
  selector:
    app: my-app
```

Применение:
* StatefulSet (каждый Pod должен быть адресуем напрямую: `pod-0.my-app.ns.svc.cluster.local`),
* кастомная балансировка на стороне клиента (gRPC client-side LB),
* сервисы, которые сами управляют кластеризацией (Cassandra, Elasticsearch).

### 1.5 Session Affinity

По умолчанию Service балансирует раунд-робин (iptables — случайный Pod на каждое соединение). `sessionAffinity: ClientIP` закрепляет клиента за одним Pod'ом:

```yaml
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800   # 3 часа
```

Применение: приложения с локальным состоянием, которое не хочется реплицировать.

---

## 2. Ingress

### 2.1 Что такое Ingress

**Ingress** — объект Kubernetes для HTTP(S)-маршрутизации внешнего трафика к Service'ам **по доменам и путям**.

Без Ingress для каждого сервиса нужен отдельный LoadBalancer → дорого и неудобно. Ingress позволяет один LB и маршрутизацию:

```
                    ┌─── example.com/api/*  → api-service:80
Ingress Controller ─┼─── example.com/web/*  → web-service:80
(один внешний IP)   └─── *.example.com       → default-service:80
```

### 2.2 Ingress vs Ingress Controller

* **Ingress** — ресурс с правилами (YAML).
* **Ingress Controller** — под, который читает Ingress-ресурсы и настраивает реальный прокси (nginx, HAProxy, Traefik, Istio Gateway, AWS ALB).

Сам по себе Ingress-ресурс **не работает** без контроллера.

### 2.3 Пример Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  ingressClassName: nginx
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1
        pathType: Prefix
        backend:
          service:
            name: api-v1
            port:
              number: 80
      - path: /v2
        pathType: Prefix
        backend:
          service:
            name: api-v2
            port:
              number: 80
  - host: web.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

### 2.4 `pathType`

| Тип | Поведение |
|-----|-----------|
| `Prefix` | Префиксное совпадение (`/api` → `/api`, `/api/v1`, `/api/whatever`) |
| `Exact` | Точное совпадение (`/api` → только `/api`, не `/api/`) |
| `ImplementationSpecific` | На усмотрение контроллера |

### 2.5 TLS

```yaml
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls-secret    # Secret с сертификатом и ключом
```

Secret должен содержать ключи `tls.crt` и `tls.key`.

---

## 3. ConfigMap

### 3.1 Что такое ConfigMap

**ConfigMap** — объект для хранения **несекретной** конфигурации в виде ключ-значение. Отделяет конфигурацию от образа (один образ → разные окружения через разные ConfigMap).

### 3.2 Создание ConfigMap

**Из литералов:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  config.yaml: |
    database:
      host: postgres
      port: 5432
    cache:
      ttl: 300
```

**Из файла:**
```
kubectl create configmap app-config --from-file=config.yaml
```

### 3.3 Использование ConfigMap

**Как переменные окружения:**
```yaml
containers:
- name: app
  image: my-app:v1
  env:
  - name: LOG_LEVEL
    valueFrom:
      configMapKeyRef:
        name: app-config
        key: LOG_LEVEL
```

**Все ключи ConfigMap как env (envFrom):**
```yaml
containers:
- name: app
  envFrom:
  - configMapRef:
      name: app-config
```

**Как файл через volume:**
```yaml
containers:
- name: app
  volumeMounts:
  - name: config
    mountPath: /etc/app/config.yaml
    subPath: config.yaml       # монтируем только один ключ как файл
volumes:
- name: config
  configMap:
    name: app-config
```

### 3.4 Обновление ConfigMap

* ConfigMap, смонтированные как volumes, обновляются на Node с задержкой (kubelet sync period + cache propagation delay), но приложение **не перезапускается**.
* ConfigMap, использованные через `env` / `envFrom`, **не обновляются** после старта Pod — требуется пересоздание Pod.
* Если нужно автоматически перезапускать Pod при изменении ConfigMap — используют аннотацию с хэшем (например, через Helm или Reloader).

### 3.5 Immutable ConfigMap

```yaml
immutable: true
```

* kubelet не следит за изменениями → меньше нагрузки на API-server.
* Для обновления — удалить и создать заново.

---

## 4. Secret

### 4.1 Что такое Secret

**Secret** — объект для хранения **чувствительной** информации: пароли, токены, ключи, TLS-сертификаты.

Отличия от ConfigMap:
* Данные хранятся в **base64** (не шифрование!).
* kubelet хранит Secret в `tmpfs` (не на диске).
* Можно использовать etcd encryption at rest (рекомендуется в продакшене).
* Доступ через RBAC.

### 4.2 Типы Secret

| Тип | Назначение |
|-----|-----------|
| `Opaque` (по умолчанию) | Произвольные данные (пароли, API-ключи) |
| `kubernetes.io/tls` | TLS-сертификат и ключ (`tls.crt`, `tls.key`) |
| `kubernetes.io/dockerconfigjson` | Данные для доступа к приватному container registry |
| `kubernetes.io/basic-auth` | Basic-аутентификация (`username`, `password`) |
| `kubernetes.io/service-account-token` | ServiceAccount-токен (автоматически создаётся) |

### 4.3 Создание и использование

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  username: YWRtaW4=      # echo -n "admin" | base64
  password: cDRzc3cwcmQ=  # echo -n "p4ssw0rd" | base64
```

Или из CLI (автоматический base64):
```
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password=p4ssw0rd
```

**Монтирование как env:**
```yaml
containers:
- name: app
  env:
  - name: DB_USER
    valueFrom:
      secretKeyRef:
        name: db-credentials
        key: username
  - name: DB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: db-credentials
        key: password
```

**Монтирование как файл:**
```yaml
volumes:
- name: db-secret
  secret:
    secretName: db-credentials
```

**TLS Secret:**
```
kubectl create secret tls api-tls --cert=cert.pem --key=key.pem
```

Используется в Ingress:
```yaml
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls
```

### 4.4 Ограничения Secret

* Максимальный размер: **1 MiB** на Secret.
* Base64 — **не шифрование**, это кодирование. Любой, кто имеет доступ через API, видит данные.
* Для шифрования: etcd encryption at rest, внешний vault (HashiCorp Vault, AWS Secrets Manager), Sealed Secrets.

---

## 5. Volumes

### 5.1 Модель: Volume → VolumeMount

Pod определяет **Volume** (что за хранилище). Каждый контейнер определяет **VolumeMount** (куда смонтировать внутри контейнера).

```
Pod.spec.volumes[]       ← тип и параметры тома
Container.volumeMounts[] ← точка монтирования внутри контейнера
```

Несколько контейнеров могут монтировать один Volume — данные разделяются.

### 5.2 Типы Volumes

#### `emptyDir`

Временная директория на Node. Создаётся при старте Pod, удаляется при удалении Pod.

```yaml
volumes:
- name: temp
  emptyDir: {}
```

* **Назначение:** scratch space, кэш, обмен данными между контейнерами одного Pod.
* **Вариант:** `emptyDir.medium: Memory` — хранить в tmpfs (RAM, быстрее, но учитывается в limits по памяти).

#### `hostPath`

Монтирует файл или директорию с Node внутрь Pod.

```yaml
volumes:
- name: node-logs
  hostPath:
    path: /var/log
    type: Directory
```

* **Назначение:** DaemonSet'ы для сбора логов/метрик (Fluent Bit, Prometheus node-exporter), доступ к Docker socket.
* **Опасность:** Pod привязывается к конкретной Node (если Pod мигрирует — данные на новой Node другие).

#### `persistentVolumeClaim` (PVC)

Запрос на персистентное хранилище.

```yaml
volumes:
- name: data
  persistentVolumeClaim:
    claimName: my-pvc
```

Модель: **PV (PersistentVolume)** — физический том, **PVC (PersistentVolumeClaim)** — запрос на том.

```
Администратор создаёт PV (или динамический provisioner создаёт по StorageClass)
          ↓
Пользователь создаёт PVC (размер, access mode, StorageClass)
          ↓
Kubernetes связывает PV ↔ PVC (Binding)
          ↓
Pod использует PVC как Volume
```

**Access Modes:**
| Режим | Аббревиатура | Смысл |
|-------|-------------|-------|
| `ReadWriteOnce` | RWO | Одна Node, чтение/запись |
| `ReadOnlyMany` | ROX | Много Nodes, только чтение |
| `ReadWriteMany` | RWX | Много Nodes, чтение/запись |
| `ReadWriteOncePod` | RWOP | Один Pod, чтение/запись (K8s 1.22+) |

Пример PVC:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

#### `configMap` и `secret`

Монтируют ConfigMap/Secret как файлы (см. разделы 3 и 4).

#### `projected`

Объединяет несколько источников (ConfigMap, Secret, downwardAPI, serviceAccountToken) в одну директорию.

```yaml
volumes:
- name: projected-volume
  projected:
    sources:
    - configMap:
        name: app-config
    - secret:
        name: db-credentials
    - downwardAPI:
        items:
        - path: "labels"
          fieldRef:
            fieldPath: metadata.labels
```

### 5.3 Downward API

Позволяет Pod'у получить метаданные о себе (имя, namespace, labels, annotations, requests/limits) через env или файлы.

**Через env:**
```yaml
env:
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: CPU_LIMIT
  valueFrom:
    resourceFieldRef:
      containerName: app
      resource: limits.cpu
```

**Через файл (volume):**
```yaml
volumes:
- name: pod-info
  downwardAPI:
    items:
    - path: "labels"
      fieldRef:
        fieldPath: metadata.labels
    - path: "annotations"
      fieldRef:
        fieldPath: metadata.annotations
```

Назначение: приложение знает своё имя Pod (для метрик/логов), свои ресурсы (для тюнинга thread pool).

---

## 6. DNS в Kubernetes

### 6.1 DNS-формат Service

Каждый Service получает DNS-запись:

```
<service-name>.<namespace>.svc.cluster.local
```

Пример: `my-app.default.svc.cluster.local` разрешается в ClusterIP сервиса.

Pod'ы в том же namespace могут обращаться просто по `my-app`.

### 6.2 Headless Service DNS

При `clusterIP: None` DNS-запись возвращает **A-записи** всех Pod'ов:

```
my-app.default.svc.cluster.local → 10.244.1.5, 10.244.2.3, 10.244.1.7
```

Для StatefulSet каждый Pod получает отдельную A-запись:
```
pod-0.my-app.default.svc.cluster.local → 10.244.1.5
pod-1.my-app.default.svc.cluster.local → 10.244.2.3
```

### 6.3 CoreDNS

CoreDNS — дефолтный DNS-сервер в Kubernetes (заменил kube-dns). Работает как Deployment с Service `kube-dns` в namespace `kube-system`. Настраивается через ConfigMap `coredns`.

---

## 7. Чек-лист: Service и конфигурация

1. **Service**: тип выбран по потребности (ClusterIP внутри, LoadBalancer для прода, NodePort для отладки).
2. **Selector**: совпадает с labels Pod'ов (`.spec.template.metadata.labels` в Deployment).
3. **Port mapping**: `port` (Service) → `targetPort` (контейнер) совпадают.
4. **ConfigMap**: несекретные настройки; через volume, если нужно горячее обновление.
5. **Secret**: пароли, токены, TLS-сертификаты; etcd encryption at rest включена.
6. **Ingress**: один на несколько сервисов по доменам/путям; TLS настроен.
7. **PersistentVolume**: PVC создан, StorageClass поддерживает нужный access mode.
8. **Downward API**: Pod знает свои метаданные (имя, namespace, ресурсы).

---

## 8) Типичные ошибки

1. **Selector не совпадает с labels Pod'ов**: Service не находит endpoints → трафик не идёт. Проверять: `kubectl get endpoints <svc>`.
2. **Путать `port` и `targetPort`**: `port` — порт Service, `targetPort` — порт контейнера. Несовпадение → connection refused.
3. **NodePort без уточнения порта**: диапазон 30000–32767, может конфликтовать.
4. **ConfigMap через env не обновляется**: Pod'ы не видят изменений до пересоздания.
5. **Secret в base64 без etcd encryption**: данные доступны любому с доступом к API.
6. **PVC не создаётся**: забыт StorageClass, или нет default StorageClass.
7. **`emptyDir` для персистентных данных**: данные теряются при удалении Pod.
8. **Ingress без Ingress Controller**: ресурс создан, но правил маршрутизации нет — трафик не проходит.
9. **Headless Service без StatefulSet**: Pod'ы не имеют стабильных сетевых идентификаторов, DNS возвращает случайные IP существующих Pod'ов.