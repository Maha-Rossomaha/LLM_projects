# Helm Basics: Chart Structure and Commands

## 1. Что такое Helm

**Helm** — пакетный менеджер для Kubernetes. Аналогия: `apt`/`brew`/`pip`, но для Kubernetes-ресурсов.

Helm оперирует **чартами (charts)** — упакованными наборами YAML-манифестов с шаблонизацией и метаданными.

**Что даёт Helm:**
* единый пакет для всех ресурсов приложения (Deployment, Service, ConfigMap, Secret, Ingress и т.д.),
* параметризация через `values.yaml` (один чарт → разные окружения),
* версионирование релизов (`helm install` / `helm upgrade` / `helm rollback`),
* переиспользование (зависимости между чартами, Helm-репозитории),
* хуки (действия до/после установки: миграции БД, создание Secret).

## 2. Chart Structure

### 2.1 Дерево чарта

```
my-chart/
├── Chart.yaml           # метаданные чарта
├── values.yaml          # значения по умолчанию
├── values.schema.json   # (опционально) JSON Schema для валидации values
├── charts/              # зависимые чарты (subcharts)
├── templates/           # Go-шаблоны Kubernetes-манифестов
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── _helpers.tpl     # именованные шаблоны (вспомогательные)
│   └── NOTES.txt        # сообщение после установки
├── crds/                # Custom Resource Definitions
├── README.md
└── .helmignore          # файлы, исключённые из упаковки
```

### 2.2 Chart.yaml

```yaml
apiVersion: v2                        # версия API Helm (v2 для Helm 3)
name: my-app
version: 1.2.3                        # версия чарта (SemVer)
appVersion: "2.4.0"                   # версия приложения (информативно)
description: My application chart
type: application                     # application | library
dependencies:                         # зависимости
- name: postgresql
  version: "12.1.0"
  repository: "https://charts.bitnami.com/bitnami"
  condition: postgresql.enabled       # можно выключить через values
keywords:
- web
- api
maintainers:
- name: team-name
  email: team@example.com
```

**Поля:**
| Поле | Назначение |
|------|-----------|
| `apiVersion: v2` | Helm 3 |
| `name` | Имя чарта (должно совпадать с именем директории) |
| `version` | Версия чарта (SemVer); меняется при изменениях чарта |
| `appVersion` | Версия упакованного приложения (информативно) |
| `type` | `application` — деплой приложения; `library` — библиотека шаблонов (не устанавливается сама) |
| `dependencies` | Список чартов, от которых зависит данный |

### 2.3 values.yaml

Файл значений по умолчанию. Пользователь переопределяет через `--set` или свой `values.yaml`.

```yaml
replicaCount: 3

image:
  repository: nginx
  tag: "1.27"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: false
  host: my-app.local

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

postgresql:
  enabled: true
  auth:
    username: app
    password: secret
    database: appdb
```

**Принцип:** values.yaml содержит значения по умолчанию. При `helm install` / `helm upgrade` можно передать кастомный файл: `-f prod-values.yaml` или отдельные ключи: `--set replicaCount=5`.

### 2.4 .helmignore

Аналог `.gitignore`. Исключает файлы из пакета чарта при `helm package`.

```
.git
*.swp
*.bak
.DS_Store
```

## 3. Основные команды

### 3.1 helm install

Установка чарта как нового **релиза**.

```
helm install <release-name> <chart> [flags]
```

**Примеры:**
```
# Из локальной директории
helm install my-app ./my-chart

# С кастомным values-файлом
helm install my-app ./my-chart -f prod-values.yaml

# С переопределением значений
helm install my-app ./my-chart --set replicaCount=5 --set image.tag=1.28

# Из репозитория
helm install my-app bitnami/nginx -f values.yaml
```

**Что происходит при install:**
1. Helm читает чарт, сливает values (по умолчанию + пользовательские + `--set`).
2. Рендерит все шаблоны в `templates/` с подстановкой значений.
3. Применяет рендеренные YAML в Kubernetes (`kubectl apply`).
4. Создаёт **release record** — Secret/ConfigMap (зависит от storage backend) с метаданными релиза и рендеренными манифестами.

### 3.2 helm upgrade

Обновление существующего релиза.

```
helm upgrade my-app ./my-chart -f prod-values.yaml
```

**Поведение:**
* Если релиза нет — ошибка (нужно `--install` для upsert).
* Если чарт не изменился, но values изменились — манифесты перерендериваются.
* Kubernetes сам решает, что изменилось, и применяет diff (rolling update для Deployment, recreate для неизменяемых ресурсов).

**`--install`**: создать релиз, если его нет (upsert).
```
helm upgrade --install my-app ./my-chart -f values.yaml
```

### 3.3 helm rollback

Откат на предыдущую ревизию.

```
helm rollback my-app          # на предыдущую (release.revision - 1)
helm rollback my-app 3        # на конкретную ревизию
```

Helm хранит историю релизов (по умолчанию до 10 ревизий, настраивается `--history-max`). Каждая ревизия — полный набор рендеренных манифестов и values на момент установки.

### 3.4 helm uninstall

Удаление релиза и всех связанных ресурсов.

```
helm uninstall my-app
```

**`--keep-history`**: не удалять историю релиза (ресурсы Kubernetes удалятся, но record останется).

### 3.5 Deploy vs Redeploy: install, upgrade и «с нуля»

**Deploy (install)** — создание нового релиза, которого ещё нет в кластере:
```
helm install my-app ./my-chart
```
* Все ресурсы создаются **с нуля** (`kubectl create`).
* Helm записывает release record (ревизия 1).
* Если релиз с таким именем уже есть — ошибка.

**Redeploy (upgrade)** — обновление существующего релиза:
```
helm upgrade my-app ./my-chart
```
* Helm вычисляет **diff** между рендеренными манифестами новой и предыдущей ревизий.
* Ресурсы, которые изменились — обновляются **in-place** (`kubectl apply`).
* Ресурсы, которых больше нет в новой версии — **удаляются**.
* Новые ресурсы — **создаются**.
* Helm записывает новую ревизию (revision +1).

**Полный redeploy с нуля** — удалить и установить заново:
```
helm uninstall my-app
helm install my-app ./my-chart
```
* Все ресурсы удаляются и создаются заново.
* Pod'ы проходят полный цикл: terminate → create → pull image → start → probes.
* **Даунтайм равен времени от `helm uninstall` до `Running` новых Pod'ов**.
* Можно избежать даунтайма через:
  ```
  helm install my-app-v2 ./my-chart   # другой release name
  # переключить трафик на my-app-v2
  helm uninstall my-app               # удалить старый
  ```

**`helm upgrade --install`** — идемпотентный deploy:
```
helm upgrade --install my-app ./my-chart
```
* Если релиза нет → `install`.
* Если релиз есть → `upgrade`.
* Стандарт для CI/CD: одна команда, не нужно знать, первый это деплой или нет.

**Когда что использовать:**

| Сценарий | Команда |
|----------|---------|
| Первый деплой | `helm install` |
| Обновление конфига/образа | `helm upgrade` |
| CI/CD (любой деплой) | `helm upgrade --install` |
| Сломалось состояние, нужен чистый старт | `helm uninstall` + `helm install` |
| Смена major-версии с несовместимыми изменениями | Два релиза с разными именами + switch трафика |

**Что происходит с Pod'ами при upgrade:**
* Deployment с RollingUpdate: Pod'ы перекатываются постепенно (старые → новые), даунтайма нет.
* Deployment с Recreate: все старые Pod'ы удаляются, новые создаются — даунтайм.
* Если чарт не меняет Pod spec (image/tag, env) — Pod'ы **не пересоздаются**. Helm обновил ConfigMap, но Pod не перезапущен (нужен Reloader или `helm rollback` + `helm upgrade` с аннотацией-хэшем).

### 3.6 helm list

Список установленных релизов.

```
helm list                    # все релизы в текущем namespace
helm list -A                 # во всех namespace
helm list -a                 # включая неудачные и удалённые (с --keep-history)
```

### 3.7 helm history

История ревизий релиза.

```
helm history my-app
```

Вывод:
```
REVISION  UPDATED                   STATUS          CHART           APP VERSION  DESCRIPTION
1         Mon Jun 9 10:00:00 2026   superseded      my-app-1.0.0    2.4.0        Install complete
2         Mon Jun 9 11:00:00 2026   superseded      my-app-1.1.0    2.5.0        Upgrade complete
3         Tue Jun 10 09:00:00 2026  deployed        my-app-1.1.0    2.5.0        Rollback to 2
```

## 4. helm template и helm lint

### 4.1 helm template

Рендерит шаблоны локально, без применения в кластер. Полезно для отладки и CI/CD.

```
helm template my-app ./my-chart -f prod-values.yaml
```

Вывод — чистые Kubernetes-манифесты, которые можно направить в `kubectl apply` или сохранить в файл:
```
helm template my-app ./my-chart -f prod-values.yaml > rendered.yaml
```

### 4.2 helm lint

Проверяет чарт на ошибки.

```
helm lint ./my-chart
```

Проверяет:
* синтаксис YAML в `Chart.yaml`, `values.yaml`,
* синтаксис шаблонов,
* наличие обязательных полей,
* корректность именованных шаблонов.

## 5. helm get и helm status

### 5.1 helm status

Статус релиза и текущая ревизия.

```
helm status my-app
```

### 5.2 helm get

Извлечение информации о релизе:

```
helm get values my-app           # values, с которыми установлен релиз (включая --set)
helm get values my-app -a        # все values (включая значения по умолчанию)
helm get manifest my-app         # рендеренные манифесты текущей ревизии
helm get notes my-app            # NOTES.txt
helm get all my-app              # всё вместе
```

## 6. helm repo

Управление Helm-репозиториями.

```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo list
helm repo update                 # обновить индексы всех репозиториев
helm search repo nginx           # поиск чартов
helm repo remove bitnami
```

**Что такое Helm-репозиторий:** HTTP-сервер с `index.yaml` (список чартов и версий) и `.tgz`-архивами чартов.

## 7. helm package

Упаковывает чарт в `.tgz`-архив.

```
helm package ./my-chart           # создаёт my-chart-1.2.3.tgz
```

Архив можно загрузить в Helm-репозиторий.

## 8. Hooks

**Hooks** — действия, выполняемые в определённые моменты жизненного цикла релиза.

Типы хуков:
| Hook | Момент выполнения |
|------|------------------|
| `pre-install` | До создания ресурсов |
| `post-install` | После создания всех ресурсов |
| `pre-delete` | До удаления ресурсов |
| `post-delete` | После удаления ресурсов |
| `pre-upgrade` | До обновления |
| `post-upgrade` | После обновления |
| `pre-rollback` | До отката |
| `post-rollback` | После отката |
| `test` | При `helm test` |

**Пример: Job для миграции БД (pre-upgrade hook):**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "my-app.fullname" . }}-db-migrate
  annotations:
    "helm.sh/hook": pre-upgrade
    "helm.sh/hook-weight": "5"          # порядок выполнения
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: db-migrate
        image: "{{ .Values.image.repository }}-migrations:{{ .Values.image.tag }}"
```

**Hook delete policies:**
| Политика | Эффект |
|----------|--------|
| `before-hook-creation` | Удалить предыдущий hook перед запуском нового |
| `hook-succeeded` | Удалить hook после успешного выполнения |
| `hook-failed` | Удалить hook после ошибки |

**`hook-weight`**: хуки выполняются в порядке возрастания веса (сначала 0, потом 5, потом 10).

## 9. Чек-лист

1. **Chart.yaml**: `apiVersion: v2`, `name`, `version`, `appVersion`.
2. **values.yaml**: все параметры имеют значения по умолчанию.
3. **templates/**: каждый файл — валидный Kubernetes-ресурс.
4. **`_helpers.tpl`**: стандартные шаблоны `fullname`, `labels`, `selectorLabels`.
5. **NOTES.txt**: инструкция после установки.
6. **`.helmignore`**: исключены временные файлы.
7. **`helm lint`** проходит без ошибок.
8. **`helm template`** рендерит корректные манифесты.

## 10. Типичные ошибки

1. **Забыт `apiVersion: v2` в Chart.yaml**: Helm выдаст ошибку о несовместимости.
2. **`--set` с точкой в ключе**: экранировать: `--set "image.tag=v2"` или `--set image.tag=v2`.
3. **Путать `helm install` и `helm upgrade`**: `helm install` только для новых релизов. Использовать `helm upgrade --install` для идемпотентности.
4. **Ресурсы без аннотаций хука создаются и удаляются с релизом**: Job для миграции БД без hook-аннотации удалится вместе с `helm uninstall`.
5. **Значение в `values.yaml` не совпадает с ожидаемым в шаблоне**: проверять `helm get values` и `helm template`.
6. **Hook без `hook-delete-policy`**: старые Job остаются в кластере после отработки.