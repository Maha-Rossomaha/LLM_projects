# Helm: Templating, Dependencies and Advanced

## 1. Go Templates — синтаксис

### 1.1 Базовый синтаксис

Helm использует **Go templates** (пакет `text/template`) с расширениями из **Sprig** (библиотека функций). Шаблоны в файлах `templates/*.yaml` обрабатываются движком Helm перед отправкой в Kubernetes.

**Двойные фигурные скобки** — подстановка значения:
```
{{ .Values.image.repository }}:{{ .Values.image.tag }}
```

Рендерится в:
```
nginx:1.27
```

**Условные операторы:**
```
{{ if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
...
{{ end }}
```

**Циклы:**
```
{{ range .Values.env }}
- name: {{ .name }}
  value: {{ .value }}
{{ end }}
```

**Управление пробелами:** дефис `-` убирает пробелы:
```
{{- .Values.name -}}
```
* `{{- ` — убрать пробелы слева
* ` -}}` — убрать пробелы справа

### 1.2 Встроенные объекты

Шаблон имеет доступ к контексту `.` (точка), который содержит:

| Объект | Описание |
|--------|----------|
| `.Release` | Информация о релизе: `.Release.Name`, `.Release.Namespace`, `.Release.Revision`, `.Release.Service` |
| `.Values` | Значения из `values.yaml` и переопределений (`--set`, `-f`) |
| `.Chart` | Метаданные из `Chart.yaml`: `.Chart.Name`, `.Chart.Version`, `.Chart.AppVersion` |
| `.Files` | Доступ к файлам внутри чарта (не шаблонам) |
| `.Capabilities` | Информация о кластере: `.Capabilities.KubeVersion`, `.Capabilities.APIVersions` |
| `.Template` | Информация о текущем шаблоне: `.Template.Name`, `.Template.BasePath` |

### 1.3 Pipeline и функции

**Pipeline** — цепочка функций (аналог `|` в Unix):
```
{{ .Values.image.tag | default "latest" | quote }}
```

Рендерится в:
```
"latest"
```

**Часто используемые функции:**

| Функция | Описание | Пример ввода | Результат |
|---------|----------|-------------|-----------|
| `default` | Значение по умолчанию | `{{ .x \| default 5 }}` | `5` |
| `quote` | Взять в кавычки | `{{ "hello" \| quote }}` | `"hello"` |
| `indent` | Добавить отступ | `{{ "line" \| indent 4 }}` | `    line` |
| `nindent` | Newline + indent | `{{ "line" \| nindent 4 }}` | `\n    line` |
| `upper` / `lower` | Регистр | `{{ "Helm" \| upper }}` | `HELM` |
| `trim` | Убрать пробелы по краям | `{{ " a " \| trim }}` | `a` |
| `b64enc` / `b64dec` | Base64 кодирование/декодирование | `{{ "secret" \| b64enc }}` | `c2VjcmV0` |
| `sha256sum` | SHA-256 хэш | `{{ "data" \| sha256sum }}` | `3a6e...` |
| `toString` / `toYaml` / `toJson` | Конвертация типов | `{{ .Values.env \| toYaml }}` | YAML-представление |
| `join` | Объединить список | `{{ list "a" "b" \| join "," }}` | `a,b` |

**Логические функции:**
```
{{ if and .Values.enabled (eq .Values.env "prod") }}
{{ if or (not .Values.tls.enabled) .Values.insecure }}
{{ if and (gt .Values.replicaCount 1) (lt .Values.replicaCount 10) }}
```

| Функция | Эквивалент |
|---------|-----------|
| `eq` | `==` |
| `ne` | `!=` |
| `lt` | `<` |
| `gt` | `>` |
| `and` / `or` / `not` | Логические операторы |

### 1.4 with и области видимости

**`with`** — меняет контекст (точку) на указанный объект:
```
{{- with .Values.resources }}
resources:
  {{- toYaml . | nindent 2 }}
{{- end }}
```

Внутри `with` точка `.` = `.Values.resources`. Если `.Values.resources` пустой — блок не выполняется.

**`$` — корневой контекст:** внутри `with` или `range` можно обратиться к корню через `$`:
```
{{- range .Values.env }}
- name: {{ .name }}
  app: {{ $.Chart.Name }}
{{- end }}
```

### 1.5 range и списки

```
{{- range $index, $element := .Values.ports }}
- port: {{ $element.port }}
  protocol: {{ $element.protocol | default "TCP" | upper }}
{{- end }}
```

**range с map (словарём):**
```
{{- range $key, $value := .Values.config }}
{{ $key }}: {{ $value | quote }}
{{- end }}
```

## 2. _helpers.tpl — именованные шаблоны

### 2.1 Определение и вызов

**`_helpers.tpl`** — файл в `templates/` для переиспользуемых шаблонов (конвенция: начинается с `_`, не рендерится сам по себе).

**Определение:**
```
{{- define "my-app.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
```

**Вызов из другого шаблона:**
```
{{ include "my-app.fullname" . }}
```

### 2.2 Стандартный набор _helpers.tpl

**`my-chart.name`:**
```
{{- define "my-chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}
```

**`my-chart.fullname`:**
```
{{- define "my-chart.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
```

**`my-chart.labels`:**
```
{{- define "my-chart.labels" -}}
app.kubernetes.io/name: {{ include "my-chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ include "my-chart.chart" . }}
{{- end }}
```

**`my-chart.selectorLabels`:**
```
{{- define "my-chart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "my-chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

**Использование в deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-chart.fullname" . }}
  labels:
    {{- include "my-chart.labels" . | nindent 4 }}
spec:
  selector:
    matchLabels:
      {{- include "my-chart.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "my-chart.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
```

### 2.3 template vs include

| Команда | Поведение |
|---------|-----------|
| `{{ template "name" . }}` | Старый синтаксис, нельзя использовать с pipeline (`|`) |
| `{{ include "name" . }}` | Рекомендуемый, возвращает строку, можно `| nindent`, `| quote` |

**Всегда использовать `include`.**

## 3. Работа с файлами в чарте

### 3.1 .Files.Get

Чтение содержимого файла из чарта (не из `templates/`):

```
{{ .Files.Get "config.json" }}
```

**Типовой сценарий — ConfigMap из JSON-файла:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "my-chart.fullname" . }}-config
data:
  config.json: |
    {{ .Files.Get "files/config.json" | nindent 4 }}
```

### 3.2 .Files.Glob

Поиск файлов по паттерну:

```
{{ range $path, $content := .Files.Glob "dashboards/*.json" }}
{{ base $path }}: |
  {{ $content | nindent 2 }}
{{ end }}
```

### 3.3 .Files.AsConfig и .Files.AsSecrets

Создание ConfigMap/Secret из всех файлов в директории:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "my-chart.fullname" . }}-configs
data:
  {{- (.Files.Glob "configs/*.yaml").AsConfig | nindent 2 }}
```

## 4. tpl — шаблонизация строк в values

Функция **`tpl`** позволяет интерпретировать строку из values как шаблон:

**values.yaml:**
```yaml
host: "{{ .Release.Name }}.example.com"
```

**Шаблон:**
```
{{- $host := tpl .Values.host . }}
value: {{ $host }}
```

Применяется когда в values нужны ссылки на другие переменные (`.Release.Name`, `.Chart.Name`).

## 5. Зависимости (Dependencies)

### 5.1 Объявление зависимостей

В `Chart.yaml`:
```yaml
dependencies:
- name: postgresql
  version: "12.1.0"
  repository: "https://charts.bitnami.com/bitnami"
  condition: postgresql.enabled
  alias: db              # псевдоним (вместо postgresql.fullname → db.fullname)
  tags:
  - database
- name: redis
  version: "18.0.0"
  repository: "https://charts.bitnami.com/bitnami"
  condition: redis.enabled
  tags:
  - cache
```

**condition**: ссылается на путь в `values.yaml`. Если `false` — зависимость не устанавливается.

**tags**: включают/выключают группу зависимостей:
```yaml
# values.yaml
tags:
  database: true
  cache: false
```

### 5.2 Скачивание зависимостей

```
helm dependency update ./my-chart
```

Скачивает чарты в `charts/` и создаёт `Chart.lock` (фиксирует версии и хэши).

```
helm dependency build ./my-chart
```

Скачивает зависимости на основе `Chart.lock` (без обновления).

### 5.3 Chart.lock

Генерируется `helm dependency update`:

```yaml
dependencies:
- name: postgresql
  version: 12.1.0
  repository: https://charts.bitnami.com/bitnami
  digest: sha256:abc123...
- name: redis
  version: 18.0.0
  repository: https://charts.bitnami.com/bitnami
  digest: sha256:def456...
```

**Должен коммититься в Git.** Гарантирует воспроизводимость — все используют одни и те же версии зависимостей.

### 5.4 Передача значений в subchart

Значения в subchart передаются через ключ, соответствующий имени чарта (или alias):

```yaml
# values.yaml родительского чарта
postgresql:
  enabled: true
  auth:
    username: app
    password: secure-pass
    database: appdb
  primary:
    persistence:
      size: 10Gi

redis:
  enabled: false
```

**Глобальные значения** (доступны всем subchart):
```yaml
global:
  imageRegistry: my-registry.local
  imagePullSecrets:
  - my-secret
```

Внутри subchart доступны как:
```
{{ .Values.global.imageRegistry }}
```

## 6. Library Charts

**Library chart** (`type: library`) — чарт, который содержит только шаблоны (в `templates/`), но не устанавливается сам. Используется для переиспользования логики.

**Chart.yaml:**
```yaml
apiVersion: v2
name: my-lib
type: library
version: 1.0.0
```

**templates/_deployment.tpl:**
```
{{- define "my-lib.deployment" -}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .name }}
spec:
  replicas: {{ .replicas }}
  ...
{{- end }}
```

**Использование в application-чарте:**
```yaml
dependencies:
- name: my-lib
  version: 1.0.0
  repository: "oci://my-registry.local/charts"
```

В шаблоне:
```
{{ include "my-lib.deployment" (dict "name" (include "my-app.fullname" .) "replicas" .Values.replicaCount) }}
```

## 7. Управление Values per Environment

### 7.1 Подход: несколько values-файлов

```
helm install my-app ./my-chart \
  -f values.yaml \              # база
  -f values/prod.yaml \         # окружение
  -f values/prod-eu.yaml        # регион
```

Helm сливает файлы в порядке передачи (последний переопределяет предыдущий).

### 7.2 Подход: --set для CI/CD

```
helm upgrade --install my-app ./my-chart \
  -f values.yaml \
  -f values/prod.yaml \
  --set image.tag=$CI_COMMIT_SHORT_SHA \
  --set replicaCount=3
```

### 7.3 Схема валидации values

**values.schema.json** — JSON Schema для валидации структуры `values.yaml`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "replicaCount": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100
    },
    "image": {
      "type": "object",
      "properties": {
        "repository": { "type": "string", "minLength": 1 },
        "tag": { "type": "string", "minLength": 1 }
      },
      "required": ["repository", "tag"]
    }
  },
  "required": ["replicaCount", "image"]
}
```

Helm проверяет `values.yaml` на соответствие схеме при `helm install/upgrade/template/lint`.

## 8. Helm Secrets

Проблема: `values.yaml` хранится в Git, но пароли в открытом виде коммитить нельзя.

### 8.1 helm-secrets (плагин)

```
helm plugin install https://github.com/jkroepke/helm-secrets
```

Позволяет шифровать values-файлы через **sops** (с KMS/GPG/Age):

```
# values.yaml → values.yaml.enc
helm secrets enc values.yaml
helm secrets dec values.yaml
helm secrets view values.yaml
helm secrets install my-app ./my-chart -f values.yaml.enc
```

### 8.2 Внешний vault

Values не хранятся в Git вообще — Helm получает их из HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager через кастомный плагин или обёртку в CI/CD.

### 8.3 Sealed Secrets

Запечатанные секреты (шифруются контроллером в кластере). В Git коммитится `SealedSecret`, в кластере расшифровывается в обычный Secret.

## 9. Hooks — углублённо

### 9.1 Порядок выполнения и вес

Хуки упорядочиваются по `helm.sh/hook-weight` (число, может быть отрицательным). Выполняются от меньшего к большему.

```
pre-install, weight=-5  →  pre-install, weight=0  →  pre-install, weight=5
```

### 9.2 Удаление хуков

`helm.sh/hook-delete-policy`:
| Политика | Эффект |
|----------|--------|
| `before-hook-creation` | Удалить предыдущий экземпляр хука перед запуском нового |
| `hook-succeeded` | Удалить после успешного выполнения |
| `hook-failed` | Удалить после ошибки |

Можно комбинировать через запятую: `hook-succeeded,before-hook-creation`.

### 9.3 helm test

Запуск тестов после установки:

```
helm test my-app
```

**Пример теста (Job):**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "my-app.fullname" . }}-test
  annotations:
    "helm.sh/hook": test
spec:
  template:
    spec:
      containers:
      - name: test
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        command: ["pytest", "/app/tests/"]
      restartPolicy: Never
```

Helm дожидается завершения Job: успех (`exit 0`) или ошибка (`exit ≠ 0`).

## 10. Чек-лист

1. **_helpers.tpl**: `fullname`, `labels`, `selectorLabels` — стандартный набор.
2. **Всегда `include`** вместо `template` для возможности `| nindent`.
3. **`with` и `range`** — помнить про потерю контекста, использовать `$` для корня.
4. **`tpl`** — когда values содержат ссылки на другие переменные.
5. **chart.lock** — закоммичен, `helm dependency update` при изменении версий.
6. **values.schema.json** — для валидации структуры.
7. **Секреты** — через helm-secrets или внешний vault, не в открытом виде в Git.
8. **Хуки** — c hook-delete-policy, чтобы не оставлять мусор в кластере.

## 11. Типичные ошибки

1. **`indent` vs `nindent`**: `indent N` добавляет N пробелов; `nindent N` сначала добавляет перенос строки, потом N пробелов. Для YAML-блоков почти всегда нужен `nindent`.
2. **Использование `template` вместо `include`**: `template` нельзя использовать с пайпами (`| indent`).
3. **Потеря контекста в `with`/`range`**: обращаться к `$.Values`, `$.Release`, `$.Chart`.
4. **Chart.lock не обновлён**: зависимости не совпадают с объявленными в Chart.yaml.
5. **Забыть `helm dependency update`**: в `charts/` нет нужных subchart'ов.
6. **Секреты в values.yaml закоммичены в Git**: использовать `helm-secrets` или vault.
7. **`tpl` без указания контекста**: `tpl .Values.host .` — точка как второй аргумент обязательна.
8. **Пустые блоки из-за `with`**: если `.Values.resources` не задан, блок пропускается молча. Проверять `helm template`.