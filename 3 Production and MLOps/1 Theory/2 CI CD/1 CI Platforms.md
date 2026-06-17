# CI Platforms Overview

## 1. Что такое CI

**Continuous Integration (CI)** — практика автоматической сборки, тестирования и валидации кода при каждом изменении в репозитории. Цель: как можно раньше обнаружить ошибки и получить готовый к деплою артефакт.

**Что даёт CI:**
* автоматический запуск тестов при push/MR,
* сборка Docker-образов,
* проверка стиля кода, линтеры,
* статический анализ безопасности (SAST),
* генерация артефакта (образ, пакет, wheel).

## 2. GitLab CI

### 2.1 Модель

GitLab CI встроен в GitLab. Конфигурация — `.gitlab-ci.yml` в корне репозитория.

**Ключевые концепты:**
* **Pipeline** — последовательность stages и jobs.
* **Stage** — группа job'ов, выполняемых параллельно.
* **Job** — конкретная задача (тесты, сборка, деплой).
* **Runner** — агент, который выполняет job'ы (shared, group-specific, project-specific).
* **Artifacts** — файлы, передаваемые между stage'ами (скомпилированный код, отчёты).

### 2.2 Пример .gitlab-ci.yml

```yaml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest --junitxml=report.xml
  artifacts:
    reports:
      junit: report.xml

build:
  stage: build
  image: docker:25
  services:
    - docker:25-dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

deploy:
  stage: deploy
  image: alpine:3.19
  script:
    - apk add --no-cache curl
    - |
      curl -X POST https://argocd.example.com/api/webhook \
        -H "Authorization: Bearer $ARGOCD_TOKEN"
  only:
    - main
```

### 2.3 Ключевые директивы

| Директива | Описание |
|-----------|----------|
| `stages` | Порядок stage'ов |
| `stage` | К какому stage принадлежит job |
| `script` | Команды для выполнения |
| `image` | Docker-образ для job'а |
| `services` | Сервисные контейнеры (БД, Docker-in-Docker) |
| `before_script` / `after_script` | Действия до/после основного script |
| `variables` | Переменные окружения (глобальные или per-job) |
| `artifacts` | Файлы для передачи между stage'ами |
| `cache` | Кэшируемые директории между pipeline'ами |
| `only` / `except` | Когда запускать job (ветки, теги, MR) |
| `rules` | Более гибкая альтернатива `only/except` |
| `needs` | Запускать job без ожидания всего stage (DAG) |
| `retry` | Автоматический повтор при ошибке |
| `timeout` | Таймаут job'а |
| `tags` | Какие runner'ы могут выполнять job |

### 2.4 Environment и Deployments

```yaml
deploy-prod:
  stage: deploy
  script: kubectl apply -f manifests/
  environment:
    name: production
    url: https://my-app.example.com
  only:
    - main
```

GitLab отслеживает историю деплоев, позволяет rollback через UI.

## 3. GitHub Actions

### 3.1 Модель

GitHub Actions встроен в GitHub. Конфигурация — YAML-файлы в `.github/workflows/`.

**Ключевые концепты:**
* **Workflow** — pipeline (один файл в `.github/workflows/`).
* **Job** — группа шагов, выполняется на одном runner'е.
* **Step** — скрипт или action.
* **Action** — переиспользуемая единица (свой или из marketplace).
* **Runner** — агент (GitHub-hosted: `ubuntu-latest`, `macos-latest`; self-hosted).

### 3.2 Пример workflow

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  DOCKER_IMAGE: ghcr.io/${{ github.repository }}:${{ github.sha }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest --junitxml=report.xml
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-report
          path: report.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ env.DOCKER_IMAGE }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: |
          curl -X POST https://argocd.example.com/api/webhook \
            -H "Authorization: Bearer ${{ secrets.ARGOCD_TOKEN }}"
```

### 3.3 Ключевые особенности

* **Matrix builds**: запуск job'а на нескольких комбинациях параметров:
  ```yaml
  strategy:
    matrix:
      python-version: ['3.9', '3.10', '3.11']
      os: [ubuntu-latest, macos-latest]
  ```
* **Reusable workflows**: вызов одного workflow из другого.
* **OIDC**: аутентификация в облачных провайдерах без хранения секретов.
* **Environments**: approval gates, protection rules, secrets per environment.

## 4. Jenkins

### 4.1 Модель

**Jenkins** — standalone CI-сервер (Java). Pipeline as Code через **Jenkinsfile** (Groovy DSL) в репозитории.

**Ключевые концепты:**
* **Pipeline** — декларативный (`pipeline {}`) или скриптовый.
* **Stage** — логический блок (Build, Test, Deploy).
* **Step** — конкретная команда.
* **Agent** — где выполняется (any, label, docker).
* **Node** — машина, подключённая к Jenkins master.

### 4.2 Пример Jenkinsfile (Declarative)

```groovy
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "my-registry/my-app:${GIT_COMMIT.take(7)}"
    }

    stages {
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pytest --junitxml=report.xml'
            }
            post {
                always {
                    junit 'report.xml'
                }
            }
        }

        stage('Build') {
            steps {
                sh '''
                    docker build -t $DOCKER_IMAGE .
                    docker push $DOCKER_IMAGE
                '''
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl set image deployment/my-app my-app=$DOCKER_IMAGE'
            }
        }
    }

    post {
        failure {
            // уведомление в Slack
            slackSend(color: 'danger', message: "Pipeline failed: ${env.BUILD_URL}")
        }
    }
}
```

### 4.3 Plugins

Jenkins экосистема строится на плагинах:
* **Git** — интеграция с Git-репозиториями.
* **Docker** / **Kubernetes** — динамические агенты.
* **Blue Ocean** — современный UI.
* **Credentials Binding** — безопасная работа с секретами.
* **Slack** / **Email** — уведомления.

## 5. ArgoCD (GitOps)

### 5.1 GitOps-модель

**ArgoCD** реализует **GitOps**: Git-репозиторий — единственный источник истины о состоянии кластера. ArgoCD следит за репозиторием и синхронизирует кластер.

```
Git-репо (желаемое состояние) → ArgoCD → Kubernetes cluster (фактическое состояние)
```

### 5.2 Ключевые концепты

* **Application** — группа Kubernetes-ресурсов, определённых в Git.
* **Project** — набор приложений с общими политиками.
* **Sync** — приведение кластера к состоянию в Git.
* **Drift detection** — обнаружение расхождений (кто-то поменял `kubectl edit`).
* **Auto-sync** — автоматическая синхронизация при изменении в Git.
* **Prune** — удаление ресурсов, которых больше нет в Git.
* **Self-heal** — автоматическое исправление drift.

### 5.3 Application CRD

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/my-org/my-app-deploy
    targetRevision: main
    path: overlays/prod
  destination:
    server: https://kubernetes.default.svc
    namespace: my-app
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

### 5.4 ArgoCD vs Push-based CD

| | Push (Jenkins, GitLab CI, Actions) | Pull (ArgoCD, Flux) |
|---|---|---|
| **Направление** | CI пушит в кластер | Оператор в кластере тянет из Git |
| **Доступ к кластеру** | CI имеет kubeconfig/key | Только оператор в кластере |
| **Drift detection** | Нет (push — разовое действие) | Постоянный мониторинг, self-heal |
| **Модель** | CI → кластер | Git → ArgoCD → кластер |

Часто используют вместе: CI собирает образы, ArgoCD деплоит.

## 6. Чек-лист

1. **Pipeline as Code**: конфигурация CI в репозитории (`.gitlab-ci.yml`, `.github/workflows/`, `Jenkinsfile`).
2. **Stages**: test → build → deploy (классическая цепочка).
3. **Артефакты**: образ в registry, отчёт о тестах.
4. **Секреты**: не в коде, через переменные CI / vault.
5. **Уведомления**: успех/ошибка в Slack/Email.
6. **GitOps**: ArgoCD для деплоя, auto-sync + drift detection.

## 7. Типичные ошибки

1. **Секреты в коде или логах CI**: использовать masked variables, никогда `echo $SECRET`.
2. **Все job'ы на одном runner'е**: медленно, не масштабируется. Разделять по тегам и типам.
3. **Отсутствие таймаутов**: job висит бесконечно, блокирует pipeline.
4. **Не чистить артефакты**: disk pressure на runner'ах. Настроить expiration.
5. **Прямой доступ CI к production-кластеру**: утечка kubeconfig — компрометация кластера. Использовать ArgoCD (pull).
6. **ArgoCD auto-sync без prune**: старые ресурсы не удаляются, накапливается мусор.