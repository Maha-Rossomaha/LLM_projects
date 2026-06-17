# CD Configs: Structure and Layering

## 1. Что такое CD Config

**CD Config (Continuous Delivery config)** — набор конфигурационных файлов, которые описывают деплой приложения в конкретное окружение. Отделяет код приложения от конфигурации деплоя.

**Проблема:** один Docker-образ → много окружений (dev, staging, prod, prod-eu, prod-us). Разные реплики, ресурсы, домены, secrets. Всё это не в коде приложения, а в CD-конфигах.

## 2. Модель наслоения конфигов

```
base.yaml         ← общее для всех окружений
├── dev.yaml      ← поверх базы для dev
├── staging.yaml  ← поверх базы для staging
└── prod.yaml     ← поверх базы для prod
    ├── prod-eu.yaml   ← региональные переопределения
    └── prod-us.yaml
```

**Принцип:** более специфичный файл переопределяет значения из более общего.

### 2.1 Пример base.yaml

```yaml
# base.yaml
replicaCount: 2

image:
  repository: my-registry/my-app

service:
  type: ClusterIP
  port: 80

resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: 1000m
    memory: 512Mi

probes:
  liveness:
    path: /healthy
  readiness:
    path: /ready
```

### 2.2 Пример prod.yaml

```yaml
# prod.yaml — переопределяет базу для продакшена
replicaCount: 5

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

ingress:
  enabled: true
  host: my-app.example.com
  tls: true
```

### 2.3 Пример prod-eu.yaml

```yaml
# prod-eu.yaml — региональные переопределения
ingress:
  host: my-app.eu.example.com

nodeAffinity:
  required:
    topology.kubernetes.io/region: eu-west-1
```

### 2.4 Применение с Helm

```
helm upgrade --install my-app ./my-chart \
  -f configs/base.yaml \
  -f configs/prod.yaml \
  -f configs/prod-eu.yaml \
  --set image.tag=$CI_COMMIT_SHORT_SHA
```

Порядок: последний `-f` переопределяет предыдущий, `--set` переопределяет все.

## 3. Структура cd_config (типовая)

```
deploy/
├── charts/
│   └── my-app/              # Helm-чарт
├── configs/
│   ├── base.yaml            # общие значения
│   ├── dev.yaml
│   ├── staging.yaml
│   ├── prod.yaml
│   └── regions/
│       ├── prod-eu.yaml
│       └── prod-us.yaml
├── envs/
│   ├── dev/
│   │   └── secrets.yaml.enc # зашифрованные секреты (sops)
│   ├── staging/
│   │   └── secrets.yaml.enc
│   └── prod/
│       └── secrets.yaml.enc
└── pipelines/
    ├── .gitlab-ci.yml       # CI/CD пайплайн
    └── deploy.sh            # скрипт деплоя
```

### 3.1 Разделение по зонам ответственности

| Директория | Содержимое | Кто меняет |
|-----------|-----------|------------|
| `charts/` | Helm-чарт (шаблоны) | Platform/DevOps team |
| `configs/base.yaml` | Параметры по умолчанию | Platform team |
| `configs/prod.yaml` | Продакшен-параметры | DevOps + SRE |
| `configs/regions/` | Региональные переопределения | SRE per region |
| `envs/*/secrets.yaml.enc` | Секреты (зашифрованы) | DevOps, security team |
| `pipelines/` | CI/CD логика | DevOps |

## 4. Типовые параметры CD-конфига

### 4.1 env.yaml (параметры окружения)

```yaml
# env.yaml
replicaCount: 3
namespace: production
environment: prod

image:
  tag: ""                    # заполняется CI

ingress:
  enabled: true
  host: my-app.example.com
  tls: true
  tlsSecretName: my-app-tls

serviceAccount:
  create: true
  name: my-app-sa
```

### 4.2 affinity.yaml (правила размещения)

```yaml
# affinity.yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app: my-app
      topologyKey: kubernetes.io/hostname

  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values:
          - compute
```

### 4.3 canary.yaml (канареечный деплой)

```yaml
# canary.yaml — отдельный релиз с -canary суффиксом
replicaCount: 1                  # малая доля
image:
  tag: ""                        # новый тег

ingress:
  host: canary.my-app.example.com
  weight: 5                      # 5% трафика (через Istio)

resources:
  requests:
    cpu: 250m
    memory: 256Mi
```

**Процесс:**
1. Деплой canary с 5% трафика.
2. Мониторинг метрик (latency, error rate, memory).
3. Если ОК — promotion: обновление основного релиза, удаление canary.

### 4.4 checks.yaml (health checks окружения)

```yaml
# checks.yaml — проверки после деплоя
postDeploy:
  healthCheck:
    url: https://my-app.example.com/ready
    expectedStatus: 200
    retries: 10
    interval: 5s
  smokeTest:
    command: ["python", "-m", "tests.smoke"]
  rollbackOnFailure: true
  rollbackTimeout: 300            # 5 минут на откат
```

## 5. Secrets в CD-конфиге

### 5.1 sops + helm-secrets

```yaml
# secrets.yaml (unencrypted)
postgresql:
  auth:
    password: super-secret-pass

api:
  keys:
    openai: sk-abc123...
```

```
# шифрование
sops --encrypt secrets.yaml > secrets.yaml.enc
```

**В CI:**
```
helm secrets upgrade --install my-app ./charts/my-app \
  -f configs/base.yaml \
  -f configs/prod.yaml \
  -f envs/prod/secrets.yaml.enc
```

### 5.2 Внешний vault (HashiCorp Vault)

```yaml
# vault-config.yaml — ссылки, а не значения
postgresql:
  auth:
    password: "vault:secret/data/my-app/db#password"

api:
  keys:
    openai: "vault:secret/data/my-app/api#openai_key"
```

Helm-плагин или init-контейнер разрешает ссылки при деплое.

## 6. Multi-environment pipeline

### 6.1 GitLab CI пример

```yaml
stages:
  - test
  - build
  - deploy-dev
  - deploy-staging
  - deploy-prod

deploy-dev:
  stage: deploy-dev
  script:
    - helm upgrade --install my-app ./charts/my-app
      -f configs/base.yaml
      -f configs/dev.yaml
      --set image.tag=$CI_COMMIT_SHORT_SHA
  environment:
    name: dev
  only:
    - develop

deploy-staging:
  stage: deploy-staging
  script:
    - helm upgrade --install my-app ./charts/my-app
      -f configs/base.yaml
      -f configs/staging.yaml
      --set image.tag=$CI_COMMIT_SHORT_SHA
  environment:
    name: staging
  only:
    - main

deploy-prod:
  stage: deploy-prod
  script:
    - helm upgrade --install my-app ./charts/my-app
      -f configs/base.yaml
      -f configs/prod.yaml
      --set image.tag=$CI_COMMIT_SHORT_SHA
  environment:
    name: production
  only:
    - main
  when: manual                    # ручное подтверждение
```

## 7. Чек-лист

1. **База и слои**: base.yaml → env.yaml → region.yaml, каждый переопределяет предыдущий.
2. **Секреты отдельно**: `envs/<env>/secrets.yaml.enc` или vault, не в общих конфигах.
3. **Ресурсы окружений**: dev/staging/prod — разные requests/limits, replicaCount.
4. **Health checks**: post-deploy smoke test + автоматический откат при провале.
5. **Canary**: отдельный values-файл с малым весом трафика.
6. **Pipeline**: dev (авто) → staging (авто) → prod (ручное подтверждение).

## 8. Типичные ошибки

1. **Деплой в prod с dev-конфигом**: неверный `-f` порядок, переопределивший prod-значения.
2. **Секреты в base.yaml**: закоммичены в Git, доступны всем. Использовать `envs/*/secrets.yaml.enc`.
3. **Canary и основной релиз с одинаковым selector**: трафик идёт на оба пула Pod'ов. Разные имена или labels.
4. **Post-deploy check без таймаута**: висит бесконечно, блокирует pipeline.
5. **Жёсткие значения регионов в base.yaml**: должны быть в `regions/*.yaml`.