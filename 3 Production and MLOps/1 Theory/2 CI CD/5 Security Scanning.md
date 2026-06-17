# Security Scanning in CI/CD

## 1. Что такое Security Scanning

**Security Scanning** — автоматический анализ кода, зависимостей, контейнеров и инфраструктуры на уязвимости, встроенный в CI/CD pipeline.

**Проблема без сканирования:** уязвимость в библиотеке (CVE в `log4j`, `requests`, `pillow`) или открытый секрет в коде — компрометация прода.

**Принцип:** Shift-left security — находить уязвимости как можно раньше, на этапе написания кода и CI, а не в проде.

## 2. Уровни Security Scanning

### 2.1 SAST (Static Application Security Testing)

Анализ исходного кода без его выполнения. Ищет известные паттерны уязвимостей: SQL-инъекции, XSS, path traversal, hardcoded секреты.

**Инструменты:**
| Инструмент | Особенности |
|-----------|-------------|
| **Semgrep** | Правила на YAML, быстро, сообщество |
| **Bandit** | Только Python, базовые проверки |
| **SonarQube** | Enterprise, много языков, quality gates |
| **Checkmarx** | Enterprise, глубокий анализ |
| **CodeQL** | GitHub, семантический анализ |

**Пример Semgrep:**
```yaml
# .semgrep.yml
rules:
- id: no-exec
  pattern: exec(...)
  message: "Avoid exec() — code injection risk"
  severity: ERROR
  languages: [python]
```

**В CI:**
```yaml
sast:
  stage: test
  image: semgrep/semgrep
  script:
    - semgrep ci --config auto
```

### 2.2 SCA (Software Composition Analysis)

Анализ зависимостей (requirements.txt, package.json, go.mod) на известные уязвимости (CVE).

**Инструменты:**
| Инструмент | Особенности |
|-----------|-------------|
| **Snyk** | Огромная база CVE, интеграция с registry |
| **Safety** | Python-specific, сравнивает с vulnerability DB |
| **Trivy** | Универсальный (контейнеры + зависимости + IaC) |
| **OWASP Dependency Check** | Java/.NET фокус, бесплатный |
| **Dependabot** | GitHub native, авто-PR с обновлениями |

**Пример Safety в CI:**
```yaml
sca:
  stage: test
  image: python:3.11
  script:
    - pip install safety
    - safety check -r requirements.txt --full-report
```

**Gate:** нет CVE с severity ≥ HIGH.

### 2.3 Сканирование контейнеров

Анализ слоёв Docker-образа на уязвимости в системных пакетах и языке.

**Инструменты:**
| Инструмент | Особенности |
|-----------|-------------|
| **Trivy (Aqua)** | Бесплатный, сканирует образы, файловую систему, Git-репо |
| **Grype (Anchore)** | Бесплатный, быстрый |
| **Docker Scout** | Встроен в Docker Desktop/CLI |
| **Clair** | CoreOS/Quay, API-based |
| **Snyk Container** | Коммерческий, приоритезация |

**Пример Trivy в CI:**
```yaml
container-scan:
  stage: test
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL --no-progress $DOCKER_IMAGE
  # Если найдены HIGH/CRITICAL — exit 1
```

### 2.4 Сканирование секретов

Поиск закоммиченных секретов (API-ключей, паролей, токенов) в истории Git и текущем коде.

**Инструменты:**
| Инструмент | Особенности |
|-----------|-------------|
| **git-secrets** | AWS-специфические паттерны |
| **truffleHog** | Ищет энтропию и паттерны, проверяет GitHub |
| **Gitleaks** | Быстрый, множество правил, CI-friendly |
| **detect-secrets** | Yelp, baseline-файл |

**Пример Gitleaks:**
```yaml
secret-scan:
  stage: test
  image: zricethezav/gitleaks:latest
  script:
    - gitleaks detect --source . -v --no-git
```

### 2.5 PTB (Penetration Testing Baseline)

Автоматизированные пен-тесты на staging-окружении: OWASP top 10, проверка заголовков, CORS, TLS.

**Инструменты:**
| Инструмент | Особенности |
|-----------|-------------|
| **OWASP ZAP** | Бесплатный, активное/пассивное сканирование |
| **Burp Suite** | Коммерческий стандарт |
| **Nikto** | Быстрый веб-сканер |

```yaml
pen-test:
  stage: staging-test
  image: owasp/zap2docker-stable
  script:
    - zap-baseline.py -t https://staging.example.com -r zap-report.html
  artifacts:
    paths:
      - zap-report.html
```

## 3. Типовой Security Pipeline

```yaml
stages:
  - security-scan
  - build
  - deploy

secret-scan:
  stage: security-scan
  image: zricethezav/gitleaks:latest
  script:
    - gitleaks detect --source . -v

sast:
  stage: security-scan
  image: semgrep/semgrep
  script:
    - semgrep ci --config auto --error

sca:
  stage: security-scan
  image: python:3.11
  script:
    - pip install safety
    - safety check -r requirements.txt --full-report

build:
  stage: build
  needs: [secret-scan, sast, sca]
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

container-scan:
  stage: security-scan
  needs: [build]
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL $DOCKER_IMAGE
```

## 4. Политики severity и блокировки

| Severity | Действие в CI | Когда |
|----------|-------------|-------|
| **CRITICAL** | Блокировать pipeline (exit 1) | Всегда |
| **HIGH** | Блокировать pipeline | В prod-ветке (main) |
| **MEDIUM** | Предупреждение (не блокирует) | Всегда |
| **LOW** | Информация в логах | — |

**Исключения:** если CVE не применима в конкретном контексте (не используется уязвимый метод), создаётся exception с обоснованием и сроком действия.

## 5. Container Security в рантайме (Kubernetes)

Security scanning не заканчивается CI. В проде через Pod Security Standards / Admission:

```yaml
# Pod Security Standard: Restricted (через labels namespace)
# Включает:
# - runAsNonRoot: true
# - readOnlyRootFilesystem: true
# - seccompProfile: RuntimeDefault
# - drop ALL capabilities
# - allowPrivilegeEscalation: false

apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: my-app:v1
    securityContext:
      readOnlyRootFilesystem: true
      allowPrivilegeEscalation: false
      capabilities:
        drop:
        - ALL
```

**Проверка при деплое (OPA / Kyverno):**
```yaml
# Kyverno policy: запретить runAsRoot
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-root-user
spec:
  validationFailureAction: Enforce
  rules:
  - name: runAsNonRoot
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "Running as root is not allowed"
      pattern:
        spec:
          securityContext:
            runAsNonRoot: true
```

## 6. Чек-лист

1. **SAST**: Semgrep / SonarQube в CI, блокировка на severity ≥ HIGH.
2. **SCA**: Safety / Snyk на каждое изменение зависимостей.
3. **Secret scanning**: Gitleaks / truffleHog, блокировка при обнаружении.
4. **Container scanning**: Trivy после сборки образа.
5. **PTB**: OWASP ZAP на staging.
6. **Runtime**: Pod Security Standards, readOnlyRootFilesystem, runAsNonRoot.
7. **Exception policy**: формальный процесс для обоснованных исключений.

## 7. Типичные ошибки

1. **Сканирование без блокировки**: уязвимости найдены, но pipeline зелёный — false sense of security.
2. **Зашитые секреты в старых коммитах**: `gitleaks` без `--no-git` не проверяет историю.
3. **Игнорирование severity LOW/MEDIUM**: низкий risk сегодня — входная точка завтра.
4. **Отсутствие runtime-политик**: CI чистый, но Pod запущен от root с `privileged: true`.
5. **SCA без фиксации версий в lock-файле**: транзитивные зависимости не сканируются.
6. **Исключения без срока действия**: CVE больше не применима, но exception висит годами.