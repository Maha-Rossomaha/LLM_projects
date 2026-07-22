# Project status: MMS GraphQL MCP

Дата фиксации: 2026-06-24.

## Текущее состояние

Сформирован MVP-каркас сервиса:

- stateless MCP Streamable HTTP;
- один tool `execute_graphql`;
- опциональная проверка пользователя и роли `AppUser`;
- технический `POST /api/v1/tech-login` через mTLS;
- GET SDL и POST GraphQL через внешний MMS route;
- локальная проверка запроса по `external-api.graphql`;
- запрет mutation/subscription и introspection `__schema`/`__type`;
- лимиты query length/depth/selected fields/first/response size;
- Origin whitelist;
- санация ошибок и структурированные логи;
- Dockerfile и CI contract draft.

Snapshot пока является непроверенной проектной версией, а не готовым release.

## Открытые вопросы к владельцу MMS

1. Нужна ли проверка старого пользовательского flow и роли `AppUser` перед каждым
   запросом к внешнему техническому API, или достаточно технической mTLS-identity?
2. Точный контракт `tech-login`: поле `accessToken`, срок жизни, JWT ли это,
   присутствует ли `exp`, возможна ли досрочная ревокация.
3. Нужно ли получать новый token на каждый вызов или MMS рекомендует TTL-cache.
4. Как MMS отвечает при истёкшем token: HTTP 401, GraphQL error или иной статус.
5. Точный content type и формат GET GraphQL-схемы.
6. Обязателен ли `first` для всех корневых списковых полей; есть ли исключения.
7. Какие корневые сущности и примерные запросы использовать для smoke-тестов.
8. Какие GraphQL error codes считаются безопасными для передачи клиенту.
9. Какие ограничения нагрузки/rate limits действуют на `tech-login` и GraphQL.

## Открытые вопросы к DevOps

1. FQDN/URL MMS для каждого стенда и маршруты между namespace/стендами.
2. Egress, NetworkPolicy, service mesh и DNS-разрешение.
3. Выпуск клиентского сертификата, whitelist FQDN на стороне MMS и Secret mount.
4. Формат CA/cert/key, пути в pod, права пользователя `1001`, зашифрован ли key.
5. Ротация Secret: рестарт pod или нужен reload SSLContext без рестарта.
6. Нужны ли `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`; значение `HTTP_TRUST_ENV`.
7. Сохраняет ли ingress/service mesh входящий `Authorization` header.
8. Конкретный Origin whitelist.
9. Нужны ли proxy headers и доверенные адреса ingress.
10. Подходят ли base image/digest, `ci-docker-base-image` и Helm chart version.
11. Поддерживает ли CI BuildKit secret `pip_conf`.
12. Resource requests/limits, liveness/readiness probes и количество replicas.

## Технические вопросы по коду

1. Проверить API `mcp==1.27.1`: `Server`, `call_tool`, request context,
   `StreamableHTTPSessionManager` и параметры `stateless/json_response`.
2. Проверить совместимость точных pins Starlette/Uvicorn с `mcp==1.27.1`.
3. Проверить `graphql-core==3.2.11`: `parse(max_tokens=...)`,
   `validate(max_errors=...)`, `get_variable_values`.
4. Добавить ограничение размера исходящего GraphQL JSON body.
5. Решить, нужна ли проверка HTTPS-only для внешних URL в production.
6. Решить readiness semantics: только готовность MCP manager или также доступность
   mounted TLS files/локальной схемы/MMS.
7. Проверить политику логирования PII и отдельного audit trail.
8. Решить, нужен ли отдельный безопасный tool для описания части GraphQL-схемы.

## План тестирования

### Этап 0 — сразу, без сети

- установить зависимости в чистое Python 3.12 окружение;
- выполнить `python -m compileall src`;
- импортировать каждый модуль;
- создать `Settings` с временными файлами;
- загрузить `external-api.graphql` и построить GraphQLSchema;
- сверить API MCP SDK с текущим `transport/server.py`.

Критерий: сервис собирается и импортируется без ошибок.

### Этап 1 — unit/component с mock transport

Проверить:

- parsing env и `allowed_origins`;
- Origin policy;
- JWT payload decode и роль `AppUser`;
- отсутствие cookie/state между user login calls;
- query syntax/schema/variables/operationName;
- mutation, subscription, introspection, fragment cycles;
- query depth, selected fields, `first`, pagination;
- response-size limit;
- санацию GraphQL errors и логов;
- поведение 401/403/3xx/5xx/timeout/malformed JSON;
- отсутствие токенов/query/variables/data в логах.

Критерий: все негативные сценарии отклоняются до реального MMS либо возвращают
безопасные ошибки.

### Этап 2 — локальный mTLS component test

Поднять тестовый HTTPS server с временным тестовым CA и клиентским сертификатом.
Проверить handshake, неправильный CA/cert/key, redirect, timeout и большой response.
Это только тестовые сертификаты; production certificates не нужны.

### Этап 3 — smoke в Kubernetes DEV

Preconditions:

- DevOps смонтировали DEV Secret;
- сертификат whitelisted в MMS;
- маршрут и DNS работают;
- известны реальные URL;
- решён вопрос `AppUser`;
- есть тестовый пользователь и безопасный GraphQL query.

Порядок:

1. `/health` и `/ready`;
2. исходящий TLS/mTLS handshake;
3. `POST tech-login` без body;
4. GET GraphQL SDL;
5. один минимальный GraphQL query с `first=1`;
6. запрос с variables и operationName;
7. отрицательные 401/403/invalid query/oversized first.

### Этап 4 — MCP end-to-end

Реальным MCP-клиентом проверить:

- initialize;
- tools/list;
- tools/call `execute_graphql`;
- передачу `Authorization` через ingress;
- Origin policy;
- JSON result и `isError`;
- параллельные вызовы и отсутствие смешивания токенов/ответов.

### Этап 5 — безопасность и надёжность

- malformed Authorization/Origin;
- redirect и SSRF assumptions;
- утечки в логах;
- большие query/variables/response;
- invalid TLS chain/hostname/expired certificate;
- pod restart и несколько replicas;
- поведение при ротации Secret;
- временные 502/503/504.

### Этап 6 — нагрузка и оптимизация

Только после корректной интеграции измерить:

- стоимость `tech-login` на каждый вызов;
- стоимость нового `AsyncClient` на каждый запрос;
- latency и throughput;
- необходимость token TTL-cache;
- необходимость connection pooling;
- допустимую retry-политику.

## Что не добавлять до результатов тестов

- кэш GraphQL-ответов;
- автоматические retry без ограничений;
- общий пользовательский HTTP client с cookies;
- production сертификаты в repository/image;
- логирование токенов, query, variables или GraphQL data.

## Ближайший следующий шаг

1. Поднять чистое Python 3.12 окружение.
2. Установить `src/requirements.txt`.
3. Исправить все несовместимости с точными версиями SDK.
4. Добавить минимальный набор unit-тестов.
5. После этого готовить DEV smoke совместно с MMS owner и DevOps.
