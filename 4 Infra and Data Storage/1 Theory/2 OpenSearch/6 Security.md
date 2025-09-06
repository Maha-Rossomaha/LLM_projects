# Security

OpenSearch поставляется с встроенным **Security Plugin** (форк X-Pack Security), который обеспечивает контроль доступа, шифрование и аутентификацию.

---

## 1. Security Plugin

* Входит в OpenSearch по умолчанию (начиная с версии 1.0).
* Обеспечивает:

  * RBAC (роли, пользователи),
  * TLS (внутренний и внешний трафик),
  * Аутентификацию (Basic, SAML, OIDC),
  * Field-level / Document-level security,
  * Аудит и API доступа.

Все настройки — через файл `config/opensearch-security/config.yml` **или API**:

```bash
POST _plugins/_security/api/roles
```

---

## 2. Пользователи и роли (RBAC)

OpenSearch использует **RBAC (role-based access control)**:

* Пользователю назначаются роли.
* Роли определяют права на индексы, поля, документы и операции.

### 2.1. Пользователь

Хранится в `internal_users.yml`, если не используется внешняя аутентификация.

```yaml
user1:
  hash: "$2y$12$hashedpass..."
  roles:
    - my_reader
```

### 2.2. Роль

```yaml
my_reader:
  cluster_permissions:
    - "cluster_composite_ops"
  index_permissions:
    - index_patterns:
        - "logs-*"
      allowed_actions:
        - "read"
```

> `cluster_permissions`: права на операции кластера (мониторинг, управление).
> `index_permissions`: шаблон индексов и допустимые действия.

Полезные `allowed_actions`:

* `read` — поиск и агрегации.
* `write` — запись.
* `delete` — удаление.
* `manage` — создание индексов.
* `indices_all` — все действия над индексами.

---

## 3. Field-level & Document-level Security

### 3.1. Field-level

Ограничение доступа на уровне полей:

```yaml
index_permissions:
  - index_patterns: ["sensitive-data"]
    allowed_actions: ["read"]
    field_permissions:
      - exclude: ["ssn", "salary"]
```

### 3.2. Document-level

Ограничение доступа к документам по query DSL:

```yaml
index_permissions:
  - index_patterns: ["sales"]
    allowed_actions: ["read"]
    document_level_security:
      filter:
        term:
          region: "EMEA"
```

> Пользователь видит только документы, где `region = EMEA`.

---

## 4. Аутентификация

OpenSearch поддерживает несколько механизмов:

### 4.1. Basic Auth (по умолчанию)

* Пользователь + пароль (из internal\_users или LDAP).
* Передаётся в `Authorization: Basic ...`.

```bash
curl -u admin:admin https://localhost:9200/_cat/indices
```

### 4.2. SAML (SSO через Identity Provider)

```yaml
authc:
  saml_auth_domain:
    http_enabled: true
    type: saml
    config:
      idp:
        metadata_url: https://idp.org/metadata.xml
      sp:
        entity_id: opensearch
        kibana_url: https://dashboards.myorg.com
```

### 4.3. OIDC (OAuth2 / OpenID Connect)

```yaml
authc:
  oidc_auth_domain:
    http_enabled: true
    type: openid
    config:
      openid_connect_url: https://accounts.google.com/.well-known/openid-configuration
```

> В обоих случаях можно настраивать маппинг групп IdP в роли OpenSearch.

---

## 5. TLS: внутренняя и внешняя

### 5.1. Внешняя TLS

* Шифрует клиентский трафик (`https://localhost:9200`).
* Включается в `opensearch.yml`:

```yaml
plugins.security.ssl.http.enabled: true
plugins.security.ssl.http.pemkey_file: node-key.pem
plugins.security.ssl.http.pemcert_file: node-cert.pem
```

### 5.2. Внутренняя TLS (node-to-node)

* Шифрует трафик между узлами кластера.
* Обязателен для работы Security Plugin.

```yaml
plugins.security.ssl.transport.enabled: true
plugins.security.ssl.transport.pemkey_file: node-key.pem
plugins.security.ssl.transport.pemcert_file: node-cert.pem
plugins.security.ssl.transport.enforce_hostname_verification: false
```

> Без включённого TLS между узлами — RBAC не работает.

---

## 6. Чеклист для продакшена

- Настроен TLS для обоих каналов: `http` (внешний) и `transport` (внутренний, node-to-node).
- Удалён или переименован дефолтный пользователь `admin`, пароль заменён на сильный.
- Включён аудит действий (`plugins.security.audit.type: internal_elasticsearch`).
- Пользователи и роли настроены по принципу наименьших привилегий.
- Использованы `field_permissions` для сокрытия чувствительных полей.
- Настроен `document_level_security` для разделения данных между регионами, отделами, ролями.
- Kibana (OpenSearch Dashboards) защищён входом через SSO (SAML или OIDC).
- Использован внешний IdP, если есть корпоративная система авторизации (Keycloak, Okta, Azure AD).
- Запрещён доступ к системным API для обычных ролей (`_cluster`, `_nodes`, `_cat`).
- Проверен и протестирован fallback на Basic Auth при недоступности SAML/OIDC.
- Настроен бэкап security-конфигурации (internal\_users.yml, config.yml, trusted\_certs).
- Все настройки проверены через `/_plugins/_security/health` и `/_plugins/_security/api/permissionsinfo`.
- Разрешения для Kibana прописаны в `kibana_server` роли (если используется Dashboards).
- В конфигурации отключены `anonymous_auth_enabled`, если не требуется публичный доступ.
- Установлены ограничения на размер и срок действия токенов (`jwt`, `session`, `basic_cache`).
- Проверен кросс-доступ между индексами и ролями (например, администратор не видит документы юзеров).
- Проверена работа DLS/FLS через Dashboards и напрямую через REST API.
- Включён REST API rate-limiting через внешние reverse proxy или API Gateway (если требуется).
