# План компетенций: Agents & MCP (Model Context Protocol)

> **Фокус:** архитектура AI-агентов, паттерны оркестрации (ReAct, LangGraph, Swarm, Reflexion), протокол MCP как стандарт интеграции агентов с внешними инструментами.  
> Темы CI/CD, деплоя, Kubernetes, Istio, мониторинга и безопасности вынесены в план `3 Production and MLOps`.

---

## I. AI-агенты — общая архитектура

- **Что такое AI-агент:** система, которая получает задачу на естественном языке, планирует шаги, вызывает инструменты (tools), интерпретирует результаты и формирует ответ.
- **Компоненты агента:**
  - LLM (мозг) — принимает промпт + контекст, генерирует план действий или ответ
  - Planner — разбивает сложную задачу на подзадачи (ReAct, Tree of Thoughts, Chain-of-Thought)
  - Tool Executor — вызывает внешние инструменты через стандартизированный протокол (MCP, function calling)
  - Memory — краткосрочная (контекст диалога) и долгосрочная (векторная БД, summary)
  - Orchestrator/Coordinator — маршрутизирует запросы между агентами и MCP-серверами
- **Типы агентов:**
  - Single-agent: один LLM + набор tools
  - Multi-agent: несколько специализированных агентов с оркестратором
  - Hierarchical: агент-планировщик + агенты-исполнители
- **Примеры фреймворков:** LangChain, LangGraph, AutoGen, CrewAI, Semantic Kernel, OpenAI Swarm.

---

## II. Архитектурные паттерны агентов

> 📝 Конспектов пока нет. Планируемые темы:

- **ReAct (Reasoning + Acting)**
  - Чередование мыслей (Thought) и действий (Action): агент думает → вызывает инструмент → получает Observation → думает дальше.
  - Почему это работает лучше, чем чистый CoT или чистое function calling.
  - Проблемы: зацикливание, потеря контекста, неэффективное перепланирование.

- **LangGraph**
  - Агент как граф состояний (state machine): узлы = шаги (LLM call, tool call), рёбра = переходы (conditional edges).
  - Циклы, ветвление, human-in-the-loop, параллельные ветки.
  - Отличие от линейных цепочек LangChain: граф позволяет вернуться назад, повторить, пропустить.

- **Plan-and-Execute**
  - Planner генерирует полный план до начала выполнения, Executor выполняет шаги последовательно.
  - Плюсы: прозрачность, возможность валидировать план перед выполнением.
  - Минусы: план может устареть при изменении контекста, нет адаптивности.

- **Swarm / Multi-agent**
  - Роевая оркестрация: множество агентов работают параллельно, передают контекст друг другу.
  - Паттерны взаимодействия: broadcast (всем), round-robin, delegate (передать конкретному), handoff.
  - Фреймворки: OpenAI Swarm, CrewAI, AutoGen, Microsoft Semantic Kernel.

- **Hierarchical Agents**
  - Агент-планировщик верхнего уровня + агенты-исполнители.
  - Планировщик декомпозирует задачу на подзадачи, назначает исполнителям, агрегирует результаты.
  - Отличие от Swarm: централизованное управление vs децентрализованное.

- **Reflexion**
  - Self-reflection: агент оценивает свой ответ через verbal reinforcement.
  - Если ответ не устраивает — агент сам себе даёт обратную связь и перепланирует.
  - Применение: code generation, reasoning, long-horizon tasks.

- **Tool Use / Function Calling**
  - Нативная поддержка вызова инструментов LLM (OpenAI, Anthropic, Gemini).
  - Сравнение подходов: native function calling vs MCP vs hand-crafted JSON parsing.
  - Schema-driven tool definition: JSON Schema для параметров, ограничения, enum'ы.

- **LangChain для агентов**
  - LangChain как фреймворк для построения агентов: модуль `langchain.agents`, AgentExecutor, инструменты, памяти.
  - Цепочки (Chains): LLMChain, SequentialChain, TransformChain — когда хватает линейной логики.
  - ReAct-агенты в LangChain: zero-shot-react, multi-input, custom tools.
  - Концепции: Tool, ToolKit, AgentType, AgentExecutor, AgentFinish.
  - Интеграция с MCP через `langchain-mcp-adapters`.
  - Связь с RAG-сценариями: [LangChain RAG Search Intro](../1%20NLP%20and%20LLM/1%20Theory/6%20LLM%20in%20Search%20and%20Rec/4%20LangChain%20RAG%20Search%20Intro.md).
  - Переход к LangGraph для сложных графовых сценариев (см. LangGraph выше).

- **Memory patterns**
  - Short-term: контекст диалога в промпте.
  - Long-term: векторная БД + retrieval по релевантности, summary compression.
  - Working memory: scratchpad для промежуточных вычислений агента.

- **Guardrails and Safety**
  - Input guardrails: проверка промпта до выполнения (prompt injection detection).
  - Output guardrails: валидация ответа агента (JSON Schema, regex, content filter).
  - Tool-level: подтверждение пользователя перед вызовом деструктивных инструментов (opt-in).
  - Jailbreak detection, rate limiting, allowlists/denylists.

- **Evaluation and Observability**
  - Метрики: task success rate, tool call accuracy, latency, token cost.
  - Трассировка: цепочка вызовов (agent → tool → LLM), OpenTelemetry.
  - Бенчмарки: AgentBench, SWE-bench, WebArena.
  - Офлайн vs онлайн оценка: replay, simulated users, human eval.

- **Agentic RAG**
  - Агент сам решает, когда искать (retrieve), когда переформулировать запрос, когда ответить.
  - Multi-step retrieval: агент делает несколько запросов к разным источникам.
  - Self-RAG: агент критикует найденные документы и решает, достаточно ли информации.


---

## III. Tools, Skills, Function Calling и Workflows

> 📝 Конспектов пока нет. Этот раздел нужен до подробного изучения MCP: сначала важно разделить сами возможности агента, механизмы их вызова и протокол подключения внешних возможностей.

### Планируемые конспекты

📄 [Tools and Skills Basics](1%20Theory/3%20Tools,%20Skills%20and%20Function%20Calling/1%20Tools%20and%20Skills%20Basics.md)  
📄 [Function Calling](1%20Theory/3%20Tools,%20Skills%20and%20Function%20Calling/2%20Function%20Calling.md)  
📄 [Tools vs Skills vs Workflows](1%20Theory/3%20Tools,%20Skills%20and%20Function%20Calling/3%20Tools%20vs%20Skills%20vs%20Workflows.md)  
📄 [MCP vs Native Tool Calling](1%20Theory/3%20Tools,%20Skills%20and%20Function%20Calling/4%20MCP%20vs%20Native%20Tool%20Calling.md)  
📄 [MCP Tools, Resources and Prompts](1%20Theory/3%20Tools,%20Skills%20and%20Function%20Calling/5%20MCP%20Tools,%20Resources%20and%20Prompts.md)

### Tool — атомарная операция

- **Tool:** конкретное действие, которое может вызвать агент или host-приложение.
- Примеры:
  - `search_documents(query)`;
  - `read_file(path)`;
  - `execute_sql(query)`;
  - `send_email(to, body)`.
- У tool обычно есть:
  - уникальное имя;
  - понятное модели описание;
  - схема входных аргументов;
  - реализация;
  - структура результата;
  - ограничения доступа и политика ошибок.
- Tool отвечает на вопрос: **«Какое конкретное действие система может выполнить?»**

### Skill — составная способность

- **Skill:** более высокоуровневая способность решать определённый класс задач.
- Skill может включать:
  - инструкции и domain knowledge;
  - один или несколько tools;
  - готовый workflow;
  - шаблоны промптов;
  - правила выбора действий;
  - валидацию результата.
- Примеры:
  - поиск информации в репозитории;
  - анализ pull request;
  - подготовка исследовательского отчёта;
  - построение учебного маршрута.
- Один skill может использовать несколько tools. Например, skill `repository_review` может последовательно вызывать `get_git_diff`, `search_code`, `read_file` и `run_tests`.
- Термин **skill менее стандартизирован**, чем tool: разные платформы могут называть skill пакет инструкций, специализированного агента, набор tools или готовый workflow.

### Function calling — механизм структурированного вызова

- **Function calling / tool calling:** механизм, при котором LLM выбирает доступный tool и формирует структурированные аргументы для его вызова.
- Типовой цикл:
  ```text
  Host передаёт модели список tools и их схемы
      ↓
  LLM выбирает tool и формирует arguments
      ↓
  Host валидирует аргументы
      ↓
  Tool Executor вызывает Python-функцию или внешний API
      ↓
  Результат возвращается модели
  ```
- Важно различать:
  - LLM **предлагает вызов**;
  - host-приложение **контролирует и выполняет вызов**;
  - tool **реализует действие**.
- Что изучить:
  - JSON Schema и strict structured outputs;
  - обязательные и необязательные параметры;
  - `enum`, ограничения типов и вложенные объекты;
  - валидацию аргументов;
  - обработку ошибок;
  - параллельные tool calls;
  - повторный вызов после неудачи;
  - ограничения числа шагов и stop conditions.

### Workflow — заранее определённый процесс

- **Workflow:** заданная последовательность или граф шагов, который решает задачу предсказуемым способом.
- Пример:
  ```text
  classify request
      → retrieve documents
      → rerank
      → build context
      → generate answer
      → validate citations
  ```
- Workflow может содержать LLM-вызовы, tools, условия, циклы и human-in-the-loop.
- Главное отличие от агента:
  - workflow в основном следует заранее заданной логике;
  - агент динамически решает, какие действия выполнить и в каком порядке.
- На практике возможен гибрид: детерминированный граф содержит отдельные agentic-узлы.

### Toolkit

- **Toolkit:** логически связанный набор tools для одной предметной области.
- Примеры:
  - Git toolkit: `get_diff`, `read_commit`, `list_branches`;
  - Database toolkit: `list_tables`, `describe_schema`, `run_readonly_query`;
  - Knowledge Base toolkit: `search`, `read_document`, `find_related`.
- Toolkit сам по себе не является skill: это набор строительных блоков, который skill или agent использует для решения задачи.

### Tools, Skills, Workflows и Agents — сравнение

| Понятие | Уровень | Что определяет | Пример |
|---|---|---|---|
| **Tool** | Атомарный | Одно доступное действие | `search_documents(query)` |
| **Toolkit** | Группа операций | Набор связанных tools | Git toolkit |
| **Skill** | Способность | Как решать класс задач | Анализ репозитория |
| **Function calling** | Механизм | Как LLM запрашивает вызов tool | Structured tool call |
| **Workflow** | Процесс | Порядок и условия шагов | Retrieve → rerank → answer |
| **Agent** | Исполняющая система | Как динамически выбирать шаги | ReAct-агент с tools |
| **MCP** | Протокол интеграции | Как внешние возможности обнаруживаются и вызываются | MCP Client ↔ MCP Server |

### Native function calling vs MCP

- **Native function calling:** tools описываются непосредственно в запросе к конкретному LLM API, а host сам связывает tool name с реализацией.
- **MCP:** внешний сервер публикует возможности через стандартный протокол; host обнаруживает их через MCP Client.
- Native function calling отвечает в основном за взаимодействие **LLM ↔ host runtime**.
- MCP отвечает за взаимодействие **host ↔ внешний provider возможностей**.
- Эти подходы не исключают друг друга:
  ```text
  MCP Server публикует tool
      ↓
  MCP Client получает его схему
      ↓
  Host передаёт эту схему LLM через native function calling
      ↓
  LLM выбирает tool
      ↓
  Host вызывает его через MCP
  ```

### MCP primitives: tools, resources и prompts

- **MCP Tool:** операция с вычислением или побочным эффектом; вызывается через `tools/call`.
- **MCP Resource:** адресуемые данные для чтения по URI; клиент получает содержимое ресурса, но не вызывает его как функцию.
- **MCP Prompt:** серверный переиспользуемый шаблон сообщения или сценария, который host может запросить и включить в работу модели.
- Нужно понимать, что не вся информация должна быть tool:
  - чтение известного документа удобно оформлять как resource;
  - поиск с параметрами — как tool;
  - стандартный сценарий анализа — как prompt или skill на стороне host.

### Проектирование качественного tool

- Делать tool узким и однозначным: одна операция — одна ответственность.
- Давать описание, достаточное для выбора между похожими tools.
- Использовать строгую схему и минимальный набор аргументов.
- Возвращать компактный структурированный результат, а не необработанный огромный payload.
- Разделять read-only и destructive tools.
- Проверять авторизацию не только на уровне агента, но и внутри tool/backend.
- Делать операции идемпотентными, где это возможно.
- Не передавать модели токены, секреты, stack trace и внутренние детали.
- Вводить timeout, ограничения размера результата и понятные безопасные ошибки.

### Типовые ошибки

- Называть MCP «набором tools»: MCP — протокол, а tools — одна из предоставляемых через него возможностей.
- Считать skill синонимом tool: skill обычно шире и может объединять несколько tools и инструкций.
- Давать модели произвольный доступ к Python-функциям без схемы и авторизации.
- Создавать один универсальный tool с десятками режимов вместо нескольких ясных операций.
- Перекладывать бизнес-валидацию на LLM.
- Смешивать orchestration агента и реализацию MCP Server.
- Считать, что JSON Schema автоматически обеспечивает безопасность вызова.

### Практика

1. Реализовать три локальных Python tools: поиск, чтение документа и получение связанных файлов.
2. Описать их схемами JSON Schema и проверить валидацию корректных и ошибочных аргументов.
3. Собрать deterministic workflow `search → read → answer` без агента.
4. Добавить LLM tool calling и дать модели выбирать между tools.
5. Объединить tools в skill `answer_from_knowledge_base`.
6. Опубликовать те же операции через MCP Server.
7. Сравнить native implementation и MCP-вариант по связности, переиспользованию, latency и обработке ошибок.
8. Добавить read-only resource для полного документа и MCP prompt для grounded-ответа.

## IV. MCP (Model Context Protocol) — фундамент

### Что такое MCP
- **MCP (Model Context Protocol):** открытый протокол, стандартизирующий взаимодействие между AI-агентами (MCP Host) и внешними сервисами/инструментами (MCP Server).
- **Зачем нужен:**
  - Единый интерфейс для подключения агента к любым источникам данных и API.
  - Агент не зависит от реализации конкретного сервиса — все общаются через JSON-RPC.
  - Переиспользование MCP-серверов разными агентами.
- **Ключевая схема:**
  ```
  User → AI-Agent (MCP Host) → JSON-RPC → MCP Server → REST/API → Внешний сервис
  ```

### Архитектура MCP
- **MCP Host:** AI-агент, который инициирует соединения и вызывает tools.
- **MCP Client:** библиотека/клиент внутри Host, управляющая соединениями с серверами.
- **MCP Server:** микросервис, предоставляющий tools (инструменты) и resources (данные).
- **Транспортный слой:** как Host и Server общаются физически (см. раздел V).

### Протокол взаимодействия (JSON-RPC)
- **Базовый формат:** все сообщения — JSON-RPC 2.0.
  - Запрос: `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}`
  - Ответ: `{"jsonrpc": "2.0", "result": {...}, "id": 1}`
  - Ошибка: `{"jsonrpc": "2.0", "error": {"code": ..., "message": "..."}, "id": 1}`
- **Жизненный цикл взаимодействия:**
  1. **`initialize`** — клиент сообщает серверу свои возможности, сервер отвечает своими.
  2. **`tools/list`** — клиент запрашивает список доступных инструментов: `name`, `description`, `inputSchema` (JSON Schema).
  3. **`tools/call`** — клиент вызывает конкретный tool, сервер выполняет логику и возвращает результат.
- **Notifications:** сервер может отправлять уведомления (без `id`) — например, `notifications/tools/list_changed`.
- **Resources:** в дополнение к tools сервер может предоставлять resources — данные для чтения (read-only), доступные по URI.

---

## V. Транспорты MCP

### stdio
- **Механика:** процесс MCP-сервера запускается как дочерний процесс Host'а. Общение через stdin/stdout.
- **Плюсы:** максимальная простота, не нужна сеть, не нужен порт, безопасно (только локально).
- **Минусы:** один клиент на процесс, не масштабируется горизонтально, не подходит для удалённых вызовов.
- **Когда использовать:** локальная разработка, CLI-инструменты.

### SSE (Server-Sent Events)
- **Механика:** клиент открывает долгоживущее SSE-соединение, сервер пушит события. Запросы клиент → сервер идут через отдельный POST-эндпоинт.
- **Stateful:** требует session_id, сервер отслеживает открытые соединения.
- **Плюсы:** стандартный HTTP, поддержка браузерами, server-push.
- **Минусы:** ограничение на количество одновременных соединений, состояние на сервере усложняет масштабирование.
- **Когда использовать:** веб-клиенты, где нужен server-push, небольшое число клиентов.

### Streamable HTTP
- **Механика:** клиент отправляет POST-запросы, сервер отвечает в теле ответа. Без долгоживущих соединений.
- **Stateless:** каждый запрос самодостаточен, не требует session_id.
- **Плюсы:** простое горизонтальное масштабирование, неограниченное число клиентов.
- **Минусы:** нет server-push без запроса клиента.
- **Когда использовать:** продакшен-среда, высокая нагрузка, микросервисная архитектура.

### SSE vs Streamable HTTP — ключевые различия

| Аспект | SSE | Streamable HTTP |
|--------|-----|-----------------|
| **Тип соединения** | Долгоживущее (keep-alive) | Короткое (request-response) |
| **Состояние** | Stateful (сессия) | Stateless |
| **Session ID** | Обязателен | Не нужен |
| **Направление** | Server → Client push | Client → Server → Client response |
| **Масштабирование** | Сложнее (sticky sessions) | Простое (за балансировщиком) |
| **Лимит соединений** | Ограничен (file descriptors) | Ограничен только QPS бекенда |
| **Продакшен** | Для малого числа клиентов | Для высокой нагрузки |

---

## VI. MCP Hub / Coordinator — оркестрация

### Что такое MCP Hub
- **MCP Hub (Coordinator):** центральный узел, который:
  - Принимает запросы от AI-агента (MCP Host).
  - Определяет, к какому MCP-серверу направить запрос.
  - Маршрутизирует JSON-RPC вызовы между агентом и несколькими MCP-серверами.
  - Агрегирует результаты от разных серверов.
  - Обеспечивает сквозную аутентификацию и аудит.

### Архитектура Hub
```
                ┌─────────────────┐
                │   AI-Agent      │
                │   (MCP Host)    │
                └────────┬────────┘
                         │ JSON-RPC
                         ▼
                ┌─────────────────┐
                │  Coordinator    │
                │  (MCP Hub)      │
                └──┬──────┬──────┬┘
                   │      │      │
      ┌────────────┼──────┼──────┼────────────┐
      ▼            ▼      ▼      ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ MCP      │ │ MCP      │ │ MCP      │ │ MCP      │
│ Server A │ │ Server B │ │ Server C │ │ Server D │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

---

## VII. Типовые вопросы на собеседовании (Agents + MCP)

### Базовые
1. **«Чем tool отличается от skill?»**
   - Tool — атомарная вызываемая операция. Skill — составная способность решать класс задач, которая может включать инструкции, workflow и несколько tools.
2. **«Чем function calling отличается от MCP?»**
   - Function calling задаёт механизм, с помощью которого LLM просит host вызвать tool. MCP стандартизирует обнаружение и удалённый вызов возможностей между host и MCP Server. Они могут использоваться вместе.
3. **«Чем workflow отличается от агента?»**
   - Workflow следует заранее определённому графу шагов. Агент динамически выбирает действия и порядок их выполнения в зависимости от наблюдений.
4. **«Что такое MCP и зачем он нужен?»**
   - MCP — открытый протокол для стандартизации взаимодействия AI-агентов с внешними инструментами. Единый интерфейс вместо кастомных интеграций под каждый сервис.
5. **«Как устроено взаимодействие MCP Host и MCP Server?»**
   - Через JSON-RPC 2.0. Транспорт: stdio / SSE / Streamable HTTP. Жизненный цикл: initialize → tools/list → tools/call.
6. **«Какие транспорты MCP знаешь и в чём разница?»**
   - stdio (простой, локальный), SSE (stateful, server-push), Streamable HTTP (stateless, продакшен). Trade-off: масштабирование vs push-уведомления.
7. **«Чем отличается ReAct от Chain-of-Thought?»**
   - CoT — только рассуждение. ReAct добавляет действия: Thought → Action → Observation → Thought. Агент взаимодействует с внешним миром.

### Средние
8. **«Чем отличается stateless от stateful режим в MCP? Когда что выбрать?»**
   - Stateless: нет сессий, горизонтальное масштабирование. Stateful: сервер хранит состояние (сессии, кеш). Stateless — для продакшена, stateful — для стриминга/SSE.
9. **«Что такое MCP Hub и зачем он нужен?»**
   - Центральный оркестратор, маршрутизирующий запросы агента к нужным MCP-серверам. Агрегирует результаты, обеспечивает сквозную аутентификацию.
10. **«LangGraph vs линейные цепочки — в чём преимущество графового подхода?»**
   - Граф позволяет: циклы (повторить шаг), conditional edges (ветвление), human-in-the-loop, параллельные ветки. Линейная цепочка — только вперёд.

### Продвинутые
11. **«Как бы ты спроектировал агента для сложной multi-step задачи?»**
   - Plan-and-Execute для предсказуемых задач, ReAct + LangGraph для адаптивных. Swarm для параллельных подзадач. Memory: краткосрочная в промпте, долгосрочная в векторной БД.
12. **«Как организовать безопасность агента, вызывающего внешние инструменты?»**
   - Input guardrails (prompt injection detection), output validation (JSON Schema), tool-level opt-in для деструктивных действий, sandboxing.
13. **«Как оценивать качество агента?»**
    - Task success rate, tool call accuracy, latency, token cost. Бенчмарки: AgentBench, SWE-bench. Офлайн: replay + simulated users. Онлайн: A/B-тесты.

---

## VIII. Ресурсы

- **JSON Schema** — json-schema.org
- **OpenAI Function Calling / Tools** — platform.openai.com/docs
- **Anthropic Tool Use** — docs.anthropic.com
- **MCP спецификация** — modelcontextprotocol.io
- **MCP Python SDK** — github.com/modelcontextprotocol/python-sdk
- **JSON-RPC 2.0 спецификация** — jsonrpc.org
- **ReAct paper (Yao et al., 2022)** — arxiv.org/abs/2210.03629
- **Reflexion paper (Shinn et al., 2023)** — arxiv.org/abs/2303.11366
- **LangGraph** — langchain-ai.github.io/langgraph
- **OpenAI Swarm** — github.com/openai/swarm
- **AutoGen (Microsoft)** — microsoft.github.io/autogen
- **CrewAI** — crewai.com
- **AgentBench** — arxiv.org/abs/2308.03688
- **SWE-bench** — swebench.com