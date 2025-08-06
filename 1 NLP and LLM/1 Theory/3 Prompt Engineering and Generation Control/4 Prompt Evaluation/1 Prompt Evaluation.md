# Оценка и тестирование LLM (Prompt Evaluation)

## Зачем нужна LLM-оценка?

Оценка качества генерации LLM — ключевой элемент в:

- CI/CD-процессах для LLM-приложений
- тестировании новых промптов или моделей
- защите от регрессий и деградаций
- red teaming и robustness-проверках

Обычные метрики (BLEU, ROUGE) плохо работают с генерацией. Поэтому применяются более гибкие фреймворки — с поддержкой **LLM-as-a-judge**, **assertions**, **сравнений output'ов**, **score-рейтингов**.

---

## Популярные инструменты

| Инструмент       | Что делает                             | Подходит для          | Особенности                                                 |
| ---------------- | -------------------------------------- | --------------------- | ----------------------------------------------------------- |
| **Promptfoo**    | Тестирование промптов и LLM-апп        | CI/CD, dev workflows  | YAML-тесты, CLI/GUI, локальный запуск, любая модель         |
| **OpenAI Evals** | Запуск и регистрация кастомных eval'ов | OpenAI API интеграция | LLM-as-a-judge, шаблоны eval’ов, тесная интеграция с OpenAI |

---

## 1. Promptfoo

**Что это:** open-source CLI/библиотека для тестирования промптов и LLM-приложений.

**Фичи:**

- Тестовые таблицы: prompt × модель × вариант
- YAML-конфигурации
- Поддержка `assert:` условий (ожидаемое содержимое, длина, регулярки)
- `Red teaming`-тесты: ввод шумов, провокаций, jailbreak-подобных примеров
- Поддержка множества моделей: OpenAI, Claude, LLaMA, local
- Локальный запуск (без необходимости внешнего API)
- Визуальный репорт или CLI-отчёт

**Пример YAML:**

```yaml
prompts:
  - name: question_answering
    prompt: "Вопрос: {{question}} Ответ:"
    vars:
      - question: "Сколько ног у паука?"
        expected: "8"
assertions:
  - type: contains
    value: "8"
```

**Где используется:** в CI-пайплайнах, A/B тестах промптов, regression-защитах.

📎 [promptfoo.dev](https://promptfoo.dev)  |  [github.com/promptfoo](https://github.com/promptfoo/promptfoo)

---

## 2. OpenAI Evals

**Что это:** фреймворк от OpenAI для автоматической оценки LLM-промптов и моделей.

**Фичи:**

- Шаблоны evaluation'ов (например, qa, math, multi-turn)
- Кастомные evaluators
- Поддержка self-graded outputs (модель сама оценивает себя)
- `LLM-as-a-judge`: сравнение двух output’ов, выбор лучшего
- Интеграция с OpenAI API
- Логгирование, сохранение результатов в базе

**Пример конфигурации:**

```python
class MyEval(Eval):
    def run(self, model: Model, test_case: Dict) -> EvalResult:
        result = model.generate(test_case["prompt"])
        score = 1 if "8" in result else 0
        return EvalResult(passed=score)
```

📎 [GitHub: openai/evals](https://github.com/openai/evals)

---

## Методы оценки

- **LLM-as-a-judge**: модель сравнивает варианты и выносит оценку (напр. "кто дал лучший ответ? почему?")
- **Assert-based**: output должен удовлетворять условиям (длина, наличие фразы, структура JSON)
- **Similarity-based**: сравнение с эталоном по semsim (cosine), edit distance, etc
- **Manual review**: human-in-the-loop, валидация outputs вручную

---

## Интеграция в Prompt Engineering pipeline

1. Разработка промпта (разные формулировки)
2. Запись в таблицу / YAML-конфигурацию
3. Запуск eval (Promptfoo / OpenEval)
4. Фиксация результата (baseline)
5. Включение в CI как регрессионный тест
6. При изменении модели — сравнение с baseline, fail on drop

---

## Сравнение инструментов

|                  | Promptfoo             | OpenAI Evals         |
| ---------------- | --------------------- | -------------------- |
| Модели           | Любые                 | Только OpenAI        |
| Архитектура      | CLI + YAML            | Python API           |
| Интерфейс        | CLI + Web             | CLI                  |
| Judge-модель     | Любая (настраиваемая) | OpenAI               |
| Локальный запуск | ✅                     | ❌ (нужен OpenAI API) |
| Red-teaming      | ✅                     | ⚠️ вручную           |

---

## Выводы

- Prompt-evaluation — важный шаг для стабильной разработки с LLM
- Promptfoo хорош для любых моделей и локального CI/CD
- OpenAI Evals подходит при работе строго с OpenAI API
- Лучше комбинировать auto + manual + LLM-as-judge стратегии
- Автооценка ≠ финальная истина, но сильно помогает в масштабировании тестов

