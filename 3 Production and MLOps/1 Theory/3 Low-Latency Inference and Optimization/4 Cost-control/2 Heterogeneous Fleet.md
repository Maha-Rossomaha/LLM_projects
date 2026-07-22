# Гетерогенный парк GPU: разные типы инстансов под разные задачи

## 1. Зачем нужны разные GPU

LLM-сервис решает не одну, а несколько задач с разными требованиями:

| Задача | Требования к памяти | Требования к compute | Подходящие GPU |
|--------|--------------------|---------------------|----------------|
| Инференс 70B | 80+ GB HBM | Высокий (FLOPs) | H100, A100-80G |
| Инференс 7–13B | 16–40 GB | Средний | A100, L40S, L4 |
| Эмбеддинги (BERT) | 1–4 GB | Низкий | T4, L4, CPU |
| Ранжирование (Cross-encoder) | 2–8 GB | Средний | L4, T4, A10 |
| Препроцессинг/токенизация | 0 GB | CPU only | Любая |

Использовать H100 для эмбеддингов — то же, что ездить на Ferrari за хлебом. H100 стоит $3–4/час, T4 — $0.3–0.5/час. Разница в 10×.

## 2. Архитектура роутинга

### 2.1 Роутер на основе требований запроса

```python
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    LLM_70B = "llm_70b"
    LLM_7B = "llm_7b"
    EMBEDDING = "embedding"
    RERANKER = "reranker"

@dataclass
class GPUCluster:
    name: str
    gpu_type: str
    endpoint: str
    max_seq_len: int
    cost_per_hour: float
    healthy: bool = True

class HeterogeneousRouter:
    def __init__(self):
        self.clusters = {
            TaskType.LLM_70B: GPUCluster(
                name="h100-cluster",
                gpu_type="H100",
                endpoint="http://h100-cluster:8080",
                max_seq_len=8192,
                cost_per_hour=3.5,
            ),
            TaskType.LLM_7B: GPUCluster(
                name="a100-cluster",
                gpu_type="A100-40G",
                endpoint="http://a100-cluster:8080",
                max_seq_len=4096,
                cost_per_hour=1.5,
            ),
            TaskType.EMBEDDING: GPUCluster(
                name="l4-cluster",
                gpu_type="L4",
                endpoint="http://l4-cluster:8080",
                max_seq_len=512,
                cost_per_hour=0.5,
            ),
            TaskType.RERANKER: GPUCluster(
                name="t4-cluster",
                gpu_type="T4",
                endpoint="http://t4-cluster:8080",
                max_seq_len=512,
                cost_per_hour=0.35,
            ),
        }
        self.health_check_interval = 10  # секунд

    async def route(self, request) -> GPUCluster:
        """Выбрать кластер на основе типа задачи и требований."""
        task_type = self._classify(request)

        # Проверить здоровье кластера
        cluster = self.clusters[task_type]
        if not cluster.healthy:
            # Fallback: next best cluster
            cluster = self._find_fallback(task_type)

        # Проверить, помещается ли запрос
        if request.max_tokens > cluster.max_seq_len:
            cluster = self._find_fallback(task_type)

        return cluster

    def _classify(self, request) -> TaskType:
        """Определить тип задачи."""
        if request.model == "llama-70b":
            return TaskType.LLM_70B
        elif request.model == "llama-7b" or request.model == "mistral-7b":
            return TaskType.LLM_7B
        elif request.model in ("bge-base", "text-embedding-ada"):
            return TaskType.EMBEDDING
        elif request.model in ("bge-reranker", "cross-encoder"):
            return TaskType.RERANKER
        else:
            return TaskType.LLM_7B  # default fallback

    def _find_fallback(self, task_type: TaskType) -> GPUCluster:
        """Найти запасной кластер, если основной недоступен."""
        fallback_map = {
            TaskType.LLM_70B: TaskType.LLM_7B,  # 70B с уменьшенным контекстом
            TaskType.LLM_7B: TaskType.EMBEDDING,
            TaskType.EMBEDDING: TaskType.LLM_7B,  # дорого, но работает
            TaskType.RERANKER: TaskType.EMBEDDING,
        }
        fallback = fallback_map.get(task_type)
        return self.clusters.get(fallback, self.clusters[TaskType.LLM_7B])
```

### 2.2 Weighted random routing (по стоимости)

Если несколько кластеров подходят, выбирать с вероятностью, обратной стоимости:

```python
import random
import math

class CostAwareRouter:
    def __init__(self, clusters: list[GPUCluster]):
        self.clusters = clusters

    def pick(self) -> GPUCluster:
        """Выбрать кластер пропорционально 1/cost."""
        weights = [1.0 / c.cost_per_hour for c in self.clusters]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(self.clusters, weights=probs, k=1)[0]
```

### 2.3 fallback при перегрузке

Если кластер перегружен (tail latency растёт), часть трафика перенаправляется на другой кластер:

```python
class LoadAwareRouter:
    def __init__(self, primary: GPUCluster, fallback: GPUCluster, threshold_p99_ms=1000):
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold_p99_ms
        self.fallback_ratio = 0.0  # 0–1, доля трафика на fallback

    async def route(self, request):
        # Адаптивно увеличивать fallback ratio при перегрузке
        p99 = await self._get_p99_latency(self.primary)
        if p99 > self.threshold:
            self.fallback_ratio = min(1.0, self.fallback_ratio + 0.1)
        else:
            self.fallback_ratio = max(0.0, self.fallback_ratio - 0.05)

        if random.random() < self.fallback_ratio:
            return await self._call(self.fallback, request)
        return await self._call(self.primary, request)
```

## 3. Мониторинг гетерогенного парка

```python
class HeterogeneousMonitor:
    def __init__(self):
        self.metrics = {}  # cluster_name → metrics

    async def collect(self, clusters: list[GPUCluster]):
        for cluster in clusters:
            self.metrics[cluster.name] = {
                "cost_per_hour": cluster.cost_per_hour,
                "gpu_utilization": await self._gpu_util(cluster),
                "p99_latency_ms": await self._p99_latency(cluster),
                "throughput_tokens_per_s": await self._throughput(cluster),
                "cost_per_million_tokens": await self._cost_per_mtok(cluster),
            }

    def suggest_optimization(self):
        """Предложить переместить нагрузку на более дешёвый кластер."""
        for name, m in self.metrics.items():
            if m["gpu_utilization"] < 30:
                print(f"{name}: низкая загрузка ({m['gpu_utilization']}%)")
                print("  → рассмотреть перемещение трафика или уменьшение реплик")
            if m["cost_per_million_tokens"] > 1.0:
                print(f"{name}: высокая стоимость токена (${m['cost_per_million_tokens']:.2f}/M)")
                print("  → рассмотреть более дешёвый GPU тип")
```

## 4. Типичные ошибки

- **Роутинг без fallback.** Если H100-кластер упал, запросы на 70B должны идти на A100 (с уменьшенным контекстом), а не падать с ошибкой.
- **Игнорировать разницу в tokenizer.** Модели на разных GPU могут использовать разные tokenizer'ы. Клиент должен получать консистентный интерфейс.
- **Смешивать задачи на одном кластере.** LLM-генерация и эмбеддинги на одном H100 — эмбеддинги будут платить за H100, хотя могли на L4.
- **Не учитывать network latency.** Если роутер в us-east-1, а L4-кластер в eu-west-1, задержка 100 мс может убить смысл дешёвого инференса.

## 5. Вопросы для самопроверки

1. Почему держать все задачи на H100 — неоптимально по стоимости?
2. Как роутер определяет, какой GPU подходит для запроса?
3. Как работает fallback при недоступности кластера?
4. Как адаптивно перераспределять трафик при перегрузке одного из кластеров?
5. Какие метрики нужно мониторить в гетерогенном парке?