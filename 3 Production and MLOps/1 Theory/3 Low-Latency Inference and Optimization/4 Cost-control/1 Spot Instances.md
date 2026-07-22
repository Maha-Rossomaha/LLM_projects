# Спот-инстансы для GPU-инференса

## 1. Что такое спот-инстансы

Спот-инстансы (preemptible, spot) — избыточные вычислительные мощности облачного провайдера, которые сдаются с большой скидкой (60–90%) в обмен на право прервать в любой момент с коротким предупреждением.

| Провайдер | Название | Скидка | Предупреждение | Лимит прерываний |
|-----------|----------|--------|----------------|-----------------|
| AWS | Spot Instance | 60–90% | 2 мин (SIGTERM) | Нет |
| GCP | Preemptible VM | 60–80% | 30 сек (ACPI G2) | 24 часа макс. |
| Azure | Low-priority | 60–80% | 30 сек | Нет |
| RunPod | Spot | 60–80% | 30 сек | Нет |

## 2. Архитектура graceful shutdown

При сигнале прерывания нужно корректно завершить активные запросы, не потеряв данные:

```python
import signal
import sys
import json
import asyncio
from pathlib import Path

class SpotGracefulShutdown:
    def __init__(self, checkpoint_dir: str = "/tmp/spot-checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._active_requests: dict[str, dict] = {}
        self._shutting_down = False

        # Перехват сигналов
        signal.signal(signal.SIGTERM, self._handle_sigterm)   # AWS, Kubernetes
        # GCP: preemptible VMs посылают ACPI G2, который можно ловить через OS

    def _handle_sigterm(self, signum, frame):
        print("Spot interruption: SIGTERM received")
        self._shutting_down = True
        asyncio.create_task(self._shutdown_gracefully())

    async def _shutdown_gracefully(self):
        """Завершить активные запросы и сохранить состояние."""
        # 1. Остановить приём новых запросов
        await self._stop_accepting()

        # 2. Сохранить KV-кэш для каждого активного запроса
        for req_id, req_data in self._active_requests.items():
            checkpoint = {
                "prompt": req_data["prompt"],
                "generated": req_data["generated_tokens"],
                "kv_cache_path": str(self.checkpoint_dir / f"{req_id}.pt"),
            }
            # Сохранить KV-кэш
            torch.save(req_data["kv_cache"], checkpoint["kv_cache_path"])
            # Сохранить мета-информацию
            with open(self.checkpoint_dir / f"{req_id}.json", "w") as f:
                json.dump(checkpoint, f)

        print(f"Saved {len(self._active_requests)} checkpoints")
        sys.exit(0)

    async def _stop_accepting(self):
        """Предотвратить приём новых запросов."""
        # Остановить HTTP-сервер
        server.should_exit = True

        # Подождать завершения активных, но не более 30 секунд
        try:
            await asyncio.wait_for(
                self._wait_for_active(),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            print("Timeout waiting for active requests, forcing shutdown")
```

**Восстановление после прерывания:**

```python
class SpotResumer:
    def __init__(self, checkpoint_dir: str = "/tmp/spot-checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)

    def get_pending(self) -> list[dict]:
        """Найти незавершённые запросы после перезапуска."""
        pending = []
        for f in self.checkpoint_dir.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
            pending.append(data)
        return pending

    async def resume(self, model, pending: list[dict]):
        """Продолжить генерацию с сохранённого KV-кэша."""
        results = []
        for data in pending:
            kv_cache = torch.load(data["kv_cache_path"])
            # Продолжить генерацию с сохранённого состояния
            out = await model.generate(
                data["prompt"],
                past_key_values=kv_cache,
                max_new_tokens=data["max_tokens"] - len(data["generated"]),
            )
            results.append(out)
        return results
```

## 3. Стратегии использования

### 3.1 Только batch inference (recommended)

Спот-инстансы идеальны для batch-задач, где прерывание не критично:

```python
class SpotBatchJob:
    def __init__(self, task_queue: asyncio.Queue):
        self.queue = task_queue
        self.max_retries = 3

    async def run_batch(self):
        while not self.queue.empty():
            task = await self.queue.get()
            for attempt in range(self.max_retries):
                try:
                    result = await self._process_task(task)
                    self._save_result(result)
                    break
                except SpotInterrupted:
                    print(f"Task interrupted, retry {attempt + 1}")
                    # Задача будет подхвачена другой spot-репликой
                    await self.queue.put(task)
                    break
```

### 3.2 Гибридный парк: on-demand + spot

Базовый парк на on-demand (покрывает min трафик), пиковый — на spot:

```yaml
# Kubernetes: два пула узлов
apiVersion: v1
kind: NodeSelector
metadata:
  name: hybrid-pool
spec:
  - name: on-demand-pool
    nodeSelector:
      instance-type: on-demand
    replicas: 2  # базовый парк, всегда включён
  - name: spot-pool
    nodeSelector:
      instance-type: spot
    replicas: 10  # расширение при пиках
```

Распределение трафика:

```python
class HybridRouter:
    def __init__(self):
        self.on_demand = "http://on-demand:8080"
        self.spot = "http://spot:8080"
        self.spot_healthy = True

    async def route(self, req):
        # Критичные запросы — только on-demand
        if req.priority == "high":
            return await self._call(self.on_demand, req)

        # Некритичные — spot (если жив)
        if self.spot_healthy:
            try:
                return await self._call(self.spot, req)
            except SpotInterrupted:
                self.spot_healthy = False
                return await self._call(self.on_demand, req)

        return await self._call(self.on_demand, req)
```

### 3.3 Спот для RL-генераций

GRPO/PPO генерации — идеальный кандидат для спота:
- Генерации не критичны по времени (batch mode).
- При прерывании можно перезапустить генерацию для этого промпта.
- RL-цикл потребляет много compute (16–128 генераций на промпт) — спот даёт 60–90% экономии.

## 4. Мониторинг спот-инстансов

```python
class SpotMonitor:
    def __init__(self, prometheus_url: str):
        self.prometheus = prometheus_url

    async def spot_interruption_rate(self) -> float:
        """Доля прерванных спот-инстансов за последний час."""
        query = """
        sum(rate(kube_pod_status_reason{reason="Evicted"}[1h]))
        /
        sum(rate(kube_pod_status_phase{phase="Running"}[1h]))
        """
        result = await self._query_prometheus(query)
        return result

    async def spot_cost_savings(self) -> dict:
        """Экономия от использования спотов."""
        query = """
        (avg(on_demand_price) - avg(spot_price))
        * sum(kube_pod_status_phase{instance_type="spot", phase="Running"})
        * 730  # часов в месяце
        """
        savings = await self._query_prometheus(query)
        return {"monthly_savings_usd": savings}
```

## 5. Типичные ошибки

- **Спот для real-time без fallback.** Если спот прервали, а on-demand реплик нет — сервис падает.
- **Игнорировать checkpoint.** При прерывании без сохранения состояния — потеря всех активных генераций.
- **Слишком высокая доля спота.** Если > 80% парка на споте, при массовом прерывании (например, при скачке цен) падает пропускная способность.
- **Не мониторить spot interruption rate.** Если провайдер прерывает споты > 5–10% времени, экономия не окупает падения качества.

## 6. Вопросы для самопроверки

1. Какие сигналы посылают AWS и GCP при прерывании спот-инстанса? Как их обработать?
2. Почему batch inference — лучший сценарий для спот-инстансов?
3. Как организовать гибридный парк on-demand + spot на Kubernetes?
4. Какие метрики нужно мониторить для спот-инстансов?
5. Почему высокая доля спота в парке — риск для real-time сервиса?