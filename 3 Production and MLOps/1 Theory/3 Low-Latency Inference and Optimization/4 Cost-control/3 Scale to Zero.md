# Scale-to-Zero: выключение GPU-реплик при отсутствии трафика

## 1. Зачем выключать реплики

GPU — дорогой ресурс, который потребляет деньги даже в простое. Для сервисов с неравномерной нагрузкой:

- Dev/Staging: 8–12 часов простоя в сутки (ночь, выходные).
- Batch inference: часы между задачами.
- Демо/прототипы: недели без единого запроса.

**Scale-to-zero** — полное выключение реплик при отсутствии трафика. Экономия — 100% стоимости GPU во время простоя.

## 2. Когда scale-to-zero оправдан

| Сценарий | Scale-to-zero | Почему |
|----------|--------------|--------|
| Dev/Staging | ✅ | Допустимо ждать 1–5 мин при первом запросе |
| Batch inference | ✅ | Задача запускается по триггеру, не нужно держать GPU постоянно |
| Демо | ✅ | Нет SLA, пользователь понимает, что сервис «холодный» |
| Production real-time | ❌ | Холодный старт > 1 сек нарушает SLA |
| Real-time с низким трафиком | ⚠️ | Компромисс: 1 warm реплика + scale-to-zero для остальных |

### 2.1 Dev/Staging

```yaml
# KEDA: scale-to-zero для dev-окружения (с 20:00 до 8:00 — 0 реплик)
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-dev-scaler
spec:
  scaleTargetRef:
    name: llm-dev
  minReplicaCount: 0   # scale-to-zero разрешён
  maxReplicaCount: 2
  triggers:
    - type: cron
      metadata:
        timezone: Europe/Moscow
        start: 0 8 * * 1-5   # 8:00 — поднять
        end: 0 20 * * 1-5    # 20:00 — выключить
        desiredReplicas: "1"
```

### 2.2 Batch inference (job-based)

Вместо постоянного сервиса — запуск задачи по требованию:

```yaml
# Kubernetes Job: поднимается, выполняет инференс, завершается
apiVersion: batch/v1
kind: Job
metadata:
  name: batch-inference-job
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: inference
          image: inference:latest
          command: ["python", "run_batch.py"]
          resources:
            limits:
              nvidia.com/gpu: 1
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: instance-type
                    operator: In
                    values:
                      - spot  # спот-инстанс, дёшево
```

Запуск по триггеру (Airflow, Argo Workflows, Lambda):

```python
# AWS Lambda → запуск batch job при появлении файла в S3
def lambda_handler(event, context):
    # Определить параметры задачи
    s3_key = event["Records"][0]["s3"]["object"]["key"]

    # Запустить Kubernetes Job
    batch_client.create_job(
        image="inference:latest",
        command=["python", "run_batch.py", "--input", s3_key],
        gpu_count=1,
        instance_type="spot",
        ttl_seconds_after_finished=3600,  # auto-delete после завершения
    )
```

## 3. Холодный старт и его цена

Холодный старт (cold start) — время от получения запроса до готовности модели отвечать:

```python
class ColdStartTracker:
    stages = {
        "pod_schedule": "планирование пода (секунды)",
        "image_pull": "скачивание Docker-образа (5–30 сек)",
        "model_load": "загрузка модели с HF Hub (30–300 сек)",
        "model_warmup": "первый forward pass (1–10 сек)",
        "total": "итого: 1–5 минут",
    }
```

**Как сократить холодный старт:**

| Метод | Экономия времени | Цена |
|-------|-----------------|------|
| Pre-pulled image | 5–30 сек | Больше места на node |
| Model cache (PVC) | 30–300 сек | PersistentVolumeClaim |
| Snapshot (CudaGraph) | 1–10 сек | Доп. память |
| Always-on warm replica | 0 сек | 1 GPU постоянно |

```
Без оптимизаций:   pod_schedule → image_pull → model_load → warmup = 3 мин
С cache + snapshot: pod_schedule → (image уже есть) → model_load (cached) → warmup (snapshot) = 30 сек
```

## 4. Serverless GPU

Облачные провайдеры предлагают serverless GPU-инференс со встроенным scale-to-zero:

### 4.1 AWS SageMaker Serverless

```bash
aws sagemaker create-endpoint-config \
    --endpoint-config-name llm-serverless \
    --production-variants '[{
        "VariantName": "default",
        "ModelName": "llama-7b",
        "ServerlessConfig": {
            "MemorySizeInMB": 6144,
            "MaxConcurrency": 5,
            "ProvisionedConcurrency": 0  # scale-to-zero
        }
    }]'
```

- **Provisioned Concurrency** — 0 означает, что при отсутствии трафика реплики выключаются.
- **MaxConcurrency** — макс. параллельных запросов. При превышении — 429 Too Many Requests.
- **Cold start** — 10–60 секунд для LLM (зависит от размера модели).

### 4.2 RunPod Serverless

```bash
# RunPod: serverless endpoint с scale-to-zero
runpodctl create endpoint \
    --name llm-inference \
    --gpu-type H100 \
    --min-workers 0 \
    --max-workers 5 \
    --idle-timeout 120  # секунд без трафика → scale to 0
```

### 4.3 GCP Cloud Run + GPU (2025+)

```bash
gcloud run deploy llm-service \
    --image gcr.io/project/llm:latest \
    --cpu 8 --memory 32Gi \
    --concurrency 1 \
    --min-instances 0 \
    --max-instances 10 \
    --execution-environment gen2
```

**Ограничение:** Cloud Run с GPU доступен не во всех регионах, max instance lifetime — 60 минут.

## 5. Warm pool (компромисс)

Scale-to-zero с холодным стартом в 3 минуты неприемлем для production. Решение — **warm pool**: 1–2 реплики всегда включены, остальные — scale-to-zero.

```yaml
# HPA + KEDA: 1 warm, остальные по требованию
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 1          # всегда 1 тёплая реплика
  maxReplicas: 10
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 мин без трафика → выключить
      policies:
        - type: Pods
          value: 1
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0    # мгновенный scale-up
      policies:
        - type: Pods
          value: 2
          periodSeconds: 15
```

**Экономия:** если средняя нагрузка — 2 реплики, а пиковая — 8, то при minReplicas=1 экономия = (2 - 1) / 2 = 50% против фиксированных 2 реплик. При minReplicas=0 (scale-to-zero) — 100% в простое, но холодный старт.

## 6. Мониторинг scale-to-zero

```python
class ScaleToZeroMonitor:
    def __init__(self, prometheus_url: str):
        self.prometheus = prometheus_url

    async def cold_start_duration(self) -> dict:
        """Длительность холодного старта (p50, p95, p99)."""
        query = """
        histogram_quantile(0.99,
          sum(rate(cold_start_duration_seconds_bucket[1h])) by (le)
        )
        """
        result = await self._query(query)
        return result

    async def idle_time_ratio(self) -> float:
        """Доля времени с 0 репликами."""
        query = """
        count(kube_pod_status_phase{phase="Running"}) == 0
        /
        count(kube_pod_status_phase)
        """
        return await self._query(query)

    async def savings_estimate(self) -> dict:
        """Оценка экономии от scale-to-zero."""
        query = """
        (avg(gpu_cost_per_hour)
         * sum(kube_pod_status_phase{phase="Running"})
         * 730)
        -
        (avg(gpu_cost_per_hour)
         * sum(kube_pod_status_phase{phase="Running"})
         * on(instance) group_left() avg(cold_start_ratio)
         * 730)
        """
        return await self._query(query)
```

## 7. Типичные ошибки

- **Scale-to-zero для real-time.** Холодный старт 1–5 минут несовместим с latency SLA < 1 сек.
- **Без warm pool.** Даже 30 секунд холодного старта — неприемлемо для пользовательского трафика.
- **Не настроить stabilizationWindow.** Если scale-down происходит слишком быстро, реплики будут выключаться между запросами → постоянные холодные старты.
- **Игнорировать холодные старты в мониторинге.** Если не измерять cold start duration, невозможно оценить влияние на пользователей.
- **Scale-to-zero для долгих генераций.** Если запрос генерирует 10K токенов (40 секунд), а scale-to-zero выключил реплику — запрос упадёт.

## 8. Вопросы для самопроверки

1. В каких сценариях scale-to-zero оправдан, а в каких нет?
2. Из чего складывается холодный старт GPU-инференса?
3. Как сократить холодный старт (pre-pulled image, model cache, snapshot)?
4. Чем warm pool отличается от полного scale-to-zero?
5. Как настроить KEDA для scale-to-zero с cron-триггером?
6. Какие метрики нужно мониторить для оценки влияния scale-to-zero?