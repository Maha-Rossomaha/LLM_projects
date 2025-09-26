# Docker / Compose Cheatsheet

### Запуск и остановка сервисов

```bash
# Запустить все сервисы из docker-compose.yml (в фоне)
docker compose up -d

# Пересобрать образы и запустить
docker compose up --build -d

# Остановить и удалить контейнеры
docker compose down

# Остановить и удалить контейнеры + тома (volumes)
docker compose down -v
```

### Состояние контейнеров

```bash
# Список работающих контейнеров
docker ps

# Список всех контейнеров (включая остановленные)
docker ps -a

# Остановить/запустить/перезапустить контейнер
docker stop <container>
docker start <container>
docker restart <container>

# Удалить контейнер
docker rm <container>
```

### Логи и отладка

```bash
# Логи конкретного контейнера (follow)
docker logs -f <container>

# Логи всех сервисов из compose
docker compose logs -f

# Зайти в контейнер (bash)
docker exec -it <container> bash
```

### Образы (images)

```bash
# Показать локальные образы
docker images

# Скачать образ из реестра
docker pull <image>:<tag>

# Удалить образ
docker rmi <image>:<tag>

# Удалить «висячие» (dangling) образы
docker image prune -f

# Очистить builder cache (если место забито)
docker builder prune -f
```

### Сборка образов

```bash
# Собрать образ из Dockerfile в текущей папке
docker build -t <image>:<tag> .

# Пересобрать без кеша
docker build --no-cache -t <image>:<tag> .

# В составе compose: пересборка сервисов
docker compose build
```

### Томá (volumes) и сети (networks)

```bash
# Список томов / удалить том
docker volume ls
docker volume rm <volume>

# Очистить неиспользуемые тома
docker volume prune -f

# Сети: список / информация
docker network ls
docker network inspect <network>
```

### Ресурсы и мониторинг

```bash
# Использование CPU/RAM всеми контейнерами
docker stats

# Диск, сетевые интерфейсы, лимиты — см. docker inspect
docker inspect <container>
```

