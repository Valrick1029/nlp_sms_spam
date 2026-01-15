## SMS Spam Detection Service (FastText Edition)

Профессиональный микросервис для фильтрации SMS-спама на базе библиотеки FastText от Facebook. Система включает в себя полноценный API, контейнеризацию и механизм автоматического дообучения на основе обратной связи от пользователей.
Ключевые особенности

    Высокая производительность: Инференс за <5 мс благодаря FastText.

    Производственный API: Реализован на FastAPI с валидацией данных через Pydantic.

    MLOps ready: Встроенный механизм Feedback Loop для сбора ошибок и скрипт автоматического переобучения.

    Контейнеризация: Полная изоляция среды через Docker.

    Надежность: Покрытие Unit и Интеграционными тестами.

### Технологический стек

    ML Core: FastText

    API Framework: FastAPI, Uvicorn

    Data Science: Pandas, Scikit-learn (для EDA и базовых метрик)

    DevOps: Docker, Pytest

    Logging: Python Logging Module

### Структура проекта

.
├── data/               # Датасеты (исходные и архивные)
├── models/             # Веса моделей (.bin файлы)
│   └── v1/             # Текущая версия модели
├── src/                # Исходный код сервиса
│   ├── main.py         # Основной API сервер
│   └── pipeline.py     # Логика обработки и предсказания
├── tests/              # Автоматические тесты
├── Dockerfile          # Инструкция для сборки контейнера
├── requirements.txt    # Зависимости проекта
└── retrain.py          # Скрипт автоматического переобучения

### Быстрый запуск
С использованием Docker (рекомендуется)

    Сборка образа:

docker build -t spam-filter:v1 .

Запуск контейнера:

    docker run -p 8000:8000 spam-filter:v1

Локальный запуск

    Установите зависимости: pip install -r requirements.txt

    Запустите сервер: uvicorn src.main:app --reload

Использование API
1. Предсказание (Prediction)

Endpoint: POST /predict Запрос:

{
  "text": "Congratulations! You've won a $1000 Walmart gift card. Click here to claim!"
}

Ответ:

{
  "prediction": {
    "label": "spam",
    "probability": 0.9854
  },
  "latency": 0.0032
}

2. Обратная связь (Feedback)

Если модель ошиблась, отправьте правильную метку для улучшения системы. Endpoint: POST /feedback Запрос:

{
  "text": "Meeting moved to 4 PM",
  "correct_label": "0"
}

Цикл переобучения (Retraining)

Для актуализации модели при изменении тактик спамеров:

    Соберите данные через эндпоинт /feedback.

    Запустите скрипт переобучения:

    python retrain.py

Скрипт объединит основной датасет с новыми данными, создаст бэкап старой модели и обучит новую.
Тестирование

Для запуска всех проверок выполните:

pytest tests/

### Результаты исследования

В ходе разработки были протестированы три архитектуры:

    TF-IDF + LogReg: Baseline (AUC: 0.982)

    FastText: Оптимальный выбор для Prod (AUC: 0.989)

    DistilBERT: Максимальная точность (AUC: 0.998), но высокая задержка.

Выбран FastText из-за идеального баланса скорости (в 100 раз быстрее BERT) и точности.













Интерактивная документация: http://localhost:8000/docs или:
	curl -X 'POST' \
	  'http://localhost:8000/predict' \
	  -H 'Content-Type: application/json' \
	  -d '{"text": "Congratulations! You won a free prize, call now!"}'