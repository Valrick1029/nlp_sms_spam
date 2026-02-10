## Сервис детекции спама

Микросервис для фильтрации SMS-спама на базе DistilBERT. Система включает в себя полноценный API, контейнеризацию и механизм автоматического дообучения на основе обратной связи от пользователей.

### Ключевые особенности

    Производственный API: Реализован на FastAPI с валидацией данных через Pydantic.

    MLOps ready: Встроенный механизм Feedback Loop для сбора ошибок и скрипт автоматического переобучения.

    Контейнеризация: Полная изоляция среды через Docker.

    Надежность: Покрытие Unit и Интеграционными тестами.

### Технологический стек

    ML Core: DistilBERT

    API Framework: FastAPI, Uvicorn

    Data Science: Pandas, Scikit-learn, Transformers (для EDA и базовых метрик)

    DevOps: Docker, Pytest

    Logging: Python Logging Module

### Структура проекта

.
├── src/                # Исходный код сервиса
│   ├── main.py         # Основной API сервер
│   └── pipeline.py     # Логика обработки и предсказания
├── tests/              # Автоматические тесты
├── Dockerfile          # Инструкция для сборки контейнера
├── requirements.txt    # Зависимости проекта
└── retrain.py          # Скрипт автоматического переобучения


### Быстрый запуск

Сборка образа:

	docker build -t spam-filter:v1 .

Запуск контейнера:

	docker run -p 8000:8000 spam-filter:v1

Локальный запуск

    Установите зависимости: pip install -r requirements.txt

    Запустите сервер: uvicorn src.main:app --reload

### Использование API
Использование API
1. Предсказание
Запрос:

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

Обратная связь
Если модель ошиблась, можно отправить правильную метку для улучшения системы. 
Запрос:

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

2. Обратная связь

	{
	  "text": "Meeting moved to 4 PM",
	  "correct_label": "0"
	}

### Цикл переобучения
Для актуализации модели при изменении тактик спамеров необходимо:
Собрать данные через эндпоинт.
Запустить скрипт переобучения:

	python retrain.py
    {
      "text": "Meeting moved to 4 PM",
      "correct_label": "0"
    }


### Тестирование

	pytest tests/
    Соберите данные через эндпоинт /feedback.

    Запустите скрипт переобучения:

    python retrain.py


Для запуска всех проверок выполните:

    pytest tests/

### Результаты исследования

В ходе разработки были протестированы три архитектуры:

- TF-IDF + LogReg: Baseline (AUC: 0.982)

- FastText: Оптимальный выбор для Prod (AUC: 0.989)

- DistilBERT: Максимальная точность (AUC: 0.998).


Выбран DistilBERT, так как эта модель выдала показатели метрик, гораздо лучшие, чем модели-конкуренты.

