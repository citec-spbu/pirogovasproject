```text
backend/
│
├── app/                       # основной пакет приложения
│   ├── __init__.py
│   ├── main.py                 # точка входа: создаёт FastAPI()
│   │
│   ├── api/                    # все эндпоинты (ручки)
│   │   ├── __init__.py
│   │   ├── v1/                  # версионирование (v1 — первая версия)
│   │   │   ├── __init__.py
│   │   │   ├── registration.py     # регистрация, аутентификация
│   │   └── dependencies.py       # общие зависимости
│   │
│   ├── core/                    # конфигурация, подключения к внешним сервисам
│   │   ├── __init__.py
│   │   ├── config.py             # загрузка настроек из .env
│   │   ├── database.py           # создание движка БД, сессии
│   │   ├── celery.py             # инициализация Celery (если нужна)
│   │   └── minio.py              # клиент для MinIO
│   │
│   ├── models/                   # модели SQLAlchemy / SQLModel
│   │   ├── __init__.py
│   │   ├── user.py                # таблица пользователей
│   │   └── report.py              # таблица отчётов
│   │
│   ├── schemas/                   # Pydantic-схемы (запросы/ответы)
│   │   ├── __init__.py
│   │   ├── user.py                 # схемы для регистрации, логина
│   │   └── report.py               # схемы для отчётов, отзывов
│   │
│   ├── services/                   # бизнес-логика
│   │   ├── __init__.py
│   │   ├── auth_service.py          # хеширование паролей, создание токенов
│   │   ├── report_service.py        # получение списка отчётов, генерация PDF
│   │   ├── llm_service.py           # генерация промпта, вызов LLM
│   │   └── diagnosis_service.py     # запись фото, метаданных в БД
│   │
│   ├── tasks/                       # Celery-задачи (длительные операции)
│   │   ├── __init__.py
│   │   └── report_tasks.py          # (возможно генерация PDF в фоне)
│   │
│   ├── utils/                        # вспомогательные функции
│   │   ├── __init__.py
│   │   ├── file_handler.py           # разархивирование, сохранение на диск
│   │   └── pdf_generator.py          # генерация PDF из HTML (Jinja2)
│   │
│   └── tests/                         # тесты
│       ├── __init__.py
│       └── test_registration.py
│
├── .env.example                     # шаблон переменных окружения
├── requirements.txt                  # зависимости
└── README.md                         # описание проекта
```