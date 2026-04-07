```txt
backend/
│
├── app                             # основной пакет приложения
│   ├── __init__.py
│   ├── main.py                     # точка входа: создаёт FastAPI()
│   ├── api                         # все эндпоинты (ручки)
│   │   ├── dependencies.py         # общие зависимости
│   │   ├── __init__.py
│   │   └── v1                      # версионирование (v1 — первая версия)
│   │       ├── __init__.py
│   │       ├── llm.py              # Генерация отчёта с помощью запроса в llm
│   │       ├── reports.py          # Создание отчёта, списка отчётов
│   │       └── users.py            # регистрация, аутентификация
│   │
│   ├── core                        # конфигурация, подключения к внешним сервисам
│   │   ├── __init__.py
│   │   ├── config.py             # загрузка настроек из .env
│   │   ├── database.py           # создание движка БД, сессии
│   │   ├── security.py           # кеширование паролей
│   │   ├── celery.py             # инициализация Celery (если нужна)
│   │   └── minio.py              # клиент для MinIO
│   │
│   │
│   ├── models                          # модели SQLAlchemy / SQLModel
│   │   ├── __init__.py
│   │   ├── report.py                   # таблица отчётов
│   │   └── user.py                     # таблица пользователей
│   │
│   ├── schemas                         # Pydantic-схемы (запросы/ответы)
│   │   ├── __init__.py
│   │   ├── report.py                    # схемы для регистрации, логина
│   │   └── user.py                      # схемы для отчётов, отзывов
│   │
│   ├── services                        # бизнес-логика
│   │   ├── __init__.py
│   │   ├── llm_service.py              # генерация промпта, вызов LLM
│   │   ├── report_service.py           # получение списка отчётов
│   │   └── user_service.py             # хеширование паролей, создание токенов
│   │
│   ├── utils                           # вспомогательные функции
│   │    ├── file_handler.py            # разархивирование, сохранение на диск
│   │    ├── html_report_generator.py   #html генератор
│   │    ├── __init__.py
│   │    └── pdf_generator.py           # генерация PDF из HTML (Jinja2)
│   │
│   ├── tasks/                       # Celery-задачи (длительные операции)
│   │   ├── __init__.py
│   │   └── report_tasks.py          # (возможно генерация PDF в фоне)
│   │
│   └── tests/                         # тесты
│       ├── __init__.py
│       └── test_registration.py
│ 
├── README.md                   # описание бэкенда
├── .env.example                     # шаблон переменных окружения
└── requirements.txt                # зависимости
```
