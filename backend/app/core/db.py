from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from .config import settings

# Создаем движок базы данных
engine = create_engine(
    settings.DATABASE_URL,
    # Для SQLite добавляем специфичные настройки
    connect_args={
        "check_same_thread": False,
        "timeout": 20
    } if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DEBUG,  # показывать SQL запросы в DEBUG режиме
    pool_pre_ping=True,  # проверяем соединение перед использованием
    pool_recycle=3600    # пересоздаем соединения каждый час
)

# Создаем фабрику сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для всех моделей
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency для получения сессии базы данных
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Явно коммитим транзакцию
    except Exception as e:
        db.rollback()  # Откатываем при ошибке
        raise
    finally:
        db.close()


def create_tables():
    """
    Создание всех таблиц в базе данных
    """
    Base.metadata.create_all(bind=engine)