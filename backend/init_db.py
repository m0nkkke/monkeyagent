#!/usr/bin/env python3
"""
Скрипт для инициализации базы данных и предварительной загрузки моделей
"""
import sys
import os

# Добавляем путь к приложению
sys.path.append(os.path.dirname(__file__))

from app.core.db import create_tables, engine
from app.core.model_manager import initialize_models
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Инициализация базы данных"""
    try:
        logger.info("Creating database tables...")
        create_tables()
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


def download_models():
    """Загрузка и кэширование моделей"""
    try:
        logger.info("Downloading and caching AI models...")
        logger.info("This may take several minutes on first run...")
        
        success = initialize_models()
        if success:
            logger.info("All models downloaded and cached successfully")
            return True
        else:
            logger.error("Failed to download some models")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False


def main():
    """Основная функция инициализации"""
    print("=" * 60)
    print("Document AI Assistant - Database and Models Initialization")
    print("=" * 60)
    
    # Создаем необходимые директории
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Database: {settings.DATABASE_URL}")
    print(f"  Upload directory: {settings.UPLOAD_DIR}")
    print(f"  Models directory: {settings.MODELS_DIR}")
    print(f"  Vector DB directory: {settings.VECTOR_DB_PATH}")
    print()
    
    # Инициализация БД
    print("1. Initializing database...")
    db_success = init_database()
    
    if not db_success:
        print("❌ Database initialization failed!")
        return 1
    
    print("✅ Database initialized successfully")
    print()
    
    # Загрузка моделей
    print("2. Downloading AI models...")
    print("   This will download several GB of models on first run...")
    
    user_input = input("   Continue? [Y/n]: ").strip().lower()
    if user_input and user_input != 'y' and user_input != 'yes':
        print("   Skipping model download. Models will be downloaded on first use.")
        models_success = True
    else:
        models_success = download_models()
    
    if not models_success:
        print("⚠️  Model download failed, but application can still run")
        print("   Models will be downloaded automatically on first use")
    else:
        print("✅ Models downloaded successfully")
    
    print()
    print("=" * 60)
    print("Initialization completed!")
    print("=" * 60)
    print("You can now start the application with:")
    print(f"  cd {os.path.dirname(__file__)}")
    print("  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
    print()
    print("API will be available at: http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    
    return 0


if __name__ == "__main__":
    exit(main())