from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = "sqlite:///./documents.db"
    
    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    
    # File storage settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".pptx", ".docx", ".txt"]
    
    # Local AI Models settings
    MODELS_DIR: str = "./models"  # директория для локальных моделей
    
    # Русскоязычные модели
    SUMMARIZATION_MODEL: str = "IlyaGusev/ru_sum_gazeta"
    EMBEDDING_MODEL: str = "cointegrated/rubert-tiny2"
    QA_MODEL: str = "IlyaGusev/multilingual_en_ru_qa_base"
    
    # Vector DB settings (FAISS)
    VECTOR_DB_PATH: str = "./vector_db"
    VECTOR_INDEX_FILE: str = "document_index.faiss"
    
    # Настройки для обработки текста
    MAX_CHUNK_SIZE: int = 512  # максимальный размер чанка для векторизации
    CHUNK_OVERLAP: int = 50    # перекрытие между чанками
    
    # OCR settings
    TESSERACT_PATH: Optional[str] = "C:/Program Files/Tesseract-OCR/tesseract.exe"  # путь к tesseract если нужен
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Создаем глобальный экземпляр настроек
settings = Settings()

# Создаем директории если их нет
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)