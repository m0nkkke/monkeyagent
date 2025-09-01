from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import uvicorn

from .core.config import settings
from .core.db import create_tables
from .core.model_manager import model_manager, initialize_models
from .api.router import api_router

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    
    # Startup
    logger.info("Starting Document AI Assistant...")
    
    # Создаем таблицы БД
    try:
        create_tables()
        logger.info("Database tables created")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise
    
    # Инициализируем модели (в фоновом режиме)
    try:
        logger.info("Initializing AI models...")
        # Загружаем только эмбеддинг модель при старте
        # Остальные модели будут загружаться по мере необходимости
        model_manager.load_embedding_model()
        logger.info("Essential models loaded")
    except Exception as e:
        logger.warning(f"Error loading models: {e}")
        # Продолжаем работу даже если модели не загрузились
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document AI Assistant...")
    
    # Выгружаем модели для освобождения памяти
    try:
        model_manager.unload_all_models()
        logger.info("Models unloaded")
    except Exception as e:
        logger.error(f"Error unloading models: {e}")
    
    logger.info("Application shutdown completed")


# Создание приложения FastAPI
app = FastAPI(
    title="Document AI Assistant",
    description="Локальный ИИ-ассистент для работы с документами: конспектирование, Q&A и OCR",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Настройка CORS для работы с десктопным клиентом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене следует ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение API роутов
app.include_router(api_router, prefix="/api")

# Статические файлы (для загруженных документов)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Document AI Assistant API",
        "version": "0.1.0",
        "status": "running",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Проверка состояния приложения"""
    try:
        model_info = model_manager.get_model_info()
        
        return {
            "status": "healthy",
            "timestamp": "2025-08-31T12:00:00Z",  # В реальном приложении использовать datetime.utcnow()
            "models": model_info,
            "database": "connected",
            "upload_dir": settings.UPLOAD_DIR,
            "vector_db": settings.VECTOR_DB_PATH
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.get("/models/info")
async def models_info():
    """Информация о доступных моделях"""
    return {
        "models": {
            "summarization": {
                "name": settings.SUMMARIZATION_MODEL,
                "loaded": model_manager._summarization_loaded,
                "purpose": "Создание конспектов документов на русском языке"
            },
            "embedding": {
                "name": settings.EMBEDDING_MODEL,
                "loaded": model_manager._embedding_loaded,
                "purpose": "Векторизация текста для семантического поиска"
            },
            "qa": {
                "name": settings.QA_MODEL,
                "loaded": model_manager._qa_loaded,
                "purpose": "Ответы на вопросы по документам (рус/англ)"
            }
        },
        "device": model_manager.device,
        "cuda_available": model_manager.device == "cuda"
    }


@app.post("/models/load/{model_type}")
async def load_model(model_type: str, background_tasks: BackgroundTasks):
    """Загрузка конкретной модели"""
    if model_type not in ["summarization", "embedding", "qa", "all"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    def load_model_task():
        try:
            if model_type == "summarization":
                model_manager.load_summarization_model()
            elif model_type == "embedding":
                model_manager.load_embedding_model()
            elif model_type == "qa":
                model_manager.load_qa_model()
            elif model_type == "all":
                model_manager.load_all_models()
            
            logger.info(f"Model {model_type} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
    
    background_tasks.add_task(load_model_task)
    
    return {
        "message": f"Loading {model_type} model in background",
        "status": "started"
    }


@app.post("/models/unload/{model_type}")
async def unload_model(model_type: str):
    """Выгрузка модели из памяти"""
    if model_type not in ["summarization", "embedding", "qa", "all"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    try:
        if model_type == "all":
            model_manager.unload_all_models()
        else:
            model_manager.unload_model(model_type)
        
        return {
            "message": f"Model {model_type} unloaded successfully",
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Error unloading model {model_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Запуск сервера для разработки
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )