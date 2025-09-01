import os
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForQuestionAnswering,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import logging
from typing import Optional, Dict, Any

from .config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Менеджер для управления локальными моделями"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # Флаги загруженности моделей
        self._summarization_loaded = False
        self._embedding_loaded = False
        self._qa_loaded = False
    
    def _download_and_cache_model(self, model_name: str, model_type: str = "auto"):
        """Загрузка и кэширование модели"""
        cache_dir = os.path.join(settings.MODELS_DIR, model_name.replace("/", "_"))
        
        try:
            if model_type == "summarization":
                tokenizer = T5Tokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir
                )
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                
            elif model_type == "qa":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                
            elif model_type == "embedding":
                model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir,
                    device=self.device
                )
                tokenizer = None
                
            else:  # auto
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def load_summarization_model(self):
        """Загрузка модели для суммаризации"""
        if self._summarization_loaded:
            return
        
        logger.info(f"Loading summarization model: {settings.SUMMARIZATION_MODEL}")
        
        model, tokenizer = self._download_and_cache_model(
            settings.SUMMARIZATION_MODEL, 
            "summarization"
        )
        
        self.models["summarization"] = model
        self.tokenizers["summarization"] = tokenizer
        self._summarization_loaded = True
        
        logger.info("Summarization model loaded successfully")
    
    def load_embedding_model(self):
        """Загрузка модели для эмбеддингов"""
        if self._embedding_loaded:
            return
        
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        model, _ = self._download_and_cache_model(
            settings.EMBEDDING_MODEL,
            "embedding"
        )
        
        self.models["embedding"] = model
        self._embedding_loaded = True
        
        logger.info("Embedding model loaded successfully")
    
    def load_qa_model(self):
        """Загрузка модели для вопросов-ответов"""
        if self._qa_loaded:
            return
        
        logger.info(f"Loading QA model: {settings.QA_MODEL}")
        
        model, tokenizer = self._download_and_cache_model(
            settings.QA_MODEL,
            "qa"
        )
        
        self.models["qa"] = model
        self.tokenizers["qa"] = tokenizer
        self._qa_loaded = True
        
        logger.info("QA model loaded successfully")
    
    def load_all_models(self):
        """Загрузка всех моделей"""
        logger.info("Loading all models...")
        
        self.load_embedding_model()
        self.load_summarization_model()
        self.load_qa_model()
        
        logger.info("All models loaded successfully")
    
    def get_summarization_model(self):
        """Получение модели суммаризации"""
        if not self._summarization_loaded:
            self.load_summarization_model()
        return self.models["summarization"], self.tokenizers["summarization"]
    
    def get_embedding_model(self):
        """Получение модели эмбеддингов"""
        if not self._embedding_loaded:
            self.load_embedding_model()
        return self.models["embedding"]
    
    def get_qa_model(self):
        """Получение модели Q&A"""
        if not self._qa_loaded:
            self.load_qa_model()
        return self.models["qa"], self.tokenizers["qa"]
    
    def unload_model(self, model_type: str):
        """Выгрузка модели из памяти"""
        if model_type in self.models:
            del self.models[model_type]
            if model_type in self.tokenizers:
                del self.tokenizers[model_type]
            
            # Освобождение памяти GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Обновляем флаги
            if model_type == "summarization":
                self._summarization_loaded = False
            elif model_type == "embedding":
                self._embedding_loaded = False
            elif model_type == "qa":
                self._qa_loaded = False
            
            logger.info(f"Model {model_type} unloaded from memory")
    
    def unload_all_models(self):
        """Выгрузка всех моделей"""
        for model_type in list(self.models.keys()):
            self.unload_model(model_type)
        
        logger.info("All models unloaded from memory")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о загруженных моделях"""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "models_loaded": {
                "summarization": self._summarization_loaded,
                "embedding": self._embedding_loaded,
                "qa": self._qa_loaded
            },
            "model_paths": {
                "summarization": settings.SUMMARIZATION_MODEL,
                "embedding": settings.EMBEDDING_MODEL,
                "qa": settings.QA_MODEL
            },
            "models_dir": settings.MODELS_DIR
        }


# Создаем глобальный экземпляр менеджера моделей
model_manager = ModelManager()


def initialize_models():
    """Инициализация моделей при запуске приложения"""
    try:
        logger.info("Initializing models...")
        model_manager.load_all_models()
        logger.info("Models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return False


if __name__ == "__main__":
    # Скрипт для предварительной загрузки моделей
    logging.basicConfig(level=logging.INFO)
    
    print("Starting model download and caching...")
    success = initialize_models()
    
    if success:
        print("All models downloaded and cached successfully!")
        info = model_manager.get_model_info()
        print(f"Models info: {info}")
    else:
        print("Failed to download models. Check logs for details.")