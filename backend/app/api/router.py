from fastapi import APIRouter

from .v1.document import router as document_router

# Создаем основной роутер для API
api_router = APIRouter()

# Подключаем роутеры версии v1
api_router.include_router(
    document_router, 
    prefix="/v1/documents", 
    tags=["documents"]
)