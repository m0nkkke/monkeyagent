from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
import shutil
import logging

from ...core.db import get_db
from ...core.config import settings
from ...schemas.document import (
    DocumentResponse, DocumentWithContent, DocumentListResponse,
    DocumentStatsResponse, SummarizeRequest, SummarizeResponse,
    QuestionRequest, QuestionResponse, OCRRequest, OCRResponse,
    DocumentUploadResponse, ProcessingStatus, SummaryType
)
from ...crud.document import document_crud, summary_crud, qa_crud
from ...services.document_loader import document_loader
from ...services.summarizer import document_summarizer
from ...services.indexer import document_indexer, vector_search_engine
from ...services.qa import qa_service
from ...services.ocr import ocr_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Загрузка документа"""
    try:
        # Валидация файла
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Проверяем размер файла
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Создаем уникальное имя файла
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Создаем запись в БД
        from ...schemas.document import DocumentCreate
        document_create = DocumentCreate(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_type=file_extension.lstrip('.'),
            file_size=file_size
        )
        
        db_document = document_crud.create_document(db, document_create)
        
        # Запускаем обработку документа в фоне
        background_tasks.add_task(process_document, db_document.id, file_path)
        
        logger.info(f"Document uploaded: {file.filename} -> {unique_filename}")
        
        return DocumentUploadResponse(
            message="Document uploaded successfully",
            document_id=db_document.id,
            status=ProcessingStatus.PENDING
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {db_document.id}: {e}")
        
        # Обновляем статус на "ошибка"
        from ...schemas.document import DocumentUpdate
        document_crud.update_document(
            db,
            db_document.id,
            DocumentUpdate(
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
        )
    finally:
        db.close()


@router.get("/", response_model=DocumentListResponse)
def get_documents(
    skip: int = 0,
    limit: int = 100,
    status: Optional[ProcessingStatus] = None,
    db: Session = Depends(get_db)
):
    """Получение списка документов"""
    try:
        documents = document_crud.get_documents(db, skip=skip, limit=limit, status=status)
        total = document_crud.get_documents_count(db, status=status)
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            page=skip // limit + 1,
            per_page=limit
        )
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DocumentStatsResponse)
def get_documents_stats(db: Session = Depends(get_db)):
    """Получение статистики по документам"""
    try:
        stats = document_crud.get_documents_stats(db)
        return DocumentStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    """Получение документа по ID"""
    document = document_crud.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/content", response_model=DocumentWithContent)
def get_document_with_content(document_id: int, db: Session = Depends(get_db)):
    """Получение документа с полным содержимым"""
    document = document_crud.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Удаление документа"""
    try:
        document = document_crud.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Удаляем файл
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Удаляем из векторного индекса
        document_indexer.remove_document_from_index(document_id)
        
        # Удаляем из БД
        success = document_crud.delete_document(db, document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/summarize", response_model=SummarizeResponse)
async def summarize_document(
    document_id: int,
    request: SummarizeRequest,
    db: Session = Depends(get_db)
):
    """Создание конспекта документа"""
    try:
        # Проверяем существование документа
        document = document_crud.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.extracted_text:
            raise HTTPException(status_code=400, detail="Document has no extracted text")
        
        # Создаем конспект
        summary_result = document_summarizer.summarize_document(
            document_text=document.extracted_text,
            summary_type=request.summary_type
        )
        
        # Сохраняем в БД
        from ...schemas.document import DocumentSummaryCreate
        summary_create = DocumentSummaryCreate(
            document_id=document_id,
            summary_text=summary_result["summary_text"],
            summary_type=request.summary_type,
            model_used=summary_result["model_used"],
            tokens_used=summary_result["tokens_used"],
            generation_time=summary_result["generation_time"]
        )
        
        db_summary = summary_crud.create_summary(db, summary_create)
        
        return SummarizeResponse(
            message="Summary created successfully",
            summary_id=db_summary.id,
            summary=db_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """Ответ на вопрос по документам"""
    try:
        # Валидация вопроса
        is_valid, error_msg = qa_service.validate_question(request.question)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Проверяем существование документов
        for doc_id in request.document_ids:
            document = document_crud.get_document(db, doc_id)
            if not document:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            if not document.extracted_text:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document {doc_id} has no extracted text"
                )
        
        # Создаем или получаем сессию Q&A
        if request.session_id:
            qa_session = qa_crud.get_session(db, request.session_id)
            if not qa_session:
                raise HTTPException(status_code=404, detail="QA session not found")
        else:
            # Создаем новую сессию для первого документа
            from ...schemas.document import QASessionCreate
            session_create = QASessionCreate(
                document_id=request.document_ids[0],
                session_name=f"Q&A: {request.question[:50]}..."
            )
            qa_session = qa_crud.create_session(db, session_create)
        
        # Получаем ответ
        answer_result = qa_service.answer_question(
            question=request.question,
            document_ids=request.document_ids
        )
        
        # Сохраняем пару вопрос-ответ
        from ...schemas.document import QAPairCreate
        qa_pair_create = QAPairCreate(
            session_id=qa_session.id,
            question=request.question,
            answer=answer_result["answer"],
            confidence_score=answer_result.get("confidence_score"),
            model_used=answer_result.get("model_used"),
            tokens_used=answer_result.get("tokens_used"),
            response_time=answer_result.get("response_time"),
            context_snippets=str(answer_result.get("context_sources", []))
        )
        
        qa_pair = qa_crud.create_qa_pair(db, qa_pair_create)
        
        return QuestionResponse(
            answer=answer_result["answer"],
            confidence_score=answer_result.get("confidence_score"),
            context_snippets=answer_result.get("context_snippets", []),
            qa_pair_id=qa_pair.id,
            session_id=qa_session.id,
            response_time=answer_result.get("response_time")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", response_model=OCRResponse)
async def extract_text_from_image(request: OCRRequest):
    """Извлечение текста из изображения (OCR)"""
    try:
        # Валидация изображения
        is_valid, error_msg = ocr_service.validate_image_data(request.image_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Обработка изображения
        result = ocr_service.extract_text_from_screenshot(request.image_data)
        
        return OCRResponse(
            extracted_text=result["extracted_text"],
            processing_time=result["processing_time"],
            confidence_score=result.get("confidence_score")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing OCR request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/summaries")
def get_document_summaries(document_id: int, db: Session = Depends(get_db)):
    """Получение всех конспектов документа"""
    try:
        document = document_crud.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        summaries = summary_crud.get_summaries_by_document(db, document_id)
        return {"summaries": summaries}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summaries for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/qa-sessions")
def get_document_qa_sessions(document_id: int, db: Session = Depends(get_db)):
    """Получение всех сессий Q&A документа"""
    try:
        document = document_crud.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        sessions = qa_crud.get_sessions_by_document(db, document_id)
        return {"qa_sessions": sessions}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting QA sessions for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{document_id}")
def get_similar_documents(document_id: int, top_k: int = 3):
    """Поиск похожих документов"""
    try:
        similar_docs = vector_search_engine.find_similar_documents(document_id, top_k)
        return {"similar_documents": similar_docs}
        
    except Exception as e:
        logger.error(f"Error finding similar documents for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search_documents(
    query: str,
    document_ids: Optional[List[int]] = None,
    top_k: int = 5
):
    """Семантический поиск по документам"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = document_indexer.search_similar_chunks(
            query_text=query,
            top_k=top_k,
            document_ids=document_ids
        )
        
        return {"search_results": results, "query": query}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/stats")
def get_index_stats():
    """Получение статистики векторного индекса"""
    try:
        stats = document_indexer.get_index_stats()
        return {"index_stats": stats}
        
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/rebuild")
def rebuild_index():
    """Перестройка векторного индекса"""
    try:
        success = document_indexer.rebuild_index()
        if success:
            return {"message": "Index rebuilt successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to rebuild index")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_document(document_id: int, file_path: str):
    """Фоновая обработка документа"""
    db = next(get_db())
    
    try:
        # Обновляем статус на "обработка"
        from ...schemas.document import DocumentUpdate
        document_crud.update_document(
            db, 
            document_id, 
            DocumentUpdate(processing_status=ProcessingStatus.PROCESSING)
        )
        
        # Извлекаем текст
        logger.info(f"Processing document {document_id}")
        extracted_text, page_count = document_loader.extract_text(file_path)
        
        # Обновляем документ с извлеченным текстом
        document_crud.update_document(
            db,
            document_id,
            DocumentUpdate(
                extracted_text=extracted_text,
                page_count=page_count,
                processing_status=ProcessingStatus.COMPLETED
            )
        )
        
        # Добавляем в векторный индекс
        document = document_crud.get_document(db, document_id)
        if document:
            vector_ids = document_indexer.add_document_to_index(
                document_id=document_id,
                document_text=extracted_text,
                document_title=document.original_filename
            )
            logger.info(f"Added {len(vector_ids)} vectors to index for document {document_id}")
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        
        from ...schemas.document import DocumentUpdate
        document_crud.update_document(
            db,
            document_id,
            DocumentUpdate(
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
        )
    finally:
        db.close()
