from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryType(str, Enum):
    GENERAL = "general"
    DETAILED = "detailed"
    BRIEF = "brief"


# Базовые схемы для Document
class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str = Field(..., pattern="^(pdf|pptx|docx|txt)$")


class DocumentCreate(DocumentBase):
    file_path: str
    file_size: int


class DocumentUpdate(BaseModel):
    is_processed: Optional[bool] = None
    processing_status: Optional[ProcessingStatus] = None
    error_message: Optional[str] = None
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None


class DocumentResponse(DocumentBase):
    id: int
    file_size: int
    is_processed: bool
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    page_count: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class DocumentWithContent(DocumentResponse):
    """Документ с полным содержимым"""
    extracted_text: Optional[str] = None


# Схемы для Summary
class DocumentSummaryBase(BaseModel):
    summary_text: str
    summary_type: SummaryType = SummaryType.GENERAL


class DocumentSummaryCreate(DocumentSummaryBase):
    document_id: int
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    generation_time: Optional[float] = None


class DocumentSummaryResponse(DocumentSummaryBase):
    id: int
    document_id: int
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    generation_time: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# Схемы для QA
class QAPairBase(BaseModel):
    question: str
    answer: str


class QAPairCreate(QAPairBase):
    session_id: int
    confidence_score: Optional[float] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    context_snippets: Optional[str] = None  # JSON строка


class QAPairResponse(QAPairBase):
    id: int
    session_id: int
    confidence_score: Optional[float] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    context_snippets: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class QASessionBase(BaseModel):
    session_name: str = "Q&A Session"


class QASessionCreate(QASessionBase):
    document_id: int


class QASessionResponse(QASessionBase):
    id: int
    document_id: int
    created_at: datetime
    qa_pairs: List[QAPairResponse] = []
    
    class Config:
        from_attributes = True


# Схемы для запросов API
class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    status: ProcessingStatus


class SummarizeRequest(BaseModel):
    document_id: int
    summary_type: SummaryType = SummaryType.GENERAL


class SummarizeResponse(BaseModel):
    message: str
    summary_id: int
    summary: DocumentSummaryResponse


class QuestionRequest(BaseModel):
    document_ids: List[int] = Field(..., min_items=1, description="ID документов для поиска ответа")
    question: str = Field(..., min_length=3, description="Вопрос для поиска ответа")
    session_id: Optional[int] = None  # Если не указан, создастся новая сессия


class QuestionResponse(BaseModel):
    answer: str
    confidence_score: Optional[float] = None
    context_snippets: Optional[List[str]] = None
    qa_pair_id: int
    session_id: int
    response_time: Optional[float] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    per_page: int


class DocumentStatsResponse(BaseModel):
    total_documents: int
    processed_documents: int
    pending_documents: int
    failed_documents: int
    total_summaries: int
    total_qa_pairs: int


# Схемы для OCR (скриншот экрана)
class OCRRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v or len(v) < 100:  # минимальная длина для base64 изображения
            raise ValueError('Invalid image data')
        return v


class OCRResponse(BaseModel):
    extracted_text: str
    processing_time: float
    confidence_score: Optional[float] = None