from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from ..core.db import Base


class Document(Base):
    """
    Модель для хранения загруженных документов
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False)  # pdf, pptx, docx, txt
    file_size = Column(Integer, nullable=False)  # размер в байтах
    
    # Метаданные обработки
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Извлеченный контент
    extracted_text = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    
    # Время создания и обновления
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Связи с другими таблицами
    summaries = relationship("DocumentSummary", back_populates="document", cascade="all, delete-orphan")
    qa_sessions = relationship("QASession", back_populates="document", cascade="all, delete-orphan")
    embeddings = relationship("DocumentEmbedding", back_populates="document", cascade="all, delete-orphan")


class DocumentSummary(Base):
    """
    Модель для хранения конспектов документов
    """
    __tablename__ = "document_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Контент конспекта
    summary_text = Column(Text, nullable=False)
    summary_type = Column(String(50), default="general")  # general, detailed, brief
    
    # Метаданные генерации
    model_used = Column(String(100), nullable=True)  # какая модель использовалась
    tokens_used = Column(Integer, nullable=True)
    generation_time = Column(Float, nullable=True)  # время генерации в секундах
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Связи
    document = relationship("Document", back_populates="summaries")


class QASession(Base):
    """
    Модель для хранения сессий вопросов-ответов
    """
    __tablename__ = "qa_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    session_name = Column(String(255), default="Q&A Session")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Связи
    document = relationship("Document", back_populates="qa_sessions")
    qa_pairs = relationship("QAPair", back_populates="session", cascade="all, delete-orphan")


class QAPair(Base):
    """
    Модель для хранения пар вопрос-ответ
    """
    __tablename__ = "qa_pairs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("qa_sessions.id"), nullable=False)
    
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    
    # Метаданные
    confidence_score = Column(Float, nullable=True)  # уверенность модели в ответе
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    response_time = Column(Float, nullable=True)  # время ответа в секундах
    
    # Контекст для ответа (фрагменты документа, которые использовались)
    context_snippets = Column(Text, nullable=True)  # JSON со ссылками на фрагменты
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Связи
    session = relationship("QASession", back_populates="qa_pairs")


class DocumentEmbedding(Base):
    """
    Модель для хранения эмбеддингов документов (для векторного поиска)
    """
    __tablename__ = "document_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Фрагмент текста и его эмбеддинг
    text_chunk = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # порядковый номер чанка в документе
    
    # Метаданные чанка
    page_number = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)  # начальная позиция в документе
    end_char = Column(Integer, nullable=True)    # конечная позиция в документе
    
    # Информация об эмбеддинге
    embedding_model = Column(String(100), nullable=False)
    vector_id = Column(String(255), nullable=True)  # ID в векторной БД
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Связи
    document = relationship("Document", back_populates="embeddings")