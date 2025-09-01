from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime

from ..models.document import (
    Document, DocumentSummary, QASession, QAPair, DocumentEmbedding
)
from ..schemas.document import (
    DocumentCreate, DocumentUpdate, 
    DocumentSummaryCreate, QASessionCreate, QAPairCreate,
    ProcessingStatus
)


class DocumentCRUD:
    
    def create_document(self, db: Session, document: DocumentCreate) -> Document:
        """Создание нового документа"""
        try:
            db_document = Document(
                filename=document.filename,
                original_filename=document.original_filename,
                file_path=document.file_path,
                file_type=document.file_type,
                file_size=document.file_size,
                processing_status=ProcessingStatus.PENDING
            )
            db.add(db_document)
            db.flush()  # Получаем ID без коммита
            db.refresh(db_document)
            return db_document
        except Exception as e:
            db.rollback()
            raise e
    
    def get_document(self, db: Session, document_id: int) -> Optional[Document]:
        """Получение документа по ID"""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                # Принудительно загружаем все атрибуты
                _ = document.id, document.filename, document.processing_status
            return document
        except Exception as e:
            db.rollback()
            raise e
    
    def get_documents(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[ProcessingStatus] = None
    ) -> List[Document]:
        """Получение списка документов с пагинацией"""
        try:
            query = db.query(Document)
            
            if status:
                query = query.filter(Document.processing_status == status)
            
            documents = query.order_by(desc(Document.created_at)).offset(skip).limit(limit).all()
            
            # Явно загружаем данные для избежания lazy loading проблем
            for doc in documents:
                _ = doc.id, doc.filename, doc.processing_status
            
            return documents
        except Exception as e:
            db.rollback()
            raise e
    
    def get_documents_count(
        self, 
        db: Session, 
        status: Optional[ProcessingStatus] = None
    ) -> int:
        """Подсчет количества документов"""
        try:
            query = db.query(Document)
            
            if status:
                query = query.filter(Document.processing_status == status)
            
            count = query.count()
            return count
        except Exception as e:
            db.rollback()
            raise e
    
    def update_document(
        self, 
        db: Session, 
        document_id: int, 
        document_update: DocumentUpdate
    ) -> Optional[Document]:
        """Обновление документа"""
        try:
            db_document = self.get_document(db, document_id)
            if not db_document:
                return None
            
            update_data = document_update.dict(exclude_unset=True)
            
            # Автоматически устанавливаем processed_at при завершении обработки
            if document_update.processing_status == ProcessingStatus.COMPLETED:
                from datetime import datetime
                update_data["processed_at"] = datetime.utcnow()
            
            for field, value in update_data.items():
                setattr(db_document, field, value)
            
            db.flush()
            db.refresh(db_document)
            return db_document
            
        except Exception as e:
            db.rollback()
            raise e
    
    def delete_document(self, db: Session, document_id: int) -> bool:
        """Удаление документа"""
        db_document = self.get_document(db, document_id)
        if not db_document:
            return False
        
        db.delete(db_document)
        db.commit()
        return True
    
    def get_documents_stats(self, db: Session) -> dict:
        """Получение статистики по документам"""
        total_documents = db.query(Document).count()
        processed_documents = db.query(Document).filter(
            Document.processing_status == ProcessingStatus.COMPLETED
        ).count()
        pending_documents = db.query(Document).filter(
            Document.processing_status == ProcessingStatus.PENDING
        ).count()
        failed_documents = db.query(Document).filter(
            Document.processing_status == ProcessingStatus.FAILED
        ).count()
        
        total_summaries = db.query(DocumentSummary).count()
        total_qa_pairs = db.query(QAPair).count()
        
        return {
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "pending_documents": pending_documents,
            "failed_documents": failed_documents,
            "total_summaries": total_summaries,
            "total_qa_pairs": total_qa_pairs
        }


class SummaryCRUD:
    
    def create_summary(
        self, 
        db: Session, 
        summary: DocumentSummaryCreate
    ) -> DocumentSummary:
        """Создание конспекта документа"""
        db_summary = DocumentSummary(**summary.dict())
        db.add(db_summary)
        db.commit()
        db.refresh(db_summary)
        return db_summary
    
    def get_summary(self, db: Session, summary_id: int) -> Optional[DocumentSummary]:
        """Получение конспекта по ID"""
        return db.query(DocumentSummary).filter(DocumentSummary.id == summary_id).first()
    
    def get_summaries_by_document(
        self, 
        db: Session, 
        document_id: int
    ) -> List[DocumentSummary]:
        """Получение всех конспектов документа"""
        return db.query(DocumentSummary).filter(
            DocumentSummary.document_id == document_id
        ).order_by(desc(DocumentSummary.created_at)).all()
    
    def delete_summary(self, db: Session, summary_id: int) -> bool:
        """Удаление конспекта"""
        db_summary = self.get_summary(db, summary_id)
        if not db_summary:
            return False
        
        db.delete(db_summary)
        db.commit()
        return True


class QACrud:
    
    def create_session(self, db: Session, session: QASessionCreate) -> QASession:
        """Создание новой сессии Q&A"""
        db_session = QASession(**session.dict())
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session
    
    def get_session(self, db: Session, session_id: int) -> Optional[QASession]:
        """Получение сессии Q&A по ID"""
        return db.query(QASession).filter(QASession.id == session_id).first()
    
    def get_sessions_by_document(
        self, 
        db: Session, 
        document_id: int
    ) -> List[QASession]:
        """Получение всех сессий Q&A для документа"""
        return db.query(QASession).filter(
            QASession.document_id == document_id
        ).order_by(desc(QASession.created_at)).all()
    
    def create_qa_pair(self, db: Session, qa_pair: QAPairCreate) -> QAPair:
        """Создание пары вопрос-ответ"""
        db_qa_pair = QAPair(**qa_pair.dict())
        db.add(db_qa_pair)
        db.commit()
        db.refresh(db_qa_pair)
        return db_qa_pair
    
    def get_qa_pair(self, db: Session, qa_pair_id: int) -> Optional[QAPair]:
        """Получение пары Q&A по ID"""
        return db.query(QAPair).filter(QAPair.id == qa_pair_id).first()
    
    def get_qa_pairs_by_session(
        self, 
        db: Session, 
        session_id: int
    ) -> List[QAPair]:
        """Получение всех пар Q&A в сессии"""
        return db.query(QAPair).filter(
            QAPair.session_id == session_id
        ).order_by(QAPair.created_at).all()
    
    def delete_session(self, db: Session, session_id: int) -> bool:
        """Удаление сессии Q&A (каскадно удалит все пары)"""
        db_session = self.get_session(db, session_id)
        if not db_session:
            return False
        
        db.delete(db_session)
        db.commit()
        return True


class EmbeddingCRUD:
    
    def create_embeddings(
        self, 
        db: Session, 
        document_id: int,
        text_chunks: List[str],
        embedding_model: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[DocumentEmbedding]:
        """Создание эмбеддингов для чанков документа"""
        embeddings = []
        
        for i, chunk in enumerate(text_chunks):
            embedding = DocumentEmbedding(
                document_id=document_id,
                text_chunk=chunk,
                chunk_index=i,
                page_number=page_numbers[i] if page_numbers else None,
                embedding_model=embedding_model
            )
            embeddings.append(embedding)
        
        db.add_all(embeddings)
        db.commit()
        
        for embedding in embeddings:
            db.refresh(embedding)
        
        return embeddings
    
    def get_embeddings_by_document(
        self, 
        db: Session, 
        document_id: int
    ) -> List[DocumentEmbedding]:
        """Получение всех эмбеддингов документа"""
        return db.query(DocumentEmbedding).filter(
            DocumentEmbedding.document_id == document_id
        ).order_by(DocumentEmbedding.chunk_index).all()
    
    def delete_embeddings_by_document(self, db: Session, document_id: int) -> bool:
        """Удаление всех эмбеддингов документа"""
        result = db.query(DocumentEmbedding).filter(
            DocumentEmbedding.document_id == document_id
        ).delete()
        db.commit()
        return result > 0


# Создаем экземпляры CRUD классов
document_crud = DocumentCRUD()
summary_crud = SummaryCRUD()
qa_crud = QACrud()
embedding_crud = EmbeddingCRUD()