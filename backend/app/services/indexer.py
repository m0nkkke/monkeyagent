import os
import pickle
import numpy as np
import faiss
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from ..core.model_manager import model_manager
from ..core.config import settings
from .document_loader import document_loader

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Сервис для создания векторных индексов документов с использованием FAISS"""
    
    def __init__(self):
        self.index_path = os.path.join(settings.VECTOR_DB_PATH, settings.VECTOR_INDEX_FILE)
        self.metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.pkl")
        
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[int, Dict] = {}  # метаданные для каждого вектора
        self.dimension: Optional[int] = None
        
        # Загружаем существующий индекс если есть
        self._load_index()
    
    def _load_index(self):
        """Загрузка существующего индекса из файла"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Загружаем FAISS индекс
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                
                # Загружаем метаданные
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors, dimension {self.dimension}")
            else:
                logger.info("No existing index found, will create new one")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = None
            self.metadata = {}
    
    def _save_index(self):
        """Сохранение индекса в файл"""
        try:
            if self.index is not None:
                # Сохраняем FAISS индекс
                faiss.write_index(self.index, self.index_path)
                
                # Сохраняем метаданные
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self.metadata, f)
                
                logger.info(f"Saved index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Создание векторных представлений текстов"""
        try:
            model = model_manager.get_embedding_model()
            
            # Создаем эмбеддинги
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # нормализуем для косинусного расстояния
            )
            
            return embeddings.astype('float32')
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def add_document_to_index(
        self, 
        document_id: int, 
        document_text: str,
        document_title: str = ""
    ) -> List[int]:
        """
        Добавление документа в векторный индекс
        
        Returns:
            List[int]: список ID добавленных векторов
        """
        try:
            # Разбиваем документ на чанки
            text_chunks = document_loader.chunk_text(document_text)
            
            if not text_chunks:
                logger.warning(f"No text chunks found for document {document_id}")
                return []
            
            logger.info(f"Creating embeddings for {len(text_chunks)} chunks of document {document_id}")
            
            # Создаем эмбеддинги
            embeddings = self._create_embeddings(text_chunks)
            
            # Инициализируем индекс если нужно
            if self.index is None:
                self.dimension = embeddings.shape[1]
                # Используем IndexFlatIP для косинусного расстояния
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created new FAISS index with dimension {self.dimension}")
            
            # Получаем начальный ID для новых векторов
            start_vector_id = self.index.ntotal
            
            # Добавляем векторы в индекс
            self.index.add(embeddings)
            
            # Сохраняем метаданные для каждого вектора
            vector_ids = []
            for i, chunk in enumerate(text_chunks):
                vector_id = start_vector_id + i
                vector_ids.append(vector_id)
                
                self.metadata[vector_id] = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk,
                    "document_title": document_title,
                    "chunk_length": len(chunk)
                }
            
            # Сохраняем индекс
            self._save_index()
            
            logger.info(f"Added {len(text_chunks)} vectors to index for document {document_id}")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Error adding document {document_id} to index: {e}")
            raise
    
    def search_similar_chunks(
        self, 
        query_text: str, 
        top_k: int = 5,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Поиск похожих чанков текста
        
        Args:
            query_text: текст запроса
            top_k: количество лучших результатов
            document_ids: список ID документов для поиска (если None - поиск по всем)
        
        Returns:
            List[Dict]: список найденных чанков с метаданными
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized")
            return []
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self._create_embeddings([query_text])
            
            # Поиск в индексе
            scores, indices = self.index.search(query_embedding, top_k * 2)  # берем больше для фильтрации
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS возвращает -1 для отсутствующих результатов
                    continue
                
                vector_metadata = self.metadata.get(idx, {})
                
                # Фильтруем по документам если указаны
                if document_ids and vector_metadata.get("document_id") not in document_ids:
                    continue
                
                result = {
                    "vector_id": int(idx),
                    "score": float(score),
                    "document_id": vector_metadata.get("document_id"),
                    "chunk_index": vector_metadata.get("chunk_index"),
                    "text": vector_metadata.get("text", ""),
                    "document_title": vector_metadata.get("document_title", ""),
                    "chunk_length": vector_metadata.get("chunk_length", 0)
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            # Сортируем по релевантности (score)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            raise
    
    def remove_document_from_index(self, document_id: int) -> bool:
        """
        Удаление документа из индекса
        Note: FAISS не поддерживает удаление векторов напрямую,
        поэтому пересоздаем индекс без удаленного документа
        """
        if self.index is None:
            return False
        
        try:
            # Находим все векторы которые НЕ принадлежат удаляемому документу
            keep_vectors = []
            keep_metadata = {}
            new_vector_id = 0
            
            for vector_id, metadata in self.metadata.items():
                if metadata.get("document_id") != document_id:
                    # Получаем вектор из индекса
                    vector = self.index.reconstruct(vector_id)
                    keep_vectors.append(vector)
                    keep_metadata[new_vector_id] = metadata
                    new_vector_id += 1
            
            if not keep_vectors:
                # Если все векторы удаляются, создаем пустой индекс
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = {}
            else:
                # Пересоздаем индекс с оставшимися векторами
                new_embeddings = np.vstack(keep_vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(new_embeddings)
                self.metadata = keep_metadata
            
            self._save_index()
            
            logger.info(f"Removed document {document_id} from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {document_id} from index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, any]:
        """Получение статистики индекса"""
        if self.index is None:
            return {
                "total_vectors": 0,
                "dimension": 0,
                "documents_count": 0,
                "index_size_mb": 0
            }
        
        # Подсчитываем количество уникальных документов
        unique_docs = set()
        for metadata in self.metadata.values():
            if "document_id" in metadata:
                unique_docs.add(metadata["document_id"])
        
        # Размер индекса
        index_size = 0
        if os.path.exists(self.index_path):
            index_size = os.path.getsize(self.index_path) / (1024 * 1024)  # в MB
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension or 0,
            "documents_count": len(unique_docs),
            "index_size_mb": round(index_size, 2)
        }
    
    def rebuild_index(self) -> bool:
        """Полная перестройка индекса (для оптимизации)"""
        try:
            if not self.metadata:
                logger.info("No metadata found, nothing to rebuild")
                return True
            
            # Группируем метаданные по документам
            documents_data = {}
            for vector_id, metadata in self.metadata.items():
                doc_id = metadata.get("document_id")
                if doc_id not in documents_data:
                    documents_data[doc_id] = []
                documents_data[doc_id].append(metadata)
            
            # Пересоздаем индекс
            all_vectors = []
            new_metadata = {}
            new_vector_id = 0
            
            for doc_id, chunks_metadata in documents_data.items():
                # Сортируем чанки по порядку
                chunks_metadata.sort(key=lambda x: x.get("chunk_index", 0))
                
                for chunk_meta in chunks_metadata:
                    # Получаем вектор
                    old_vector_id = None
                    for vid, meta in self.metadata.items():
                        if meta == chunk_meta:
                            old_vector_id = vid
                            break
                    
                    if old_vector_id is not None:
                        vector = self.index.reconstruct(old_vector_id)
                        all_vectors.append(vector)
                        new_metadata[new_vector_id] = chunk_meta
                        new_vector_id += 1
            
            if all_vectors:
                # Создаем новый индекс
                embeddings_array = np.vstack(all_vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings_array)
                self.metadata = new_metadata
                
                self._save_index()
                
                logger.info("Index rebuilt successfully")
                return True
            else:
                logger.warning("No vectors to rebuild index")
                return False
                
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False


class VectorSearchEngine:
    """Движок для семантического поиска по документам"""
    
    def __init__(self, indexer: DocumentIndexer):
        self.indexer = indexer
    
    def find_relevant_context(
        self, 
        question: str, 
        document_ids: Optional[List[int]] = None,
        max_chunks: int = 5
    ) -> List[Dict]:
        """
        Поиск релевантного контекста для ответа на вопрос
        
        Args:
            question: вопрос пользователя
            document_ids: список документов для поиска
            max_chunks: максимальное количество чанков в контексте
        
        Returns:
            List[Dict]: список релевантных чанков с метаданными
        """
        try:
            # Ищем похожие чанки
            similar_chunks = self.indexer.search_similar_chunks(
                question, 
                top_k=max_chunks * 2,  # берем больше для последующей фильтрации
                document_ids=document_ids
            )
            
            # Фильтруем по релевантности (score > 0.3 обычно означает хорошую релевантность)
            relevant_chunks = [
                chunk for chunk in similar_chunks 
                if chunk["score"] > 0.3
            ][:max_chunks]
            
            # Добавляем дополнительную информацию для контекста
            for chunk in relevant_chunks:
                chunk["relevance_reason"] = self._explain_relevance(question, chunk["text"])
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for question")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error finding relevant context: {e}")
            return []
    
    def _explain_relevance(self, question: str, text: str) -> str:
        """Простое объяснение релевантности (можно улучшить с помощью NLP)"""
        question_words = set(question.lower().split())
        text_words = set(text.lower().split())
        
        common_words = question_words.intersection(text_words)
        
        if len(common_words) > 2:
            return f"Содержит ключевые слова: {', '.join(list(common_words)[:5])}"
        else:
            return "Семантическая близость"
    
    def get_document_summary_context(self, document_id: int, max_chunks: int = 10) -> List[str]:
        """Получение контекста документа для создания конспекта"""
        try:
            # Получаем все чанки документа
            all_chunks = []
            for vector_id, metadata in self.indexer.metadata.items():
                if metadata.get("document_id") == document_id:
                    all_chunks.append({
                        "chunk_index": metadata.get("chunk_index", 0),
                        "text": metadata.get("text", "")
                    })
            
            # Сортируем по порядку чанков
            all_chunks.sort(key=lambda x: x["chunk_index"])
            
            # Если чанков слишком много, берем равномерно распределенные
            if len(all_chunks) > max_chunks:
                step = len(all_chunks) // max_chunks
                selected_chunks = [all_chunks[i] for i in range(0, len(all_chunks), step)][:max_chunks]
            else:
                selected_chunks = all_chunks
            
            return [chunk["text"] for chunk in selected_chunks]
            
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return []
    
    def find_similar_documents(self, document_id: int, top_k: int = 3) -> List[Dict]:
        """Поиск похожих документов"""
        try:
            # Получаем все чанки целевого документа
            target_chunks = []
            for vector_id, metadata in self.indexer.metadata.items():
                if metadata.get("document_id") == document_id:
                    target_chunks.append(metadata.get("text", ""))
            
            if not target_chunks:
                return []
            
            # Объединяем чанки в один текст для поиска
            combined_text = " ".join(target_chunks[:5])  # берем первые 5 чанков
            
            # Ищем похожие чанки из других документов
            similar_chunks = self.indexer.search_similar_chunks(
                combined_text,
                top_k=top_k * 5
            )
            
            # Группируем по документам и исключаем исходный документ
            doc_scores = {}
            for chunk in similar_chunks:
                chunk_doc_id = chunk["document_id"]
                if chunk_doc_id != document_id:
                    if chunk_doc_id not in doc_scores:
                        doc_scores[chunk_doc_id] = {
                            "document_id": chunk_doc_id,
                            "max_score": chunk["score"],
                            "avg_score": chunk["score"],
                            "chunk_count": 1,
                            "title": chunk["document_title"]
                        }
                    else:
                        doc_scores[chunk_doc_id]["max_score"] = max(
                            doc_scores[chunk_doc_id]["max_score"], 
                            chunk["score"]
                        )
                        doc_scores[chunk_doc_id]["avg_score"] = (
                            doc_scores[chunk_doc_id]["avg_score"] * doc_scores[chunk_doc_id]["chunk_count"] + 
                            chunk["score"]
                        ) / (doc_scores[chunk_doc_id]["chunk_count"] + 1)
                        doc_scores[chunk_doc_id]["chunk_count"] += 1
            
            # Сортируем по средней релевантности
            similar_docs = list(doc_scores.values())
            similar_docs.sort(key=lambda x: x["avg_score"], reverse=True)
            
            return similar_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []


# Создаем глобальные экземпляры
document_indexer = DocumentIndexer()
vector_search_engine = VectorSearchEngine(document_indexer)