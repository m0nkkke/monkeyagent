import torch
import time
import logging
import json
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from ..core.model_manager import model_manager
from ..core.config import settings
from .indexer import vector_search_engine

logger = logging.getLogger(__name__)


class QuestionAnsweringService:
    """Сервис для ответов на вопросы по документам"""
    
    def __init__(self):
        self.max_context_length = 512  # максимальная длина контекста для модели
        self.min_confidence_threshold = 0.1  # минимальный порог уверенности
        
    def answer_question(
        self, 
        question: str, 
        document_ids: List[int],
        context_chunks_limit: int = 5
    ) -> Dict[str, any]:
        """
        Ответ на вопрос на основе документов
        
        Args:
            question: вопрос пользователя
            document_ids: список ID документов для поиска ответа
            context_chunks_limit: максимальное количество чанков контекста
        
        Returns:
            Dict с ответом и метаданными
        """
        start_time = time.time()
        
        try:
            # Находим релевантный контекст
            relevant_chunks = vector_search_engine.find_relevant_context(
                question=question,
                document_ids=document_ids,
                max_chunks=context_chunks_limit
            )
            
            if not relevant_chunks:
                return {
                    "answer": "Не удалось найти релевантную информацию в указанных документах для ответа на вопрос.",
                    "confidence_score": 0.0,
                    "context_snippets": [],
                    "response_time": time.time() - start_time,
                    "model_used": settings.QA_MODEL,
                    "error": "No relevant context found"
                }
            
            # Объединяем контекст
            context_text = self._prepare_context(relevant_chunks)
            
            # Генерируем ответ
            answer_result = self._generate_answer(question, context_text)
            
            # Формируем результат
            result = {
                "answer": answer_result["answer"],
                "confidence_score": answer_result["confidence"],
                "context_snippets": [chunk["text"] for chunk in relevant_chunks],
                "response_time": time.time() - start_time,
                "model_used": settings.QA_MODEL,
                "tokens_used": answer_result.get("tokens_used", 0),
                "relevant_chunks_count": len(relevant_chunks),
                "context_sources": self._format_context_sources(relevant_chunks)
            }
            
            logger.info(f"Question answered in {result['response_time']:.2f}s with confidence {result['confidence_score']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Произошла ошибка при обработке вопроса: {str(e)}",
                "confidence_score": 0.0,
                "context_snippets": [],
                "response_time": time.time() - start_time,
                "model_used": settings.QA_MODEL,
                "error": str(e)
            }
    
    def _prepare_context(self, relevant_chunks: List[Dict]) -> str:
        """Подготовка контекста для модели QA"""
        context_parts = []
        total_length = 0
        
        for chunk in relevant_chunks:
            chunk_text = chunk["text"]
            
            # Проверяем, не превысим ли лимит длины
            if total_length + len(chunk_text) > self.max_context_length * 3:  # примерно в символах
                break
            
            # Добавляем источник для лучшего понимания
            doc_title = chunk.get("document_title", f"Документ {chunk['document_id']}")
            formatted_chunk = f"[{doc_title}]: {chunk_text}"
            
            context_parts.append(formatted_chunk)
            total_length += len(formatted_chunk)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> Dict[str, any]:
        """Генерация ответа с использованием модели QA"""
        try:
            # Загружаем модель
            model, tokenizer = model_manager.get_qa_model()
            
            # Подготавливаем входные данные
            inputs = tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                max_length=self.max_context_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(model_manager.device)
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Находим начало и конец ответа
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                
                # Получаем лучшие позиции для начала и конца
                start_idx = torch.argmax(start_scores)
                end_idx = torch.argmax(end_scores)
                
                # Проверяем валидность позиций
                if end_idx < start_idx:
                    end_idx = start_idx + 10  # берем небольшой фрагмент
                
                # Извлекаем ответ
                input_ids = inputs["input_ids"][0]
                answer_tokens = input_ids[start_idx:end_idx + 1]
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Вычисляем уверенность
                start_prob = torch.softmax(start_scores, dim=-1)[0][start_idx].item()
                end_prob = torch.softmax(end_scores, dim=-1)[0][end_idx].item()
                confidence = (start_prob + end_prob) / 2
                
                # Подсчитываем использованные токены
                tokens_used = len(input_ids) + len(answer_tokens)
            
            # Постобработка ответа
            answer = self._postprocess_answer(answer, question)
            
            return {
                "answer": answer,
                "confidence": float(confidence),
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "Не удалось сгенерировать ответ на вопрос.",
                "confidence": 0.0,
                "tokens_used": 0
            }
    
    def _postprocess_answer(self, answer: str, question: str) -> str:
        """Постобработка ответа"""
        answer = answer.strip()
        
        # Если ответ пустой или слишком короткий
        if not answer or len(answer) < 10:
            return "Не удалось найти конкретный ответ на вопрос в предоставленном контексте."
        
        # Убираем повторения вопроса в ответе
        question_words = set(question.lower().split())
        answer_words = answer.lower().split()
        
        # Если ответ начинается с части вопроса, убираем это
        if len(answer_words) > 3:
            first_words = set(answer_words[:3])
            if len(first_words.intersection(question_words)) >= 2:
                # Ищем первое слово, которого нет в вопросе
                for i, word in enumerate(answer_words):
                    if word.lower() not in question_words:
                        answer = ' '.join(answer_words[i:])
                        break
        
        # Делаем первую букву заглавной
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Добавляем точку в конце если нет
        if answer and not answer[-1] in '.!?':
            answer += '.'
        
        return answer
    
    def _format_context_sources(self, relevant_chunks: List[Dict]) -> List[Dict]:
        """Форматирование источников контекста для JSON хранения"""
        sources = []
        
        for chunk in relevant_chunks:
            source = {
                "document_id": chunk["document_id"],
                "document_title": chunk.get("document_title", ""),
                "chunk_index": chunk["chunk_index"],
                "score": round(chunk["score"], 3),
                "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            }
            sources.append(source)
        
        return sources
    
    def validate_question(self, question: str) -> Tuple[bool, Optional[str]]:
        """Валидация вопроса"""
        question = question.strip()
        
        if not question:
            return False, "Вопрос не может быть пустым"
        
        if len(question) < 3:
            return False, "Вопрос слишком короткий"
        
        if len(question) > 1000:
            return False, "Вопрос слишком длинный (максимум 1000 символов)"
        
        # Проверяем наличие вопросительных слов/знаков
        question_indicators = ["что", "где", "когда", "как", "почему", "зачем", "кто", "какой", "?"]
        has_question_indicator = any(indicator in question.lower() for indicator in question_indicators)
        
        if not has_question_indicator:
            return False, "Текст не похож на вопрос. Используйте вопросительные слова или знак вопроса."
        
        return True, None
    
    def get_answer_suggestions(self, document_ids: List[int]) -> List[str]:
        """Генерация предлагаемых вопросов на основе содержимого документов"""
        try:
            # Получаем несколько случайных чанков из документов
            sample_chunks = []
            for vector_id, metadata in list(self.indexer.metadata.items())[:20]:  # берем первые 20
                if metadata.get("document_id") in document_ids:
                    sample_chunks.append(metadata.get("text", ""))
            
            if not sample_chunks:
                return []
            
            # Простые шаблоны вопросов
            suggestion_templates = [
                "Что говорится о {}?",
                "Как объясняется {}?",
                "Какие примеры {} приводятся?",
                "В чем заключается суть {}?",
                "Какие выводы делаются о {}?"
            ]
            
            # Извлекаем ключевые фразы из чанков (простая эвристика)
            key_phrases = self._extract_key_phrases(sample_chunks[:5])
            
            suggestions = []
            for phrase in key_phrases[:3]:  # берем топ-3 фразы
                for template in suggestion_templates[:2]:  # и топ-2 шаблона
                    suggestion = template.format(phrase)
                    suggestions.append(suggestion)
            
            return suggestions[:5]  # возвращаем максимум 5 предложений
            
        except Exception as e:
            logger.error(f"Error generating answer suggestions: {e}")
            return []
    
    def _extract_key_phrases(self, texts: List[str]) -> List[str]:
        """Простое извлечение ключевых фраз из текстов"""
        # Объединяем все тексты
        combined_text = " ".join(texts).lower()
        
        # Разбиваем на слова и считаем частоту
        words = combined_text.split()
        word_freq = {}
        
        for word in words:
            # Фильтруем короткие слова и стоп-слова
            if len(word) > 3 and word not in ["это", "что", "как", "для", "при", "или", "все", "быть", "мочь"]:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Сортируем по частоте и берем топ слова
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [word[0] for word in top_words]


# Создаем глобальный экземпляр сервиса
qa_service = QuestionAnsweringService()