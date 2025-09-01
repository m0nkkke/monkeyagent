import torch
import time
import logging
from typing import List, Dict, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ..core.model_manager import model_manager
from ..core.config import settings
from ..schemas.document import SummaryType

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """Сервис для создания конспектов документов с использованием локальной модели"""
    
    def __init__(self):
        self.max_input_length = 512  # максимальная длина входного текста для модели
        self.max_output_length = 256  # максимальная длина выходного текста
        
        # Настройки генерации для разных типов конспектов
        self.generation_configs = {
            SummaryType.BRIEF: {
                "max_length": 128,
                "min_length": 50,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            },
            SummaryType.GENERAL: {
                "max_length": 256,
                "min_length": 100,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            SummaryType.DETAILED: {
                "max_length": 512,
                "min_length": 200,
                "temperature": 0.9,
                "do_sample": True,
                "top_p": 0.95,
                "repetition_penalty": 1.1
            }
        }
    
    def _prepare_text_for_summarization(self, text: str) -> List[str]:
        """Подготовка текста для суммаризации - разбиение на подходящие чанки"""
        # Удаляем лишние пробелы и переносы
        text = ' '.join(text.split())
        
        # Разбиваем на чанки по предложениям, соблюдая лимит токенов
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Примерная оценка длины в токенах (1 токен ≈ 4 символа для русского)
            sentence_token_length = len(sentence) // 3
            
            if current_length + sentence_token_length > self.max_input_length and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                current_chunk = [sentence]
                current_length = sentence_token_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_token_length
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks
    
    def _summarize_chunk(
        self, 
        text_chunk: str, 
        summary_type: SummaryType,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer
    ) -> str:
        """Суммаризация одного чанка текста"""
        
        # Подготавливаем prompt для русской модели
        prompt = f"summarize: {text_chunk}"
        
        # Токенизация
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
            padding=True
        ).to(model_manager.device)
        
        # Получаем конфигурацию генерации
        gen_config = self.generation_configs[summary_type]
        
        # Генерация конспекта
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=gen_config["max_length"],
                min_length=gen_config["min_length"],
                temperature=gen_config["temperature"],
                do_sample=gen_config["do_sample"],
                top_p=gen_config["top_p"],
                repetition_penalty=gen_config["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # избегаем повторений
                early_stopping=True
            )
        
        # Декодирование результата
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary.strip()
    
    def summarize_document(
        self, 
        document_text: str, 
        summary_type: SummaryType = SummaryType.GENERAL
    ) -> Dict[str, any]:
        """
        Создание конспекта документа
        
        Returns:
            Dict с информацией о созданном конспекте
        """
        start_time = time.time()
        
        try:
            # Загружаем модель если не загружена
            model, tokenizer = model_manager.get_summarization_model()
            
            # Подготавливаем текст
            text_chunks = self._prepare_text_for_summarization(document_text)
            logger.info(f"Document split into {len(text_chunks)} chunks for summarization")
            
            # Суммаризируем каждый чанк
            chunk_summaries = []
            total_tokens = 0
            
            for i, chunk in enumerate(text_chunks):
                try:
                    chunk_summary = self._summarize_chunk(chunk, summary_type, model, tokenizer)
                    chunk_summaries.append(chunk_summary)
                    
                    # Подсчитываем токены (приблизительно)
                    tokens_used = len(tokenizer.encode(chunk)) + len(tokenizer.encode(chunk_summary))
                    total_tokens += tokens_used
                    
                    logger.info(f"Summarized chunk {i+1}/{len(text_chunks)}")
                    
                except Exception as e:
                    logger.error(f"Error summarizing chunk {i}: {e}")
                    chunk_summaries.append(f"[Ошибка обработки фрагмента {i+1}]")
            
            # Объединяем конспекты чанков
            if len(chunk_summaries) > 1:
                # Если чанков много, создаем финальный конспект из конспектов чанков
                combined_summaries = "\n\n".join(chunk_summaries)
                
                if len(combined_summaries) > self.max_input_length * 3:  # если слишком длинно
                    # Создаем конспект конспектов
                    final_summary = self._summarize_chunk(
                        combined_summaries, 
                        summary_type, 
                        model, 
                        tokenizer
                    )
                else:
                    final_summary = combined_summaries
            else:
                final_summary = chunk_summaries[0] if chunk_summaries else "Не удалось создать конспект"
            
            generation_time = time.time() - start_time
            
            result = {
                "summary_text": final_summary,
                "model_used": settings.SUMMARIZATION_MODEL,
                "tokens_used": total_tokens,
                "generation_time": generation_time,
                "chunks_processed": len(text_chunks)
            }
            
            logger.info(f"Document summarization completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during document summarization: {e}")
            raise
    
    def get_summary_preview(self, document_text: str, max_length: int = 200) -> str:
        """Быстрый предварительный просмотр документа"""
        # Берем первые несколько предложений
        sentences = document_text.split('.')[:5]
        preview = '. '.join(sentences)
        
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        
        return preview
    
    def estimate_processing_time(self, text_length: int) -> float:
        """Оценка времени обработки документа"""
        # Примерная оценка: 1000 символов ≈ 2-3 секунды
        chunks_count = (text_length // (self.max_input_length * 3)) + 1
        estimated_time = chunks_count * 3.0
        
        return estimated_time


class TextProcessor:
    """Утилиты для обработки текста"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста от мусора"""
        # Удаляем лишние пробелы и переносы
        text = ' '.join(text.split())
        
        # Удаляем повторяющиеся символы
        import re
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_key_sentences(text: str, count: int = 5) -> List[str]:
        """Извлечение ключевых предложений из текста"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Простая эвристика: берем самые длинные предложения
        # (обычно они содержат больше информации)
        sentences_with_length = [(s, len(s)) for s in sentences]
        sentences_with_length.sort(key=lambda x: x[1], reverse=True)
        
        key_sentences = [s[0] for s in sentences_with_length[:count]]
        
        return key_sentences
    
    @staticmethod
    def count_words(text: str) -> Dict[str, int]:
        """Подсчет статистики текста"""
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        return {
            "characters": len(text),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len(paragraphs)
        }


# Создаем глобальные экземпляры
document_summarizer = DocumentSummarizer()
text_processor = TextProcessor()