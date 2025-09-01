import base64
import time
import logging
import io
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from typing import Dict, List, Optional, Tuple

from ..core.config import settings

logger = logging.getLogger(__name__)


class OCRService:
    """Сервис для оптического распознавания символов"""
    
    def __init__(self):
        # Настройка Tesseract
        if settings.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
        
        # Конфигурации OCR для разных типов контента
        self.ocr_configs = {
            "default": "--oem 3 --psm 6",  # автоматическое определение структуры
            "single_line": "--oem 3 --psm 8",  # одна строка текста
            "single_word": "--oem 3 --psm 10",  # одно слово
            "sparse_text": "--oem 3 --psm 11",  # разреженный текст
            "dense_text": "--oem 3 --psm 6"   # плотный текст
        }
    
    def extract_text_from_image(
        self, 
        image_data: str,
        config_type: str = "default",
        preprocess: bool = True
    ) -> Dict[str, any]:
        """
        Извлечение текста из изображения
        
        Args:
            image_data: base64 encoded изображение
            config_type: тип конфигурации OCR
            preprocess: применять ли предобработку изображения
        
        Returns:
            Dict с результатами OCR
        """
        start_time = time.time()
        
        try:
            # Декодируем base64 изображение
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            logger.info(f"Processing image: {image.size}, mode: {image.mode}")
            
            # Предобработка изображения для лучшего OCR
            if preprocess:
                image = self._preprocess_image(image)
            
            # Конфигурация OCR с поддержкой русского языка
            ocr_config = self.ocr_configs.get(config_type, self.ocr_configs["default"])
            
            # Распознавание текста
            extracted_text = pytesseract.image_to_string(
                image, 
                lang='rus+eng',  # русский и английский языки
                config=ocr_config
            )
            
            # Получаем данные о уверенности
            ocr_data = pytesseract.image_to_data(
                image,
                lang='rus+eng',
                config=ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Вычисляем среднюю уверенность
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            
            # Очищаем текст
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            result = {
                "extracted_text": cleaned_text,
                "processing_time": processing_time,
                "confidence_score": avg_confidence / 100.0,  # нормализуем к 0-1
                "word_count": len(cleaned_text.split()),
                "character_count": len(cleaned_text),
                "image_size": image.size,
                "preprocessing_applied": preprocess
            }
            
            logger.info(f"OCR completed in {processing_time:.2f}s, confidence: {avg_confidence:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Предобработка изображения для улучшения OCR"""
        try:
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Увеличиваем контрастность
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Увеличиваем резкость
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Конвертируем в градации серого для лучшего OCR
            image = image.convert('L')
            
            # Увеличиваем размер если изображение маленькое
            if image.width < 300 or image.height < 300:
                scale_factor = max(300 / image.width, 300 / image.height)
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Применяем фильтр для уменьшения шума
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image  # возвращаем оригинал если предобработка не удалась
    
    def _clean_ocr_text(self, text: str) -> str:
        """Очистка текста после OCR"""
        if not text:
            return ""
        
        # Удаляем лишние пробелы и переносы
        text = ' '.join(text.split())
        
        # Убираем артефакты OCR
        import re
        
        # Убираем одиночные символы и цифры
        text = re.sub(r'\b[a-zA-Z0-9]\b', '', text)
        
        # Убираем повторяющиеся знаки препинания
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        
        # Убираем лишние пробелы
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    def extract_text_from_screenshot(
        self, 
        image_data: str,
        auto_detect_text_type: bool = True
    ) -> Dict[str, any]:
        """
        Специализированный метод для обработки скриншотов экрана
        
        Args:
            image_data: base64 encoded скриншот
            auto_detect_text_type: автоматически определять тип текста
        
        Returns:
            Dict с результатами OCR оптимизированными для скриншотов
        """
        try:
            # Пробуем разные конфигурации OCR
            configs_to_try = ["default", "dense_text", "sparse_text"]
            
            best_result = None
            best_score = 0
            
            for config in configs_to_try:
                try:
                    result = self.extract_text_from_image(
                        image_data=image_data,
                        config_type=config,
                        preprocess=True
                    )
                    
                    # Оцениваем качество результата
                    score = self._evaluate_ocr_quality(result)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_result["config_used"] = config
                
                except Exception as e:
                    logger.warning(f"OCR config {config} failed: {e}")
                    continue
            
            if best_result is None:
                return {
                    "extracted_text": "",
                    "processing_time": 0,
                    "confidence_score": 0.0,
                    "error": "Не удалось распознать текст ни с одной конфигурацией"
                }
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            raise
    
    def _evaluate_ocr_quality(self, ocr_result: Dict) -> float:
        """Оценка качества результата OCR"""
        text = ocr_result.get("extracted_text", "")
        confidence = ocr_result.get("confidence_score", 0.0)
        
        if not text:
            return 0.0
        
        # Факторы качества:
        # 1. Уверенность модели
        confidence_score = confidence * 0.4
        
        # 2. Длина текста (больше текста обычно = лучше)
        length_score = min(len(text) / 100.0, 1.0) * 0.3
        
        # 3. Наличие осмысленных слов
        words = text.split()
        meaningful_words = [w for w in words if len(w) > 2]
        word_score = min(len(meaningful_words) / 10.0, 1.0) * 0.3
        
        total_score = confidence_score + length_score + word_score
        
        return total_score
    
    def detect_text_regions(self, image_data: str) -> List[Dict]:
        """Определение областей с текстом на изображении"""
        try:
            # Декодируем изображение
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Получаем данные о расположении слов
            ocr_data = pytesseract.image_to_data(
                image,
                lang='rus+eng',
                config="--oem 3 --psm 6",
                output_type=pytesseract.Output.DICT
            )
            
            # Группируем слова в текстовые регионы
            text_regions = []
            current_region = None
            
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 30:  # только уверенные результаты
                    word_data = {
                        "text": ocr_data['text'][i],
                        "x": ocr_data['left'][i],
                        "y": ocr_data['top'][i],
                        "width": ocr_data['width'][i],
                        "height": ocr_data['height'][i],
                        "confidence": int(ocr_data['conf'][i])
                    }
                    
                    if current_region is None:
                        current_region = {
                            "words": [word_data],
                            "bbox": [word_data["x"], word_data["y"], 
                                    word_data["x"] + word_data["width"],
                                    word_data["y"] + word_data["height"]]
                        }
                    else:
                        # Проверяем, относится ли слово к текущему региону
                        if self._words_are_close(current_region["words"][-1], word_data):
                            current_region["words"].append(word_data)
                            # Обновляем bounding box
                            current_region["bbox"][0] = min(current_region["bbox"][0], word_data["x"])
                            current_region["bbox"][1] = min(current_region["bbox"][1], word_data["y"])
                            current_region["bbox"][2] = max(current_region["bbox"][2], 
                                                           word_data["x"] + word_data["width"])
                            current_region["bbox"][3] = max(current_region["bbox"][3], 
                                                           word_data["y"] + word_data["height"])
                        else:
                            # Начинаем новый регион
                            if len(current_region["words"]) > 0:
                                current_region["text"] = " ".join([w["text"] for w in current_region["words"]])
                                text_regions.append(current_region)
                            
                            current_region = {
                                "words": [word_data],
                                "bbox": [word_data["x"], word_data["y"], 
                                        word_data["x"] + word_data["width"],
                                        word_data["y"] + word_data["height"]]
                            }
            
            # Добавляем последний регион
            if current_region and len(current_region["words"]) > 0:
                current_region["text"] = " ".join([w["text"] for w in current_region["words"]])
                text_regions.append(current_region)
            
            logger.info(f"Detected {len(text_regions)} text regions")
            return text_regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    def _words_are_close(self, word1: Dict, word2: Dict, threshold: int = 50) -> bool:
        """Проверка, находятся ли два слова близко друг к другу"""
        # Вычисляем расстояние между центрами слов
        center1_x = word1["x"] + word1["width"] // 2
        center1_y = word1["y"] + word1["height"] // 2
        center2_x = word2["x"] + word2["width"] // 2
        center2_y = word2["y"] + word2["height"] // 2
        
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        return distance < threshold
    
    def extract_text_from_region(
        self, 
        image_data: str, 
        region_coords: Tuple[int, int, int, int]
    ) -> Dict[str, any]:
        """
        Извлечение текста из конкретной области изображения
        
        Args:
            image_data: base64 encoded изображение
            region_coords: (x, y, width, height) координаты области
        
        Returns:
            Dict с результатами OCR для области
        """
        try:
            # Декодируем изображение
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Вырезаем область
            x, y, width, height = region_coords
            region_image = image.crop((x, y, x + width, y + height))
            
            # Обрабатываем вырезанную область
            region_base64 = self._image_to_base64(region_image)
            
            result = self.extract_text_from_image(
                image_data=region_base64,
                config_type="dense_text",  # для выделенных областей обычно плотный текст
                preprocess=True
            )
            
            result["region_coords"] = region_coords
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from region: {e}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Конвертация PIL Image в base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def batch_process_regions(
        self, 
        image_data: str, 
        regions: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        """Пакетная обработка нескольких областей изображения"""
        results = []
        
        for i, region in enumerate(regions):
            try:
                result = self.extract_text_from_region(image_data, region)
                result["region_index"] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing region {i}: {e}")
                results.append({
                    "region_index": i,
                    "region_coords": region,
                    "extracted_text": "",
                    "error": str(e)
                })
        
        return results
    
    def validate_image_data(self, image_data: str) -> Tuple[bool, Optional[str]]:
        """Валидация base64 изображения"""
        try:
            if not image_data:
                return False, "Отсутствуют данные изображения"
            
            # Проверяем base64 формат
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                return False, "Неверный формат base64"
            
            # Проверяем что это валидное изображение
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.verify()  # проверяем целостность
            except Exception:
                return False, "Поврежденное изображение"
            
            # Проверяем размер
            image = Image.open(io.BytesIO(image_bytes))  # открываем заново после verify
            if image.width * image.height > 10000000:  # ~10MP лимит
                return False, "Изображение слишком большое"
            
            if image.width < 50 or image.height < 50:
                return False, "Изображение слишком маленькое"
            
            return True, None
            
        except Exception as e:
            return False, f"Ошибка валидации: {str(e)}"
    
    def get_ocr_capabilities(self) -> Dict[str, any]:
        """Информация о возможностях OCR"""
        try:
            # Проверяем доступность Tesseract
            version = pytesseract.get_tesseract_version()
            
            # Получаем список поддерживаемых языков
            languages = pytesseract.get_languages()
            
            return {
                "tesseract_version": str(version),
                "supported_languages": languages,
                "russian_supported": "rus" in languages,
                "english_supported": "eng" in languages,
                "available_configs": list(self.ocr_configs.keys()),
                "tesseract_path": pytesseract.pytesseract.tesseract_cmd
            }
            
        except Exception as e:
            logger.error(f"Error getting OCR capabilities: {e}")
            return {
                "error": str(e),
                "tesseract_available": False
            }


class ScreenCapture:
    """Утилиты для работы со скриншотами экрана"""
    
    @staticmethod
    def capture_screen_area(x: int, y: int, width: int, height: int) -> str:
        """
        Захват области экрана и возврат в base64
        
        Args:
            x, y: координаты левого верхнего угла
            width, height: размеры области
        
        Returns:
            str: base64 encoded изображение
        """
        try:
            from PIL import ImageGrab
            
            # Захватываем область экрана
            bbox = (x, y, x + width, y + height)
            screenshot = ImageGrab.grab(bbox)
            
            # Конвертируем в base64
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error capturing screen area: {e}")
            raise
    
    @staticmethod
    def capture_full_screen() -> str:
        """Захват всего экрана"""
        try:
            from PIL import ImageGrab
            
            screenshot = ImageGrab.grab()
            
            # Конвертируем в base64
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error capturing full screen: {e}")
            raise
    
    @staticmethod
    def get_screen_resolution() -> Tuple[int, int]:
        """Получение разрешения экрана"""
        try:
            from PIL import ImageGrab
            
            # Захватываем маленький скриншот для определения размера
            screenshot = ImageGrab.grab()
            return screenshot.size
            
        except Exception as e:
            logger.error(f"Error getting screen resolution: {e}")
            return (1920, 1080)  # значение по умолчанию


# Создаем глобальные экземпляры
ocr_service = OCRService()
screen_capture = ScreenCapture()