#!/usr/bin/env python3
"""
Document AI Assistant - Desktop GUI Client
"""

import sys
import os
import json
import base64
import requests
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QListWidget, QListWidgetItem,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QGroupBox, QLineEdit, QComboBox, QScrollArea, QFrame,
    QSystemTrayIcon, QMenu, QDialog, QGridLayout, QSpinBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QSize, QBuffer
)
from PyQt6.QtGui import (
    QFont, QIcon, QPixmap, QAction, QShortcut, QKeySequence,
    QPainter, QPen, QColor, QCursor
)


class APIClient:
    """Клиент для взаимодействия с FastAPI backend"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    def check_health(self) -> bool:
        """Проверка доступности API"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def upload_document(self, file_path: str) -> Optional[Dict]:
        """Загрузка документа"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f)}
                response = self.session.post(
                    f"{self.base_url}/api/v1/documents/upload",
                    files=files
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def get_documents(self) -> List[Dict]:
        """Получение списка документов"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/documents/")
            if response.status_code == 200:
                return response.json().get("documents", [])
            return []
        except:
            return []
    
    def get_document(self, document_id: int) -> Optional[Dict]:
        """Получение документа по ID"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/documents/{document_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def summarize_document(self, document_id: int, summary_type: str = "general") -> Optional[Dict]:
        """Создание конспекта"""
        try:
            payload = {
                "document_id": document_id,
                "summary_type": summary_type
            }
            response = self.session.post(
                f"{self.base_url}/api/v1/documents/{document_id}/summarize",
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def ask_question(self, document_ids: List[int], question: str) -> Optional[Dict]:
        """Задать вопрос по документам"""
        try:
            payload = {
                "document_ids": document_ids,
                "question": question
            }
            response = self.session.post(
                f"{self.base_url}/api/v1/documents/question",
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def ocr_image(self, image_data: str) -> Optional[Dict]:
        """OCR изображения"""
        try:
            payload = {"image_data": image_data}
            response = self.session.post(
                f"{self.base_url}/api/v1/documents/ocr",
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}


class ScreenCaptureWidget(QWidget):
    """Виджет для выделения области экрана"""
    
    captured = pyqtSignal(QRect)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setMouseTracking(True)
        
        # Получаем скриншот экрана
        screen = QApplication.primaryScreen()
        self.screenshot = screen.grabWindow(0)
        
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing = False
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Рисуем затемненный скриншот
        painter.drawPixmap(self.rect(), self.screenshot, self.screenshot.rect())
        
        # Затемняем весь экран
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        # Если выделяем область, показываем ее без затемнения
        if self.drawing and not self.begin.isNull() and not self.end.isNull():
            selection_rect = QRect(self.begin, self.end).normalized()
            painter.drawPixmap(selection_rect, self.screenshot, selection_rect)
            
            # Рисуем рамку выделения
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(selection_rect)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin = event.position().toPoint()
            self.end = self.begin
            self.drawing = True
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end = event.position().toPoint()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            
            selection_rect = QRect(self.begin, self.end).normalized()
            
            # Минимальный размер выделения
            if selection_rect.width() > 10 and selection_rect.height() > 10:
                self.captured.emit(selection_rect)
            
            self.close()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


class WorkerThread(QThread):
    """Поток для выполнения длительных операций"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.operation(*self.args, **self.kwargs)
            self.finished.emit(result or {})
        except Exception as e:
            self.error.emit(str(e))


class DocumentWidget(QFrame):
    """Виджет для отображения документа"""
    
    summarize_requested = pyqtSignal(int, str)
    question_requested = pyqtSignal(list, str)
    
    def __init__(self, document_data: Dict):
        super().__init__()
        self.document_data = document_data
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Заголовок документа
        title_label = QLabel(self.document_data.get("original_filename", "Неизвестный документ"))
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Информация о документе
        info_text = f"""
        Размер: {self.document_data.get('file_size', 0) // 1024} KB
        Страниц: {self.document_data.get('page_count', 'N/A')}
        Статус: {self.document_data.get('processing_status', 'unknown')}
        """
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info_label)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        # Кнопка конспектирования
        summarize_btn = QPushButton("📝 Законспектировать")
        summarize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        summarize_btn.clicked.connect(self.on_summarize_clicked)
        buttons_layout.addWidget(summarize_btn)
        
        # Кнопка для вопросов
        question_btn = QPushButton("❓ Задать вопрос")
        question_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        question_btn.clicked.connect(self.on_question_clicked)
        buttons_layout.addWidget(question_btn)
        
        layout.addLayout(buttons_layout)
    
    def on_summarize_clicked(self):
        """Обработчик кнопки конспектирования"""
        document_id = self.document_data.get("id")
        if document_id:
            self.summarize_requested.emit(document_id, "general")
    
    def on_question_clicked(self):
        """Обработчик кнопки вопросов"""
        document_id = self.document_data.get("id")
        if document_id:
            self.question_requested.emit([document_id], "")


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.api_client = APIClient()
        self.current_documents = []
        self.capture_widget = None
        
        self.setup_ui()
        self.setup_shortcuts()
        self.setup_system_tray()
        
        # Таймер для проверки статуса документов
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.refresh_documents)
        self.status_timer.start(5000)  # каждые 5 секунд
        
        self.refresh_documents()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("Document AI Assistant")
        self.setGeometry(100, 100, 1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QHBoxLayout(central_widget)
        
        # Создаем разделитель
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Левая панель - управление документами
        self.setup_left_panel(splitter)
        
        # Правая панель - рабочая область
        self.setup_right_panel(splitter)
        
        # Настройка пропорций
        splitter.setSizes([400, 800])
    
    def setup_left_panel(self, parent):
        """Настройка левой панели"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Заголовок
        title_label = QLabel("📚 Документы")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left_layout.addWidget(title_label)
        
        # Кнопка загрузки
        upload_btn = QPushButton("📁 Загрузить документ")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        upload_btn.clicked.connect(self.upload_document)
        left_layout.addWidget(upload_btn)
        
        # Список документов
        self.documents_list = QListWidget()
        self.documents_list.itemClicked.connect(self.on_document_selected)
        left_layout.addWidget(self.documents_list)
        
        # Кнопка обновления
        refresh_btn = QPushButton("🔄 Обновить")
        refresh_btn.clicked.connect(self.refresh_documents)
        left_layout.addWidget(refresh_btn)
        
        # Статистика
        self.stats_label = QLabel("Статистика загружается...")
        self.stats_label.setStyleSheet("color: #666; font-size: 10px;")
        left_layout.addWidget(self.stats_label)
        
        parent.addWidget(left_panel)
    
    def setup_right_panel(self, parent):
        """Настройка правой панели"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Вкладки
        self.tab_widget = QTabWidget()
        right_layout.addWidget(self.tab_widget)
        
        # Вкладка конспектов
        self.setup_summary_tab()
        
        # Вкладка Q&A
        self.setup_qa_tab()
        
        # Вкладка OCR
        self.setup_ocr_tab()
        
        parent.addWidget(right_panel)
    
    def setup_summary_tab(self):
        """Настройка вкладки конспектов"""
        summary_tab = QWidget()
        layout = QVBoxLayout(summary_tab)
        
        # Заголовок
        header = QLabel("📝 Конспекты документов")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Контролы для конспектирования
        controls_group = QGroupBox("Настройки конспектирования")
        controls_layout = QGridLayout(controls_group)
        
        controls_layout.addWidget(QLabel("Тип конспекта:"), 0, 0)
        self.summary_type_combo = QComboBox()
        self.summary_type_combo.addItems(["general", "brief", "detailed"])
        controls_layout.addWidget(self.summary_type_combo, 0, 1)
        
        # Кнопка "Законспектировать в один клик"
        self.one_click_summary_btn = QPushButton("✨ Законспектировать в один клик")
        self.one_click_summary_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E55A2B;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.one_click_summary_btn.clicked.connect(self.one_click_summarize)
        self.one_click_summary_btn.setEnabled(False)
        controls_layout.addWidget(self.one_click_summary_btn, 1, 0, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Область для отображения конспектов
        self.summary_display = QTextEdit()
        self.summary_display.setPlaceholderText("Конспекты будут отображаться здесь...")
        self.summary_display.setReadOnly(True)
        layout.addWidget(self.summary_display)
        
        # Прогресс бар
        self.summary_progress = QProgressBar()
        self.summary_progress.setVisible(False)
        layout.addWidget(self.summary_progress)
        
        self.tab_widget.addTab(summary_tab, "📝 Конспекты")
    
    def setup_qa_tab(self):
        """Настройка вкладки Q&A"""
        qa_tab = QWidget()
        layout = QVBoxLayout(qa_tab)
        
        # Заголовок
        header = QLabel("❓ Вопросы и ответы")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Область для истории Q&A
        self.qa_history = QTextEdit()
        self.qa_history.setPlaceholderText("История вопросов и ответов...")
        self.qa_history.setReadOnly(True)
        layout.addWidget(self.qa_history)
        
        # Ввод вопроса
        question_group = QGroupBox("Задать вопрос")
        question_layout = QVBoxLayout(question_group)
        
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Введите ваш вопрос...")
        self.question_input.returnPressed.connect(self.ask_question)
        question_layout.addWidget(self.question_input)
        
        question_buttons = QHBoxLayout()
        
        ask_btn = QPushButton("🔍 Найти ответ")
        ask_btn.clicked.connect(self.ask_question)
        ask_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        question_buttons.addWidget(ask_btn)
        
        clear_btn = QPushButton("🗑️ Очистить")
        clear_btn.clicked.connect(lambda: self.qa_history.clear())
        question_buttons.addWidget(clear_btn)
        
        question_layout.addLayout(question_buttons)
        layout.addWidget(question_group)
        
        # Прогресс бар для Q&A
        self.qa_progress = QProgressBar()
        self.qa_progress.setVisible(False)
        layout.addWidget(self.qa_progress)
        
        self.tab_widget.addTab(qa_tab, "❓ Q&A")
    
    def setup_ocr_tab(self):
        """Настройка вкладки OCR"""
        ocr_tab = QWidget()
        layout = QVBoxLayout(ocr_tab)
        
        # Заголовок
        header = QLabel("👁️ Распознавание текста с экрана")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Инструкции
        instructions = QLabel("""
        <b>Горячие клавиши:</b><br>
        • <b>Ctrl+Shift+S</b> - Выделить область экрана<br>
        • <b>Ctrl+Shift+Q</b> - Задать вопрос по выделенному тексту<br><br>
        
        <i>Выделите область с текстом на экране, и AI найдет ответ в ваших документах!</i>
        """)
        instructions.setStyleSheet("color: #666; padding: 10px; background-color: #f9f9f9; border-radius: 4px;")
        layout.addWidget(instructions)
        
        # Кнопка выделения области
        capture_btn = QPushButton("📷 Выделить область экрана")
        capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        capture_btn.clicked.connect(self.capture_screen_area)
        layout.addWidget(capture_btn)
        
        # Область для отображения распознанного текста
        ocr_group = QGroupBox("Распознанный текст")
        ocr_group_layout = QVBoxLayout(ocr_group)
        
        self.ocr_result = QTextEdit()
        self.ocr_result.setPlaceholderText("Распознанный текст будет отображаться здесь...")
        ocr_group_layout.addWidget(self.ocr_result)
        
        # Кнопка для поиска ответа по OCR тексту
        ocr_question_btn = QPushButton("🔍 Найти ответ по этому тексту")
        ocr_question_btn.clicked.connect(self.ask_question_from_ocr)
        ocr_question_btn.setEnabled(False)
        ocr_group_layout.addWidget(ocr_question_btn)
        self.ocr_question_btn = ocr_question_btn
        
        layout.addWidget(ocr_group)
        
        # Прогресс бар для OCR
        self.ocr_progress = QProgressBar()
        self.ocr_progress.setVisible(False)
        layout.addWidget(self.ocr_progress)
        
        self.tab_widget.addTab(ocr_tab, "👁️ OCR")
    
    def setup_shortcuts(self):
        """Настройка горячих клавиш"""
        # Ctrl+Shift+S - выделение области экрана
        capture_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        capture_shortcut.activated.connect(self.capture_screen_area)
        
        # Ctrl+Shift+Q - вопрос по выделенному тексту
        question_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        question_shortcut.activated.connect(self.capture_and_question)
    
    def setup_system_tray(self):
        """Настройка системного трея"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            
            # Иконка (можно заменить на кастомную)
            self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
            
            # Меню трея
            tray_menu = QMenu()
            
            show_action = QAction("Показать", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)
            
            quit_action = QAction("Выход", self)
            quit_action.triggered.connect(QApplication.instance().quit)
            tray_menu.addAction(quit_action)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
            
            # Двойной клик для показа окна
            self.tray_icon.activated.connect(self.tray_icon_activated)
    
    def tray_icon_activated(self, reason):
        """Обработчик клика по иконке в трее"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()
    
    def upload_document(self):
        """Загрузка документа"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите документ",
            "",
            "Документы (*.pdf *.pptx *.docx *.txt);;Все файлы (*)"
        )
        
        if file_path:
            self.upload_document_async(file_path)
    
    def upload_document_async(self, file_path: str):
        """Асинхронная загрузка документа"""
        self.show_status("Загрузка документа...")
        
        def upload_operation():
            return self.api_client.upload_document(file_path)
        
        self.worker = WorkerThread(upload_operation)
        self.worker.finished.connect(self.on_upload_finished)
        self.worker.error.connect(self.on_operation_error)
        self.worker.start()
    
    def on_upload_finished(self, result: Dict):
        """Обработчик завершения загрузки"""
        if "error" in result:
            self.show_error(f"Ошибка загрузки: {result['error']}")
        else:
            self.show_success(f"Документ загружен успешно! ID: {result.get('document_id')}")
            self.refresh_documents()
    
    def refresh_documents(self):
        """Обновление списка документов"""
        def get_documents():
            return self.api_client.get_documents()
        
        self.worker = WorkerThread(get_documents)
        self.worker.finished.connect(self.on_documents_loaded)
        self.worker.start()
    
    def on_documents_loaded(self, documents: List[Dict]):
        """Обработчик загрузки списка документов"""
        self.current_documents = documents if isinstance(documents, list) else []
        self.update_documents_list()
        self.update_stats()
    
    def update_documents_list(self):
        """Обновление отображения списка документов"""
        self.documents_list.clear()
        
        for doc in self.current_documents:
            item_text = f"📄 {doc.get('original_filename', 'Unknown')}"
            status = doc.get('processing_status', 'unknown')
            
            if status == 'completed':
                item_text += " ✅"
            elif status == 'processing':
                item_text += " ⏳"
            elif status == 'failed':
                item_text += " ❌"
            else:
                item_text += " ⏸️"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, doc)
            self.documents_list.addItem(item)
        
        # Обновляем доступность кнопки "один клик"
        completed_docs = [d for d in self.current_documents if d.get('processing_status') == 'completed']
        self.one_click_summary_btn.setEnabled(len(completed_docs) > 0)
    
    def update_stats(self):
        """Обновление статистики"""
        total = len(self.current_documents)
        completed = len([d for d in self.current_documents if d.get('processing_status') == 'completed'])
        processing = len([d for d in self.current_documents if d.get('processing_status') == 'processing'])
        
        stats_text = f"Всего: {total} | Обработано: {completed} | Обрабатывается: {processing}"
        self.stats_label.setText(stats_text)
    
    def on_document_selected(self, item: QListWidgetItem):
        """Обработчик выбора документа"""
        document = item.data(Qt.ItemDataRole.UserRole)
        if document:
            self.show_document_info(document)
    
    def show_document_info(self, document: Dict):
        """Отображение информации о документе"""
        info_text = f"""
        <h3>📄 {document.get('original_filename', 'Неизвестный документ')}</h3>
        
        <b>Информация:</b><br>
        • ID: {document.get('id')}<br>
        • Размер: {document.get('file_size', 0) // 1024} KB<br>
        • Тип: {document.get('file_type', 'unknown').upper()}<br>
        • Страниц: {document.get('page_count', 'N/A')}<br>
        • Статус: {document.get('processing_status', 'unknown')}<br>
        • Загружен: {document.get('created_at', 'N/A')[:19]}<br>
        """
        
        if document.get('processing_status') == 'failed':
            info_text += f"<br><b style='color: red;'>Ошибка:</b> {document.get('error_message', 'Неизвестная ошибка')}"
        
        # Показываем информацию в области конспектов (временно)
        self.summary_display.setHtml(info_text)
    
    def one_click_summarize(self):
        """Конспектирование в один клик для всех обработанных документов"""
        completed_docs = [d for d in self.current_documents if d.get('processing_status') == 'completed']
        
        if not completed_docs:
            self.show_error("Нет обработанных документов для конспектирования")
            return
        
        # Берем последний загруженный документ
        latest_doc = max(completed_docs, key=lambda x: x.get('created_at', ''))
        
        self.summarize_document_async(latest_doc['id'], self.summary_type_combo.currentText())
    
    def summarize_document_async(self, document_id: int, summary_type: str):
        """Асинхронное создание конспекта"""
        self.show_progress("Создание конспекта...", self.summary_progress)
        
        def summarize_operation():
            return self.api_client.summarize_document(document_id, summary_type)
        
        self.worker = WorkerThread(summarize_operation)
        self.worker.finished.connect(self.on_summarize_finished)
        self.worker.error.connect(self.on_operation_error)
        self.worker.start()
    
    def on_summarize_finished(self, result: Dict):
        """Обработчик завершения конспектирования"""
        self.hide_progress(self.summary_progress)
        
        if "error" in result:
            self.show_error(f"Ошибка конспектирования: {result['error']}")
        else:
            summary_data = result.get('summary', {})
            summary_text = summary_data.get('summary_text', 'Конспект не найден')
            
            # Форматируем и отображаем конспект
            summary_text_html = summary_text.replace("\n", "<br>")

            formatted_summary = f"""
                <h3>📝 Конспект документа</h3>
                <p><b>Тип:</b> {summary_data.get('summary_type', 'general')}</p>
                <p><b>Модель:</b> {summary_data.get('model_used', 'unknown')}</p>
                <p><b>Время генерации:</b> {summary_data.get('generation_time', 0):.2f} сек</p>
                <hr>
                <div style="line-height: 1.6; font-size: 14px;">
                {summary_text_html}
                </div> """

            self.summary_display.setHtml(formatted_summary)
            self.show_success("Конспект создан успешно!")
    
    def ask_question(self):
        """Задать вопрос по документам"""
        question = self.question_input.text().strip()
        if not question:
            self.show_error("Введите вопрос")
            return
        
        # Получаем ID обработанных документов
        completed_docs = [d['id'] for d in self.current_documents if d.get('processing_status') == 'completed']
        
        if not completed_docs:
            self.show_error("Нет обработанных документов для поиска ответа")
            return
        
        self.ask_question_async(completed_docs, question)
    
    def ask_question_async(self, document_ids: List[int], question: str):
        """Асинхронный поиск ответа"""
        self.show_progress("Поиск ответа...", self.qa_progress)
        
        def question_operation():
            return self.api_client.ask_question(document_ids, question)
        
        self.worker = WorkerThread(question_operation)
        self.worker.finished.connect(lambda result: self.on_question_answered(result, question))
        self.worker.error.connect(self.on_operation_error)
        self.worker.start()
    
    def on_question_answered(self, result: Dict, question: str):
        """Обработчик получения ответа"""
        self.hide_progress(self.qa_progress)
        
        if "error" in result:
            self.show_error(f"Ошибка поиска ответа: {result['error']}")
        else:
            answer = result.get('answer', 'Ответ не найден')
            confidence = result.get('confidence_score', 0)
            
            # Добавляем Q&A в историю
            qa_entry = f"""
            <div style="border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px;">
                <p><b>❓ Вопрос:</b> {question}</p>
                <p><b>✅ Ответ:</b> {answer}</p>
                <p style="color: #666; font-size: 12px;">
                    Уверенность: {confidence:.2f} | Время: {result.get('response_time', 0):.2f}с
                </p>
            </div>
            """
            
            current_html = self.qa_history.toHtml()
            self.qa_history.setHtml(current_html + qa_entry)
            
            # Очищаем поле ввода
            self.question_input.clear()
            
            self.show_success("Ответ найден!")
    
    def capture_screen_area(self):
        """Захват области экрана"""
        self.hide()  # скрываем главное окно
        
        # Небольшая задержка для скрытия окна
        QTimer.singleShot(500, self.start_screen_capture)
    
    def start_screen_capture(self):
        """Запуск виджета захвата экрана"""
        self.capture_widget = ScreenCaptureWidget()
        self.capture_widget.captured.connect(self.on_screen_captured)
        self.capture_widget.show()
    
    def on_screen_captured(self, rect: QRect):
        """Обработчик захвата области экрана"""
        self.show()  # показываем главное окно обратно
        
        try:
            # Захватываем выделенную область
            screen = QApplication.primaryScreen()
            pixmap = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
            
            # Конвертируем в base64
            byte_array = bytearray()
            buffer = QBuffer(byte_array)
            buffer.open(QBuffer.OpenModeFlag.WriteOnly)
            pixmap.save(buffer, "PNG")
            image_base64 = base64.b64encode(byte_array).decode('utf-8')
            
            # Отправляем на OCR
            self.process_ocr_async(image_base64)
            
        except Exception as e:
            self.show_error(f"Ошибка захвата экрана: {e}")
    
    def capture_and_question(self):
        """Захват области и сразу задать вопрос"""
        # Сначала захватываем область
        self.capture_screen_area()
        # После OCR автоматически зададим вопрос
        self._auto_question_after_ocr = True
    
    def process_ocr_async(self, image_data: str):
        """Асинхронная обработка OCR"""
        self.show_progress("Распознавание текста...", self.ocr_progress)
        self.tab_widget.setCurrentIndex(2)  # переключаемся на вкладку OCR
        
        def ocr_operation():
            return self.api_client.ocr_image(image_data)
        
        self.worker = WorkerThread(ocr_operation)
        self.worker.finished.connect(self.on_ocr_finished)
        self.worker.error.connect(self.on_operation_error)
        self.worker.start()
    
    def on_ocr_finished(self, result: Dict):
        """Обработчик завершения OCR"""
        self.hide_progress(self.ocr_progress)
        
        if "error" in result:
            self.show_error(f"Ошибка OCR: {result['error']}")
        else:
            extracted_text = result.get('extracted_text', '')
            confidence = result.get('confidence_score', 0)
            
            if extracted_text.strip():
                # Отображаем распознанный текст
                extracted_text_html = extracted_text.replace("\n", "<br>")

                ocr_display = f"""
                    <h4>👁️ Распознанный текст</h4>
                    <p><b>Уверенность:</b> {confidence:.2f}</p>
                    <p><b>Время обработки:</b> {result.get('processing_time', 0):.2f}с</p>
                    <hr>
                    <div style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
                    {extracted_text_html}
                    </div>
                    """

                
                self.ocr_result.setHtml(ocr_display)
                self.ocr_question_btn.setEnabled(True)
                
                # Если это автоматический вопрос после захвата
                if hasattr(self, '_auto_question_after_ocr') and self._auto_question_after_ocr:
                    self._auto_question_after_ocr = False
                    self.ask_question_from_ocr()
                
                self.show_success("Текст распознан успешно!")
            else:
                self.ocr_result.setPlainText("Текст не распознан или область не содержит текста.")
                self.ocr_question_btn.setEnabled(False)
                self.show_error("Не удалось распознать текст")
    
    def ask_question_from_ocr(self):
        """Задать вопрос на основе OCR текста"""
        ocr_text = self.ocr_result.toPlainText()
        
        # Извлекаем только распознанный текст (убираем HTML разметку)
        import re
        text_match = re.search(r'<div[^>]*>(.*?)</div>', ocr_text, re.DOTALL)
        if text_match:
            clean_text = text_match.group(1).replace('<br>', ' ').strip()
        else:
            clean_text = ocr_text.strip()
        
        if not clean_text:
            self.show_error("Нет распознанного текста для поиска ответа")
            return
        
        # Получаем ID обработанных документов
        completed_docs = [d['id'] for d in self.current_documents if d.get('processing_status') == 'completed']
        
        if not completed_docs:
            self.show_error("Нет обработанных документов для поиска ответа")
            return
        
        # Переключаемся на вкладку Q&A
        self.tab_widget.setCurrentIndex(1)
        
        # Задаем вопрос
        self.ask_question_async(completed_docs, clean_text)
    
    def show_progress(self, message: str, progress_bar: QProgressBar):
        """Показать прогресс операции"""
        progress_bar.setVisible(True)
        progress_bar.setRange(0, 0)  # индикатор неопределенного прогресса
        self.statusBar().showMessage(message)
    
    def hide_progress(self, progress_bar: QProgressBar):
        """Скрыть прогресс операции"""
        progress_bar.setVisible(False)
        self.statusBar().clearMessage()
    
    def show_status(self, message: str):
        """Показать статусное сообщение"""
        self.statusBar().showMessage(message, 3000)
    
    def show_success(self, message: str):
        """Показать сообщение об успехе"""
        self.statusBar().showMessage(f"✅ {message}", 5000)
    
    def show_error(self, message: str):
        """Показать сообщение об ошибке"""
        self.statusBar().showMessage(f"❌ {message}", 10000)
        
        # Также показываем в диалоге для важных ошибок
        if "ошибка" in message.lower() or "error" in message.lower():
            QMessageBox.warning(self, "Ошибка", message)
    
    def on_operation_error(self, error_message: str):
        """Обработчик ошибок операций"""
        self.hide_progress(self.summary_progress)
        self.hide_progress(self.qa_progress)
        self.hide_progress(self.ocr_progress)
        self.show_error(f"Ошибка операции: {error_message}")
    
    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        if hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
            # Сворачиваем в трей вместо закрытия
            self.hide()
            self.tray_icon.showMessage(
                "Document AI Assistant",
                "Приложение свернуто в системный трей",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
            event.ignore()
        else:
            event.accept()


class SettingsDialog(QDialog):
    """Диалог настроек приложения"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setFixedSize(400, 300)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # API настройки
        api_group = QGroupBox("API сервер")
        api_layout = QGridLayout(api_group)
        
        api_layout.addWidget(QLabel("Адрес сервера:"), 0, 0)
        self.server_input = QLineEdit("http://127.0.0.1:8000")
        api_layout.addWidget(self.server_input, 0, 1)
        
        api_layout.addWidget(QLabel("Таймаут (сек):"), 1, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        self.timeout_spin.setValue(30)
        api_layout.addWidget(self.timeout_spin, 1, 1)
        
        layout.addWidget(api_group)
        
        # OCR настройки
        ocr_group = QGroupBox("OCR настройки")
        ocr_layout = QGridLayout(ocr_group)
        
        ocr_layout.addWidget(QLabel("Предобработка:"), 0, 0)
        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItems(["Автоматически", "Включена", "Отключена"])
        ocr_layout.addWidget(self.preprocess_combo, 0, 1)
        
        layout.addWidget(ocr_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)


def check_server_connection():
    """Проверка подключения к серверу"""
    client = APIClient()
    return client.check_health()


def main():
    """Главная функция приложения"""
    app = QApplication(sys.argv)
    app.setApplicationName("Document AI Assistant")
    app.setApplicationVersion("0.1.0")
    
    # Проверяем подключение к серверу
    if not check_server_connection():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Ошибка подключения")
        msg.setText("Не удается подключиться к серверу API")
        msg.setInformativeText(
            "Убедитесь что backend сервер запущен:\n"
            "python start_server.py\n\n"
            "Или запустите его из директории backend/"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Close)
        
        if msg.exec() == QMessageBox.StandardButton.Retry:
            if not check_server_connection():
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()
    
    # Показываем приветственное сообщение
    welcome_msg = QMessageBox()
    welcome_msg.setIcon(QMessageBox.Icon.Information)
    welcome_msg.setWindowTitle("Добро пожаловать!")
    welcome_msg.setText("Document AI Assistant готов к работе!")
    welcome_msg.setInformativeText(
        "Горячие клавиши:\n"
        "• Ctrl+Shift+S - Выделить область экрана\n"
        "• Ctrl+Shift+Q - Выделить область и найти ответ\n\n"
        "Загрузите документы и начните работу!"
    )
    welcome_msg.exec()
    
    return app.exec()


if __name__ == "__main__":
    exit(main())