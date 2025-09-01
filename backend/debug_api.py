#!/usr/bin/env python3
"""
Скрипт для отладки API и проверки состояния базы данных
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from app.core.db import SessionLocal, engine
from app.models.document import Document
from app.crud.document import document_crud
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_database():
    """Проверка состояния базы данных"""
    print("🔍 Проверка базы данных...")
    
    try:
        # Проверяем соединение
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Подключение к БД работает")
        
        # Проверяем таблицы
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"📋 Найдено таблиц: {len(tables)}")
        for table in tables:
            print(f"   - {table}")
        
        # Проверяем документы
        db = SessionLocal()
        try:
            documents = db.query(Document).all()
            print(f"📄 Документов в БД: {len(documents)}")
            
            for doc in documents:
                print(f"   - ID: {doc.id}, Файл: {doc.original_filename}, Статус: {doc.processing_status}")
        finally:
            db.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка БД: {e}")
        return False


def check_api_endpoints():
    """Проверка API эндпоинтов"""
    print("\n🌐 Проверка API эндпоинтов...")
    
    import requests
    
    base_url = "http://127.0.0.1:8000"
    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/api/v1/documents/", "GET"),
        ("/models/info", "GET")
    ]
    
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {method} {endpoint} - OK")
                if endpoint == "/api/v1/documents/":
                    data = response.json()
                    print(f"   📊 Найдено документов: {data.get('total', 0)}")
            else:
                print(f"❌ {method} {endpoint} - {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {method} {endpoint} - Сервер не доступен")
        except Exception as e:
            print(f"❌ {method} {endpoint} - {e}")


def fix_database_sessions():
    """Исправление проблем с сессиями БД"""
    print("\n🔧 Исправление проблем с сессиями...")
    
    try:
        # Закрываем все активные соединения
        engine.dispose()
        print("✅ Активные соединения закрыты")
        
        # Пересоздаем таблицы если нужно
        from app.core.db import Base
        Base.metadata.create_all(bind=engine)
        print("✅ Таблицы проверены/созданы")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка исправления: {e}")
        return False


def test_crud_operations():
    """Тестирование CRUD операций"""
    print("\n🧪 Тестирование CRUD операций...")
    
    db = SessionLocal()
    try:
        # Тест получения документов
        documents = document_crud.get_documents(db, skip=0, limit=10)
        print(f"✅ Получено документов: {len(documents)}")
        
        # Тест подсчета
        count = document_crud.get_documents_count(db)
        print(f"✅ Общее количество: {count}")
        
        # Тест статистики
        stats = document_crud.get_documents_stats(db)
        print(f"✅ Статистика: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка CRUD: {e}")
        return False
    finally:
        db.close()


def create_test_document():
    """Создание тестового документа"""
    print("\n📝 Создание тестового документа...")
    
    db = SessionLocal()
    try:
        from app.schemas.document import DocumentCreate
        
        test_doc = DocumentCreate(
            filename="test_doc.txt",
            original_filename="Тестовый документ.txt", 
            file_path="./test_doc.txt",
            file_type="txt",
            file_size=1024
        )
        
        # Создаем тестовый файл
        with open("test_doc.txt", "w", encoding="utf-8") as f:
            f.write("Это тестовый документ для проверки работы системы.")
        
        doc = document_crud.create_document(db, test_doc)
        print(f"✅ Тестовый документ создан с ID: {doc.id}")
        
        return doc.id
        
    except Exception as e:
        print(f"❌ Ошибка создания тестового документа: {e}")
        return None
    finally:
        db.close()


def cleanup_test_documents():
    """Очистка тестовых документов"""
    print("\n🧹 Очистка тестовых документов...")
    
    db = SessionLocal()
    try:
        # Удаляем тестовые документы
        test_docs = db.query(Document).filter(Document.filename.like("test_%")).all()
        
        for doc in test_docs:
            # Удаляем файл если существует
            if os.path.exists(doc.file_path):
                os.remove(doc.file_path)
            
            db.delete(doc)
        
        db.commit()
        print(f"✅ Удалено тестовых документов: {len(test_docs)}")
        
    except Exception as e:
        print(f"❌ Ошибка очистки: {e}")
        db.rollback()
    finally:
        db.close()


def main():
    """Главная функция отладки"""
    print("=" * 60)
    print("Document AI Assistant - Database Debug Tool")
    print("=" * 60)
    
    # 1. Проверка БД
    if not check_database():
        print("\n🔧 Попытка исправления...")
        fix_database_sessions()
        if not check_database():
            print("❌ Не удалось исправить проблемы с БД")
            return 1
    
    # 2. Тест CRUD
    if not test_crud_operations():
        print("❌ CRUD операции не работают")
        return 1
    
    # 3. Проверка API (если сервер запущен)
    check_api_endpoints()
    
    # 4. Интерактивные команды
    print("\n" + "=" * 60)
    print("Доступные команды:")
    print("1. Создать тестовый документ")
    print("2. Очистить тестовые документы") 
    print("3. Показать все документы")
    print("4. Выход")
    
    while True:
        try:
            choice = input("\nВведите номер команды: ").strip()
            
            if choice == "1":
                doc_id = create_test_document()
                if doc_id:
                    print(f"Тестовый документ создан с ID: {doc_id}")
            
            elif choice == "2":
                cleanup_test_documents()
            
            elif choice == "3":
                db = SessionLocal()
                try:
                    docs = db.query(Document).all()
                    print(f"\n📚 Все документы ({len(docs)}):")
                    for doc in docs:
                        print(f"   ID: {doc.id} | {doc.original_filename} | {doc.processing_status}")
                finally:
                    db.close()
            
            elif choice == "4":
                break
            
            else:
                print("Неверный выбор")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}")
    
    print("\n👋 Отладка завершена")
    return 0


if __name__ == "__main__":
    exit(main())