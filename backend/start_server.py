#!/usr/bin/env python3
"""
Стартовый скрипт для Document AI Assistant
"""
import sys
import os
import subprocess
import argparse

# Добавляем путь к приложению
sys.path.append(os.path.dirname(__file__))

from app.core.config import settings


def check_dependencies():
    """Проверка установленных зависимостей"""
    try:
        import torch
        import transformers
        import fastapi
        import sqlalchemy
        import fitz  # PyMuPDF
        import pytesseract
        print(f"✅ All dependencies installed")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Transformers: {transformers.__version__}")
        print(f"   FastAPI: {fastapi.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def check_tesseract():
    """Проверка Tesseract OCR"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Tesseract OCR: {version_line}")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️  Tesseract OCR not found")
    print("   Install: sudo apt-get install tesseract-ocr tesseract-ocr-rus")
    print("   Or set TESSERACT_PATH in .env file")
    return False


def start_server(host: str, port: int, reload: bool = False):
    """Запуск FastAPI сервера"""
    try:
        import uvicorn
        
        print(f"🚀 Starting Document AI Assistant...")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Debug mode: {reload}")
        print(f"   API Docs: http://{host}:{port}/docs")
        print()
        
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="debug"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Document AI Assistant Server')
    parser.add_argument('--host', default=settings.HOST, 
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=settings.PORT,
                       help='Port to bind to')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--no-checks', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Document AI Assistant - Server Startup")
    print("=" * 60)
    
    if not args.no_checks:
        print("Checking dependencies...")
        if not check_dependencies():
            return 1
        
        check_tesseract()
        print()
    
    # Проверяем существование БД
    if not os.path.exists("documents.db"):
        print("⚠️  Database not found!")
        print("   Run: python init_db.py")
        user_input = input("   Continue anyway? [y/N]: ").strip().lower()
        if user_input != 'y' and user_input != 'yes':
            return 1
    
    return start_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    exit(main())