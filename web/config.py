import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """Flask 앱 설정 클래스"""
    DEBUG = True  # 디버그 모드 활성화
    DB_HOST = os.getenv("DB_HOST") 
    DB_USER = os.getenv("DB_USER") 
    DB_PASSWORD = os.getenv("DB_PASSWORD")  
    DB_NAME = os.getenv("DB_NAME") 
