from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

def get_db_connection():
    # 환경 변수에서 DB 연결 정보 읽기
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")

    # SQLAlchemy 연결 문자열 생성
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)
    return engine
