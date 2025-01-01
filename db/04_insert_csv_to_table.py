import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# .env 변수 가져오기
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# MySQL 연결 문자열 생성
db_url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy 엔진 생성
engine = create_engine(db_url)

# 동적 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 경로
project_root = os.path.dirname(script_dir)  # 프로젝트 루트 경로
data_dir = os.path.join(project_root, 'data')  # data 폴더 경로
csv_file_path = os.path.join(data_dir, 'clustered_seoul.csv')  # CSV 파일 경로

# CSV 파일에서 df1 불러오기
try:
    df1 = pd.read_csv(csv_file_path)
    print("CSV 파일이 성공적으로 로드되었습니다!")
    print(df1.head())  # 데이터 확인
except FileNotFoundError as e:
    print("CSV 파일을 찾을 수 없습니다. 경로를 확인하세요:", csv_file_path)
    print("오류 메시지:", e)
    exit()

# df1 데이터를 MySQL 테이블에 삽입
try:
    table_name = "clustered_seoul"  # 삽입할 테이블 이름
    df1.to_sql(name=table_name, con=engine, if_exists="append", index=False)
    print("데이터 삽입 완료!")
except Exception as e:
    print("데이터 삽입 중 오류 발생:", e)
