import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 MySQL 연결 설정 가져오기
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_NAME')

# SQLAlchemy 엔진 생성
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}')

# 데이터베이스 및 테이블 생성 SQL
setup_sql = f"""
DROP DATABASE IF EXISTS {database};
CREATE DATABASE {database};
USE {database};

CREATE TABLE market_info (
    market_id INT AUTO_INCREMENT PRIMARY KEY,
    market_name VARCHAR(100),
    location VARCHAR(200)
);

CREATE TABLE auction_data (
    auction_id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    category VARCHAR(100),
    item VARCHAR(100),
    variety VARCHAR(100),
    market VARCHAR(100),
    quantity_kg FLOAT,
    price_won INT,
    FOREIGN KEY (market) REFERENCES market_info(market_id)
);

CREATE TABLE weather_data (
    weather_id INT AUTO_INCREMENT PRIMARY KEY,
    region VARCHAR(100),
    date DATE NOT NULL,
    avg_temp FLOAT,
    min_temp FLOAT,
    max_temp FLOAT,
    daily_rainfall FLOAT,
    max_wind FLOAT,
    avg_wind FLOAT,
    avg_humidity FLOAT,
    avg_pressure FLOAT,
    sunshine_hours FLOAT,
    max_snow_depth FLOAT
);
"""

# 데이터베이스 및 테이블 설정
try:
    with engine.connect() as connection:
        # 여러 SQL 문을 개별적으로 실행
        for statement in setup_sql.split(';'):
            if statement.strip():
                connection.execute(text(statement))
    print("데이터베이스 및 테이블이 성공적으로 생성되었습니다.")
except SQLAlchemyError as e:
    print(f"데이터베이스 설정 중 오류 발생: {e}")
