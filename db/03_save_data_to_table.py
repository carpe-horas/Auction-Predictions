# agromarket_climate.csv 상위 30개 품목 외 열 삭제
# 계절별 품목(seasonal_items) 컬럼 추가 후 db 테이블 생성 후 저장

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error

# .env 파일 로드
load_dotenv()

# 현재 스크립트 상위 폴더 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data')

# MySQL 데이터베이스 연결 설정
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT', 3306)
database = os.getenv('DB_NAME')


csv_file_path = os.path.join(data_dir, 'agromarket_climate.csv')
data = pd.read_csv(csv_file_path)

# 계절별 품목 리스트
valid_items = [
    '딸기', '시금치', '상추', '열무', '얼갈이배추', '부추', '수박', '참외', '오이', '호박', 
    '풋고추', '토마토', '방울토마토', '양배추', '복숭아', '사과', '배', '포도', '단감', 
    '고구마', '감자', '배추', '무', '마늘', '양파', '당근', '대파', '감귤', '바나나', '새송이'
]

# valid_items에 없는 item 행 제거
data = data[data['item'].isin(valid_items)]

# 계절별 품목 분류
seasonal_items = {
    '봄': ['딸기', '시금치', '상추', '열무', '얼갈이배추', '부추'],
    '여름': ['수박', '참외', '오이', '호박', '풋고추', '토마토', '방울토마토', '양배추', '복숭아'],
    '가을': ['사과', '배', '포도', '단감', '고구마', '감자'],
    '겨울': ['배추', '무', '마늘', '양파', '당근', '대파', '감귤'],
    '연중/특이': ['바나나', '새송이']
}

# 품목을 계절로 분류하는 함수
def classify_season(item, seasonal_items):
    for season, items in seasonal_items.items():
        if item in items:
            return season
    return None

# item_season 컬럼 추가
data['item_season'] = data['item'].apply(lambda x: classify_season(x, seasonal_items))

# 데이터 삽입 전 타입 변환
data = data.astype({
    col: 'object' if data[col].dtype == 'O' else 'float' if data[col].dtype == 'float64' else 'int'
    for col in data.columns
})

# 데이터베이스에 연결 및 테이블 생성
def create_connection():
    try:
        connection = mysql.connector.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
        if connection.is_connected():
            print("MySQL 데이터베이스에 연결되었습니다.")
            return connection
    except Error as e:
        print(f"에러 발생: {e}")
        return None

# 데이터프레임을 MySQL에 삽입
def insert_data_to_mysql(connection, dataframe, table_name, batch_size=1000):
    cursor = connection.cursor()
    # 테이블 생성 쿼리 (필요시 수정)
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join([f'`{col}` TEXT' for col in dataframe.columns])}
    ) ENGINE=InnoDB;
    """
    try:
        # 테이블 생성
        cursor.execute(create_table_query)
        connection.commit()
        print(f"테이블 '{table_name}'이(가) 생성되었습니다.")

        # 배치 단위로 데이터 삽입
        rows = dataframe.to_records(index=False)
        total_rows = len(rows)
        for i in range(0, total_rows, batch_size):
            batch = [tuple(map(lambda x: int(x) if isinstance(x, np.int64) else x, row)) for row in rows[i:i+batch_size]]
            insert_query = f"""
            INSERT INTO {table_name} ({', '.join([f'`{col}`' for col in dataframe.columns])}) 
            VALUES ({', '.join(['%s'] * len(dataframe.columns))});
            """
            cursor.executemany(insert_query, batch)
            connection.commit()
            print(f"{i + len(batch)} / {total_rows}개의 데이터가 삽입되었습니다.")

        print("모든 데이터가 성공적으로 삽입되었습니다.")
    except Error as e:
        print(f"데이터 삽입 중 에러 발생: {e}")
        connection.rollback()
    finally:
        cursor.close()

if __name__ == "__main__":
    table_name = "agromarket_climate"
    connection = create_connection()
    if connection:
        try:
            insert_data_to_mysql(connection, data, table_name)
        finally:
            connection.close()
            print("MySQL 연결이 종료되었습니다.")
