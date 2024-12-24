# pip install mysql-connector-python

import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os

# env 파일 로드
load_dotenv()

# 환경 변수 가져오기
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# CSV 파일 읽기
csv_file_path = "C:/Users/carpe/OneDrive/Desktop/workspace/project/Aution/Auction/data/climate_data.csv"
data = pd.read_csv(csv_file_path)

# MySQL 연결 설정
connection = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE
)
cursor = connection.cursor()

# 테이블 생성
create_table_query = """
CREATE TABLE IF NOT EXISTS weather_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    region VARCHAR(10),
    date DATE,
    average_temperature FLOAT,
    min_temperature FLOAT,
    max_temperature FLOAT,
    daily_rainfall FLOAT,
    max_wind_speed FLOAT,
    average_wind_speed FLOAT,
    average_humidity FLOAT,
    average_local_pressure FLOAT,
    total_sunshine_hours FLOAT,
    max_snow_depth FLOAT
);
"""
cursor.execute(create_table_query)

# 데이터 삽입
insert_query = """
INSERT INTO weather_data (
    region, date, average_temperature, min_temperature, max_temperature, daily_rainfall, max_wind_speed, average_wind_speed, average_humidity, average_local_pressure, total_sunshine_hours, max_snow_depth
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

for _, row in data.iterrows():
    cursor.execute(insert_query, tuple(row))

# 변경 사항 저장 및 연결 종료
connection.commit()
cursor.close()
connection.close()

print("CSV 데이터를 MySQL에 성공적으로 삽입했습니다.")
