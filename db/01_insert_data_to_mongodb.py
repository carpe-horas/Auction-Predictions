# csv 합쳐서 mongodb에 저장

# activate 가상환경
# pip install pymongo
# pip install python-dotenv
# python preprocessing/saved_csv_to_mongodb.py로 실행

import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
MONGO_URI = os.getenv('MONGO_URI')  # MongoDB URI
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')  # MongoDB 데이터베이스 이름

# MongoDB 연결
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]  # 데이터베이스 선택

print(f"데이터베이스 '{MONGO_DB_NAME}'에 연결되었습니다.")

# CSV 파일이 저장된 폴더 경로
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 경로
folder_path = os.path.join(current_dir, "../data/processed/agromarket_yearandseason")

# 모든 CSV 파일 합치기
all_data = pd.DataFrame()  # 빈 DataFrame 생성
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # CSV 읽기 (헤더가 중복 삽입되지 않도록 기본 헤더를 사용)
        data = pd.read_csv(file_path, header=0)  # 첫 번째 행을 헤더로 인식
        all_data = pd.concat([all_data, data], ignore_index=True)  # 모든 파일 데이터를 합침

# 중복된 헤더 제거 (만약 데이터 내에 헤더 행이 포함된 경우 처리)
if all_data.columns[0] == all_data.iloc[0].values[0]:  # 첫 번째 행이 헤더일 경우
    all_data = all_data.iloc[1:]  # 첫 번째 행 제거

# 하나로 합친 데이터를 MongoDB에 저장
collection_name = "agromarket"  # 저장할 컬렉션 이름
collection = db[collection_name]

# MongoDB에 삽입
if not all_data.empty:  # 데이터가 비어 있지 않은 경우만 삽입
    collection.insert_many(all_data.to_dict('records'))  # MongoDB에 삽입
    print(f"'{collection_name}' 컬렉션에 데이터가 삽입되었습니다.")
else:
    print("폴더에 CSV 파일이 없거나 데이터가 비어 있습니다.")
