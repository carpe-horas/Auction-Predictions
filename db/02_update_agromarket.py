# 소수점 둘째자리로 db 업데이트

from pymongo import MongoClient
from bson.decimal128 import Decimal128, create_decimal128_context
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
MONGO_URI = os.getenv('MONGO_URI')  # MongoDB URI
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')  

# MongoDB 연결
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]  
collection = db['agromarket']

# 업데이트 대상 필드
numeric_fields = ['총물량(kg)', '평균물량(kg)', '총금액(원)', '평균금액(원)', '총단가(원/kg)', '평균단가(원/kg)']

# 각 필드에 대해 소수점 둘째 자리 반올림 업데이트
for field in numeric_fields:
    # 해당 필드가 존재하는 모든 문서 검색
    for document in collection.find({field: {"$exists": True}}):
        original_value = document.get(field)
        
        # 값이 숫자일 경우 소수점 둘째 자리로 반올림
        if isinstance(original_value, (int, float)):
            rounded_value = round(original_value, 2)
            
            # MongoDB 업데이트
            collection.update_one(
                {"_id": document["_id"]},  # 문서 식별
                {"$set": {field: rounded_value}}  # 반올림된 값 저장
            )

print("소수점 둘째 자리 처리가 완료되었습니다.")
