import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

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

# 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'data', 'processed', 'clustered_seoul.csv')

# 데이터 불러오기
data = pd.read_csv(data_path)

# 파생 변수 생성
data['sales_amount'] = data['total_quantity'] * data['average_price']
data['quantity_price_ratio'] = data['total_quantity'] / data['average_price']
data['price_volatility'] = data['average_price'].pct_change().fillna(0).abs()
data['quantity_volatility'] = data['total_quantity'].pct_change().fillna(0).abs()

# 칼럼 순서 조정
columns_order = [
    'id', 'item', 'date', 'total_quantity', 'average_price', 'temperature_range',
    'average_temperature', 'daily_rainfall', 'season', 'cluster',
    'sales_amount', 'quantity_price_ratio', 'price_volatility', 'quantity_volatility'
]
data.reset_index(inplace=True)
data.rename(columns={"index": "id"}, inplace=True)
data = data[columns_order]

# MySQL DB에 저장
def save_to_db(data, table_name, engine):
    with engine.connect() as connection:
        data.to_sql(table_name, con=connection, if_exists='replace', index=False)

# DB 연결 및 저장
engine = get_db_connection()
save_to_db(data, 'clustered_seoul', engine)
print("데이터가 MySQL DB에 성공적으로 저장되었습니다.")

save_path = os.path.join(base_dir, 'data', 'last_clustered_seoul.csv')
data.to_csv(save_path, index=False, encoding='utf-8-sig')
