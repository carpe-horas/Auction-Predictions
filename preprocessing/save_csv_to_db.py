import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 현재 스크립트 상위 폴더 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data')
processed_dir = os.path.join(data_dir, 'processed')

# MySQL 데이터베이스 연결 설정
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_NAME')

# SQLAlchemy 엔진 생성
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}')

# 1. 날씨 데이터 처리 및 저장
climate_file_path = os.path.join(processed_dir, 'preprocessed_climate.csv')
df_climate = pd.read_csv(climate_file_path, index_col=False)

# 날씨 데이터 칼럼명 매핑
climate_column_mapping = {
    '지점명': 'region',
    '일시': 'date',
    '평균기온(°C)': 'avg_temp',
    '최저기온(°C)': 'min_temp',
    '최고기온(°C)': 'max_temp',
    '일강수량(mm)': 'daily_rainfall',
    '최대 풍속(m/s)': 'max_wind',
    '평균 풍속(m/s)': 'avg_wind',
    '평균 상대습도(%)': 'avg_humidity',
    '평균 현지기압(hPa)': 'avg_pressure',
    '합계 일조시간(hr)': 'sunshine_hours',
    '일 최심신적설(cm)': 'max_snow_depth'
}

# 칼럼명 변경
df_climate.rename(columns=climate_column_mapping, inplace=True)

# 데이터프레임을 MySQL 테이블로 저장
weather_table_name = 'weather_data'
df_climate.reset_index(inplace=True)
df_climate.rename(columns={'index': 'weather_id'}, inplace=True)
df_climate.to_sql(name=weather_table_name, con=engine, if_exists='replace', index=False)
print(f"데이터가 MySQL 데이터베이스의 {weather_table_name} 테이블에 저장되었습니다.")

# 2. 경매 데이터 처리 및 저장
market_file_path = os.path.join(processed_dir, 'preprocessed_agromarket.csv')
df_market = pd.read_csv(market_file_path, index_col=False)

# 경매 데이터 칼럼명 매핑
market_column_mapping = {
    '일자': 'date',
    '부류': 'category',
    '품목': 'item',
    '품종': 'variety',
    '도매시장': 'market',
    '물량(kg)': 'quantity_kg',
    '금액(원)': 'price_won'
}

# 칼럼명 변경
df_market.rename(columns=market_column_mapping, inplace=True)

# 데이터프레임을 MySQL 테이블로 저장
auction_table_name = 'auction_data'
df_market.reset_index(inplace=True)
df_market.rename(columns={'index': 'auction_id'}, inplace=True)
df_market.to_sql(name=auction_table_name, con=engine, if_exists='replace', index=False)
print(f"데이터가 MySQL 데이터베이스의 {auction_table_name} 테이블에 저장되었습니다.")