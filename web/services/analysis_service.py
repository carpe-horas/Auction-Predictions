import pandas as pd
import logging
from services.db_connection import get_db_connection

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 전처리된 데이터를 메모리에 저장
cached_data = None

def preload_data():
    """애플리케이션 초기화 시 데이터 로드"""
    global cached_data
    logging.info("애플리케이션 초기화 - 데이터 로드 시작")
    cached_data = fetch_and_preprocess_data()
    logging.info("애플리케이션 초기화 - 데이터 로드 완료")

def fetch_and_preprocess_data(filter_item=None):
    """데이터를 DB에서 가져와 전처리하거나 캐싱된 데이터 반환"""
    global cached_data

    # 캐싱된 데이터가 있으면 반환
    if cached_data is not None:
        logging.info("캐싱된 데이터 반환")
        if filter_item:
            filtered_data = cached_data[cached_data['item'] == filter_item]
            logging.info(f"필터링된 데이터 크기: {filtered_data.shape}")
            return filtered_data
        return cached_data

    # DB 연결
    try:
        logging.info("데이터베이스 연결 시작")
        engine = get_db_connection()
        logging.info("데이터베이스 연결 완료")
    except Exception as e:
        logging.error(f"데이터베이스 연결 실패: {e}")
        raise

    # 쿼리 실행
    query = """
        SELECT 
            date, category, item, market AS wholesale_market,
            SUM(quantity_kg) AS total_quantity_kg,
            AVG(quantity_kg) AS avg_quantity_kg,
            SUM(price_won) AS total_price_won,
            AVG(price_won) AS avg_price_won
        FROM auction_data
        WHERE date >= '2019-01-03' AND date <= '2024-12-20'
        GROUP BY date, category, item, market
    """
    try:
        logging.info("쿼리 실행 시작")
        df = pd.read_sql(query, con=engine)
        logging.info(f"쿼리 실행 완료 - 데이터 크기: {df.shape}")
    except Exception as e:
        logging.error(f"쿼리 실행 실패: {e}")
        raise

    # 데이터 처리
    logging.info("데이터 처리 시작")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df["unit_price_per_kg"] = df["total_price_won"] / df["total_quantity_kg"].replace(0, pd.NA)
    df["avg_unit_price_per_kg"] = df["avg_price_won"] / df["avg_quantity_kg"].replace(0, pd.NA)
    df = df.fillna(0)
    df['region'] = df['wholesale_market'].str[:2]
    df = df[[
        'date', 'category', 'item', 'region', 'wholesale_market',
        'total_quantity_kg', 'avg_quantity_kg', 'total_price_won', 'avg_price_won',
        'unit_price_per_kg', 'avg_unit_price_per_kg'
    ]]
    logging.info("데이터 처리 완료")

    # 캐시 업데이트
    cached_data = df
    logging.info("데이터 캐싱 완료")
    
    if filter_item:
        filtered_data = cached_data[cached_data['item'] == filter_item]
        logging.info(f"필터링된 데이터 크기: {filtered_data.shape}")
        return filtered_data

    return cached_data

def reset_analysis_cache():
    """캐시 초기화"""
    global cached_data
    cached_data = None
    logging.info("캐시 초기화 완료")
