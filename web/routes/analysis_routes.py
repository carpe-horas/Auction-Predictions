from flask import Blueprint, render_template, jsonify, request
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# Blueprint 설정
analysis_routes = Blueprint('analysis_routes', __name__)

# DB 연결 설정
def get_db_connection():
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)
    return engine

# Analysis 페이지 렌더링
@analysis_routes.route('/', methods=['GET'])
def analysis_home():
    try:
        engine = get_db_connection()
        query = "SELECT DISTINCT item FROM agromarket_climate"
        df = pd.read_sql(query, engine)
        items = df['item'].tolist()
        return render_template('analysis.html', items=items)
    except Exception as e:
        print(f"오류 발생: {e}")
        return render_template('analysis.html', items=[])

# 품목 데이터 가져오기
@analysis_routes.route('/get-item-data', methods=['GET'])
def get_item_data():
    try:
        item = request.args.get('item', None)
        if not item:
            return jsonify({"error": "No item specified"}), 400

        engine = get_db_connection()
        query = f"SELECT * FROM agromarket_climate WHERE item = '{item}'"
        df = pd.read_sql(query, engine)

        regional_analysis = (
            df.groupby('region')[['avg_unit_price_per_kg', 'avg_quantity_kg']].mean().reset_index()
        )
        seasonal_analysis = (
            df.groupby('season')[['avg_unit_price_per_kg', 'avg_quantity_kg']].mean().reset_index()
        )

        return jsonify({
            "regional_analysis": regional_analysis.to_dict(orient='list'),
            "seasonal_analysis": seasonal_analysis.to_dict(orient='list')
        })
    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"error": str(e)}), 500
