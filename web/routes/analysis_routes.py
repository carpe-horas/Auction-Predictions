from flask import Blueprint, render_template, jsonify, request
import pandas as pd
import numpy as np
from services.db_connection import get_db_connection

# Blueprint 설정
analysis_routes = Blueprint('analysis_routes', __name__)

# Analysis 페이지 렌더링
# @analysis_routes.route('/', methods=['GET'])
# def analysis_home():
#     try:
#         engine = get_db_connection()
#         query = "SELECT DISTINCT item FROM agromarket_climate"
#         df = pd.read_sql(query, engine)
#         items = df['item'].tolist()
#         return render_template('analysis.html', items=items)
#     except Exception as e:
#         print(f"오류 발생: {e}")
#         return render_template('analysis.html', items=[])



# 품목 데이터 가져오기
# @analysis_routes.route('/get-item-data', methods=['GET'])
# def get_item_data():
#     try:
#         item = request.args.get('item', None)
#         if not item:
#             return jsonify({"error": "No item specified"}), 400

#         engine = get_db_connection()
#         query = f"SELECT * FROM agromarket_climate WHERE item = '{item}'"
#         df = pd.read_sql(query, engine)

#         regional_analysis = (
#             df.groupby('region')[['avg_unit_price_per_kg', 'avg_quantity_kg']].mean().reset_index()
#         )
#         seasonal_analysis = (
#             df.groupby('season')[['avg_unit_price_per_kg', 'avg_quantity_kg']].mean().reset_index()
#         )

#         return jsonify({
#             "regional_analysis": regional_analysis.to_dict(orient='list'),
#             "seasonal_analysis": seasonal_analysis.to_dict(orient='list')
#         })
#     except Exception as e:
#         print(f"오류 발생: {e}")
#         return jsonify({"error": str(e)}), 500


# v2 품목 데이터 가져오기
@analysis_routes.route('/', methods=['GET'])
def analysis_v2_home():
    try:
        engine = get_db_connection()
        query = "SELECT DISTINCT item FROM clustered_seoul"
        df = pd.read_sql(query, engine)
        items = df['item'].tolist()
        return render_template('analysis_v2.html', items=items)
    except Exception as e:
        print(f"오류 발생: {e}")
        return render_template('analysis_v2.html', items=[])

# v2 품목 데이터 가져오기
@analysis_routes.route('/get-item-data-v2', methods=['GET'])
def get_item_data_v2():
    try:
        # 요청에서 item과 analysis_type 가져오기
        item = request.args.get('item', None)
        analysis_type = request.args.get('analysis_type', None)

        # 필수 파라미터 검증
        if not item or not analysis_type:
            return jsonify({"error": "No item or analysis type specified"}), 400

        # DB 연결 및 데이터 조회
        engine = get_db_connection()
        query = f"SELECT * FROM clustered_seoul WHERE item = '{item}'"
        df = pd.read_sql(query, engine)

        # date 컬럼을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().any():
            return jsonify({"error": "Invalid date values detected"}), 400

        # 분석 유형에 따른 처리
        if analysis_type == 'daily_quantity_price_trend':
            # 일별 물량 및 가격 추세 분석
            analysis_result = (
                df.groupby('date')[['average_price', 'total_quantity']]
                .mean()
                .reset_index()
                .sort_values('date')
            )
            return jsonify({
                "analysis_result": {
                    "date": analysis_result['date'].dt.strftime('%Y-%m-%d').tolist(),
                    "average_price": analysis_result['average_price'].tolist(),
                    "total_quantity": analysis_result['total_quantity'].tolist()
                }
            })

        elif analysis_type == 'quantity_price_correlation':
            # 물량 대비 가격 상관관계 분석
            correlation_coefficient = df['total_quantity'].corr(df['average_price'])
            scatter_data = df[['total_quantity', 'average_price']].dropna().to_dict(orient='records')
            return jsonify({
                "analysis_result": {
                    "correlation_coefficient": correlation_coefficient,
                    "scatter_data": scatter_data
                }
            })

        elif analysis_type == 'item_price_extremes':
            # 품목별 최고가 및 최저가 분석
            max_price = df['average_price'].max()
            max_price_date = df.loc[df['average_price'].idxmax(), 'date']
            min_price = df['average_price'].min()
            min_price_date = df.loc[df['average_price'].idxmin(), 'date']
            return jsonify({
                "analysis_result": {
                    "max_price": max_price,
                    "max_price_date": max_price_date.strftime('%Y-%m-%d'),
                    "min_price": min_price,
                    "min_price_date": min_price_date.strftime('%Y-%m-%d')
                }
            })

        elif analysis_type == 'seasonal_average_price_quantity':
            # 계절별 평균 단가 및 물량 분석
            season_order = ['봄', '여름', '가을', '겨울']
            df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
            analysis_result = (
                df.groupby('season')[['average_price', 'total_quantity']].mean().reset_index()
            )
            return jsonify({
                "analysis_result": analysis_result.to_dict(orient='list')
            })

        elif analysis_type == 'monthly_average_price_quantity':
            # 월별 평균 단가 및 물량 분석
            df['month'] = df['date'].dt.month
            analysis_result = (
                df.groupby('month')[['average_price', 'total_quantity']].mean().reset_index()
            )
            return jsonify({
                "analysis_result": analysis_result.to_dict(orient='list')
            })

        elif analysis_type == 'recent_30_days_average_price_quantity':
            # 최근 30일간 평균 단가 및 물량 분석
            recent_30_days = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
            if recent_30_days.empty:
                return jsonify({"error": "최근 30일간의 데이터가 없습니다."}), 404

            analysis_result = (
                recent_30_days.groupby('date')[['average_price', 'total_quantity']]
                .mean()
                .reset_index()
                .sort_values('date')
            )
            return jsonify({
                "analysis_result": {
                    "date": analysis_result['date'].dt.strftime('%Y-%m-%d').tolist(),
                    "average_price": analysis_result['average_price'].tolist(),
                    "total_quantity": analysis_result['total_quantity'].tolist()
                }
            })

        elif analysis_type == 'temperature_average_price_quantity':
            # 평균 기온 구간 생성 및 분석 (5도 단위)
            bins = np.arange(df['average_temperature'].min(), df['average_temperature'].max() + 5, 5)
            labels = [int(b) for b in bins[:-1]]
            df['temperature_range'] = pd.cut(df['average_temperature'], bins, labels=labels)
            analysis_result = (
                df.groupby('temperature_range')[['average_price', 'total_quantity']].mean().reset_index()
            )
            return jsonify({
                "analysis_result": analysis_result.to_dict(orient='list')
            })

        elif analysis_type == 'rainfall_average_price_quantity':
            # 일 강수량 구간 생성 및 분석 (5mm 간격)
            bins = np.arange(0, df['daily_rainfall'].max() + 5, 5)
            labels = [f"{int(b)}-{int(b+5)}" for b in bins[:-1]]
            df['rainfall_range'] = pd.cut(
                df['daily_rainfall'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            analysis_result = (
                df.groupby('rainfall_range')[['average_price', 'total_quantity']].mean().reset_index()
            )
            analysis_result = analysis_result.dropna()
            return jsonify({
                "analysis_result": {
                    "rainfall_range": analysis_result['rainfall_range'].astype(str).tolist(),
                    "average_price": analysis_result['average_price'].tolist(),
                    "total_quantity": analysis_result['total_quantity'].tolist()
                }
            })

        else:
            # 잘못된 분석 유형 처리
            return jsonify({"error": "Invalid analysis type"}), 400

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"error": str(e)}), 500