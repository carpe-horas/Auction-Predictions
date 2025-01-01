from flask import Blueprint, render_template, request, jsonify
from services.db_connection import get_db_connection
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from uuid import uuid4
from datetime import datetime, timedelta

# 환경 변수 설정
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Blueprint 생성
prediction_routes = Blueprint("prediction", __name__)

# 경로 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_PATH = os.path.join(BASE_DIR, '..', 'static')

# 모델 및 스케일러 로드 함수
def get_model_and_scaler(cluster_id):
    model_path = os.path.join(BASE_DIR, '..', '..', 'saved_models', 'LSTM', f'cluster_{cluster_id}_units_100_dropout_0.3_epochs_100_batch_64', 'lstm_model.keras')
    scaler_path = os.path.join(BASE_DIR, '..', '..', 'saved_models', 'LSTM', f'cluster_{cluster_id}_units_100_dropout_0.3_epochs_100_batch_64', 'scaler.npy')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}. 경로를 확인하고 파일이 존재하는지 확인하세요.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}. 경로를 확인하고 파일이 존재하는지 확인하세요.")

    model = load_model(model_path)
    try:
        scaler_params = np.load(scaler_path, allow_pickle=True).item()
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = scaler_params['mean'], scaler_params['scale']
    except Exception as e:
        raise ValueError(f"스케일러 파일을 로드하는 중 오류가 발생했습니다: {e}")

    return model, scaler

# 예측 함수
def prepare_input_sequence(data, sequence_length=30):
    """데이터를 시퀀스로 변환."""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# 컬럼명 매핑 정의 
COLUMN_MAPPING = {
    'date': '날짜',
    'total_quantity': '총물량',
    'average_price': '평균단가',
    'temperature_range': '온도차',
    'average_temperature': '평균기온',
    'daily_rainfall': '일 강수량',
    'sales_amount': '매출액',
    'quantity_price_ratio': '물량_단가_비율',
    'price_volatility': '단가_변동성',
    'quantity_volatility': '물량_변동성'
}

@prediction_routes.route("/", methods=["GET"])
def show_form():
    try:
        engine = get_db_connection()

        query_items = "SELECT DISTINCT item FROM clustered_seoul"
        items = pd.read_sql(query_items, engine)['item'].tolist()

        # GET 요청: 폼만 렌더링
        return render_template("prediction.html", items=items)

    except Exception as e:
        return jsonify({'error': f"오류가 발생했습니다: {str(e)}"})

# @prediction_routes.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # DB 연결
#         engine = get_db_connection()

#         # 품목 리스트 가져오기
#         query_items = "SELECT DISTINCT item FROM clustered_seoul"
#         items = pd.read_sql(query_items, engine)['item'].tolist()

#         # 사용자 입력 수신
#         item = request.form.get("item")
#         prediction_date = request.form.get("prediction_date")

#         # 입력 날짜 유효성 검증
#         try:
#             prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d").date()
#         except ValueError:
#             return jsonify({'error': "날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식으로 입력하세요."})

#         # DB에서 데이터 가져오기 (예측 날짜 이전 데이터)
#         query = """
#             SELECT date, total_quantity, average_price, temperature_range, average_temperature, 
#                    daily_rainfall, sales_amount, quantity_price_ratio, price_volatility, quantity_volatility
#             FROM clustered_seoul
#             WHERE item = %s AND date <= %s
#             ORDER BY date DESC
#             LIMIT 30
#         """
#         df = pd.read_sql(query, engine, params=(item, prediction_date))

#         # 컬럼명 매핑 적용
#         df.rename(columns=COLUMN_MAPPING, inplace=True)

#         # 데이터 검증
#         if df.empty:
#             return jsonify({'error': f"{prediction_date} 이전의 데이터가 없습니다. 다른 날짜를 선택하세요."})
#         if len(df) < 30:
#             return jsonify({'error': f"예측을 위해 최소 30개의 데이터가 필요합니다. 현재 데이터 개수: {len(df)}"})

#         # 필요한 특성 정의 (학습 코드와 동일하게)
#         feature_columns = ['평균단가', '총물량', '온도차', '평균기온', '일 강수량',
#                            '매출액', '물량_단가_비율', '단가_변동성', '물량_변동성']

#         # 데이터 스케일링
#         try:
#             scaled_features = scaler.transform(df[feature_columns])
#         except ValueError as e:
#             return jsonify({
#                 'error': f"스케일링 오류: 입력 데이터와 스케일러의 크기가 일치하지 않습니다. "
#                          f"입력 데이터 크기: {df[feature_columns].shape}, "
#                          f"스케일러 기대 크기: {len(scaler.mean_)}. 세부 오류: {e}"
#             })

#         # 입력 시퀀스 준비
#         input_sequence = prepare_input_sequence(scaled_features, sequence_length=30)

#         # 예측 수행
#         try:
#             prediction = model.predict(input_sequence[-1:])  # 가장 최근 데이터를 기반으로 예측
#             predicted_price = round(prediction[0][0], 2)
#         except Exception as e:
#             return jsonify({'error': f"LSTM 모델 예측 오류: {e}"})

#         # 결과 반환
#         return render_template(
#             "prediction.html",
#             items=items,
#             predicted_price=predicted_price,
#             prediction_date=prediction_date,
#             item=item
#         )

#     except Exception as e:
#         return jsonify({'error': f"오류가 발생했습니다: {str(e)}"})

@prediction_routes.route("/items", methods=["GET"])
def get_items():
    """
    품목 리스트 조회 API
    """
    try:
        engine = get_db_connection()
        query = "SELECT DISTINCT item FROM clustered_seoul"
        items = pd.read_sql(query, engine)['item'].tolist()
        return jsonify({"items": items}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_routes.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        item = data.get("item")
        prediction_date = data.get("date")

        if not item or not prediction_date:
            return jsonify({"error": "item과 date는 필수입니다."}), 400

        # 입력 날짜 유효성 검증
        try:
            prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식으로 입력하세요."}), 400

        # DB 연결
        engine = get_db_connection()
        query = """
            SELECT date, total_quantity, average_price, temperature_range, average_temperature, 
                   daily_rainfall, sales_amount, quantity_price_ratio, price_volatility, quantity_volatility, season, cluster
            FROM clustered_seoul
            WHERE item = %s AND date <= %s
            ORDER BY date DESC
            LIMIT 30
        """
        df = pd.read_sql(query, engine, params=(item, prediction_date))

        # 데이터 검증
        if df.empty:
            return jsonify({"error": "해당 품목에 대한 데이터가 존재하지 않습니다."}), 404

        # 클러스터 ID 가져오기
        cluster_id = df['cluster'].iloc[0] if 'cluster' in df.columns and not df['cluster'].isnull().all() else None

        # 클러스터 ID가 없는 경우 기본값 사용
        if cluster_id is None:
            default_cluster_id = 1  # 기본 클러스터 ID 설정
            cluster_id = default_cluster_id

        # 클러스터에 맞는 모델 및 스케일러 로드
        try:
            model, scaler = get_model_and_scaler(cluster_id)
        except Exception as e:
            return jsonify({"error": f"모델 또는 스케일러를 로드하는 중 오류가 발생했습니다: {str(e)}"}), 500

        # 데이터 스케일링 및 시퀀스 준비
        feature_columns = [
            'total_quantity', 'temperature_range',
            'average_temperature', 'daily_rainfall', 'sales_amount',
            'quantity_price_ratio', 'price_volatility', 'quantity_volatility'
        ]

        if not df.empty:
            # Season encoding
            encoder = OneHotEncoder(sparse_output=False, categories=[['봄', '여름', '가을', '겨울']])
            season_encoded = encoder.fit_transform(df[['season']])
            season_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(['season']))
            df = pd.concat([df.reset_index(drop=True), season_df.reset_index(drop=True)], axis=1)

            scaled_features = scaler.transform(df[feature_columns + list(season_df.columns)].to_numpy())
            input_sequence = prepare_input_sequence(scaled_features, sequence_length=30)
        else:
            # 초기 데이터 생성 (모델 입력을 위한 임의 값)
            initial_sequence = np.zeros((30, len(feature_columns)))  # 기본값으로 0 사용
            input_sequence = initial_sequence[np.newaxis, :, :]  # 모델 입력 형태로 변환

        # 미래 데이터를 생성하기 위해 필요한 부분
        future_predictions = []
        for i in range(10):  # 10일 예측
            prediction = model.predict(input_sequence[:, -30:, :], verbose=0)  # 예측값
            prediction = prediction.flatten()  # 예측값을 1차원으로 변환
            future_predictions.append(float(prediction[0]))  # float32 -> float 변환

            # 새로운 데이터를 기존 특성 수에 맞게 확장
            expanded_prediction = np.zeros((1, 1, input_sequence.shape[2]))  # (1, 1, feature_count)
            expanded_prediction[0, 0, 0] = prediction[0]  # 첫 번째 특성에 예측값 추가

            # 새로운 데이터를 입력 시퀀스에 추가
            input_sequence = np.concatenate([input_sequence[:, 1:, :], expanded_prediction], axis=1)  # (1, 30, feature_count)

        # 미래 데이터프레임 생성
        future_dates = [prediction_date + timedelta(days=i) for i in range(1, 11)]
        future_df = pd.DataFrame({
            "date": future_dates,
            "average_price": future_predictions
        })

        # 과거 데이터와 미래 데이터 병합
        if not df.empty:
            combined_df = pd.concat([df[['date', 'average_price']], future_df])
        else:
            combined_df = future_df

        # NaN 값 기본값으로 대체
        combined_df.fillna(0, inplace=True)

        # datetime 형식으로 변환
        combined_df["date"] = pd.to_datetime(combined_df["date"])

        # 날짜 순 정렬
        combined_df.sort_values(by="date", inplace=True)

        # 단가 변동성 계산 (최근 5일 기준)
        combined_df["price_volatility"] = (
            combined_df["average_price"]
            .rolling(window=5, min_periods=1)  # 최근 5일 기준
            .std()
            .fillna(0)  # NaN 값을 0으로 대체
        )

        # 응답 데이터 구성
        response_data = {
            "item": item,
            "date": prediction_date.strftime("%Y-%m-%d"),
            "predicted_price": float(future_predictions[0]),
            "historical_data": {
                "dates": combined_df["date"].dt.strftime("%Y-%m-%d").tolist(),
                "prices": [float(price) for price in combined_df["average_price"]],
                "volatilities": [float(volatility) for volatility in combined_df["price_volatility"]],
            }
        }


        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
