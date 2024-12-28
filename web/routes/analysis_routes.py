from flask import Blueprint, render_template, jsonify, request
from services.analysis_service import fetch_and_preprocess_data, reset_analysis_cache
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

analysis_routes = Blueprint("analysis", __name__)

@analysis_routes.route("/")
def analysis_page():
    return render_template("analysis.html")

@analysis_routes.route("/preprocess", methods=["POST"])
def start_preprocessing():
    """동기적으로 데이터를 전처리"""
    logging.info("데이터 전처리 시작")
    processed_data = fetch_and_preprocess_data()  # 데이터 전처리 동기 실행
    logging.info("데이터 전처리 완료")
    return jsonify({"message": "데이터 전처리가 완료되었습니다."})


@analysis_routes.route("/data", methods=["GET"])
def get_analysis_data():
    """요청한 품목의 데이터를 반환"""
    requested_item = request.args.get("item", default="배추")
    logging.info(f"요청된 품목: {requested_item}")
    
    # 데이터 가져오기
    processed_data = fetch_and_preprocess_data(filter_item=requested_item)
    
    # JSON 변환
    logging.info("JSON 변환 시작")
    data = processed_data.to_dict(orient="records")
    logging.info("JSON 변환 완료")
    
    return jsonify(data)

@analysis_routes.route("/reset-cache", methods=["POST"])
def reset_cache():
    """캐시 초기화"""
    reset_analysis_cache()  
    logging.info("캐시 초기화 완료")
    return jsonify({"message": "캐시 초기화 완료"})
