from flask import Blueprint, render_template

qna_routes = Blueprint("qna", __name__)

@qna_routes.route("/")
def qna():
    faq_data = [
        {"question": "이 프로젝트는 무엇인가요?", "answer": "경매 가격을 예측하는 웹 애플리케이션입니다."},
        {"question": "예측은 어떻게 이루어지나요?", "answer": "딥러닝 모델과 과거 데이터를 활용해 예측합니다."},
        {"question": "데이터 출처는 어디인가요?", "answer": "기상청 데이터와 도매시장 거래 데이터를 사용합니다."}
    ]
    return render_template("qna.html", faq_data=faq_data)
