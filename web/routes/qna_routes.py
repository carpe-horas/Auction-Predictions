from flask import Blueprint, render_template

qna_routes = Blueprint("qna", __name__)

@qna_routes.route("/")
def qna():
    faq_data = [
        {"question": "이 프로젝트는 무엇인가요?", "answer": "경매 가격을 예측하는 웹 애플리케이션입니다."},
        {"question": "예측은 어떻게 이루어지나요?", "answer": "딥러닝 모델과 과거 데이터를 활용해 예측합니다."},
        {"question": "데이터 출처는 어디인가요?", "answer": "기상청 데이터와 도매시장 거래 데이터를 사용합니다."},
        {"question": "회원 가입이 필요한가요?", "answer": "회원 가입 없이도 서비스를 이용할 수 있지만, 일부 기능은 회원만 이용할 수 있습니다."},
        {"question": "서비스의 주요 기능은 무엇인가요?", "answer": "이 서비스는 농산물 경매 가격 예측, 과거 거래 데이터 분석, 실시간 예측 결과 제공 기능을 제공합니다."},
        {"question": "딥러닝 모델은 어떤 알고리즘을 사용하나요?", "answer": "LSTM 모델을 사용하여 시계열 데이터 예측을 최적화하고 있습니다."},
    ]
    return render_template("qna.html", faq_data=faq_data)
