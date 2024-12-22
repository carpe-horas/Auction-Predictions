# pip install python-dotenv

import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os
from datetime import datetime
import re

# .env 파일 로드
load_dotenv()

# 네이버 API 키를 환경 변수에서 가져오기
client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")

# 검색 키워드와 API 요청 설정
keyword = "농산물 가격"  # 고정된 검색 키워드
base_url = "https://openapi.naver.com/v1/search/news.json"
headers = {
    "X-Naver-Client-Id": client_id,
    "X-Naver-Client-Secret": client_secret
}

# HTML 태그 제거 함수
def clean_html_tags(text):
    return re.sub(r'<[^>]*>', '', text)

# 날짜 형식 변환 함수
def convert_pub_date(pub_date):
    return datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z').strftime('%Y-%m-%d %H:%M:%S')

# 수집할 뉴스 데이터 저장 리스트
news_list = []

# 페이지네이션: 최대 1000개까지 (start=1, 101, 201, ...)
for start in range(1, 1001, 100):
    params = {
        "query": keyword,
        "display": 100,  # 한 번에 가져올 기사 수 (최대 100)
        "start": start,  # 시작 위치
        "sort": "date"   # 최신순 정렬
    }
    
    try:
        # API 요청
        response = requests.get(base_url, headers=headers, params=params)
        
        # 응답 상태 확인
        if response.status_code == 200:
            news_data = response.json()
            
            # 각 뉴스 아이템 처리
            for item in news_data['items']:
                news_list.append({
                    "Title": clean_html_tags(item['title']),  # HTML 태그 제거
                    "Link": item['link'],
                    "Summary": clean_html_tags(item['description']),  # HTML 태그 제거
                    "Publication Date": convert_pub_date(item['pubDate'])  # 날짜 형식 변환
                })
            
            print(f"{start}번째 뉴스 데이터 수집 완료")
            
        else:
            print(f"Error Code: {response.status_code}")
            break
        
        # 요청 제한을 피하기 위해 딜레이 추가
        time.sleep(1)
        
    except Exception as e:
        print(f"에러 발생: {e}")
        break

# 수집한 데이터를 데이터프레임으로 변환
news_df = pd.DataFrame(news_list)

# 중복 제거: 제목과 링크를 기준으로 중복 제거
news_df = news_df.drop_duplicates(subset=['Title', 'Link'])

# 결과를 CSV 파일로 저장
# 파일 경로와 이름 설정
output_dir = "/data/raw"
os.makedirs(output_dir, exist_ok=True)  # 디렉토리 없으면 생성

# 파일 이름 수정
output_file = os.path.join(output_dir, "agricultural_price_news.csv")

# 데이터 저장
news_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"중복 제거 후 뉴스 데이터가 {output_file} 파일로 저장되었습니다.")
