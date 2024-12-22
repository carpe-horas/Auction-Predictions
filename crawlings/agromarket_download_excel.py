import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import datetime
import chromedriver_autoinstaller

# chromedriver 자동 설치 및 경로 설정
chromedriver_autoinstaller.install()

# 경로 설정
base_dir = os.getcwd()  # 현재 실행 경로
tmp_dir = os.path.join(base_dir, r"data\raw\tmp")  # 임시 다운로드 폴더
final_dir = os.path.join(base_dir, r"data\raw\agromarket")  # 최종 저장 폴더

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

# Chrome 옵션 설정
options = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": tmp_dir,  # 다운로드 경로 설정
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
options.add_experimental_option("prefs", prefs)
#options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# 웹 드라이버 초기화
driver = webdriver.Chrome(options=options)

try:
    # 시작 URL
    driver.get("https://at.agromarket.kr/domeinfo/smallTrade.do")

    # 날짜 범위 및 도매시장 설정
    start_date = datetime.date(2019, 1, 3)
    end_date = datetime.date(2024, 12, 20)
    delta = datetime.timedelta(days=30)

    whsal_list = [
        "서울가락", "서울강서", "수원", "안양", "안산", "구리", "인천남촌", "인천삼산", "순천",
        "광주각화", "광주서부", "정읍", "익산", "전주", "대전오정", "대전노은", "청주", "천안",
        "충주", "안동", "구미", "대구북부", "진주", "창원팔용", "창원내서", "부산엄궁", "부산반여",
        "울산", "포항", "원주", "춘천", "강릉", "부산국제수산"
    ]

    # 데이터 다운로드 루프
    for whsal in whsal_list:
        current_start_date = start_date
        while current_start_date <= end_date:
            current_end_date = min(current_start_date + delta - datetime.timedelta(days=1), end_date)
            try:
                # 시작일 및 종료일 설정 (JavaScript 사용)
                driver.execute_script("document.getElementById('startDate').value = arguments[0];", current_start_date.strftime("%Y-%m-%d"))
                driver.execute_script("document.getElementById('endDate').value = arguments[0];", current_end_date.strftime("%Y-%m-%d"))

                # 도매시장 선택
                whsalCd_select = Select(driver.find_element(By.ID, "whsalCd"))
                whsalCd_select.select_by_visible_text(whsal)

                # 엑셀 다운로드 버튼 클릭
                download_button = driver.find_element(By.CLASS_NAME, "btn_down")
                download_button.click()

                # 다운로드 대기
                time.sleep(5)

                # 다운로드된 파일 이름 확인 및 변경
                downloaded_files = os.listdir(tmp_dir)
                if not downloaded_files:
                    raise FileNotFoundError(f"다운로드된 파일이 없습니다. 시장: {whsal}, 기간: {current_start_date} ~ {current_end_date}")
                
                for file in downloaded_files:
                    if file.endswith(".xls") or file.endswith(".xlsx"):  # 파일 형식 확인
                        old_path = os.path.join(tmp_dir, file)
                        new_file_name = f"{whsal}_{current_start_date.strftime('%Y%m%d')}_{current_end_date.strftime('%Y%m%d')}.xls"
                        new_path = os.path.join(final_dir, new_file_name)
                        os.rename(old_path, new_path)
                        print(f"'{whsal}' 도매시장: {current_start_date} ~ {current_end_date} 데이터 저장 -> {new_file_name}")
                        break
                else:
                    raise FileNotFoundError("다운로드된 파일이 예상되는 형식이 아닙니다.")
            
            except Exception as e:
                print(f"오류 발생: 시장: {whsal}, 기간: {current_start_date} ~ {current_end_date}, 오류: {e}")
            
            # 다음 날짜로 이동
            current_start_date += delta

finally:
    # 드라이버 종료
    driver.quit()

print(f"모든 데이터가 '{final_dir}'에 저장되었습니다.")
