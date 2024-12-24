import os
import pandas as pd

# 현재 스크립트 상위 폴더 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  
data_dir = os.path.join(project_root, 'data')
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')

# 엑셀 파일을 데이터프레임으로 읽기
climate_file_path = os.path.join(raw_dir,'OBS_ASOS_DD_20241220160421.xls.xlsx')
df = pd.read_excel(climate_file_path)

# 필터링할 컬럼 목록
columns_to_keep = [
    '지점명',
    '일시',
    '평균기온(°C)',
    '최저기온(°C)',
    '최고기온(°C)',
    '일강수량(mm)',
    '최대 풍속(m/s)',
    '평균 풍속(m/s)',
    '평균 상대습도(%)',
    '평균 현지기압(hPa)',
    '합계 일조시간(hr)',
    '일 최심신적설(cm)'
]

# 해당 컬럼들만 선택
df_filtered = df[columns_to_keep]

agromarket_file_path = os.path.join(processed_dir, 'preprocessed_agromarket.csv')
df2 = pd.read_csv(agromarket_file_path, index_col=False)

# '도매시장' 칼럼의 유니크 값 추출
unique_market_values = set(df2['도매시장'].unique())
unique_market_values = set([value[:2] for value in unique_market_values])

# '지점명' 칼럼의 유니크 값 추출
unique_branch_names = set(df_filtered['지점명'].unique())

# '도매시장'에만 있는 값 찾기
exclusive_market_values = unique_market_values - unique_branch_names

print("df2['도매시장']에만 있는 값:")
print(exclusive_market_values)

special_map = {
    '익산': '군산',
    '구리': '서울',
    '안양': '수원',
    '안산': '인천'
}

# 2) 도매시장 이름에 대해 매핑 or 앞 2글자 처리 함수
def map_or_prefix(market_name):
    if market_name in special_map:
        return special_map[market_name]
    else:
        return market_name[:2]

# 3) df2['도매시장']의 유니크 값 리스트
unique_market_list = df2['도매시장'].unique().tolist()

# 4) 매핑 또는 앞 2글자 처리 이후 집합으로 변환하여 중복 제거
mapped_list = [map_or_prefix(market_name) for market_name in unique_market_list]
mapped_set = set(mapped_list)

# 5) df_filtered를 '지점명' 기준으로 필터링
df_filtered_mapped = df_filtered[df_filtered['지점명'].isin(mapped_set)]

l1 = sorted(df_filtered_mapped['지점명'].unique().tolist())
l2 = sorted(mapped_set)

# l1과 l2가 같은지 확인
are_equal = set(l1) == set(l2)

# 결과 출력
print("l1과 l2가 같은지 여부:", are_equal)

os.makedirs(processed_dir, exist_ok=True)
output_file = os.path.join(processed_dir, 'preprocessed_climate.csv')
df_filtered_mapped.to_csv(output_file, index=False)

# 결과 확인
print("df_filtered_mapped 크기:", df_filtered_mapped.shape)
print(f"CSV 파일로 저장됨: {output_file}")