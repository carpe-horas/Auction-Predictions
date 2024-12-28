# import os
# import pandas as pd
# from tkinter import Tk, filedialog

# # 현재 스크립트 상위 폴더 경로
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)  
# data_dir = os.path.join(project_root, 'data')
# raw_dir = os.path.join(data_dir, 'raw')
# processed_dir = os.path.join(data_dir, 'processed')

# # Tkinter GUI 비활성화
# root = Tk()
# root.withdraw()  # Tkinter GUI 창 숨기기
# root.attributes('-topmost', True)  # Tk 창을 맨 위로

# # 폴더 선택 다이얼로그를 raw_dir로 기본 설정
# folder_path = filedialog.askdirectory(title="폴더를 선택하세요", initialdir=raw_dir)

# # 폴더 안의 모든 .xls, .xlsx 파일 리스트
# excel_files = [f for f in os.listdir(folder_path) 
#                if f.endswith('.xls') or f.endswith('.xlsx')]

# all_dfs = []
# columns_reference = None  # 모든 파일의 칼럼명이 같은지 비교하기 위한 레퍼런스

# for excel_file in excel_files:
#     file_path = os.path.join(folder_path, excel_file)
#     df = pd.read_excel(file_path)
    
#     # (1) 빈 파일 체크
#     if df.empty:
#         print(f"[주의] 빈 파일 발견: {excel_file} (병합에서 제외)")
#         continue  # 비어있는 파일은 병합하지 않고 넘어감
    
#     # (2) 칼럼명 기준 설정 및 불일치 체크
#     if columns_reference is None:
#         columns_reference = list(df.columns)
#     else:
#         if list(df.columns) != columns_reference:
#             print(f"[주의] 칼럼 불일치 발생: {excel_file}")
#             print(f"기준 칼럼      : {columns_reference}")
#             print(f"현재 파일 칼럼 : {list(df.columns)}\n")
#             raise ValueError(f"칼럼명이 일치하지 않아 병합할 수 없습니다. 파일: {file_path}")
    
#     # (3) 문제 없으면 병합 리스트에 추가
#     all_dfs.append(df)

# # 모든 파일에서 문제가 없으면 DataFrame 병합
# merged_df = pd.concat(all_dfs, ignore_index=True)

# print("병합된 데이터프레임 크기:", merged_df.shape)

# # 1) '법인' 칼럼 drop
# merged_df.drop(columns=['법인'], inplace=True)

# # 2) '품목' 별 '물량(kg)' 상위 30개 품목만 필터링
# top30_items = (
#     merged_df.groupby('품목')['물량(kg)']
#     .sum()
#     .sort_values(ascending=False)
#     .head(30)
#     .index
# )
# merged_df = merged_df[merged_df['품목'].isin(top30_items)]

# # 3) 0 이하 데이터(0과 음수 값) 제거
# merged_df = merged_df[
#     (merged_df['물량(kg)'] > 0) &
#     (merged_df['금액(원)'] > 0)
# ]

# # 4) '부류', '품목' 칼럼의 결측치 데이터 삭제
# merged_df.dropna(subset=['부류', '품목'], inplace=True)

# # CSV로 저장
# os.makedirs(processed_dir, exist_ok=True)
# output_file = os.path.join(processed_dir, 'preprocessed_agromarket.csv')
# merged_df.to_csv(output_file, index=False)

# print(f"전처리 후 최종 데이터프레임 크기: {merged_df.shape}")
# print(f"CSV 파일로 저장됨: {output_file}")




import os
import pandas as pd

# 현재 스크립트 상위 폴더 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  
data_dir = os.path.join(project_root, 'data')
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')

# 폴더 경로를 직접 지정
folder_path = raw_dir  # raw_dir 경로를 기본 폴더로 사용

# 폴더 안의 모든 .xls, .xlsx 파일 리스트
excel_files = [f for f in os.listdir(folder_path) 
               if f.endswith('.xls') or f.endswith('.xlsx')]

all_dfs = []
columns_reference = None  # 모든 파일의 칼럼명이 같은지 비교하기 위한 레퍼런스

for excel_file in excel_files:
    file_path = os.path.join(folder_path, excel_file)
    df = pd.read_excel(file_path)
    
    # (1) 빈 파일 체크
    if df.empty:
        print(f"[주의] 빈 파일 발견: {excel_file} (병합에서 제외)")
        continue  # 비어있는 파일은 병합하지 않고 넘어감
    
    # (2) 칼럼명 기준 설정 및 불일치 체크
    if columns_reference is None:
        columns_reference = list(df.columns)
    else:
        if list(df.columns) != columns_reference:
            print(f"[주의] 칼럼 불일치 발생: {excel_file}")
            print(f"기준 칼럼      : {columns_reference}")
            print(f"현재 파일 칼럼 : {list(df.columns)}\n")
            raise ValueError(f"칼럼명이 일치하지 않아 병합할 수 없습니다. 파일: {file_path}")
    
    # (3) 문제 없으면 병합 리스트에 추가
    all_dfs.append(df)

# 모든 파일에서 문제가 없으면 DataFrame 병합
merged_df = pd.concat(all_dfs, ignore_index=True)

print("병합된 데이터프레임 크기:", merged_df.shape)

# 1) '법인' 칼럼 drop
merged_df.drop(columns=['법인'], inplace=True)

# 2) '품목' 별 '물량(kg)' 상위 30개 품목만 필터링
top30_items = (
    merged_df.groupby('품목')['물량(kg)']
    .sum()
    .sort_values(ascending=False)
    .head(30)
    .index
)
merged_df = merged_df[merged_df['품목'].isin(top30_items)]

# 3) 0 이하 데이터(0과 음수 값) 제거
merged_df = merged_df[
    (merged_df['물량(kg)'] > 0) & 
    (merged_df['금액(원)'] > 0)
]

# 4) '부류', '품목' 칼럼의 결측치 데이터 삭제
merged_df.dropna(subset=['부류', '품목'], inplace=True)

# CSV로 저장
os.makedirs(processed_dir, exist_ok=True)
output_file = os.path.join(processed_dir, 'preprocessed_agromarket.csv')
merged_df.to_csv(output_file, index=False)

print(f"전처리 후 최종 데이터프레임 크기: {merged_df.shape}")
print(f"CSV 파일로 저장됨: {output_file}")
