# 라이브러리 불러오기
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
import joblib

# 한글 폰트 경로 설정
font_path = r"C:\Windows\Fonts\batang.ttc"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

print(f"설정된 폰트: {font_name}")

# 데이터 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
processed_dir = os.path.join(data_dir, 'processed')
output_dir = os.path.join(base_dir, 'output')
clustering_dir = os.path.join(output_dir, 'clustering')
os.makedirs(clustering_dir, exist_ok=True)
file_path = os.path.join(processed_dir, 'daily_average_price_volume_by_product_seoul.csv')

# 데이터 불러오기
df = pd.read_csv(file_path)

# 파생 변수 생성 함수
def create_derived_features(df):
    df['매출액'] = df['총물량'] * df['평균단가']
    df['물량_단가_비율'] = df['총물량'] / df['평균단가']
    df['단가_변동성'] = df.groupby('품목')['평균단가'].pct_change().fillna(0)
    return df

df = create_derived_features(df)

# 품목별 데이터 요약 함수
def summarize_data(df):
    summary = df.groupby('품목').agg({
        '총물량': ['mean', 'std', 'max', 'min'],
        '평균단가': ['mean', 'std', 'max', 'min'],
        '매출액': ['mean', 'std'],
        '물량_단가_비율': ['mean', 'std'],
        '단가_변동성': ['mean', 'std']
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary

item_level_data = summarize_data(df)

# 상관 관계 분석
correlation_matrix = item_level_data.corr()

# 상관 관계 시각화 저장
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for item_level_data")
correlation_matrix_path = os.path.join(clustering_dir, "correlation_matrix.png")
plt.savefig(correlation_matrix_path)
plt.close()

print("Correlation Matrix:")
print(correlation_matrix)

# 상관 관계를 기준으로 대표 변수 선택 함수
def select_features(correlation_matrix, threshold=0.8):
    selected_features = set()
    processed_features = set()
    for col in correlation_matrix.columns:
        if col not in processed_features:
            high_corr_features = correlation_matrix[col][correlation_matrix[col].abs() >= threshold].index
            selected_features.add(col)
            processed_features.update(high_corr_features)
    return list(selected_features)

# 대표 변수 선정
selected_features = select_features(correlation_matrix)
print("Selected Features for K-Means Clustering:", selected_features)

# 선택된 변수 필터링
filtered_data = item_level_data[selected_features]

# PCA 적용 함수
def apply_pca(data, n_components=2):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data_scaled), scaler, pca

# PCA 적용
pca_data, scaler, pca_model = apply_pca(filtered_data, n_components=2)

# K-Means 클러스터링
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(pca_data)

# 클러스터링 결과 저장 함수
def save_clustering_results(data, labels, clustering_dir, item_names):
    result_df = pd.DataFrame(data, columns=['PCA1', 'PCA2'])
    result_df['Item'] = item_names
    result_df['Cluster'] = labels
    result_df['Cluster'] = labels
    result_path = os.path.join(clustering_dir, 'clustering_results.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"Clustering results saved to: {result_path}")

    # 클러스터링 시각화 저장
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(result_df['PCA1'], result_df['PCA2'], c=result_df['Cluster'], cmap='viridis', alpha=0.7)
    for i, item in enumerate(result_df['Item']):
        plt.text(result_df['PCA1'].iloc[i], result_df['PCA2'].iloc[i], item, fontsize=8, alpha=0.8)
    plt.title("K-Means Clustering Visualization")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.colorbar(scatter, label='Cluster')
    plot_path = os.path.join(clustering_dir, 'clustering_visualization.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Clustering visualization saved to: {plot_path}")

# 모델 구성 요소 저장 함수
def save_model_components(scaler, pca_model, kmeans_model, selected_features, clustering_dir):
    joblib.dump(scaler, os.path.join(clustering_dir, 'scaler.pkl'))
    joblib.dump(pca_model, os.path.join(clustering_dir, 'pca_model.pkl'))
    joblib.dump(kmeans_model, os.path.join(clustering_dir, 'kmeans_model.pkl'))
    features_path = os.path.join(clustering_dir, 'selected_features.txt')
    with open(features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print("Model components saved.")

# 새로운 품종에 대한 군집 예측 함수
def predict_new_sample(new_sample, clustering_dir):
    scaler = joblib.load(os.path.join(clustering_dir, 'scaler.pkl'))
    pca_model = joblib.load(os.path.join(clustering_dir, 'pca_model.pkl'))
    kmeans_model = joblib.load(os.path.join(clustering_dir, 'kmeans_model.pkl'))
    features_path = os.path.join(clustering_dir, 'selected_features.txt')
    with open(features_path, 'r') as f:
        selected_features = [line.strip() for line in f]

    new_sample = create_derived_features(new_sample)  # 파생 변수 생성
    new_sample_summary = summarize_data(new_sample)  # 데이터 요약
    new_sample_filtered = new_sample_summary[selected_features]  # 필터링

    new_sample_scaled = scaler.transform(new_sample_filtered)
    new_sample_pca = pca_model.transform(new_sample_scaled)
    predictions = kmeans_model.predict(new_sample_pca)

    # 기존 결과 확인 및 추가
    existing_results_path = os.path.join(clustering_dir, 'clustering_results.csv')
    existing_results = pd.read_csv(existing_results_path)

    new_sample_summary['Cluster'] = predictions
    updated_results = pd.concat([existing_results, new_sample_summary]).drop_duplicates(subset=new_sample_summary.index.names, keep='first')
    updated_results.to_csv(existing_results_path, index=False, encoding='utf-8-sig')

    print(f"New sample predictions added to: {existing_results_path}")
    return predictions

# 결과 저장 및 모델 저장 호출
save_clustering_results(pca_data, labels, clustering_dir, item_level_data.index)
save_model_components(scaler, pca_model, kmeans, selected_features, clustering_dir)
