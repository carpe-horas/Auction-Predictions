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
from sklearn.metrics import silhouette_score

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

# 이너셔와 실루엣 점수 평가 함수
def evaluate_clustering(data, max_clusters=10):
    inertias = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        
        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
    
    evaluation_df = pd.DataFrame({
        'Number of Clusters': list(cluster_range),
        'Inertia': inertias,
        'Silhouette Score': silhouette_scores
    })
    return evaluation_df

# 평가 수행
evaluation_df = evaluate_clustering(pca_data, max_clusters=10)

# 평가 결과 저장
evaluation_path = os.path.join(clustering_dir, 'clustering_evaluation.csv')
evaluation_df.to_csv(evaluation_path, index=False, encoding='utf-8-sig')
print(f"Clustering evaluation results saved to: {evaluation_path}")

# 이너셔 및 실루엣 점수 시각화
plt.figure(figsize=(14, 7))

# 이너셔 시각화
plt.subplot(1, 2, 1)
plt.plot(evaluation_df['Number of Clusters'], evaluation_df['Inertia'], marker='o', label='Inertia')
plt.title('Inertia vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid()
plt.legend()

# 실루엣 점수 시각화
plt.subplot(1, 2, 2)
plt.plot(evaluation_df['Number of Clusters'], evaluation_df['Silhouette Score'], marker='o', color='orange', label='Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.legend()

# 시각화 저장
evaluation_plot_path = os.path.join(clustering_dir, 'clustering_evaluation_plot.png')
plt.savefig(evaluation_plot_path)
plt.close()

print(f"Evaluation plots saved to: {evaluation_plot_path}")

# 결과 저장 및 모델 저장 호출
save_clustering_results(pca_data, labels, clustering_dir, item_level_data.index)
