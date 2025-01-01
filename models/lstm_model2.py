# 표준 라이브러리
import os

# 서드파티 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit  # Used for splitting time-series data into train and test sets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import json

# 프로젝트 루트 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'data', 'processed', 'daily_average_price_volume_by_product_seoul.csv')
clustering_path = os.path.join(base_dir, 'output', 'clustering', 'clustering_results.csv')
saved_models_dir = os.path.join(base_dir, 'saved_models', 'LSTM')
os.makedirs(saved_models_dir, exist_ok=True)

# 데이터 불러오기
data = pd.read_csv(data_path)
clusters = pd.read_csv(clustering_path)

# 데이터와 군집 결과 병합
data = data.merge(clusters[['Item', 'Cluster']], left_on='품목', right_on='Item', how='inner')  # Merge clustering results with main data
data.drop(columns=['Item'], inplace=True)

# 파생 변수 생성 함수
def create_derived_features(data):
    data['매출액'] = data['총물량'] * data['평균단가']
    data['물량_단가_비율'] = data['총물량'] / data['평균단가']
    data['단가_변동성'] = data.groupby('품목')['평균단가'].pct_change().fillna(0)
    data['물량_변동성'] = data.groupby('품목')['총물량'].pct_change().fillna(0)
    return data

# 이상치 제거 및 파생 변수 생성
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

data = remove_outliers(data, '평균단가')
data = create_derived_features(data)

# 데이터 전처리 함수 개선
def preprocess_data(cluster_data, target_col, sequence_length=30):
    scaler = StandardScaler()

    features = cluster_data.drop(columns=['일자', '품목', '계절', 'Cluster', target_col])
    target = cluster_data[target_col]

    features_scaled = scaler.fit_transform(features)
    sequences, targets = [], []

    for i in range(len(features_scaled) - sequence_length):
        sequences.append(features_scaled[i:i + sequence_length])
        targets.append(target.values[i + sequence_length])

    return np.array(sequences), np.array(targets), scaler

# 훈련 및 검증 데이터 분리 (TimeSeriesSplit 사용)
def split_time_series(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

# 모델 성능 시각화 저장
def plot_training_history(history, cluster_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')  # Label for the x-axis showing training epochs
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(cluster_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved at {plot_path}.")

# LSTM 모델 정의 함수
def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 모델 정의 기록 함수
def log_model_definition(cluster_dir, lstm_units, dropout_rate):
    model_definition = {
        'LSTM_Units': lstm_units,
        'Dropout_Rate': dropout_rate,
        'Optimizer': 'adam',
        'Loss': 'mse',
        'Metrics': ['mae']
    }
    definition_path = os.path.join(cluster_dir, 'model_definition.json')
    with open(definition_path, 'w') as f:
        json.dump(model_definition, f, indent=4)
    print(f"Model definition saved at {definition_path}.")

# 모델 성능 기록 함수
def log_model_performance(log_path, cluster, params, train_loss, train_mae, val_loss, val_mae, r2, rmse, mape):
    record = {
        'Cluster': cluster,
        'LSTM_Units': params['lstm_units'],
        'Dropout_Rate': params['dropout_rate'],
        'Batch_Size': params['batch_size'],
        'Epochs': params['epochs'],
        'Train_Loss': train_loss,
        'Train_MAE': train_mae,
        'Validation_Loss': val_loss,
        'Validation_MAE': val_mae,
        'R2_Score': r2,
        'RMSE': rmse,
        'MAPE': mape
    }
    if not os.path.exists(log_path):
        pd.DataFrame([record]).to_csv(log_path, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame([record]).to_csv(log_path, index=False, mode='a', header=False, encoding='utf-8-sig')
    if not os.path.exists(log_path):
        pd.DataFrame([record]).to_csv(log_path, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame([record]).to_csv(log_path, index=False, mode='a', header=False, encoding='utf-8-sig')

# 하이퍼파라미터 반복 실행 훈련 함수
def train_lstm_clusters_with_hyperparams(data, cluster_col, target_col, sequence_length=30, hyperparams=None):
    if hyperparams is None:
        hyperparams = [
            {'lstm_units': 50, 'dropout_rate': 0.2, 'epochs': 50, 'batch_size': 32},
            {'lstm_units': 100, 'dropout_rate': 0.3, 'epochs': 100, 'batch_size': 64}
        ]

    clusters = data[cluster_col].unique()
    log_path = os.path.join(saved_models_dir, 'model_performance_log.csv')

    for cluster in clusters:
        cluster_data = data[data[cluster_col] == cluster].sort_values('일자')

        for params in hyperparams:
            print(f"\nTraining for Cluster {cluster} with params {params}...")
            X, y, scaler = preprocess_data(cluster_data, target_col, sequence_length)

            # 시계열 검증 데이터 분리
            X_train, X_test, y_train, y_test = split_time_series(X, y)

            # 모델 생성
            model = build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate']
            )

            # 콜백 정의
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # 모델 훈련
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=1
            )

            # 성능 기록
            train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
            y_pred = model.predict(X_test)
            r2 = np.corrcoef(y_test, y_pred.flatten())[0, 1] ** 2
            rmse = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))
            mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
            val_loss, val_mae = model.evaluate(X_test, y_test, verbose=0)
            log_model_performance(log_path, cluster, params, train_loss, train_mae, val_loss, val_mae, r2, rmse, mape)

            # 모델 및 시각화 저장
            cluster_dir = os.path.join(saved_models_dir, f'cluster_{cluster}_units_{params['lstm_units']}_dropout_{params['dropout_rate']}_epochs_{params['epochs']}_batch_{params['batch_size']}')
            os.makedirs(cluster_dir, exist_ok=True)
            model.save(os.path.join(cluster_dir, 'lstm_model.keras'))
            log_model_definition(cluster_dir, params['lstm_units'], params['dropout_rate'])
            np.save(os.path.join(cluster_dir, 'scaler.npy'), scaler.mean_)
            plot_training_history(history, cluster_dir)

            print(f"Model for Cluster {cluster} with params {params} saved at {cluster_dir}.")

# 훈련 실행
hyperparams = [
    {'lstm_units': 50, 'dropout_rate': 0.2, 'epochs': 50, 'batch_size': 32},
    {'lstm_units': 100, 'dropout_rate': 0.3, 'epochs': 100, 'batch_size': 64}
]

target_column = '평균단가'
train_lstm_clusters_with_hyperparams(data, 'Cluster', target_column, hyperparams=hyperparams)
