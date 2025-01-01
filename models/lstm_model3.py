import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import json
from multiprocessing import Pool, freeze_support

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'data', 'last_clustered_seoul.csv')  # CSV 파일 경로
saved_models_dir = os.path.join(base_dir, 'saved_models', 'LSTM')
os.makedirs(saved_models_dir, exist_ok=True)

# 데이터 불러오기
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")
data = pd.read_csv(data_path)

# 데이터 정렬
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['item', 'date'])

# 계절 변수 원핫 인코딩
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(data[['season']])
season_columns = encoder.get_feature_names_out(['season'])
season_df = pd.DataFrame(season_encoded, columns=season_columns)
data = pd.concat([data.reset_index(drop=True), season_df.reset_index(drop=True)], axis=1)

# 데이터 전처리 함수 (필요한 특성만 사용)
def preprocess_data(cluster_data, target_col, sequence_length=30):
    scaler = StandardScaler()
    feature_columns = [
        'total_quantity', 'temperature_range',
        'average_temperature', 'daily_rainfall', 'sales_amount',
        'quantity_price_ratio', 'price_volatility', 'quantity_volatility'
    ] + list(season_columns)

    features = cluster_data[feature_columns]
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
        yield X[train_index], X[test_index], y[train_index], y[test_index]

# 모델 성능 시각화 저장
def plot_training_history(history, cluster_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
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
    model = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    return model

# 모델 정의 기록 함수
def log_model_definition(cluster_dir, lstm_units, dropout_rate):
    model_definition = {
        'LSTM_Units': lstm_units,
        'Dropout_Rate': dropout_rate,
        'Optimizer': 'adam',
        'Loss': 'mse',
        'Metrics': ['mae', 'mape']
    }
    definition_path = os.path.join(cluster_dir, 'model_definition.json')
    with open(definition_path, 'w') as f:
        json.dump(model_definition, f, indent=4)
    print(f"Model definition saved at {definition_path}.")

# 모델 성능 기록 함수
def log_model_performance(log_path, cluster, params, train_loss, train_mae, train_mape, val_loss, val_mae, r2, rmse, mape):
    record = {
        'Cluster': cluster,
        'LSTM_Units': params['lstm_units'],
        'Dropout_Rate': params['dropout_rate'],
        'Batch_Size': params['batch_size'],
        'Epochs': params['epochs'],
        'Train_Loss': train_loss,
        'Train_MAE': train_mae,
        'Train_MAPE': train_mape,
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

# 클러스터별 학습 함수
def train_single_cluster(cluster_data, cluster, hyperparams, log_path, target_col):
    for params in hyperparams:
        print(f"\nTraining for Cluster {cluster} with params {params}...")
        X, y, scaler = preprocess_data(cluster_data, target_col, sequence_length=30)

        for X_train, X_test, y_train, y_test in split_time_series(X, y):
            # 모델 생성
            model = build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate']
            )

            # 콜백 정의
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True)

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
            train_loss, train_mae, train_mape = model.evaluate(X_train, y_train, verbose=0)
            y_pred = model.predict(X_test)
            r2 = np.corrcoef(y_test, y_pred.flatten())[0, 1] ** 2
            rmse = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))
            mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

            val_loss, val_mae, _ = model.evaluate(X_test, y_test, verbose=0)
            log_model_performance(log_path, cluster, params, train_loss, train_mae, train_mape, val_loss, val_mae, r2, rmse, mape)

            # 모델 및 시각화 저장
            cluster_dir = os.path.join(saved_models_dir, f'cluster_{cluster}_units_{params["lstm_units"]}_dropout_{params["dropout_rate"]}_epochs_{params["epochs"]}_batch_{params["batch_size"]}')
            os.makedirs(cluster_dir, exist_ok=True)
            model.save(os.path.join(cluster_dir, 'lstm_model.keras'))
            log_model_definition(cluster_dir, params['lstm_units'], params['dropout_rate'])
            scaler_path = os.path.join(cluster_dir, "scaler.npy")
            np.save(scaler_path, {'mean': scaler.mean_, 'scale': scaler.scale_})
            plot_training_history(history, cluster_dir)
            print(f"Model for Cluster {cluster} saved at {cluster_dir}")

# 병렬 처리로 클러스터별 학습
def train_lstm_in_parallel(data, cluster_col, target_col, hyperparams, log_path):
    clusters = data[cluster_col].unique()
    cluster_data_list = [(data[data[cluster_col] == cluster], cluster, hyperparams, log_path, target_col) for cluster in clusters]

    with Pool(processes=4) as pool:  # 병렬 처리 프로세스 개수
        pool.starmap(train_single_cluster, cluster_data_list)

# 실행
if __name__ == '__main__':
    freeze_support()  # Windows 환경에서 병렬 처리 안전 실행
    hyperparams = [
        {'lstm_units': 50, 'dropout_rate': 0.2, 'epochs': 50, 'batch_size': 32},
        {'lstm_units': 100, 'dropout_rate': 0.3, 'epochs': 100, 'batch_size': 64}
    ]

    target_column = 'average_price'
    log_path = os.path.join(saved_models_dir, 'model_performance_log.csv')
    train_lstm_in_parallel(data, 'cluster', target_column, hyperparams, log_path)
