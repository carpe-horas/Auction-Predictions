import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# 데이터 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
processed_dir = os.path.join(data_dir, 'processed')
file_path = os.path.join(processed_dir, 'daily_average_price_volume_by_product_seoul.csv')

# 모델 저장 경로
saved_model_dir = os.path.join(base_dir, 'saved_models')
os.makedirs(saved_model_dir, exist_ok=True)

# 데이터 로드
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found at {file_path}")
data = pd.read_csv(file_path)

# 컬럼명 매핑
column_mapping = {
    '일자': 'date',
    '품목': 'product',
    '평균단가': 'average_price',
    '총물량': 'volume',
    # 기후 데이터가 필요한 경우 추가
    '평균기온(°C)': 'avg_temp',
    '최저기온(°C)': 'min_temp',
    '최고기온(°C)': 'max_temp',
    '일강수량(mm)': 'precipitation'
}
data.rename(columns=column_mapping, inplace=True)

# 데이터 전처리
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['product', 'date'])

# 성능 결과 저장용 리스트
performance_results = []

# 각 품목별로 처리
def preprocess_and_train(grouped_data, product_name):
    # 필요한 컬럼 선택 (기후 데이터를 포함할지 결정)
    data_subset = grouped_data[['date', 'average_price', 'volume']]

    # 결측치 처리
    data_subset = data_subset.ffill().bfill()

    # 스케일링
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_subset[['average_price', 'volume']])
    
    # 시계열 데이터 생성
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :])
            y.append(data[i+seq_length, 0])  # average_price 예측
        return np.array(X), np.array(y)

    sequence_length = 30
    X, y = create_sequences(scaled_data, sequence_length)

    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"Processing fold {fold} for product: {product_name}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 모델 구성
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 2)),
            LSTM(50, return_sequences=False),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # 체크포인트 설정
        checkpoint_path = os.path.join(saved_model_dir, f'{product_name}_fold{fold}_model.keras')
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

        # 모델 훈련
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])

        # 성능 평가
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        performance_results.append({"product": product_name, "fold": fold, "mse": mse})
        print(f"Performance for {product_name}, fold {fold}: MSE = {mse}")
        print(f"Model for {product_name}, fold {fold} saved at {checkpoint_path}")

        fold += 1

# 그룹화하여 각 품목별로 처리
grouped = data.groupby('product')

for product, group in grouped:
    print(f"Processing product: {product}")
    preprocess_and_train(group, product)

# 성능 결과 저장
performance_df = pd.DataFrame(performance_results)
performance_csv_path = os.path.join(saved_model_dir, 'performance_results.csv')
performance_df.to_csv(performance_csv_path, index=False)
print(f"Performance results saved to {performance_csv_path}")
