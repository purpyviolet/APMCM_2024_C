import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import SelectFromModel
file_path = 'Data1.xlsx'
china_data = pd.read_excel(file_path, sheet_name='中国')

cats_series = china_data['猫(万)']
dogs_series = china_data['狗(万)']
years = china_data['宠物/年份'].astype(int)
scaler_cats = MinMaxScaler()
scaler_dogs = MinMaxScaler()

cats_series_normalized = scaler_cats.fit_transform(cats_series.values.reshape(-1, 1)).flatten()
dogs_series_normalized = scaler_dogs.fit_transform(dogs_series.values.reshape(-1, 1)).flatten()

def create_lagged_features(series, lag=3):
    df = pd.DataFrame({'y': series})
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    return df.dropna()

cats_features = create_lagged_features(cats_series_normalized)
dogs_features = create_lagged_features(dogs_series_normalized)

X_cats = cats_features.drop('y', axis=1)
y_cats = cats_features['y']

X_dogs = dogs_features.drop('y', axis=1)
y_dogs = dogs_features['y']

def select_top_features(X, y, top_n=10):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-top_n:]  # 选择前 top_n 个特征
    selected_features = X.columns[indices]
    return selected_features

# 获取重要特征
top_features_cats = select_top_features(X_cats, y_cats, top_n=10)
top_features_dogs = select_top_features(X_dogs, y_dogs, top_n=10)

X_cats = X_cats[top_features_cats]
X_dogs = X_dogs[top_features_dogs]

X_cats_train, X_cats_test, y_cats_train, y_cats_test = train_test_split(X_cats, y_cats, test_size=0.2, random_state=42)
X_dogs_train, X_dogs_test, y_dogs_train, y_dogs_test = train_test_split(X_dogs, y_dogs, test_size=0.2, random_state=42)

model_cats = RandomForestRegressor(n_estimators=100, random_state=42)
model_dogs = RandomForestRegressor(n_estimators=100, random_state=42)

loo = LeaveOneOut()
cv_cats = cross_val_score(model_cats, X_cats, y_cats, cv=loo, scoring='neg_mean_squared_error')
cv_dogs = cross_val_score(model_dogs, X_dogs, y_dogs, cv=loo, scoring='neg_mean_squared_error')

model_cats.fit(X_cats_train, y_cats_train)
model_dogs.fit(X_dogs_train, y_dogs_train)

# 创建未来特征用于预测未来三年
def create_future_features(series, model, steps, lag=3):
    future_features = []
    current_values = list(series[-lag:])  # 从最近的滞后值开始
    for _ in range(steps):
        features = np.array(current_values[-lag:]).reshape(1, -1)
        next_value = model.predict(features)[0]
        future_features.append(next_value)
        current_values.append(next_value)  # 更新当前值
    return future_features

cats_future_predictions_normalized = create_future_features(cats_series_normalized, model_cats, steps=3, lag=len(top_features_cats))
dogs_future_predictions_normalized = create_future_features(dogs_series_normalized, model_dogs, steps=3, lag=len(top_features_dogs))

cats_future_predictions = scaler_cats.inverse_transform(np.array(cats_future_predictions_normalized).reshape(-1, 1)).flatten()
dogs_future_predictions = scaler_dogs.inverse_transform(np.array(dogs_future_predictions_normalized).reshape(-1, 1)).flatten()

print("Cats Forecast (2024-2026):", cats_future_predictions)
print("Dogs Forecast (2024-2026):", dogs_future_predictions)

# 预测
y_cats_pred_normalized = model_cats.predict(X_cats_test)
y_dogs_pred_normalized = model_dogs.predict(X_dogs_test)

mse_cats_normalized = mean_squared_error(y_cats_test, y_cats_pred_normalized)
mse_dogs_normalized = mean_squared_error(y_dogs_test, y_dogs_pred_normalized)

print(f"Cats Model MSE (Normalized): {mse_cats_normalized:.6f}")
print(f"Dogs Model MSE (Normalized): {mse_dogs_normalized:.6f}")

years_extended = np.append(years, [2024, 2025, 2026])

cats_series_extended = np.append(cats_series.values, cats_future_predictions)
dogs_series_extended = np.append(dogs_series.values, dogs_future_predictions)

plt.figure(figsize=(10, 6))

# 猫的时间序列及预测
plt.plot(years_extended, cats_series_extended, label="Cats (Actual + Forecast)", linestyle='-', marker='o')

# 狗的时间序列及预测
plt.plot(years_extended, dogs_series_extended, label="Dogs (Actual + Forecast)", linestyle='-', marker='s')

plt.title("China Pet Industry Prediction (2019-2026) Using Random Forest", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Pets (10,000s)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
