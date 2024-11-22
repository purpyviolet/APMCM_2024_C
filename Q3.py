import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
file_path = 'Data1.xlsx'
china_data = pd.read_excel(file_path, sheet_name='中国')

# 数据探索与预处理
# print(china_data.head())
features = ['中国宠物食品出口总值（美元）', '人口增长（年度百分比）', '农村人口', '出生人口性别比（每1000名男性的女性）',
            '向高收入经济体的商品出口（占商品出口总额的百分比）', '宠物市场规模 (亿美元)', '总税率（占商业利润的百分比）',
            '狗(万)', '网络影响力指数', '食品出口（占商品出口的百分比）']
target_production = '中国宠物食品总产值（人民币）'
target_export = '中国宠物食品出口总值（美元）'

X = china_data[features]
y_production = china_data[target_production]
y_export = china_data[target_export]

X.fillna(X.mean(), inplace=True)

for feature in X.columns:
    X[f'{feature}_diff'] = X[feature].diff().fillna(0)

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train_prod, y_test_prod = train_test_split(X_normalized, y_production, test_size=0.2, random_state=42)

_, _, y_train_exp, y_test_exp = train_test_split(X_normalized, y_export, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_search_prod = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_prod.fit(X_train, y_train_prod)

best_gbr_prod = grid_search_prod.best_estimator_

y_pred_prod = best_gbr_prod.predict(X_test)
mse_prod = mean_squared_error(y_test_prod, y_pred_prod)

future_data_prod = X_normalized[-1].reshape(1, -1)
future_predictions_prod = []

for year in range(3):
    next_year_pred = best_gbr_prod.predict(future_data_prod)[0]
    future_predictions_prod.append(next_year_pred)
    future_data_prod = np.append(future_data_prod[:, 1:], [[next_year_pred]], axis=1)

years = china_data['宠物/年份']
years_extended = np.append(years, [2024, 2025, 2026])

plt.figure(figsize=(12, 6))
plt.plot(years_extended, np.append(y_production, future_predictions_prod),
         label="China Pet Food Production (Actual + Forecast)", linestyle='-', marker='o', color='b')
plt.fill_between([2024, 2025, 2026], np.array(future_predictions_prod) * 0.95, np.array(future_predictions_prod) * 1.05,
                 color='blue', alpha=0.1)
plt.title("China Pet Food Production Prediction (2019-2026)", fontsize=16, weight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Production Value (RMB)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

grid_search_exp = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_exp.fit(X_train, y_train_exp)
best_gbr_exp = grid_search_exp.best_estimator_

y_pred_exp = best_gbr_exp.predict(X_test)
mse_exp = mean_squared_error(y_test_exp, y_pred_exp)

future_data_exp = X_normalized[-1].reshape(1, -1)
future_predictions_exp = []

for year in range(3):
    next_year_pred = best_gbr_exp.predict(future_data_exp)[0]
    future_predictions_exp.append(next_year_pred)
    future_data_exp = np.append(future_data_exp[:, 1:], [[next_year_pred]], axis=1)

plt.figure(figsize=(12, 6))
plt.plot(years_extended, np.append(y_export, future_predictions_exp),
         label="China Pet Food Export (Actual + Forecast)", linestyle='-', marker='o', color='g')
plt.fill_between([2024, 2025, 2026], np.array(future_predictions_exp) * 0.95, np.array(future_predictions_exp) * 1.05,
                 color='green', alpha=0.1)
plt.title("China Pet Food Export Prediction (2019-2026)", fontsize=16, weight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Export Value (USD)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nFuture Predictions for China Pet Food Production (2024-2026):")
for year, value in zip(range(2024, 2027), future_predictions_prod):
    print(f"Year {year}: {value:.2f} RMB")

print("\nFuture Predictions for China Pet Food Export (2024-2026):")
for year, value in zip(range(2024, 2027), future_predictions_exp):
    print(f"Year {year}: {value:.2f} USD")

