import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# 定义误差计算函数
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 1. 数据定义
years = np.array([2019, 2020, 2021, 2022, 2023])  # 历史年份
cat_count = np.array([1470, 1570, 1670, 1520, 1570])  # 猫数量（千）
dog_count = np.array([1010, 1070, 1030, 1060, 1050])  # 狗数量（千）
total_pet_count = np.array([34.3, 35.0, 36.0, 37.2, 38.5])  # 宠物总数量（百万）
pet_food_expenditure = np.array([3.0, 3.2, 3.4, 3.6, 3.8])  # 宠物食品开支（亿美元）
gdp_per_capita = np.array([46805.14, 46749.48, 51426.75, 48717.99, 52745.76])  # 人均GDP（现价美元）
population_growth = np.array([0.226, 0.082, 0.042, 0.721, 0.813])  # 人口增长率（年度百分比）
import_volume_index = np.array([106.47, 99.90, 108.21, 109.72, 111.24])  # 进口物量指数（2015年 = 100）
food_production_index = np.array([94.41, 95.45, 94.49, 92.71, 96.23])  # 食品生产指数（2014-2016 = 100）

# 2. 预测未来三年各指标
years_future = np.array([2024, 2025, 2026])
all_years = np.concatenate((years, years_future))

# 2.1 人口增长率 - 假设未来三年平均增长率为 0.3%
population_growth_future = np.array([0.3, 0.3, 0.3])

# 预测未来三年的人口增长率和人均GDP
gdp_per_capita_future = gdp_per_capita[-1] * (1 + 0.05) ** np.arange(1, 4)  # 人均GDP年增长率5%

# 构建多元线性回归模型
X = np.column_stack((population_growth, gdp_per_capita))
X_future = np.column_stack((population_growth_future, gdp_per_capita_future))

# 线性回归模型预测
def linear_regression_forecast(X, y, X_future):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X_future)

pet_food_expenditure_regression_future = linear_regression_forecast(X, pet_food_expenditure, X_future)

# 多项式回归预测
poly_order = 2
def polynomial_regression_forecast(years, y, years_future, poly_order):
    poly = PolynomialFeatures(degree=poly_order)
    X_poly = poly.fit_transform(years.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    X_future_poly = poly.fit_transform(years_future.reshape(-1, 1))
    return model.predict(X_future_poly)

pet_food_expenditure_poly_future = polynomial_regression_forecast(years, pet_food_expenditure, years_future, poly_order)

# 非线性回归预测（指数回归示例）
def log_model(x, a, b):
    return a * np.log(x) + b

def nonlinear_regression_forecast(years, y, years_future):
    try:
        popt, _ = curve_fit(log_model, years, y, maxfev=10000)
        return log_model(years_future, *popt)
    except RuntimeError as e:
        print(f"Nonlinear regression failed: {e}")
        return np.full(len(years_future), np.nan)

pet_food_expenditure_nonlinear_future = nonlinear_regression_forecast(years, pet_food_expenditure, years_future)

# ARIMA模型预测
def arima_forecast(y, steps):
    model = ARIMA(y, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

pet_food_expenditure_arima_future = arima_forecast(pet_food_expenditure, len(years_future))

# 随机森林回归预测
def random_forest_forecast(X, y, X_future):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

pet_food_expenditure_rf_future = random_forest_forecast(X, pet_food_expenditure, X_future)

# 梯度提升树回归预测
def gradient_boosting_forecast(X, y, X_future):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

pet_food_expenditure_gb_future = gradient_boosting_forecast(X, pet_food_expenditure, X_future)

# 支持向量回归预测
def svr_forecast(X, y, X_future):
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X, y)
    return model.predict(X_future)

pet_food_expenditure_svr_future = svr_forecast(X, pet_food_expenditure, X_future)

# 神经网络回归预测
def mlp_forecast(X, y, X_future):
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

pet_food_expenditure_mlp_future = mlp_forecast(X, pet_food_expenditure, X_future)

# 优化加权平均预测
w0 = np.repeat(1/8, 8)

def objective(w):
    weighted_pred = (
        w[0] * pet_food_expenditure_regression_future +
        w[1] * pet_food_expenditure_poly_future +
        w[2] * pet_food_expenditure_nonlinear_future +
        w[3] * pet_food_expenditure_arima_future +
        w[4] * pet_food_expenditure_rf_future +
        w[5] * pet_food_expenditure_gb_future +
        w[6] * pet_food_expenditure_svr_future +
        w[7] * pet_food_expenditure_mlp_future
    )
    return np.mean((weighted_pred - pet_food_expenditure[-len(years_future):]) ** 2)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

bounds = [(0, 1)] * 8

result = minimize(objective, w0, bounds=bounds, constraints=constraints, method='SLSQP')
optimal_weights = result.x

pet_food_expenditure_weighted_future = (
    optimal_weights[0] * pet_food_expenditure_regression_future +
    optimal_weights[1] * pet_food_expenditure_poly_future +
    optimal_weights[2] * pet_food_expenditure_nonlinear_future +
    optimal_weights[3] * pet_food_expenditure_arima_future +
    optimal_weights[4] * pet_food_expenditure_rf_future +
    optimal_weights[5] * pet_food_expenditure_gb_future +
    optimal_weights[6] * pet_food_expenditure_svr_future +
    optimal_weights[7] * pet_food_expenditure_mlp_future
)

# 将原始数据和预测数据合并
pet_food_expenditure_regression = np.concatenate((pet_food_expenditure, pet_food_expenditure_regression_future))
pet_food_expenditure_poly = np.concatenate((pet_food_expenditure, pet_food_expenditure_poly_future))
pet_food_expenditure_nonlinear = np.concatenate((pet_food_expenditure, pet_food_expenditure_nonlinear_future))
pet_food_expenditure_arima = np.concatenate((pet_food_expenditure, pet_food_expenditure_arima_future))
pet_food_expenditure_rf = np.concatenate((pet_food_expenditure, pet_food_expenditure_rf_future))
pet_food_expenditure_gb = np.concatenate((pet_food_expenditure, pet_food_expenditure_gb_future))
pet_food_expenditure_svr = np.concatenate((pet_food_expenditure, pet_food_expenditure_svr_future))
pet_food_expenditure_mlp = np.concatenate((pet_food_expenditure, pet_food_expenditure_mlp_future))
pet_food_expenditure_weighted = np.concatenate((pet_food_expenditure, pet_food_expenditure_weighted_future))

models = {
'Linear Regression': pet_food_expenditure_regression,
'Polynomial Regression': pet_food_expenditure_poly,
'Nonlinear Regression': pet_food_expenditure_nonlinear,
'ARIMA': pet_food_expenditure_arima,
'Random Forest': pet_food_expenditure_rf,
'Gradient Boosting': pet_food_expenditure_gb,
'SVR': pet_food_expenditure_svr,
# 'MLP': pet_food_expenditure_mlp,
'Optimized Weighted': pet_food_expenditure_weighted
}

plt.figure(figsize=(12, 8))
for name, predictions in models.items():
    plt.plot(np.concatenate((years, years_future)), predictions, label=name)

plt.plot(years, pet_food_expenditure, 'ko', label='Actual Data')
plt.title('Pet Food Expenditure Predictions of France')
plt.xlabel('Year')
plt.ylabel('Pet Food Expenditure (Billion USD)')
plt.legend()
plt.grid(True)
plt.savefig('model_predictions_Q2_France.png', dpi=300, bbox_inches='tight')
plt.show()

mse_values = []
mape_values = []
model_names = []

future_predictions = {
'Linear Regression': pet_food_expenditure_regression_future,
'Polynomial Regression': pet_food_expenditure_poly_future,
'Nonlinear Regression': pet_food_expenditure_nonlinear_future,
'ARIMA': pet_food_expenditure_arima_future,
'Random Forest': pet_food_expenditure_rf_future,
'Gradient Boosting': pet_food_expenditure_gb_future,
'SVR': pet_food_expenditure_svr_future,
# 'MLP': pet_food_expenditure_mlp_future,
'Optimized Weighted': pet_food_expenditure_weighted_future
}

for name, predictions in future_predictions.items():
    mse_values.append(mse(pet_food_expenditure[-len(years_future):], predictions))  # 只计算未来三年的MSE
    mape_values.append(mape(pet_food_expenditure[-len(years_future):], predictions))  # 只计算未来三年的MAPE
    model_names.append(name)

plt.figure(figsize=(16, 6))  # 增加图表的宽度

# MSE Comparison
plt.subplot(1, 2, 1)
plt.bar(model_names, mse_values, color='skyblue')
plt.title('MSE Comparison')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签45度，并向右对齐

# MAPE Comparison
plt.subplot(1, 2, 2)
plt.bar(model_names, mape_values, color='lightgreen')
plt.title('MAPE Comparison')
plt.xlabel('Model')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签45度，并向右对齐
plt.tight_layout()  # 调整子图布局，以防止标签重叠
plt.savefig('model_comparison_Q2_France.png', dpi=300, bbox_inches='tight')
plt.show()
print(pet_food_expenditure_weighted_future)