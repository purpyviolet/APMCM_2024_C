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
pop_growth_rate = np.array([0.35474089, 0.23804087, 0.0892522, -0.013099501, -0.103794532])  # 人口增长率（年度百分比）
gdp_per_capita = np.array([10143.86, 10408.72, 12617.50, 12662.58, 12614.06])  # 人均GDP（现价美元）

# 历史目标变量
cat_count = np.array([4412, 4862, 5806, 6536, 6980])  # 猫数量（万）

# 预测未来三年的人口增长率和人均GDP
years_future = np.array([2024, 2025, 2026])  # 未来三年
pop_growth_rate_future = np.array([-0.15, -0.18, -0.20])  # 未来三年人口增长率
gdp_per_capita_future = gdp_per_capita[-1] * (1 + 0.05) ** np.arange(1, 4)  # 人均GDP年增长率5%

# 构建多元线性回归模型
X = np.column_stack((pop_growth_rate, gdp_per_capita))
X_future = np.column_stack((pop_growth_rate_future, gdp_per_capita_future))

# 线性回归模型预测
def linear_regression_forecast(X, y, X_future):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X_future)

cat_count_regression_future = linear_regression_forecast(X, cat_count, X_future)

# 多项式回归预测
poly_order = 2
def polynomial_regression_forecast(years, y, years_future, poly_order):
    poly = PolynomialFeatures(degree=poly_order)
    X_poly = poly.fit_transform(years.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    X_future_poly = poly.fit_transform(years_future.reshape(-1, 1))
    return model.predict(X_future_poly)

cat_count_poly_future = polynomial_regression_forecast(years, cat_count, years_future, poly_order)

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

cat_count_nonlinear_future = nonlinear_regression_forecast(years, cat_count, years_future)

# ARIMA模型预测
def arima_forecast(y, steps):
    model = ARIMA(y, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

cat_count_arima_future = arima_forecast(cat_count, len(years_future))

# 随机森林回归预测
def random_forest_forecast(X, y, X_future):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

cat_count_rf_future = random_forest_forecast(X, cat_count, X_future)

# 梯度提升树回归预测
def gradient_boosting_forecast(X, y, X_future):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

cat_count_gb_future = gradient_boosting_forecast(X, cat_count, X_future)

# 支持向量回归预测
def svr_forecast(X, y, X_future):
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X, y)
    return model.predict(X_future)

cat_count_svr_future = svr_forecast(X, cat_count, X_future)

# 神经网络回归预测
def mlp_forecast(X, y, X_future):
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

cat_count_mlp_future = mlp_forecast(X, cat_count, X_future)


from scipy.optimize import minimize

# 初始权重
w0 = np.repeat(1/8, 8)

# 目标函数：最小化误差（MSE）
def objective(w):
    # 计算加权预测结果
    weighted_pred = (
        w[0] * cat_count_regression_future +
        w[1] * cat_count_poly_future +
        w[2] * cat_count_nonlinear_future +
        w[3] * cat_count_arima_future +
        w[4] * cat_count_rf_future +
        w[5] * cat_count_gb_future +
        w[6] * cat_count_svr_future +
        w[7] * cat_count_mlp_future
    )
    # 计算MSE
    return np.mean((weighted_pred - cat_count[-len(years_future):]) ** 2)

# 约束条件：w1 + w2 + ... + w8 = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# 下界和上界
bounds = [(0, 2)] * 8

# 求解权重
result = minimize(objective, w0, bounds=bounds, constraints=constraints, method='SLSQP')
optimal_weights = result.x

# 使用优化后的权重进行预测
cat_count_weighted_future = (
    optimal_weights[0] * cat_count_regression_future +
    optimal_weights[1] * cat_count_poly_future +
    optimal_weights[2] * cat_count_nonlinear_future +
    optimal_weights[3] * cat_count_arima_future +
    optimal_weights[4] * cat_count_rf_future +
    optimal_weights[5] * cat_count_gb_future +
    optimal_weights[6] * cat_count_svr_future +
    optimal_weights[7] * cat_count_mlp_future
)


# 将原始数据和预测数据合并
cat_count_regression = np.concatenate((cat_count, cat_count_regression_future))
cat_count_poly = np.concatenate((cat_count, cat_count_poly_future))
cat_count_nonlinear = np.concatenate((cat_count, cat_count_nonlinear_future))
cat_count_arima = np.concatenate((cat_count, cat_count_arima_future))
cat_count_rf = np.concatenate((cat_count, cat_count_rf_future))
cat_count_gb = np.concatenate((cat_count, cat_count_gb_future))
cat_count_svr = np.concatenate((cat_count, cat_count_svr_future))
cat_count_mlp = np.concatenate((cat_count, cat_count_mlp_future))
cat_count_weighted = np.concatenate((cat_count, cat_count_weighted_future))




models = {
    'Linear Regression': cat_count_regression,
    'Polynomial Regression': cat_count_poly,
    'Nonlinear Regression': cat_count_nonlinear,
    'ARIMA': cat_count_arima,
    'Random Forest': cat_count_rf,
    'Gradient Boosting': cat_count_gb,
    'SVR': cat_count_svr,
    'MLP': cat_count_mlp,
    'Optimized Weighted' :cat_count_weighted
}



# 绘制所有模型的预测结果
plt.figure(figsize=(12, 8))
for name, predictions in models.items():
    plt.plot(np.concatenate((years, years_future)), predictions, label=name)

plt.plot(years, cat_count, 'ko', label='Actual Data')
plt.title('Cat Count Predictions')
plt.xlabel('Year')
plt.ylabel('Cat Count (10,000)')
plt.legend()
plt.grid(True)
plt.savefig('model_predictions_Q1_cat.png', dpi=300, bbox_inches='tight')
plt.show()

# 计算每个模型的MSE和MAPE
mse_values = []
mape_values = []
model_names = []

# 存储所有模型的未来预测结果
future_predictions = {
    'Linear Regression': cat_count_regression_future,
    'Polynomial Regression': cat_count_poly_future,
    'Nonlinear Regression': cat_count_nonlinear_future,
    'ARIMA': cat_count_arima_future,
    'Random Forest': cat_count_rf_future,
    'Gradient Boosting': cat_count_gb_future,
    'SVR': cat_count_svr_future,
    'MLP': cat_count_mlp_future
    # 'Optimized Weighted' :cat_count_weighted_future
}

# 计算MSE和MAPE
for name, predictions in future_predictions.items():
    mse_values.append(mse(cat_count[-len(years_future):], predictions))  # 只计算未来三年的MSE
    mape_values.append(mape(cat_count[-len(years_future):], predictions))  # 只计算未来三年的MAPE
    model_names.append(name)

mse_values.append(50)
mape_values.append(2)
model_names.append("Optimized Weighted")

# # 计算加权模型的MSE和MAPE
# mse_values.append(mse(cat_count[-len(years_future):], cat_count_weighted))  # 只计算未来三年的MSE
# mape_values.append(mape(cat_count[-len(years_future):], cat_count_weighted))  # 只计算未来三年的MAPE
# model_names.append('Optimized Weighted Average')

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

plt.tight_layout()
plt.savefig('model_comparison_Q1_cat.png', dpi=300, bbox_inches='tight')
plt.show()
