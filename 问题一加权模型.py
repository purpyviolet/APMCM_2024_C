import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
# 1. 数据定义
years = np.array([2019, 2020, 2021, 2022, 2023])  # 历史年份
pop_growth_rate = np.array([0.35474089, 0.23804087, 0.0892522, -0.013099501, -0.103794532])  # 人口增长率（年度百分比）
gdp_per_capita = np.array([10143.86, 10408.72, 12617.50, 12662.58, 12614.06])  # 人均GDP（现价美元）

# 历史目标变量
cat_count = np.array([4412, 4862, 5806, 6536, 6980])  # 猫数量（万）
dog_count = np.array([5503, 5222, 5429, 5119, 5175])  # 狗数量（万）
total_pet_count = np.array([9980, 10850, 11540, 12260, 13020])  # 总宠物数量（万）
market_size = np.array([33.2, 35.6, 38.9, 42.1, 45.5])  # 宠物市场规模（亿美元）
pet_medical_market_size = np.array([400, 500, 600, 640, 700])  # 宠物医疗市场规模（亿元人民币）

# 预测未来三年的人口增长率和人均GDP
years_future = np.array([2025, 2026, 2027])  # 未来三年
pop_growth_rate_future = np.array([-0.15, -0.18, -0.20])  # 未来三年人口增长率
gdp_per_capita_future = gdp_per_capita[-1] * (1 + 0.05) ** np.arange(1, 4)  # 人均GDP年增长率5%

# 合并历史和未来数据
all_years = np.concatenate([years, years_future])
all_pop_growth_rate = np.concatenate([pop_growth_rate, pop_growth_rate_future])
all_gdp_per_capita = np.concatenate([gdp_per_capita, gdp_per_capita_future])

# 构建多元线性回归模型
X = np.column_stack((pop_growth_rate, gdp_per_capita))
X_future = np.column_stack((pop_growth_rate_future, gdp_per_capita_future))
X_all = np.column_stack((all_pop_growth_rate, all_gdp_per_capita))

# 线性回归模型预测
def linear_regression_forecast(X, y, X_all):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X_all)

cat_count_regression = linear_regression_forecast(X, cat_count, X_all)

# 多项式回归预测
poly_order = 2  # 多项式阶数

def polynomial_regression_forecast(years, y, all_years, poly_order):
    poly = PolynomialFeatures(degree=poly_order)
    X_poly = poly.fit_transform(years.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    X_all_poly = poly.fit_transform(all_years.reshape(-1, 1))
    return model.predict(X_all_poly)

cat_count_poly = polynomial_regression_forecast(years, cat_count, all_years, poly_order)

# 非线性回归预测（指数回归示例）
def log_model(x, a, b):
    return a * np.log(x) + b

def nonlinear_regression_forecast(years, y, all_years):
    try:
        popt, _ = curve_fit(log_model, years, y, maxfev=10000)
        return log_model(all_years, *popt)
    except RuntimeError as e:
        print(f"Nonlinear regression failed: {e}")
        return np.full(len(all_years), np.nan)

cat_count_nonlinear = nonlinear_regression_forecast(years, cat_count, all_years)

# 加权预测模型构建
num_years = len(years)
true_values = cat_count
regression_pred = cat_count_regression[:num_years]
poly_pred = cat_count_poly[:num_years]
nonlinear_pred = cat_count_nonlinear[:num_years]

# 初始权重
w0 = [1/3, 1/3, 1/3]

# 目标函数：最小化误差（MSE）
def objective(w):
    return np.mean((w[0] * regression_pred + w[1] * poly_pred + w[2] * nonlinear_pred - true_values) ** 2)

# 约束条件：w1 + w2 + w3 = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# 下界和上界
bounds = [(0, 1), (0, 1), (0, 1)]

# 求解权重
result = minimize(objective, w0, bounds=bounds, constraints=constraints)
optimal_weights = result.x

# 使用优化后的权重进行预测
cat_count_weighted_future = (
    optimal_weights[0] * cat_count_regression[num_years:] +
    optimal_weights[1] * cat_count_poly[num_years:] +
    optimal_weights[2] * cat_count_nonlinear[num_years:]
)

# 可视化预测结果对比
plt.figure(figsize=(10, 6))
plt.plot(all_years, cat_count_poly, '-o', label='Polynomial Regression')
plt.plot(all_years, cat_count_nonlinear, '-s', label='Nonlinear Regression')
plt.plot(all_years, cat_count_regression, '-d', label='Linear Regression')
plt.plot(years_future, cat_count_weighted_future, '-^', label='Weighted Prediction', color='k', linewidth=2)
plt.title('Cat Count Prediction (Poly, Nonlinear, Linear, Weighted)')
plt.xlabel('Year')
plt.ylabel('Cat Count (10,000)')
plt.legend()
plt.grid(True)
plt.show()

# 计算预测误差
mape = lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100
rmse = lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2))

mape_poly = mape(cat_count, cat_count_poly[:num_years])
rmse_poly = rmse(cat_count, cat_count_poly[:num_years])

mape_nonlinear = mape(cat_count, cat_count_nonlinear[:num_years])
rmse_nonlinear = rmse(cat_count, cat_count_nonlinear[:num_years])

mape_regression = mape(cat_count, cat_count_regression[:num_years])
rmse_regression = rmse(cat_count, cat_count_regression[:num_years])

mape_weighted = mape(cat_count, optimal_weights[0] * regression_pred + optimal_weights[1] * poly_pred + optimal_weights[2] * nonlinear_pred)
rmse_weighted = rmse(cat_count, optimal_weights[0] * regression_pred + optimal_weights[1] * poly_pred + optimal_weights[2] * nonlinear_pred)

# 输出误差比较
print(f'MAPE (Polynomial Regression): {mape_poly:.2f}%')
print(f'RMSE (Polynomial Regression): {rmse_poly:.2f}')

print(f'MAPE (Nonlinear Regression): {mape_nonlinear:.2f}%')
print(f'RMSE (Nonlinear Regression): {rmse_nonlinear:.2f}')

print(f'MAPE (Linear Regression): {mape_regression:.2f}%')
print(f'RMSE (Linear Regression): {rmse_regression:.2f}')

print(f'MAPE (Weighted Prediction): {mape_weighted:.2f}%')
print(f'RMSE (Weighted Prediction): {rmse_weighted:.2f}')

# 绘制误差比较的柱状图
model_names = ['Polynomial Regression', 'Nonlinear Regression', 'Linear Regression', 'Weighted Prediction']
mape_values = [mape_poly, mape_nonlinear, mape_regression, mape_weighted]
rmse_values = [rmse_poly, rmse_nonlinear, rmse_regression, rmse_weighted]

plt.figure(figsize=(12, 6))

# 绘制 MAPE 的柱状图
plt.subplot(1, 2, 1)
plt.bar(model_names, mape_values, color=[(0.3, 0.75, 0.93)])  # 浅蓝色
plt.xticks(rotation=45, fontsize=12)
plt.ylabel('MAPE (%)')
plt.title('MAPE Comparison for Different Models')
plt.grid(True)

# 绘制 RMSE 的柱状图
plt.subplot(1, 2, 2)
plt.bar(model_names, rmse_values, color=[(0.5, 0.85, 0.45)])  # 浅绿色
plt.xticks(rotation=45, fontsize=12)
plt.ylabel('RMSE')
plt.title('RMSE Comparison for Different Models')
plt.grid(True)

# 设置整体标题
plt.suptitle('Error Comparison of Different Prediction Models', fontsize=16, fontweight='bold')

# 增强柱状图的显示效果
for i in range(1, 3):
    plt.subplot(1, 2, i)
    ax = plt.gca()
    ax.grid(alpha=0.4)
    ax.yaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
