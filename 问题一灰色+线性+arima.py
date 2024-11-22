import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
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

# 构建多元线性回归模型
X = np.column_stack((pop_growth_rate, gdp_per_capita))
X_future = np.column_stack((pop_growth_rate_future, gdp_per_capita_future))

# 线性回归模型预测
def linear_regression_forecast(X, y, X_future):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X_future)

cat_count_regression = linear_regression_forecast(X, cat_count, X_future)
dog_count_regression = linear_regression_forecast(X, dog_count, X_future)
total_pet_count_regression = linear_regression_forecast(X, total_pet_count, X_future)
market_size_regression = linear_regression_forecast(X, market_size, X_future)
medical_market_size_regression = linear_regression_forecast(X, pet_medical_market_size, X_future)

# ARIMA预测
def arima_forecast(series, steps):
    model = ARIMA(series, order=(1, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

cat_count_arima = arima_forecast(cat_count, 3)
dog_count_arima = arima_forecast(dog_count, 3)
total_pet_count_arima = arima_forecast(total_pet_count, 3)
market_size_arima = arima_forecast(market_size, 3)
medical_market_size_arima = arima_forecast(pet_medical_market_size, 3)

# 可视化预测结果对比
plt.figure(figsize=(15, 18))

# 猫数量预测对比
plt.subplot(3, 2, 1)
plt.plot(np.concatenate([years, years_future]), np.concatenate([cat_count, cat_count_regression]), '-o', label='Linear Regression')
plt.plot(np.concatenate([years, years_future]), np.concatenate([cat_count, cat_count_arima]), '-d', label='ARIMA')
plt.title('Cat Count Prediction (Linear, ARIMA)')
plt.xlabel('Year')
plt.ylabel('Cat Count (10,000)')
plt.legend()
plt.grid(True)

# 狗数量预测对比
plt.subplot(3, 2, 2)
plt.plot(np.concatenate([years, years_future]), np.concatenate([dog_count, dog_count_regression]), '-o', label='Linear Regression')
plt.plot(np.concatenate([years, years_future]), np.concatenate([dog_count, dog_count_arima]), '-d', label='ARIMA')
plt.title('Dog Count Prediction (Linear, ARIMA)')
plt.xlabel('Year')
plt.ylabel('Dog Count (10,000)')
plt.legend()
plt.grid(True)

# 总宠物数量预测对比
plt.subplot(3, 2, 3)
plt.plot(np.concatenate([years, years_future]), np.concatenate([total_pet_count, total_pet_count_regression]), '-o', label='Linear Regression')
plt.plot(np.concatenate([years, years_future]), np.concatenate([total_pet_count, total_pet_count_arima]), '-d', label='ARIMA')
plt.title('Total Pet Count Prediction (Linear, ARIMA)')
plt.xlabel('Year')
plt.ylabel('Total Pet Count (10,000)')
plt.legend()
plt.grid(True)

# 宠物市场规模预测对比
plt.subplot(3, 2, 4)
plt.plot(np.concatenate([years, years_future]), np.concatenate([market_size, market_size_regression]), '-o', label='Linear Regression')
plt.plot(np.concatenate([years, years_future]), np.concatenate([market_size, market_size_arima]), '-d', label='ARIMA')
plt.title('Market Size Prediction (Linear, ARIMA)')
plt.xlabel('Year')
plt.ylabel('Market Size (Billion USD)')
plt.legend()
plt.grid(True)

# 宠物医疗市场规模预测对比
plt.subplot(3, 2, 5)
plt.plot(np.concatenate([years, years_future]), np.concatenate([pet_medical_market_size, medical_market_size_regression]), '-o', label='Linear Regression')
plt.plot(np.concatenate([years, years_future]), np.concatenate([pet_medical_market_size, medical_market_size_arima]), '-d', label='ARIMA')
plt.title('Medical Market Size Prediction (Linear, ARIMA)')
plt.xlabel('Year')
plt.ylabel('Medical Market Size (Billion CNY)')
plt.legend()
plt.grid(True)

# 设置整体标题
plt.suptitle('Prediction of China Pet Industry for Next 3 Years (2025-2027)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
