import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用Times New Roman字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 定义列表
germany = [2.6, 2.8, 3.0, 3.2, 3.4]
america = [36.9, 38.1, 39.4, 40.8, 42.3]
france = [3.0, 3.2, 3.4, 3.6, 3.8]

# 计算all列表
all = [g + u + f for g, u, f in zip(germany, america, france)]

# 定义GDP列表
gdp_germany = [3957208, 3940143, 4348297, 4163596, 4525704]
gdp_usa = [21521395, 20891000, 25744108, 25744108, 27360935]
gdp_france = [27288670, 27790922, 27790922, 27790922, 30309040]

# 计算all_gdp列表
all_gdp = [g + u + f for g, u, f in zip(gdp_germany, gdp_usa, gdp_france)]
print(len(all_gdp))


# 将all和all_gdp列表中对应位置的元素相除
rate = [a / b for a, b in zip(all, all_gdp)]

# 定义年份和rate
years = np.array([2019, 2020, 2021, 2022, 2023])


# 使用ARIMA模型进行预测
model = ARIMA(rate, order=(1, 1, 1))  # ARIMA(p, d, q)，这里使用(1, 1, 1)作为示例
model_fit = model.fit()

# 预测未来3年的rate
n = 3  # 预测未来3年
rate_pred = model_fit.forecast(steps=n)

# 打印预测结果
print("未来3年的预测rate：", rate_pred)

# 绘制历史数据和预测结果
plt.figure(figsize=(10, 6))

# 绘制实际数据


# 绘制预测数据，包括历史数据的最后一个值
future_years = np.array([2024, 2025, 2026])
plt.plot(np.concatenate((years, future_years)), np.concatenate((rate, rate_pred)), 'b*--', label='Predicted Data')
plt.plot(years, rate, 'ro-', label='Historical Data')
plt.title('Rate Predictions',fontsize=20, fontweight='bold')
plt.xlabel('Year',fontsize=12, fontweight='bold')
plt.ylabel('Rate',fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.savefig('Q2_Rate Predictions', dpi=300, bbox_inches='tight')
plt.show()

all_rate =np.concatenate((rate, rate_pred))

global_gdp_history = [
    86.6,  # 2019年
    89.2,  # 2020年
    94.9,  # 2021年
    100.6,  # 2022年
    105.4  # 2023年
]

# 将数据转换为NumPy数组
global_gdp_history = np.array(global_gdp_history)

# 使用ARIMA模型进行预测
model = ARIMA(global_gdp_history, order=(1, 1, 1))  # ARIMA(p, d, q)，这里使用(1, 1, 1)作为示例
model_fit = model.fit()

# 预测未来3年的GDP
n = 3  # 预测未来3年
global_gdp_pred = model_fit.forecast(steps=n)

all_global_gdp = np.concatenate((global_gdp_history,global_gdp_pred))

global_pet_food_expenditure = [a * b for a, b in zip(all_rate, all_global_gdp)]

print(global_pet_food_expenditure)

future_years = np.array([2024, 2025, 2026])
plt.plot(np.concatenate((years, future_years)), global_pet_food_expenditure, 'r*-', label='Estimated Data')

plt.title('Global Pet Food Expenditure Estimation',fontsize=18, fontweight='bold')
plt.xlabel('Year',fontsize=12, fontweight='bold')
plt.ylabel('Rate',fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.savefig('Q2_Global Pet Food Expenditure Estimation', dpi=300, bbox_inches='tight')
plt.show()