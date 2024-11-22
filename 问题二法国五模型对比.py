import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
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
years_future = np.array([2024,2025, 2026, 2027])
all_years = np.concatenate((years, years_future))

# 2.1 人口增长率 - 假设未来三年平均增长率为 0.5%
population_growth_future = np.array([0.5,0.5, 0.5, 0.5])

# 2.2 人均 GDP 预测 - 使用多项式回归
poly_order = 2
gdp_poly_coeff = np.polyfit(years, gdp_per_capita, poly_order)
gdp_per_capita_future = np.polyval(gdp_poly_coeff, years_future)

# 2.3 进口物量指数预测 - 使用多项式回归
import_index_poly_coeff = np.polyfit(years, import_volume_index, poly_order)
import_volume_index_future = np.polyval(import_index_poly_coeff, years_future)

# 2.4 食品生产指数预测 - 使用多项式回归
food_production_poly_coeff = np.polyfit(years, food_production_index, poly_order)
food_production_index_future = np.polyval(food_production_poly_coeff, years_future)

# 3. 构建宠物数量的预测模型
# 使用人均 GDP 和人口增长率预测未来宠物数量
X = np.column_stack((gdp_per_capita, population_growth))

# 3.1 线性回归模型
pet_count_linear_model = LinearRegression()
pet_count_linear_model.fit(X, total_pet_count)
X_future = np.column_stack((gdp_per_capita_future, population_growth_future))
pet_count_linear_future = pet_count_linear_model.predict(X_future)

# 3.2 多项式回归模型
poly_order_pet = 2
pet_count_poly_coeff = np.polyfit(years, total_pet_count, poly_order_pet)
pet_count_poly = np.polyval(pet_count_poly_coeff, all_years)

# 3.3 非线性回归模型 (指数模型)
def exp_model(x, a, b):
    return a * np.exp(b * x)

try:
    popt, _ = curve_fit(exp_model, years, total_pet_count, maxfev=10000)
    pet_count_nonlinear = exp_model(all_years, *popt)
except RuntimeError as e:
    print(f"Nonlinear regression failed: {e}")
    pet_count_nonlinear = np.full(len(all_years), np.nan)

# 检查并替换 inf 或 nan 值
pet_count_nonlinear = np.nan_to_num(pet_count_nonlinear, nan=np.mean(total_pet_count), posinf=np.max(total_pet_count), neginf=np.min(total_pet_count))

# 4. 加权组合预测未来宠物数量
# 合并历史和未来的特征
all_features = np.column_stack((gdp_per_capita, population_growth))
future_features = np.column_stack((gdp_per_capita_future, population_growth_future))
all_features_combined = np.vstack((all_features, future_features))

# 使用线性回归模型进行预测
pet_count_linear_all = pet_count_linear_model.predict(all_features_combined)

# 提取历史部分和未来部分的预测值
num_years = len(years)
regression_pred = pet_count_linear_all[:num_years]    # 历史部分的预测
pet_count_linear_future = pet_count_linear_all[num_years:]  # 未来部分的预测

# 对于多项式和非线性回归模型，继续按照之前的方法
poly_pred = pet_count_poly[:num_years]              # 历史部分的多项式预测
nonlinear_pred = pet_count_nonlinear[:num_years]    # 历史部分的非线性预测

# 初始权重
w0 = [1/3, 1/3, 1/3]

# 目标函数：最小化误差（MSE）
true_values = total_pet_count  # 历史真实值
def objective(w):
    return np.mean((w[0] * regression_pred + w[1] * poly_pred + w[2] * nonlinear_pred - true_values) ** 2)

# 约束条件：w1 + w2 + w3 = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# 下界和上界
bounds = [(0, 1), (0, 1), (0, 1)]

# 使用 scipy.optimize.minimize 进行优化
result = minimize(objective, w0, bounds=bounds, constraints=constraints)
optimal_weights = result.x

# 使用优化后的权重预测未来宠物数量
poly_future = pet_count_poly[num_years:]
nonlinear_future = pet_count_nonlinear[num_years:]

pet_count_weighted_future = (
    optimal_weights[0] * pet_count_linear_future +
    optimal_weights[1] * poly_future +
    optimal_weights[2] * nonlinear_future
)

# 5. 使用预测的各因素预测未来宠物食品开支
selected_features_historical = np.column_stack((total_pet_count, gdp_per_capita, population_growth, import_volume_index, food_production_index))
selected_features_future = np.column_stack((pet_count_weighted_future, gdp_per_capita_future, population_growth_future, import_volume_index_future, food_production_index_future))

# 构建多元线性回归模型，预测未来宠物食品开支
food_model = LinearRegression()
food_model.fit(selected_features_historical, pet_food_expenditure)
pet_food_expenditure_future = food_model.predict(selected_features_future)

# 8. 使用不同模型预测宠物食品开支
# 8.1 线性回归模型
linear_model = LinearRegression()
linear_model.fit(selected_features_historical, pet_food_expenditure)
linear_pred_future = linear_model.predict(selected_features_future)

# 8.2 多项式回归模型（使用二阶多项式进行拟合）
poly = PolynomialFeatures(degree=2)
poly_model = make_pipeline(poly, LinearRegression())
poly_model.fit(selected_features_historical, pet_food_expenditure)
poly_pred_future = poly_model.predict(selected_features_future)

# 8.3 非线性回归模型（指数模型）
try:
    popt, _ = curve_fit(exp_model, years, pet_food_expenditure, maxfev=10000)
    nonlinear_pred_future = exp_model(years_future, *popt)
except RuntimeError as e:
    print(f"Nonlinear regression failed: {e}")
    nonlinear_pred_future = np.full(len(years_future), np.nan)

# 8.4 支持向量回归（SVR）模型
svr_model = SVR(kernel='rbf')
svr_model.fit(selected_features_historical, pet_food_expenditure)
svr_pred_future = svr_model.predict(selected_features_future)

# 8.5 随机森林回归模型
random_forest_model = RandomForestRegressor(n_estimators=50, random_state=42)
random_forest_model.fit(selected_features_historical, pet_food_expenditure)
random_forest_pred_future = random_forest_model.predict(selected_features_future)

# 9. 可视化预测结果
plt.figure(figsize=(10, 6))

# 绘制历史宠物食品开支数据
plt.plot(years, pet_food_expenditure, '-o', label='Historical Pet Food Expenditure', linewidth=2, markersize=6, color='b')

# 绘制不同模型预测的未来宠物食品开支数据
plt.plot(years_future, linear_pred_future, '-^', label='Linear Regression', linewidth=2, markersize=6, color='r')
plt.plot(years_future, poly_pred_future, '-s', label='Polynomial Regression', linewidth=2, markersize=6, color='g')
plt.plot(years_future, nonlinear_pred_future, '-d', label='Nonlinear Regression (Exponential)', linewidth=2, markersize=6, color='m')
plt.plot(years_future, svr_pred_future, '-p', label='Support Vector Regression (SVR)', linewidth=2, markersize=6, color='c')
plt.plot(years_future, random_forest_pred_future, '-x', label='Random Forest Regression', linewidth=2, markersize=6, color='k')

# 图形设置
plt.title('Prediction of Pet Food Expenditure in France (2025-2027)')
plt.xlabel('Year')
plt.ylabel('Pet Food Expenditure (Billion USD)')
plt.legend()
plt.grid(True)
plt.suptitle('Forecasting Pet Food Expenditure in France using Multiple Models', fontsize=16)
plt.show()

# 10. 输出预测结果
print('Predicted Pet Food Expenditure in France (2025-2027) using Different Models:')
for i, year in enumerate(years_future):
    print(f'Year {year}:')
    print(f' - Linear Regression: {linear_pred_future[i]:.2f} Billion USD')
    print(f' - Polynomial Regression: {poly_pred_future[i]:.2f} Billion USD')
    print(f' - Nonlinear Regression (Exponential): {nonlinear_pred_future[i]:.2f} Billion USD')
    print(f' - Support Vector Regression (SVR): {svr_pred_future[i]:.2f} Billion USD')
    print(f' - Random Forest Regression: {random_forest_pred_future[i]:.2f} Billion USD\n')
