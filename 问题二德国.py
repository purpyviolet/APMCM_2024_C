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
import seaborn as sns

# 1. 数据定义
years = np.array([2019, 2020, 2021, 2022, 2023])  # 历史年份
cat_count = np.array([1300, 1490, 1510, 1490, 1660])  # 猫数量（千）
dog_count = np.array([740, 775, 750, 760, 990])  # 狗数量（千）
total_pet_count = np.array([27.3, 28.0, 28.8, 29.7, 30.6])  # 宠物总数量（百万）
pet_food_expenditure = np.array([2.6, 2.8, 3.0, 3.2, 3.4])  # 宠物食品开支（亿美元）
gdp_per_capita = np.array([40494.90, 39179.74, 43671.31, 40886.25, 44460.82])  # 人均GDP（现价美元）
population_growth = np.array([0.341, 0.271, 0.286, 0.305, 0.292])  # 人口增长率（年度百分比）
import_volume_index = np.array([101.70, 89.16, 98.53, 98.25, 97.97])  # 进口物量指数（2015年 = 100）
food_production_index = np.array([98.00, 93.70, 96.72, 94.02, 96.73])  # 食品生产指数（2014-2016 = 100）

# 2. 预测未来三年各指标
years_future = np.array([2025, 2026, 2027])
all_years = np.concatenate((years, years_future))

# 2.1 人口增长率 - 假设未来三年平均增长率为 0.3%
population_growth_future = np.array([0.3, 0.3, 0.3])

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

# 3. 使用不同模型预测宠物食品开支
# 3.1 构建数据特征矩阵
selected_features_historical = np.column_stack((total_pet_count, gdp_per_capita, population_growth, import_volume_index, food_production_index))
selected_features_future = np.column_stack((total_pet_count[-1] * 1.02 ** np.arange(1, 4), gdp_per_capita_future, population_growth_future, import_volume_index_future, food_production_index_future))

# 3.2 线性回归模型
linear_model = LinearRegression()
linear_model.fit(selected_features_historical, pet_food_expenditure)
linear_pred_future = linear_model.predict(selected_features_future)

# 3.3 多项式回归模型（使用二阶多项式进行拟合）
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(selected_features_historical, pet_food_expenditure)
poly_pred_future = poly_model.predict(selected_features_future)

# 3.4 非线性回归模型（指数模型）
def exp_model(x, a, b):
    return a * np.exp(b * x)

try:
    popt, _ = curve_fit(exp_model, years, pet_food_expenditure, maxfev=10000)
    nonlinear_pred_future = exp_model(years_future, *popt)
except RuntimeError as e:
    print(f"Nonlinear regression failed: {e}")
    nonlinear_pred_future = np.full(len(years_future), np.nan)

# 3.5 支持向量回归（SVR）模型
svr_model = SVR(kernel='rbf')
svr_model.fit(selected_features_historical, pet_food_expenditure)
svr_pred_future = svr_model.predict(selected_features_future)

# 3.6 随机森林回归模型
random_forest_model = RandomForestRegressor(n_estimators=50, random_state=42)
random_forest_model.fit(selected_features_historical, pet_food_expenditure)
random_forest_pred_future = random_forest_model.predict(selected_features_future)

# 4. 可视化预测结果
# 绘制预测的未来宠物食品开支情况
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
plt.title('Prediction of Pet Food Expenditure in Germany (2025-2027)')
plt.xlabel('Year')
plt.ylabel('Pet Food Expenditure (Billion EUR)')
plt.legend()
plt.grid(True)

plt.suptitle('Forecasting Pet Food Expenditure in Germany using Multiple Models', fontsize=16)
plt.show()

# 5. 输出预测结果
print('Predicted Pet Food Expenditure in Germany (2025-2027) using Different Models:')
for i, year in enumerate(years_future):
    print(f'Year {year}:')
    print(f' - Linear Regression: {linear_pred_future[i]:.2f} Billion EUR')
    print(f' - Polynomial Regression: {poly_pred_future[i]:.2f} Billion EUR')
    print(f' - Nonlinear Regression (Exponential): {nonlinear_pred_future[i]:.2f} Billion EUR')
    print(f' - Support Vector Regression (SVR): {svr_pred_future[i]:.2f} Billion EUR')
    print(f' - Random Forest Regression: {random_forest_pred_future[i]:.2f} Billion EUR\n')
