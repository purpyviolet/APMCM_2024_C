import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
import seaborn as sns
# 1. 数据定义
years = np.array([2019, 2020, 2021, 2022, 2023])  # 历史年份
cat_count = np.array([9420, 6500, 9420, 7380, 7380])  # 猫数量（千）
dog_count = np.array([8970, 8500, 8970, 8970, 8010])  # 狗数量（千）
total_pet_count = np.array([89.7, 91.8, 94.2, 96.7, 99.2])  # 宠物总数量（百万）
pet_food_expenditure = np.array([36.9, 38.1, 39.4, 40.8, 42.3])  # 宠物食品开支（亿美元）
gdp_per_capita = np.array([65548.07, 64317.40, 71055.88, 77246.67, 81695.19])  # 人均GDP（现价美元）
population_growth = np.array([0.455, 0.969, 0.157, 0.367, 0.492])  # 人口增长率（年度百分比）
import_volume_index = np.array([109.48, 105.23, 117.93, 119.17, 121.91])  # 进口物量指数（2015年 = 100）
food_production_index = np.array([99.64, 103.12, 104.57, 100.86, 105.93])  # 食品生产指数（2014-2016 = 100）

# 2. 数据可视化分析
plt.figure(figsize=(14, 10))

# 猫和狗的数量趋势
plt.subplot(2, 2, 1)
plt.plot(years, cat_count, '-o', label='Cat Count', linewidth=2, markersize=6, color='b')
plt.plot(years, dog_count, '-s', label='Dog Count', linewidth=2, markersize=6, color='r')
plt.title('Cat and Dog Population in the US (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Count (Thousand)')
plt.legend()
plt.grid(True)

# 宠物总数量与宠物食品开支趋势
plt.subplot(2, 2, 2)
plt.plot(years, total_pet_count, '-^', label='Total Pet Count', linewidth=2, markersize=6, color='g')
plt.ylabel('Total Pet Count (Million)')
plt.twinx()
plt.plot(years, pet_food_expenditure, '-d', label='Pet Food Expenditure', linewidth=2, markersize=6, color='m')
plt.ylabel('Pet Food Expenditure (Billion USD)')
plt.title('Total Pet Count and Pet Food Expenditure')
plt.xlabel('Year')
plt.legend()
plt.grid(True)

# 人均GDP和宠物食品开支的关系
plt.subplot(2, 2, 3)
plt.scatter(gdp_per_capita, pet_food_expenditure, s=100, c='blue', alpha=0.6, edgecolors='w')
plt.title('GDP per Capita vs Pet Food Expenditure')
plt.xlabel('GDP per Capita (USD)')
plt.ylabel('Pet Food Expenditure (Billion USD)')
plt.grid(True)

# 进口物量指数、食品生产指数与宠物食品开支
plt.subplot(2, 2, 4)
plt.plot(years, import_volume_index, '-o', label='Import Volume Index', linewidth=2, markersize=6, color='c')
plt.plot(years, food_production_index, '-s', label='Food Production Index', linewidth=2, markersize=6, color='k')
plt.title('Import Volume and Food Production Index')
plt.xlabel('Year')
plt.ylabel('Index')
plt.legend()
plt.grid(True)

plt.suptitle('Analysis of Pet Industry in the US (2019-2023)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 3. 计算相关性
# 将相关性分析的变量组合在一起
X = np.column_stack((cat_count, dog_count, total_pet_count, gdp_per_capita, population_growth, import_volume_index, food_production_index))
y = pet_food_expenditure  # 目标变量：宠物食品开支

# 计算皮尔逊相关系数
corr_matrix = np.corrcoef(np.column_stack((X, y)).T)

# 提取宠物食品开支与各个因素的相关系数
correlation_with_food = corr_matrix[-1, :-1]  # 最后一行是与宠物食品开支的相关系数

# 输出相关系数
correlation_df = pd.DataFrame({'Correlation Coefficients': correlation_with_food},
                              index=['Cat Count', 'Dog Count', 'Total Pet Count', 'GDP per Capita', 'Population Growth', 'Import Volume Index', 'Food Production Index'])
print('Correlation with Pet Food Expenditure:')
print(correlation_df)

# 4. 可视化相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=['Cat Count', 'Dog Count', 'Total Pet Count', 'GDP per Capita', 'Population Growth', 'Import Volume Index', 'Food Production Index', 'Pet Food Expenditure'], yticklabels=['Cat Count', 'Dog Count', 'Total Pet Count', 'GDP per Capita', 'Population Growth', 'Import Volume Index', 'Food Production Index', 'Pet Food Expenditure'])
plt.title('Correlation Matrix for Pet Industry Factors')
plt.show()

# 5. 预测未来三年各因素
years_future = np.array([2025, 2026, 2027])
all_years = np.concatenate((years, years_future))

# 5.1 人口增长率 - 假设未来三年平均增长率为 0.3%
population_growth_future = np.array([0.3, 0.3, 0.3])

# 5.2 人均 GDP 预测 - 使用多项式回归
poly_order = 2
gdp_poly_coeff = np.polyfit(years, gdp_per_capita, poly_order)
gdp_per_capita_future = np.polyval(gdp_poly_coeff, years_future)

# 5.3 进口物量指数预测 - 使用多项式回归
import_index_poly_coeff = np.polyfit(years, import_volume_index, poly_order)
import_volume_index_future = np.polyval(import_index_poly_coeff, years_future)

# 5.4 食品生产指数预测 - 使用多项式回归
food_production_poly_coeff = np.polyfit(years, food_production_index, poly_order)
food_production_index_future = np.polyval(food_production_poly_coeff, years_future)

# 6. 构建宠物数量的预测模型
X = np.column_stack((gdp_per_capita, population_growth))

# 6.1 线性回归模型
pet_count_linear_model = LinearRegression()
pet_count_linear_model.fit(X, total_pet_count)
X_future = np.column_stack((gdp_per_capita_future, population_growth_future))
pet_count_linear_future = pet_count_linear_model.predict(X_future)

# 6.2 多项式回归模型
pet_count_poly_coeff = np.polyfit(years, total_pet_count, poly_order)
pet_count_poly_future = np.polyval(pet_count_poly_coeff, all_years)

# 6.3 非线性回归模型（幂模型替代）
def power_model(x, a, b):
    return a * np.power(x, b)

try:
    popt, _ = curve_fit(power_model, years, total_pet_count, maxfev=10000)
    pet_count_nonlinear_future = power_model(all_years, *popt)
except RuntimeError as e:
    print(f"Nonlinear regression failed: {e}")
    pet_count_nonlinear_future = np.full(len(all_years), np.nan)

# 检查并替换 inf 或 nan 值
pet_count_nonlinear_future = np.nan_to_num(pet_count_nonlinear_future, nan=np.mean(total_pet_count), posinf=np.max(total_pet_count), neginf=np.min(total_pet_count))

# 7. 加权组合预测未来宠物数量
num_years = len(years)
regression_pred = total_pet_count[:num_years]
poly_pred = pet_count_poly_future[:num_years]
nonlinear_pred = pet_count_nonlinear_future[:num_years]

# 初始权重
w0 = [1/3, 1/3, 1/3]

# 目标函数：最小化误差（MSE）
def objective(w):
    return np.mean((w[0] * regression_pred + w[1] * poly_pred + w[2] * nonlinear_pred - total_pet_count) ** 2)

# 约束条件：w1 + w2 + w3 = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# 下界和上界
bounds = [(0, 1), (0, 1), (0, 1)]

# 使用 scipy.optimize.minimize 进行优化
result = minimize(objective, w0, bounds=bounds, constraints=constraints)
optimal_weights = result.x

# 使用优化后的权重预测未来宠物数量
pet_count_weighted_future = (
    optimal_weights[0] * pet_count_linear_future +
    optimal_weights[1] * pet_count_poly_future[num_years:] +
    optimal_weights[2] * pet_count_nonlinear_future[num_years:]
)

# 8. 使用预测的各因素预测未来宠物食品开支
selected_features_historical = np.column_stack((total_pet_count, gdp_per_capita, population_growth, import_volume_index, food_production_index))
selected_features_future = np.column_stack((pet_count_weighted_future, gdp_per_capita_future, population_growth_future, import_volume_index_future, food_production_index_future))

# 检查并替换 inf 或 nan 值
selected_features_future = np.nan_to_num(selected_features_future, nan=0.0, posinf=np.max(selected_features_historical), neginf=np.min(selected_features_historical))

# 构建多元线性回归模型，预测未来宠物食品开支
food_model = LinearRegression()
food_model.fit(selected_features_historical, pet_food_expenditure)
pet_food_expenditure_future = food_model.predict(selected_features_future)

# 9. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(years, pet_food_expenditure, '-o', label='Historical Pet Food Expenditure', linewidth=2, markersize=6, color='b')
plt.plot(years_future, pet_food_expenditure_future, '-^', label='Predicted Pet Food Expenditure', linewidth=2, markersize=6, color='r')
plt.title('Prediction of Pet Food Expenditure in the US (2025-2027)')
plt.xlabel('Year')
plt.ylabel('Pet Food Expenditure (Billion USD)')
plt.legend()
plt.grid(True)
plt.suptitle('Forecasting Pet Food Expenditure in the US', fontsize=16)
plt.show()

# 10. 输出预测结果
print('Predicted Pet Food Expenditure (2025-2027):')
for i, year in enumerate(years_future):
    print(f'Year {year}: {pet_food_expenditure_future[i]:.2f} Billion USD')
