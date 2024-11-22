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
pet_count = np.array([99.8, 108.5, 115.4, 122.6, 130.2])  # 宠物数量（百万）
pet_food_output_value = np.array([440.7, 727.3, 1554, 1508, 2793])  # 中国宠物食品总产值（人民币 亿）
pet_food_export_value = np.array([154.1, 9.8, 12.2, 24.7, 39.6])  # 中国宠物食品出口总值（美元 亿）
market_size = np.array([33.2, 35.6, 38.9, 42.1, 45.5])  # 宠物市场规模（美元 亿）
pet_household_penetration = np.array([0.18, 0.2, 0.2, 0.2, 0.22])  # 宠物家庭渗透率
food_export_percentage = np.array([2.8745, 2.6897, 2.3228, 2.3002, 2.4563])  # 食品出口占商品出口的百分比
import_volume_index = np.array([120.23, 126.07, 134.64, 139.98, 145.32])  # 进口物量指数
population_growth = np.array([0.3547, 0.2380, 0.0893, -0.0131, -0.1038])  # 人口增长率（年度百分比）
gdp_per_capita = np.array([10143.86, 10408.72, 12617.51, 12662.58, 12614.06])  # 人均GDP（现价美元）
global_market_size = np.array([1000, 1055, 1149.42, 1200, 1250])  # 全球市场规模（美元 亿）
# 2. 数据可视化分析
plt.figure(figsize=(14, 10))

# 宠物食品总产值和出口总值
plt.subplot(2, 2, 1)
plt.plot(years, pet_food_output_value, '-o', label='Total Pet Food Output Value (CNY)', linewidth=2, markersize=6, color='b')
plt.plot(years, pet_food_export_value, '-s', label='Total Pet Food Export Value (USD)', linewidth=2, markersize=6, color='r')
plt.title('Total Pet Food Output and Export Value (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# 市场规模和宠物家庭渗透率
plt.subplot(2, 2, 2)
plt.plot(years, market_size, '-^', label='Pet Market Size (Billion USD)', linewidth=2, markersize=6, color='g')
plt.ylabel('Market Size (Billion USD)')
plt.twinx()
plt.plot(years, pet_household_penetration, '-d', label='Pet Household Penetration Rate', linewidth=2, markersize=6, color='m')
plt.ylabel('Household Penetration Rate')
plt.title('Pet Market Size and Household Penetration Rate')
plt.xlabel('Year')
plt.legend()
plt.grid(True)

# 食品出口占比和进口物量指数
plt.subplot(2, 2, 3)
plt.plot(years, food_export_percentage, '-o', label='Food Export as % of Goods Export', linewidth=2, markersize=6, color='c')
plt.plot(years, import_volume_index, '-s', label='Import Volume Index', linewidth=2, markersize=6, color='k')
plt.title('Food Export Percentage and Import Volume Index')
plt.xlabel('Year')
plt.ylabel('Index / Percentage')
plt.legend()
plt.grid(True)

# 人口增长率和人均 GDP
plt.subplot(2, 2, 4)
plt.plot(years, population_growth, '-^', label='Population Growth Rate', linewidth=2, markersize=6, color='g')
plt.ylabel('Population Growth Rate (%)')
plt.twinx()
plt.plot(years, gdp_per_capita, '-d', label='GDP per Capita (USD)', linewidth=2, markersize=6, color='m')
plt.ylabel('GDP per Capita (USD)')
plt.title('Population Growth Rate and GDP per Capita')
plt.xlabel('Year')
plt.legend()
plt.grid(True)

plt.suptitle('Analysis of China Pet Food Industry (2019-2023)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 3. 相关性分析
X = np.column_stack((pet_count, market_size, pet_household_penetration, food_export_percentage, import_volume_index, population_growth, gdp_per_capita, global_market_size))

# 相关系数计算
corr_matrix_output = np.corrcoef(np.column_stack((X, pet_food_output_value)).T)[-1, :-1]
corr_matrix_export = np.corrcoef(np.column_stack((X, pet_food_export_value)).T)[-1, :-1]

# 输出相关系数结果
output_corr_df = pd.DataFrame({'Correlation_Coefficients': corr_matrix_output},
                              index=['Pet_Count', 'Market_Size', 'Household_Penetration', 'Food_Export_Percentage', 'Import_Volume_Index', 'Population_Growth', 'GDP_per_Capita', 'Global_Market_Size'])
export_corr_df = pd.DataFrame({'Correlation_Coefficients': corr_matrix_export},
                              index=['Pet_Count', 'Market_Size', 'Household_Penetration', 'Food_Export_Percentage', 'Import_Volume_Index', 'Population_Growth', 'GDP_per_Capita', 'Global_Market_Size'])

print('Correlation with Total Pet Food Output Value:')
print(output_corr_df)

print('Correlation with Total Pet Food Export Value:')
print(export_corr_df)

# 4. 相关性热力图可视化
corr_matrix = np.corrcoef(np.column_stack((X, pet_food_output_value, pet_food_export_value)).T)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=['Pet_Count', 'Market_Size', 'Household_Penetration', 'Food_Export_Percentage', 'Import_Volume_Index', 'Population_Growth', 'GDP per_Capita', 'Global_Market_Size', 'Pet_Food_Output', 'Pet_Food_Export'],
            yticklabels=['Pet_Count', 'Market_Size', 'Household_Penetration', 'Food_Export_Percentage', 'Import_Volume_Index', 'Population_Growth', 'GDP per_Capita', 'Global_Market_Size', 'Pet_Food_Output', 'Pet_Food_Export'])
plt.title('Correlation Heatmap of Factors Affecting Pet Food Industry')
plt.show()

# 5. 结论
# 宠物数量、市场规模和全球市场规模对中国宠物食品总产值和出口总值有明显的相关性，而人口增长率和家庭渗透率对于宠物食品的影响相对小。
# 宠物食品总产值在过去几年中显著增长，特别是在2021年到2023年之间，但出口总值在某些年度有所低起。
# 市场规模也显示了进一步的增长，同时宠物家庭渗透率也有所提高。
# 人口增长率近几年下降，但人均 GDP 仍保持稳定增长。
