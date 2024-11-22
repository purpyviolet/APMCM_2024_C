import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 1. 数据定义
years = np.array([2019, 2020, 2021, 2022, 2023])  # 年份
cat_count = np.array([4412, 4862, 5806, 6536, 6980])  # 猫数量（万）
dog_count = np.array([5503, 5222, 5429, 5119, 5175])  # 狗数量（万）
total_pet_count = np.array([9980, 10850, 11540, 12260, 13020])  # 总宠物数量（万）
market_size = np.array([33.2, 35.6, 38.9, 42.1, 45.5])  # 宠物市场规模（亿美元）
pet_food_expenditure = np.array([15.1, 16.2, 17.5, 18.9, 20.3])  # 宠物食品开支（亿美元）
vet_service_expenditure = np.array([3.4, 3.8, 4.1, 4.5, 4.9])  # 兽医服务开支（亿美元）
pet_family_penetration = np.array([0.18, 0.20, 0.20, 0.20, 0.22])  # 家庭宠物渗透率
pet_medical_market_size = np.array([400, 500, 600, 640, 700])  # 宠物医疗市场规模（亿元人民币）
total_population = np.array([1407745000, 1411100000, 1412360000, 1412175000, 1410710000])  # 人口总数
population_growth_rate = np.array([0.35474089, 0.23804087, 0.0892522, -0.013099501, -0.103794532])  # 人口增长（年度百分比）
gini_index = np.array([38.2, 37.1, 35.7, 35.54242424, 34.947669])  # 基尼系数
urban_population = np.array([848982855, 866810508, 882894483, 897578430, 910895447])  # 城镇人口
rural_population = np.array([558762145, 544289492, 529465517, 514596570, 499814553])  # 农村人口
gdp_per_capita = np.array([10143.86022, 10408.71955, 12617.5051, 12662.58317, 12614.06099])  

# 2. 创建图形（中文版本）
plt.figure(figsize=(15, 18))

# 绘制宠物数量变化（猫、狗、总宠物数量）
plt.subplot(3, 2, 1)
plt.plot(years, cat_count, 'o-', label='猫数量', linewidth=2, markersize=6, color='b')
plt.plot(years, dog_count, 's-', label='狗数量', linewidth=2, markersize=6, color='r')
plt.plot(years, total_pet_count, '^-', label='总宠物数量', linewidth=2, markersize=6, color='g')
plt.xlabel('年份')
plt.ylabel('数量（万）')
plt.title('宠物数量变化（猫、狗、总宠物）')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制宠物市场规模、宠物食品开支与兽医服务开支
plt.subplot(3, 2, 2)
plt.plot(years, market_size, 'o-', label='宠物市场规模', linewidth=2, markersize=6, color='c')
plt.plot(years, pet_food_expenditure, 's-', label='宠物食品开支', linewidth=2, markersize=6, color='m')
plt.plot(years, vet_service_expenditure, '^-', label='兽医服务开支', linewidth=2, markersize=6, color='k')
plt.xlabel('年份')
plt.ylabel('金额（亿美元）')
plt.title('宠物市场规模与相关支出')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制宠物医疗市场规模与家庭宠物渗透率
plt.subplot(3, 2, 3)
plt.plot(years, pet_medical_market_size, 'o-', label='宠物医疗市场规模', linewidth=2, markersize=6, color='g')
plt.xlabel('年份')
plt.ylabel('宠物医疗市场规模（亿元人民币）')
plt.title('宠物医疗市场规模与家庭宠物渗透率')
plt.twinx()
plt.plot(years, pet_family_penetration, 's-', label='家庭宠物渗透率', linewidth=2, markersize=6, color='b')
plt.ylabel('家庭宠物渗透率')
plt.legend(loc='upper left')
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制人均GDP趋势
plt.subplot(3, 2, 4)
plt.plot(years, gdp_per_capita, 'o-', label='人均GDP', linewidth=2, markersize=6, color='r')
plt.xlabel('年份')
plt.ylabel('人均GDP（现价美元）')
plt.title('人均GDP变化趋势')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制人口增长率与基尼系数
plt.subplot(3, 2, 5)
plt.plot(years, population_growth_rate, 'o-', label='人口增长率', linewidth=2, markersize=6, color='b')
plt.xlabel('年份')
plt.ylabel('人口增长率（年度百分比）')
plt.title('人口增长率与基尼系数')
plt.twinx()
plt.plot(years, gini_index, 's-', label='基尼系数', linewidth=2, markersize=6, color='m')
plt.ylabel('基尼系数')
plt.legend(loc='upper left')
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制城镇人口与农村人口
plt.subplot(3, 2, 6)
plt.plot(years, urban_population, 'o-', label='城镇人口', linewidth=2, markersize=6, color='g')
plt.plot(years, rural_population, 's-', label='农村人口', linewidth=2, markersize=6, color='r')
plt.xlabel('年份')
plt.ylabel('人口数量')
plt.title('城镇人口与农村人口变化')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 设置整体标题
plt.suptitle('中国宠物行业发展情况（2019-2023）', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 3. 创建英文版本图形
plt.figure(figsize=(15, 18))

# 绘制宠物数量变化（猫、狗、总宠物数量）
plt.subplot(3, 2, 1)
plt.plot(years, cat_count, 'o-', label='Cat Number', linewidth=2, markersize=6, color='b')
plt.plot(years, dog_count, 's-', label='Dog Number', linewidth=2, markersize=6, color='r')
plt.plot(years, total_pet_count, '^-', label='Total Pet Number', linewidth=2, markersize=6, color='g')
plt.xlabel('Year')
plt.ylabel('Number (10,000)')
plt.title('Pet Number Change (Cat, Dog, Total Pet)')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制宠物市场规模、宠物食品开支与兽医服务开支
plt.subplot(3, 2, 2)
plt.plot(years, market_size, 'o-', label='Pet Market Size', linewidth=2, markersize=6, color='c')
plt.plot(years, pet_food_expenditure, 's-', label='Pet Food Expenditure', linewidth=2, markersize=6, color='m')
plt.plot(years, vet_service_expenditure, '^-', label='Vet Service Expenditure', linewidth=2, markersize=6, color='k')
plt.xlabel('Year')
plt.ylabel('Expenditure (Billion USD)')
plt.title('Pet Market Size and Related Expenditures')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制宠物医疗市场规模与家庭宠物渗透率
plt.subplot(3, 2, 3)
plt.plot(years, pet_medical_market_size, 'o-', label='Pet Medical Market Size', linewidth=2, markersize=6, color='g')
plt.xlabel('Year')
plt.ylabel('Pet Medical Market Size (Billion CNY)')
plt.title('Pet Medical Market Size and Penetration Rate')
plt.twinx()
plt.plot(years, pet_family_penetration, 's-', label='Pet Penetration Rate', linewidth=2, markersize=6, color='b')
plt.ylabel('Pet Penetration Rate')
plt.legend(loc='upper left')
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制人均GDP趋势
plt.subplot(3, 2, 4)
plt.plot(years, gdp_per_capita, 'o-', label='GDP per Capita', linewidth=2, markersize=6, color='r')
plt.xlabel('Year')
plt.ylabel('GDP per Capita (USD)')
plt.title('GDP per Capita Trend')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制人口增长率与基尼系数
plt.subplot(3, 2, 5)
plt.plot(years, population_growth_rate, 'o-', label='Population Growth Rate', linewidth=2, markersize=6, color='b')
plt.xlabel('Year')
plt.ylabel('Population Growth Rate (%)')
plt.title('Population Growth Rate and Gini Index')
plt.twinx()
plt.plot(years, gini_index, 's-', label='Gini Index', linewidth=2, markersize=6, color='m')
plt.ylabel('Gini Index')
plt.legend(loc='upper left')
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 绘制城镇人口与农村人口
plt.subplot(3, 2, 6)
plt.plot(years, urban_population, 'o-', label='Urban Population', linewidth=2, markersize=6, color='g')
plt.plot(years, rural_population, 's-', label='Rural Population', linewidth=2, markersize=6, color='r')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Urban vs Rural Population Change')
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=12)

# 设置整体标题
plt.suptitle('Development of the Pet Industry in China (2019-2023)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 4. 输出数据
data_summary = pd.DataFrame({
    'Year': years,
    'Cat_Count': cat_count,
    'Dog_Count': dog_count,
    'Total_Pet_Count': total_pet_count,
    'Market_Size_Billion_USD': market_size,
    'Pet_Food_Expenditure_Billion_USD': pet_food_expenditure,
    'Veterinary_Service_Expenditure_Billion_USD': vet_service_expenditure,
    'Pet_Family_Penetration': pet_family_penetration,
    'Pet_Medical_Market_Size_Billion_CNY': pet_medical_market_size,
    'Total_Population': total_population,
    'Population_Growth_Rate': population_growth_rate,
    'Gini_Index': gini_index,
    'Urban_Population': urban_population,
    'Rural_Population': rural_population,
    'GDP_Per_Capita_USD': gdp_per_capita
})

print('Summary of Pet Industry Data (2019-2023):')
print(data_summary)
