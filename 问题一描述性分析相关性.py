# 数据准备
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

years = np.array([2019, 2020, 2021, 2022, 2023])
data_cats = np.array([4412, 4862, 5806, 6536, 6980])
data_dogs = np.array([5503, 5222, 5429, 5119, 5175])
data_pet_food_production = np.array([440.7, 727.3, 1554, 1508, 2793])
data_pet_food_export = np.array([154.1, 9.8, 12.2, 24.7, 39.6])
data_pet_quantity = np.array([99.8, 108.5, 115.4, 122.6, 130.2])
data_pet_market_size = np.array([33.2, 35.6, 38.9, 42.1, 45.5])
data_pet_food_spending = np.array([15.1, 16.2, 17.5, 18.9, 20.3])
data_vet_services_spending = np.array([3.4, 3.8, 4.1, 4.5, 4.9])
data_pet_household_penetration = np.array([0.18, 0.2, 0.2, 0.2, 0.22])
data_pet_medical_market_size = np.array([400, 500, 600, 640, 700])
data_export_residual = np.array([0.069201579, 0.065434606, 0.179612486, 0.16517555, 0.150738614])
data_export_to_latam = np.array([4.958242168, 4.682402241, 5.324499456, 5.477973205, 5.631446953])
data_export_to_high_income = np.array([67.87573629, 68.86799377, 65.88879755, 64.93652255, 63.98424755])
data_manufacturing_export = np.array([92.87408425, 93.54165793, 93.4459168, 92.64633695, 91.88215829])
data_food_export = np.array([2.87450637, 2.68970992, 2.322785721, 2.300244385, 2.456280799])

# 构造数据框用于计算相关性
all_data = pd.DataFrame({
    '猫数量(万)': data_cats,
    '狗数量(万)': data_dogs,
    '中国宠物食品总产值(人民币)': data_pet_food_production,
    '中国宠物食品出口总值(美元)': data_pet_food_export,
    '宠物数量(百万)': data_pet_quantity,
    '宠物市场规模(亿美元)': data_pet_market_size,
    '宠物食品开支(亿美元)': data_pet_food_spending,
    '兽医服务开支(亿美元)': data_vet_services_spending,
    '中国宠物家庭渗透率': data_pet_household_penetration,
    '宠物医疗市场规模(亿元人民币)': data_pet_medical_market_size,
    '报告经济体的商品出口，剩余(占商品出口总额的百分比)': data_export_residual,
    '向拉丁美洲和加勒比地区的发展中经济体的商品出口(占商品出口总额的百分比)': data_export_to_latam,
    '向高收入经济体的商品出口(占商品出口总额的百分比)': data_export_to_high_income,
    '制造业出口(占商品出口的百分比)': data_manufacturing_export,
    '食品出口(占商品出口的百分比)': data_food_export
})

# 计算相关性矩阵
correlation_matrix = all_data.corr()

# 绘制热力图表示相关性
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('指标之间的相关性矩阵热力图')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
