# 数据准备
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 原始数据 (年份和宠物食品产值与出口值)
years = np.array([2019, 2020, 2021, 2022, 2023])
data_pet_food_production = np.array([440.7, 727.3, 1554, 1508, 2793])
data_pet_food_export = np.array([154.1, 9.8, 12.2, 24.7, 39.6])

# 对生产和出口数据进行时间序列分析并进行预测
future_years = np.array([2024, 2025, 2026])

# 使用线性回归对宠物食品总产值进行预测
X = np.vstack([years, np.ones(len(years))]).T  # 构造自变量矩阵
b_production = np.linalg.lstsq(X, data_pet_food_production, rcond=None)[0]  # 最小二乘拟合

# 预测未来三年
future_X = np.vstack([future_years, np.ones(len(future_years))]).T
production_forecast = future_X @ b_production

# 使用线性回归对宠物食品出口总值进行预测
b_export = np.linalg.lstsq(X, data_pet_food_export, rcond=None)[0]  # 最小二乘拟合

# 预测未来三年
export_forecast = future_X @ b_export

# 可视化原始数据与预测结果
all_years = np.concatenate((years, future_years))
plt.figure(figsize=(10, 8))

# 生产总值预测
plt.subplot(2, 1, 1)
plt.plot(years, data_pet_food_production, 'o-b', linewidth=1.5, label='实际数据')
plt.plot(future_years, production_forecast, 'r--', linewidth=1.5, label='预测数据')
plt.title('中国宠物食品总产值的预测')
plt.xlabel('年份')
plt.ylabel('总产值 (亿元人民币)')
plt.grid(True)
plt.legend(loc='upper left')

# 出口总值预测
plt.subplot(2, 1, 2)
plt.plot(years, data_pet_food_export, 'o-g', linewidth=1.5, label='实际数据')
plt.plot(future_years, export_forecast, 'r--', linewidth=1.5, label='预测数据')
plt.title('中国宠物食品出口总值的预测')
plt.xlabel('年份')
plt.ylabel('出口总值 (亿美元)')
plt.grid(True)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# 打印预测结果
print('未来三年中国宠物食品总产值预测：')
for year, value in zip(future_years, production_forecast):
    print(f'{year} 年: {value:.2f} 亿元人民币')

print('\n未来三年中国宠物食品出口总值预测：')
for year, value in zip(future_years, export_forecast):
    print(f'{year} 年: {value:.2f} 亿美元')
