import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. 原始数据定义
years = np.array([2019, 2020, 2021, 2022, 2023])  # 年份
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 原始数据，NaN 表示缺失值
vet_service_expense = np.array([np.nan, 1.9, 2.0, np.nan, 2.4])  # 兽医服务开支（亿美元）
pet_medical_market_size = np.array([1.2, np.nan, 1.4, np.nan, 1.6])  # 宠物医疗市场规模（亿欧元）
goods_export_percentage = np.array([0.7872, 0.4558, 0.4661, np.nan, 0.2965])  # 商品出口占比（百分比）
total_fertility_rate = np.array([1.86, 1.83, 1.84, 1.794, 1.8445])  # 总生育率（女性人均生育数）
# 2. 插值补全缺失值
# 使用线性插值进行补全

# 兽医服务开支的插值
mask = ~np.isnan(vet_service_expense)
interp_func = interp1d(years[mask], vet_service_expense[mask], kind='linear', fill_value='extrapolate')
vet_service_expense_interpolated = interp_func(years)

# 宠物医疗市场规模的插值
mask = ~np.isnan(pet_medical_market_size)
interp_func = interp1d(years[mask], pet_medical_market_size[mask], kind='linear', fill_value='extrapolate')
pet_medical_market_size_interpolated = interp_func(years)

# 商品出口占比的插值
mask = ~np.isnan(goods_export_percentage)
interp_func = interp1d(years[mask], goods_export_percentage[mask], kind='linear', fill_value='extrapolate')
goods_export_percentage_interpolated = interp_func(years)

# 生育率的数据没有缺失，不需要插值

# 3. 插值前后对比可视化
plt.figure(figsize=(10, 12))

# 兽医服务开支的插值前后对比
plt.subplot(3, 1, 1)
plt.plot(years, vet_service_expense, 'ro-', label='Original (With Missing Values)', linewidth=1.5, markersize=8)
plt.plot(years, vet_service_expense_interpolated, 'b*-', label='Interpolated', linewidth=1.5, markersize=8)
plt.title('Veterinary Service Expense Interpolation (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Expense (Billion USD)')
plt.legend()
plt.grid(True)

# 宠物医疗市场规模的插值前后对比
plt.subplot(3, 1, 2)
plt.plot(years, pet_medical_market_size, 'ro-', label='Original (With Missing Values)', linewidth=1.5, markersize=8)
plt.plot(years, pet_medical_market_size_interpolated, 'b*-', label='Interpolated', linewidth=1.5, markersize=8)
plt.title('Pet Medical Market Size Interpolation (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Market Size (Billion EUR)')
plt.legend()
plt.grid(True)

# 商品出口占比的插值前后对比
plt.subplot(3, 1, 3)
plt.plot(years, goods_export_percentage, 'ro-', label='Original (With Missing Values)', linewidth=1.5, markersize=8)
plt.plot(years, goods_export_percentage_interpolated, 'b*-', label='Interpolated', linewidth=1.5, markersize=8)
plt.title('Goods Export Percentage Interpolation (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Goods Export Percentage (%)')
plt.legend()
plt.grid(True)

plt.suptitle('Comparison of Original and Interpolated Data (France)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 4. 输出插值后的数据
interpolated_data = pd.DataFrame({
    'Year': years,
    'Veterinary_Service_Expense': vet_service_expense_interpolated,
    'Pet_Medical_Market_Size': pet_medical_market_size_interpolated,
    'Goods_Export_Percentage': goods_export_percentage_interpolated,
    'Total_Fertility_Rate': total_fertility_rate
})

print('Interpolated Data:')
print(interpolated_data)
