# 数据准备
import numpy as np
import matplotlib.pyplot as plt

# 原始数据 (年份和猫狗数量)
years = np.array([2019, 2020, 2021, 2022, 2023])
data_cats = np.array([4412, 4862, 5806, 6536, 6980])
data_dogs = np.array([5503, 5222, 5429, 5119, 5175])

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# GM(1,1) 灰色预测函数定义
def GM11(data, predict_years):
    # 计算累计生成序列
    n = len(data)
    x1 = np.cumsum(data)

    # 构造数据矩阵B和向量Y
    B = np.vstack((-0.5 * (x1[:-1] + x1[1:]), np.ones(n - 1))).T
    Y = data[1:]

    # 求解参数向量a和b
    U = np.linalg.inv(B.T @ B) @ B.T @ Y
    a, b = U

    # 构造灰色预测模型的方程
    x1_hat = np.zeros(n + predict_years)
    x1_hat[0] = data[0]
    for k in range(1, n + predict_years):
        x1_hat[k] = (data[0] - b / a) * np.exp(-a * k) + b / a

    # 还原预测值
    prediction = np.diff(x1_hat, prepend=data[0])
    return prediction[:n + predict_years]


# 对猫和狗的数量进行灰色预测
# 1. 对猫数量进行GM(1,1)灰色预测
pred_cats = GM11(data_cats, 3)  # 预测未来3年的数据

# 2. 对狗数量进行GM(1,1)灰色预测
pred_dogs = GM11(data_dogs, 3)  # 预测未来3年的数据

# 可视化原始数据与预测结果
future_years = np.array([2024, 2025, 2026])
all_years = np.concatenate((years, future_years))

plt.figure(figsize=(10, 8))

# 猫的数据可视化
plt.subplot(2, 1, 1)
plt.plot(years, data_cats, 'o-b', linewidth=1.5, label='实际数据')
plt.plot(all_years, pred_cats, '-r', linewidth=1.5, label='预测数据')
plt.title('猫数量的灰色预测')
plt.xlabel('年份')
plt.ylabel('数量 (万)')
plt.grid(True)
plt.legend(loc='upper left')

# 狗的数据可视化
plt.subplot(2, 1, 2)
plt.plot(years, data_dogs, 'o-g', linewidth=1.5, label='实际数据')
plt.plot(all_years, pred_dogs, '-r', linewidth=1.5, label='预测数据')
plt.title('狗数量的灰色预测')
plt.xlabel('年份')
plt.ylabel('数量 (万)')
plt.grid(True)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# 关键步骤结果的可视化
print('关键过程结果：')
print(f'猫的预测数据：{pred_cats}')
print(f'狗的预测数据：{pred_dogs}')
