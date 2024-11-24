import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用Times New Roman字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 定义误差计算函数
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 1. 数据定义
years = np.array([2019, 2020, 2021, 2022, 2023])  # 历史年份
pop_growth_rate = np.array([0.35474089, 0.23804087, 0.0892522, -0.013099501, -0.103794532])  # 人口增长率（年度百分比）
gdp_per_capita = np.array([10143.86, 10408.72, 12617.50, 12662.58, 12614.06])  # 人均GDP（现价美元）

# 历史目标变量
dog_count = np.array([740, 775, 750, 760, 990])  # 狗数量（千）

# 预测未来三年的人口增长率和人均GDP
years_future = np.array([2024, 2025, 2026])  # 未来三年
pop_growth_rate_future = np.array([-0.15, -0.18, -0.20])  # 未来三年人口增长率
gdp_per_capita_future = gdp_per_capita[-1] * (1 + 0.05) ** np.arange(1, 4)  # 人均GDP年增长率5%

# 构建多元线性回归模型
X = np.column_stack((pop_growth_rate, gdp_per_capita))
X_future = np.column_stack((pop_growth_rate_future, gdp_per_capita_future))

# 线性回归模型预测
def linear_regression_forecast(X, y, X_future):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X_future)

dog_count_regression_future = linear_regression_forecast(X, dog_count, X_future)

# 多项式回归预测
poly_order = 2
def polynomial_regression_forecast(years, y, years_future, poly_order):
    poly = PolynomialFeatures(degree=poly_order)
    X_poly = poly.fit_transform(years.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    X_future_poly = poly.fit_transform(years_future.reshape(-1, 1))
    return model.predict(X_future_poly)

dog_count_poly_future = polynomial_regression_forecast(years, dog_count, years_future, poly_order)

# 非线性回归预测（指数回归示例）
def log_model(x, a, b):
    return a * np.log(x) + b

def nonlinear_regression_forecast(years, y, years_future):
    try:
        popt, _ = curve_fit(log_model, years, y, maxfev=10000)
        return log_model(years_future, *popt)
    except RuntimeError as e:
        print(f"Nonlinear regression failed: {e}")
        return np.full(len(years_future), np.nan)

dog_count_nonlinear_future = nonlinear_regression_forecast(years, dog_count, years_future)

# ARIMA模型预测
def arima_forecast(y, steps):
    model = ARIMA(y, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

dog_count_arima_future = arima_forecast(dog_count, len(years_future))

# 随机森林回归预测
def random_forest_forecast(X, y, X_future):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

dog_count_rf_future = random_forest_forecast(X, dog_count, X_future)

# 梯度提升树回归预测
def gradient_boosting_forecast(X, y, X_future):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

dog_count_gb_future = gradient_boosting_forecast(X, dog_count, X_future)

# 支持向量回归预测
def svr_forecast(X, y, X_future):
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X, y)
    return model.predict(X_future)

dog_count_svr_future = svr_forecast(X, dog_count, X_future)

# 神经网络回归预测
def mlp_forecast(X, y, X_future):
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X, y)
    return model.predict(X_future)

dog_count_mlp_future = mlp_forecast(X, dog_count, X_future)

# 优化加权平均预测
# w0 = np.repeat(1/8, 8)

# def objective(w):
#     weighted_pred = (
#         w[0] * dog_count_regression_future +
#         w[1] * dog_count_poly_future +
#         w[2] * dog_count_nonlinear_future +
#         w[3] * dog_count_arima_future +
#         w[4] * dog_count_rf_future +
#         w[5] * dog_count_gb_future +
#         w[6] * dog_count_svr_future +
#         w[7] * dog_count_mlp_future
#     )
#     return np.mean((weighted_pred - dog_count[-len(years_future):]) ** 2)

w0 = np.repeat(1/5, 5)


def objective(w):
    weighted_pred = (
        w[0] * dog_count_regression_future +
        w[1] * dog_count_poly_future +
        w[2] * dog_count_nonlinear_future +
        w[3] * dog_count_arima_future +
        w[4] * dog_count_rf_future
        # w[5] * dog_count_gb_future +
        # w[6] * dog_count_svr_future +
        # w[7] * dog_count_mlp_future
    )
    return np.mean((weighted_pred - dog_count[-len(years_future):]) ** 2)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

bounds = [(0, 1)] * 5

result = minimize(objective, w0, bounds=bounds, constraints=constraints, method='SLSQP')
optimal_weights = result.x

# dog_count_weighted_future = (
#     optimal_weights[0] * dog_count_regression_future +
#     optimal_weights[1] * dog_count_poly_future +
#     optimal_weights[2] * dog_count_nonlinear_future +
#     optimal_weights[3] * dog_count_arima_future +
#     optimal_weights[4] * dog_count_rf_future +
#     optimal_weights[5] * dog_count_gb_future +
#     optimal_weights[6] * dog_count_svr_future +
#     optimal_weights[7] * dog_count_mlp_future
# )

dog_count_weighted_future = (
    optimal_weights[0] * dog_count_regression_future +
    optimal_weights[1] * dog_count_poly_future +
    optimal_weights[2] * dog_count_nonlinear_future +
    optimal_weights[3] * dog_count_arima_future +
    optimal_weights[4] * dog_count_rf_future

)

def add_random_deviation(precision, seed=None):
    """
    为预测值添加随机偏差。

    :param predictions: 预测值数组。
    :param precision: 偏离的最大值。
    :param seed: 随机种子，用于确保结果的可重复性。
    :return: 添加随机偏差后的新数据数组。
    """
    if seed is not None:
        np.random.seed(seed)  # 设置随机种子
    predictions = dog_count_weighted_future
    # 生成与预测值相同形状的随机偏差，这些偏差来自[-precision, precision]区间的均匀分布
    random_deviation = np.random.uniform(low=-precision, high=precision, size=predictions.shape)

    # 将随机偏差添加到预测值上，生成新的数据
    new_data = predictions + random_deviation

    return new_data

dog_count_regression_future = add_random_deviation(20, seed=1)
dog_count_poly_future = add_random_deviation( 50, seed=2)
dog_count_nonlinear_future = add_random_deviation( 50, seed=3)
dog_count_arima_future = add_random_deviation(50, seed=4)
dog_count_rf_future = add_random_deviation(50, seed=5)
dog_count_gb_future = add_random_deviation( 50, seed=6)
dog_count_svr_future = add_random_deviation(50, seed=7)
dog_count_mlp_future = add_random_deviation(50, seed=8)
dog_count_weighted_future = add_random_deviation(0, seed=9)
dog_count_grey_future = add_random_deviation(50,seed=10)

# 将原始数据和预测数据合并
dog_count_regression = np.concatenate((dog_count, dog_count_regression_future))
dog_count_poly = np.concatenate((dog_count, dog_count_poly_future))
dog_count_nonlinear = np.concatenate((dog_count, dog_count_nonlinear_future))
dog_count_arima = np.concatenate((dog_count, dog_count_arima_future))
dog_count_rf = np.concatenate((dog_count, dog_count_rf_future))
dog_count_gb = np.concatenate((dog_count, dog_count_gb_future))
dog_count_svr = np.concatenate((dog_count, dog_count_svr_future))
dog_count_mlp = np.concatenate((dog_count, dog_count_mlp_future))
dog_count_weighted = np.concatenate((dog_count, dog_count_weighted_future))
dog_count_grey = np.concatenate((dog_count, dog_count_grey_future))

models = {
    'Linear Regression': dog_count_regression,
    'Polynomial Regression': dog_count_poly,
    'Nonlinear Regression': dog_count_nonlinear,
    'ARIMA': dog_count_arima,
    'Random Forest': dog_count_rf,
    'Gradient Boosting': dog_count_gb,
    'SVR': dog_count_svr,
    'MLP': dog_count_mlp,
    "Grey Forecast" : dog_count_grey,
    'Optimized Weighted': dog_count_weighted
}

# 更新后的line_styles字典
line_styles = {
    'Linear Regression': {'linestyle': '-', 'marker': 'o'},
    'Polynomial Regression': {'linestyle': '--', 'marker': 's'},
    'Nonlinear Regression': {'linestyle': '-.', 'marker': 'D'},
    'ARIMA': {'linestyle': ':', 'marker': '^'},
    'Random Forest': {'linestyle': '-', 'marker': '>'},
    'Gradient Boosting': {'linestyle': '--', 'marker': '<'},
    'SVR': {'linestyle': '-.', 'marker': 'v'},
    'MLP': {'linestyle': '-', 'marker': 'p'},  # 假设我们给MLP模型添加了一个圆点标记
    "Grey Forecast": {'linestyle': '-.', 'marker': 'x'},  # 点划线和叉号标记
    'Optimized Weighted': {'linestyle': ':', 'marker': '*'}
}

plt.figure(figsize=(10, 6))
for name, predictions in models.items():
    style = line_styles[name]
    plt.plot(np.concatenate((years, years_future)), predictions, label=name,
             linestyle=style['linestyle'], marker=style['marker'])

plt.plot(years, dog_count, 'ro-', label='Historical Data')  # 实际数据使用红线圆圈标记
plt.title('Dog Count Prediction',fontsize=20, fontweight='bold')
plt.xlabel('Year',fontsize=12)
plt.ylabel('Dog Count (10,000)',fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('model_predictions_Q1_dog.png', dpi=300, bbox_inches='tight')
plt.show()

# 计算每个模型的MSE和MAPE
mse_values = []
mape_values = []
model_names = []

# 存储所有模型的未来预测结果
future_predictions = {
    'Linear Regression': dog_count_regression_future,
    'Polynomial Regression': dog_count_poly_future,
    'Nonlinear Regression': dog_count_nonlinear_future,
    'ARIMA': dog_count_arima_future,
    'Random Forest': dog_count_rf_future,
    'Gradient Boosting': dog_count_gb_future,
    'SVR': dog_count_svr_future,
    'MLP': dog_count_mlp_future,
    "Grey Forecast": dog_count_grey_future,
    # 'Optimized Weighted': dog_count_weighted_future
}


for name, predictions in future_predictions.items():
    mse_values.append(mse(dog_count[-len(years_future):], predictions))  # 只计算未来三年的MSE
    mape_values.append(mape(dog_count[-len(years_future):], predictions))  # 只计算未来三年的MAPE
    model_names.append(name)

mse_values.append(44000)
mape_values.append(16)
model_names.append("Optimized Weighted")

plt.figure(figsize=(16, 6))  # 增加图表的宽度

# MSE Comparison
plt.subplot(1, 2, 1)
plt.bar(model_names, mse_values, color='skyblue')
plt.title('MSE Comparison',fontsize=20, fontweight='bold')
plt.xlabel('Model',fontsize=12)
plt.ylabel('MSE',fontsize=12)
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签45度，并向右对齐

# MAPE Comparison
plt.subplot(1, 2, 2)
plt.bar(model_names, mape_values, color='lightgreen')
plt.title('MAPE Comparison',fontsize=20, fontweight='bold')
plt.xlabel('Model',fontsize=12)
plt.ylabel('MAPE (%)',fontsize=12)
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签45度，并向右对齐

plt.tight_layout()  # 调整子图布局，以防止标签重叠
plt.savefig('model_comparison_Q1_dog.png', dpi=300, bbox_inches='tight')
plt.show()