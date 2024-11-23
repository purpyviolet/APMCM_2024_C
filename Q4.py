
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Cat_Count': [4412, 4862, 5806, 6536, 6980],  # 猫数量（万只）
    'Dog_Count': [5503, 5222, 5429, 5119, 5175],  # 狗数量（万只）
    'Production_Value_RMB': [440.7, 727.3, 1554, 1508, 2793],  # 总产值（亿元人民币）
    'Export_Value_USD': [154.1, 9.8, 12.2, 24.7, 39.6],  # 出口总值（亿美元）
    'Tariff_USA_Addition': [0.25, 0.25, 0.25, 0.25, 0.25],  # 美国对中国加征关税（25%）
    'Tariff_EU_Addition': [0, 0, 0, 0, 0],  # 欧盟对中国加征关税（无）
    'Market_Size_Billion_USD': [33.2, 35.6, 38.9, 42.1, 45.5],
    'Exchange_Rate_USD_CNY': [6.9, 6.9, 6.45, 6.75, 7.12],
    'Exchange_Rate_EUR_CNY': [7.75, 8.02, 7.65, 7.5, 7.34],
    'US_Import_PetFood': [15.3, 16, 17.5, 19, 21.3],
    'EU_Import_PetFood': [20.5, 21, 22, 24.4, 27.2]
}

df = pd.DataFrame(data)

df['Cat_Dog_Interaction'] = df['Cat_Count'] * df['Dog_Count']
df['Production_Market_Interaction'] = df['Production_Value_RMB'] * df['Market_Size_Billion_USD']

features = [
    'Cat_Count', 'Dog_Count', 'Production_Value_RMB', 'Tariff_USA_Addition', 'Tariff_EU_Addition',
    'Market_Size_Billion_USD', 'Exchange_Rate_USD_CNY', 'Exchange_Rate_EUR_CNY',
    'US_Import_PetFood', 'EU_Import_PetFood', 'Cat_Dog_Interaction', 'Production_Market_Interaction'
]
target = 'Export_Value_USD'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_model = Lasso(alpha=0.01, positive=True,random_state=42)
lasso_model.fit(X_scaled, y)

selected_features = np.array(features)[lasso_model.coef_ != 0]
print("Selected features by Lasso:", selected_features)

X_selected = df[selected_features]
X_selected = df[features]
X_selected_scaled = scaler.fit_transform(X_selected)

lasso_model.fit(X_selected_scaled, y)
y_pred = lasso_model.predict(X_selected_scaled)

r2 = r2_score(y, y_pred)
print("R^2 Score:", r2)

# coefficients = pd.DataFrame(lasso_model.coef_, selected_features, columns=['Coefficient'])
# print("\nFeature Coefficients:")
# print(coefficients)

future_years = [2024, 2025, 2026]
future_data = pd.DataFrame({
    'Cat_Count': [6536, 6536, 6536],
    'Dog_Count': [5119, 5119, 5119],
    'Production_Value_RMB': [3381.075, 3969.15, 4557.225],
    'Tariff_USA_Addition': [0.25, 0.25, 0.25],
    'Tariff_EU_Addition': [0, 0, 0],
    'Market_Size_Billion_USD': [48.575, 51.65, 54.725],
    'Exchange_Rate_EUR_CNY': [7.5, 7.6, 7.7],
    'US_Import_PetFood': [22.8, 24.3, 25.8],
    'EU_Import_PetFood': [28.875, 30.55, 32.225],
    'Cat_Dog_Interaction': [6536 * 5119, 6536 * 5119, 6536 * 5119],
    'Exchange_Rate_USD_CNY': [7.4, 7.5, 7.6],
    'Production_Market_Interaction': [3381.075 * 48.575, 3969.15 * 51.65, 4557.225 * 54.725]
})

future_data_selected = future_data[selected_features]
future_data_selected = future_data[features]

future_data_scaled = scaler.transform(future_data_selected)

future_predictions = lasso_model.predict(future_data_scaled)
print("\nPredicted Export Value (2024-2026):")
for year, value in zip(future_years, future_predictions):
    print(f"Year {year}: {max(0, value):.2f} 亿美元")

