import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# 1. Data Definition
years = np.array([2019, 2020, 2021, 2022, 2023])
# Data indicators
pet_count = np.array([99.8, 108.5, 115.4, 122.6, 130.2])  # Pet count (million)
pet_food_output_value = np.array([440.7, 727.3, 1554, 1508, 2793])  # Total pet food production value (CNY billion)
pet_food_export_value = np.array([154.1, 9.8, 12.2, 24.7, 39.6])  # Total pet food export value (USD billion)
market_size = np.array([33.2, 35.6, 38.9, 42.1, 45.5])  # Pet market size (USD billion)
pet_household_penetration = np.array([0.18, 0.2, 0.2, 0.2, 0.22])  # Pet household penetration rate
food_export_percentage = np.array([2.8745, 2.6897, 2.3228, 2.3002, 2.4563])  # Food export percentage of goods
import_volume_index = np.array([120.23, 126.07, 134.64, 139.98, 145.32])  # Import volume index
population_growth = np.array([0.3547, 0.2380, 0.0893, -0.0131, -0.1038])  # Population growth rate (annual %)
gdp_per_capita = np.array([10143.86, 10408.72, 12617.51, 12662.58, 12614.06])  # GDP per capita (USD)
global_market_size = np.array([1000, 1055, 1149.42, 1200, 1250])  # Global market size (USD billion)
exchange_rate_usd = np.array([6.9, 6.9, 6.45, 6.75, 7.12])  # Exchange rate (USD/CNY)
us_pet_food_import = np.array([15.3, 16, 17.5, 19, 21.3])  # US pet food import (USD billion)
tarrif_usa_addition = np.array([0.25, 0.25, 0.25, 0.25, 0.25])# 美国对中国加征关税（25%）
tarrif_eu_addition = np.array([0, 0, 0, 0, 0])# 欧盟对中国加征关税（无）

# 2. Construct Models
# Extract features related to export
X_export = np.column_stack((food_export_percentage, population_growth, exchange_rate_usd, us_pet_food_import))

# Scale the features
scaler = StandardScaler()
X_export_scaled = scaler.fit_transform(X_export)

# Fit SVR model
svr_export_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_export_model.fit(X_export_scaled, pet_food_export_value)

# Fit Random Forest Regressor model
rf_export_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_export_model.fit(X_export_scaled, pet_food_export_value)

# Fit Linear Regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_export_scaled, pet_food_export_value)

# Fit ARIMA model for time series prediction
arima_model = ARIMA(pet_food_export_value, order=(1, 1, 1))
arima_model_fit = arima_model.fit()

# 3. Predict future values for population growth rate, exchange rate, and US import volume
years_future = np.array([2024, 2025, 2026, 2027])  # Future four years
pop_growth_rate_future = np.array([-0.12, -0.15, -0.18, -0.20])  # Future four years population growth rate
exchange_rate_usd_future = exchange_rate_usd[-1] * (1 + 0.02) ** np.arange(1, 5)  # Exchange rate (USD/CNY) assuming 2% annual growth
us_pet_food_import_future = us_pet_food_import[-1] * (1 + 0.03) ** np.arange(1, 5)  # US pet food import volume assuming 3% annual growth
food_export_percentage_future = food_export_percentage[-1] * (1 + 0.01) ** np.arange(1, 5)  # Food export percentage assuming 1% annual growth
tarrif_usa_addition_future = tarrif_usa_addition[-1] * (1 + 0.01) ** np.arange(1, 5)  # 美国对中国加征关税（25%）每年1%增长
tarrif_eu_addition_future = tarrif_eu_addition[-1] * (1 + 0.01) ** np.arange(1, 5)  # 欧盟对中国加征关税（无）每年1%增长

# Future features matrix
gdp_future_features = np.column_stack((food_export_percentage_future, pop_growth_rate_future, exchange_rate_usd_future, us_pet_food_import_future))
gdp_future_features_scaled = scaler.transform(gdp_future_features)

# 4. Use Models to predict future pet food export values
# SVR Prediction
pet_food_export_value_svr_future = svr_export_model.predict(gdp_future_features_scaled)

# Random Forest Prediction
pet_food_export_value_rf_future = rf_export_model.predict(gdp_future_features_scaled)

# Linear Regression Prediction
pet_food_export_value_lr_future = linear_regression_model.predict(gdp_future_features_scaled)

# ARIMA Prediction
pet_food_export_value_arima_future = arima_model_fit.forecast(steps=4)

# 5. Visualization of prediction results
plt.figure(figsize=(12, 8))

# Pet food export prediction (Historical)
plt.plot(years, pet_food_export_value, '-o', label='Historical Pet Food Export Value', linewidth=2, markersize=6, color='b')

# SVR prediction
plt.plot(years_future, pet_food_export_value_svr_future, '-s', label='Support Vector Regression (SVR)', linewidth=2, markersize=6, color='r')

# Random Forest prediction
plt.plot(years_future, pet_food_export_value_rf_future, '-^', label='Random Forest Regressor', linewidth=2, markersize=6, color='g')

# Linear Regression prediction
plt.plot(years_future, pet_food_export_value_lr_future, '-d', label='Linear Regression', linewidth=2, markersize=6, color='m')

# ARIMA prediction
plt.plot(years_future, pet_food_export_value_arima_future, '-x', label='ARIMA', linewidth=2, markersize=6, color='c')

# Plot settings
plt.title('Prediction of Pet Food Export Value (2019-2027)')
plt.xlabel('Year')
plt.ylabel('Export Value (USD Billion)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print forecast results
print('Forecast of Pet Food Export Value in China for the next four years:')
for i in range(len(years_future)):
    print(f'{years_future[i]}:')
    print(f'  SVR: {pet_food_export_value_svr_future[i]:.2f} USD Billion')
    print(f'  Random Forest: {pet_food_export_value_rf_future[i]:.2f} USD Billion')
    print(f'  Linear Regression: {pet_food_export_value_lr_future[i]:.2f} USD Billion')
    print(f'  ARIMA: {pet_food_export_value_arima_future[i]:.2f} USD Billion')
