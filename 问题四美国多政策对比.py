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
exchange_rate_usd_future_stable = exchange_rate_usd[-1] * (1 + 0.00) ** np.arange(1, 5)  # Exchange rate (USD/CNY) assuming no change
exchange_rate_usd_future_increase = exchange_rate_usd[-1] * (1 + 0.10) ** np.arange(1, 5)  # Exchange rate (USD/CNY) assuming large increase (10% annually)
exchange_rate_usd_future_decrease = exchange_rate_usd[-1] * (1 - 0.10) ** np.arange(1, 5)  # Exchange rate (USD/CNY) assuming large decrease (10% annually)
us_pet_food_import_future = us_pet_food_import[-1] * (1 + 0.03) ** np.arange(1, 5)  # US pet food import volume assuming 3% annual growth
food_export_percentage_future = food_export_percentage[-1] * (1 + 0.01) ** np.arange(1, 5)  # Food export percentage assuming 1% annual growth

# Future features matrix for different exchange rate scenarios
gdp_future_features_stable = np.column_stack((food_export_percentage_future, pop_growth_rate_future, exchange_rate_usd_future_stable, us_pet_food_import_future))
gdp_future_features_increase = np.column_stack((food_export_percentage_future, pop_growth_rate_future, exchange_rate_usd_future_increase, us_pet_food_import_future))
gdp_future_features_decrease = np.column_stack((food_export_percentage_future, pop_growth_rate_future, exchange_rate_usd_future_decrease, us_pet_food_import_future))

# Scale future features
gdp_future_features_scaled_stable = scaler.transform(gdp_future_features_stable)
gdp_future_features_scaled_increase = scaler.transform(gdp_future_features_increase)
gdp_future_features_scaled_decrease = scaler.transform(gdp_future_features_decrease)

# 4. Use SVR to predict future pet food export values for different scenarios
# SVR Prediction (Stable Exchange Rate)
pet_food_export_value_svr_future_stable = svr_export_model.predict(gdp_future_features_scaled_stable)

# SVR Prediction (Increased Exchange Rate)
pet_food_export_value_svr_future_increase = svr_export_model.predict(gdp_future_features_scaled_increase)

# SVR Prediction (Decreased Exchange Rate)
pet_food_export_value_svr_future_decrease = svr_export_model.predict(gdp_future_features_scaled_decrease)

# 5. Visualization of prediction results
plt.figure(figsize=(14, 10))

# Pet food export prediction (Historical)
plt.plot(years, pet_food_export_value, '-o', label='Historical Pet Food Export Value', linewidth=2, markersize=6, color='b')

# SVR prediction (Stable Exchange Rate)
plt.plot(years_future, pet_food_export_value_svr_future_stable, '-s', label='SVR (Stable Exchange Rate)', linewidth=2, markersize=6, color='r')

# SVR prediction (Increased Exchange Rate)
plt.plot(years_future, pet_food_export_value_svr_future_increase, '-^', label='SVR (Increased Exchange Rate)', linewidth=2, markersize=6, color='g')

# SVR prediction (Decreased Exchange Rate)
plt.plot(years_future, pet_food_export_value_svr_future_decrease, '-d', label='SVR (Decreased Exchange Rate)', linewidth=2, markersize=6, color='m')

# Plot settings
plt.title('Prediction of Pet Food Export Value (2019-2027) with Different Exchange Rate Scenarios')
plt.xlabel('Year')
plt.ylabel('Export Value (USD Billion)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print forecast results for different exchange rate scenarios
print('Forecast of Pet Food Export Value in China for the next four years with Different Exchange Rate Scenarios:')
for i in range(len(years_future)):
    print(f'{years_future[i]}:')
    print(f'  SVR (Stable Exchange Rate): {pet_food_export_value_svr_future_stable[i]:.2f} USD Billion')
    print(f'  SVR (Increased Exchange Rate): {pet_food_export_value_svr_future_increase[i]:.2f} USD Billion')
    print(f'  SVR (Decreased Exchange Rate): {pet_food_export_value_svr_future_decrease[i]:.2f} USD Billion')
