import numpy as np
import matplotlib.pyplot as plt
# Data Preparation
years = np.array([2019, 2020, 2021, 2022, 2023])
data_pet_food_production = np.array([440.7, 727.3, 1554, 1508, 2793])
data_pet_food_export = np.array([154.1, 9.8, 12.2, 24.7, 39.6])

# Number of years and future years
n_years = len(years)
future_years = np.array([2024, 2025, 2026])

# Linear Regression for Pet Food Production
X = np.vstack([years, np.ones(n_years)]).T  # Construct independent variable matrix
b_production = np.linalg.lstsq(X, data_pet_food_production, rcond=None)[0]  # Least squares fitting

# Forecast for the next three years
future_X = np.vstack([future_years, np.ones(len(future_years))]).T
production_forecast = future_X.dot(b_production)

# Linear Regression for Pet Food Export
b_export = np.linalg.lstsq(X, data_pet_food_export, rcond=None)[0]  # Least squares fitting

# Forecast for the next three years
export_forecast = future_X.dot(b_export)

# Visualization of original data and forecast results
all_years = np.concatenate([years, future_years])

plt.figure(figsize=(10, 8))

# Pet Food Production Plot
plt.subplot(2, 1, 1)
plt.plot(years, data_pet_food_production, 'o-b', linewidth=1.5, label='Actual Data')
plt.plot(future_years, production_forecast, 'r--', linewidth=1.5, label='Forecast Data')
plt.title('Prediction of Total Pet Food Production in China')
plt.xlabel('Year')
plt.ylabel('Total Production (CNY Billion)')
plt.grid(True)
plt.legend(loc='upper left')

# Pet Food Export Plot
plt.subplot(2, 1, 2)
plt.plot(years, data_pet_food_export, 'o-g', linewidth=1.5, label='Actual Data')
plt.plot(future_years, export_forecast, 'r--', linewidth=1.5, label='Forecast Data')
plt.title('Prediction of Pet Food Export in China')
plt.xlabel('Year')
plt.ylabel('Export Value (USD Billion)')
plt.grid(True)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Print forecast results
print('Forecast of Total Pet Food Production in China for the next three years:')
for i in range(len(future_years)):
    print(f'{future_years[i]}: {production_forecast[i]:.2f} CNY Billion')

print('\nForecast of Pet Food Export in China for the next three years:')
for i in range(len(future_years)):
    print(f'{future_years[i]}: {export_forecast[i]:.2f} USD Billion')
