import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv(
    r'c:\Users\21377\OneDrive\Cityu BS 26-27\Sem A 26-27\GE2340\GE2340 Project\GE2340 Phase 2\canteen_orders_synthetic.csv')
df['date'] = pd.to_datetime(df['date'])

# 1. Filter for ONE specific time series (e.g., Window 1, 12:00 slot)
series_df = df[(df['window_id'] == 1) & (
    df['slot_start'] == '12:00')].sort_values('date')
series = series_df['order_count'].values

# 2. Split: first 9 weeks (45 days) for training, last 3 weeks (15 days) for testing
train_data = series[:45]
test_data = series[45:]

# 3. Fit ARIMA model (order=(p,d,q) - you can use (1,1,1) as a starting point)
model = ARIMA(train_data, order=(1, 1, 1))
fitted_model = model.fit()

# 4. Forecast the next 15 days
forecast = fitted_model.forecast(steps=15)

# 5. Evaluate
rmse = np.sqrt(mean_squared_error(test_data, forecast))
print(f"ARIMA RMSE for Window 1 (12:00): {rmse:.4f}")
