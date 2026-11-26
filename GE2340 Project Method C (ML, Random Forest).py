# -*- coding: utf-8 -*-
"""
Method C (Adjusted) -- Random Forest (corrected version)
========================================================
On top of the original Random Forest, we add only one step, "exam-week online
recalibration"; the features and parameters are unchanged.

Note: in the original script, is_exam_week is one of the features, but it is
always 0 across the 9 training weeks, so the trees never split on it
(importance = 0) and keeping or dropping it makes almost no difference.
We keep it here (to match the original script); the exam week is actually
handled by the recalibration in step 4.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ---------------- 0. Load data ----------------
df = pd.read_csv('canteen_orders_synthetic.csv', parse_dates=['date'])
split = pd.to_datetime('2025-11-01')
train = df[df.date < split].copy()
test = df[df.date >= split].copy()

# ---------------- 1. Features and training ----------------
features = ['window_id', 'slot_start', 'day_of_week', 'is_exam_week']
Xtr = pd.get_dummies(train[features], columns=['window_id', 'slot_start', 'day_of_week'])
Xte = pd.get_dummies(test[features], columns=['window_id', 'slot_start', 'day_of_week'])
Xtr, Xte = Xtr.align(Xte, join='left', axis=1, fill_value=0)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(Xtr, train['order_count'])
pred = rf.predict(Xte)

# ---------------- 2. Exam-week online recalibration (same as Method B) ----------------
t = test.copy()
t['pred'] = pred
pred_adj = pred.copy()
uplift_pool = []
exam_days = sorted(t[t.is_exam_week == 1]['date'].unique())

for d in exam_days:
    m = (t['date'] == d).values
    k = 1.0 if not uplift_pool else float(np.clip(np.mean(uplift_pool), 1.0, 1.35))
    pred_adj[m] = pred_adj[m] * k
    uplift_pool += list(t.loc[m, 'order_count'].values / t.loc[m, 'pred'].values)

# ---------------- 3. Results ----------------
y = test['order_count'].values
rmse_no = np.sqrt(mean_squared_error(y, pred))
rmse_adj = np.sqrt(mean_squared_error(y, pred_adj))
print(f"Method C (adjusted)  RMSE  without exam-week recalibration = {rmse_no:.2f}")
print(f"Method C (adjusted)  RMSE  with    exam-week recalibration = {rmse_adj:.2f}")
print(f"Improvement = {rmse_no - rmse_adj:.2f} orders / window / slot")

# ---------------- 4. Feature importance (for the report) ----------------
imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
print("\nTop 8 feature importances:")
print(imp.head(8).round(4).to_string())
print(f"\nis_exam_week importance = {imp.get('is_exam_week', 0.0):.4f}"
      " (constant 0 in training, so the trees never learn it)")
