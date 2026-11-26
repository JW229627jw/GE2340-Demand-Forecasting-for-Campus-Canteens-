# -*- coding: utf-8 -*-
"""
Method B (Adjusted) -- calendar-adjusted rule (corrected version)
=================================================================
Only two things changed from the original Method B; the overall idea is unchanged.

1. Weekday coefficient corrected
   Original: day_coeff = (global mean for a weekday) / (global mean), multiplied
   onto the window x slot cell mean.
   Problem: the weekday effect was counted twice (the global mean already
   contains the weekday structure).
   Fix: first compute, for each row, the ratio to its OWN cell (window x slot),
        then average those ratios by weekday to get multipliers around 1.0.
        Multiplying these onto the cell mean no longer double-counts the weekday.

2. Exam-week online recalibration
   The 9 training weeks contain no exam week (is_exam_week is always 0), so
   "how much to scale up for exam week" cannot be learned at training time.
   Fix: during the exam week, estimate a scale factor k each day from
   (observed actuals / base forecast) of the days already seen, and use it to
   scale that day's and the following days' forecasts.
   This is the report's answer to the TA's "rare-event calendar adjustment".
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# ---------------- 0. Load data ----------------
# Put this script and canteen_orders_synthetic.csv in the same folder,
# or change the path below to your full path.
df = pd.read_csv('canteen_orders_synthetic.csv', parse_dates=['date'])

# Per the proposal: first 9 weeks train, last 3 weeks test
split = pd.to_datetime('2025-11-01')
train = df[df.date < split].copy()
test = df[df.date >= split].copy()

# ---------------- 1. Base value: mean per window x slot ----------------
base = train.groupby(['window_id', 'slot_start'])['order_count'].mean().rename('base')

# ---------------- 2. Weekday coefficient (corrected) ----------------
cell_mean = train.groupby(['window_id', 'slot_start'])['order_count'].transform('mean')
train['ratio'] = train['order_count'] / cell_mean          # each row relative to its own cell
dow_coeff = train.groupby('day_of_week')['ratio'].mean()   # average by weekday, around 1.0

# ---------------- 3. Base forecast: base x weekday coefficient ----------------
t = test.merge(base, on=['window_id', 'slot_start'], how='left')
t['base'] = t['base'].fillna(train['order_count'].mean())  # unseen combinations -> global mean
t['pred'] = t['base'] * t['day_of_week'].map(dow_coeff)

# ---------------- 4. Exam-week online recalibration ----------------
pred_adj = t['pred'].values.copy()
uplift_pool = []                                  # observed exam-week actual/base ratios so far
exam_days = sorted(t[t.is_exam_week == 1]['date'].unique())

for d in exam_days:
    m = (t['date'] == d).values
    # Day one has no prior, so use k = 1.0; afterwards use the mean of past days
    # (clamped to 1.0~1.35 so one odd day cannot drag it off).
    k = 1.0 if not uplift_pool else float(np.clip(np.mean(uplift_pool), 1.0, 1.35))
    pred_adj[m] = pred_adj[m] * k
    uplift_pool += list(t.loc[m, 'order_count'].values / t.loc[m, 'pred'].values)

# ---------------- 5. Results ----------------
y = t['order_count'].values
rmse_no = np.sqrt(mean_squared_error(y, t['pred'].values))
rmse_adj = np.sqrt(mean_squared_error(y, pred_adj))

print("Weekday coefficients (corrected, around 1.0):")
print(dow_coeff.round(3).to_string())
print()
print(f"Method B (adjusted)  RMSE  without exam-week recalibration = {rmse_no:.2f}")
print(f"Method B (adjusted)  RMSE  with    exam-week recalibration = {rmse_adj:.2f}")
print(f"Improvement = {rmse_no - rmse_adj:.2f} orders / window / slot")
