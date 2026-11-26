import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

print("开始运行 Method A: 纯历史平均法...")

# ==========================================
# 1. 加载数据 (已使用绝对路径，避免找不到文件)
# ==========================================
# 注意：如果路径中有任何变动，请修改下面这行
file_path = r'c:\Users\21377\OneDrive\Cityu BS 26-27\Sem A 26-27\GE2340\GE2340 Project\GE2340 Phase 2\canteen_orders_synthetic.csv'

try:
    df = pd.read_csv(file_path)
    print("数据加载成功！")
except FileNotFoundError:
    print(f"❌ 报错：依然找不到文件。请检查路径是否正确：\n{file_path}")
    exit()

df['date'] = pd.to_datetime(df['date'])

# 按照 Proposal：前 9 周训练，后 3 周测试
split_date = pd.to_datetime('2025-11-01')
train_df = df[df['date'] < split_date].copy()
test_df = df[df['date'] >= split_date].copy()

# ==========================================
# 2. 计算 Method A 的基础值
# ==========================================
# Method A 完全不考虑星期、考试周等因素。
# 它只算每个窗口(window_id)在特定时间段(slot_start)的历史平均订单量。

# 计算训练集的历史平均值
base_mean = train_df.groupby(['window_id', 'slot_start'])[
    'order_count'].mean().reset_index()
# 重命名列名（去掉了导致警告的 inplace=True）
base_mean = base_mean.rename(columns={'order_count': 'predicted_count'})

# ==========================================
# 3. 在测试集上生成预测
# ==========================================
# 将历史平均值直接合并到测试集上
test_pred = pd.merge(test_df, base_mean, on=[
                     'window_id', 'slot_start'], how='left')

# 极少数情况下，测试集可能出现训练集没见过的组合，用整体平均值填补
overall_mean = train_df['order_count'].mean()
test_pred['predicted_count'] = test_pred['predicted_count'].fillna(
    overall_mean)

# ==========================================
# 4. 计算 RMSE 评估结果
# ==========================================
# 注意：Method A 的预测结果只有几套固定的数值（因为没有日历调整）
rmse = np.sqrt(mean_squared_error(
    test_pred['order_count'], test_pred['predicted_count']))

print(f"\n--- Method A 评估结果 ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} 订单/窗口/时间段")

print("\n预测结果抽样对比 (真实值 vs 预测值):")
print(test_pred[['date', 'window_id', 'slot_start',
      'order_count', 'predicted_count']].head(10))
