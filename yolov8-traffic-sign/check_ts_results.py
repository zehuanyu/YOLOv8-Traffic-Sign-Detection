"""检查纯YOLO-TS训练结果"""
import csv

with open('runs/traffic_sign/yolo_ts_100epochs/results.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 清理列名
clean_rows = []
for row in rows:
    clean_row = {k.strip(): v for k, v in row.items()}
    clean_rows.append(clean_row)

# 找到最佳mAP50
best_row = max(clean_rows, key=lambda x: float(x['metrics/mAP50(B)']))
last_row = clean_rows[-1]

print('=' * 60)
print('纯YOLO-TS 训练结果')
print('=' * 60)
print(f'训练轮数: {len(clean_rows)} epochs')
print(f'最佳Epoch: {best_row["epoch"]}')
print()
print('最佳指标:')
print(f'  mAP@0.5:       {float(best_row["metrics/mAP50(B)"])*100:.1f}%')
print(f'  mAP@0.5:0.95:  {float(best_row["metrics/mAP50-95(B)"])*100:.1f}%')
print(f'  Precision:     {float(best_row["metrics/precision(B)"])*100:.1f}%')
print(f'  Recall:        {float(best_row["metrics/recall(B)"])*100:.1f}%')
print('=' * 60)
