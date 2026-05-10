"""对比IMCMD和IMCMD-TS训练结果"""
import pandas as pd

print('=' * 75)
print('IMCMD vs IMCMD-TS 训练结果对比')
print('=' * 75)

models = [
    ('baseline_yolov8s_50epochs', 'YOLOv8s Baseline'),
    ('imcmd_small_100epochs', 'IMCMD Small'),
    ('imcmd_ts_small_100epochs', 'IMCMD-TS Small'),
]

results = []
for folder, name in models:
    try:
        df = pd.read_csv(f'runs/traffic_sign/{folder}/results.csv')
        df.columns = df.columns.str.strip()
        final = df.iloc[-1]
        best_idx = df['metrics/mAP50(B)'].idxmax()
        best = df.iloc[best_idx]
        results.append({
            'name': name,
            'epochs': len(df),
            'final_map50': final['metrics/mAP50(B)'],
            'final_map50_95': final['metrics/mAP50-95(B)'],
            'final_prec': final['metrics/precision(B)'],
            'final_rec': final['metrics/recall(B)'],
            'best_map50': best['metrics/mAP50(B)'],
            'best_epoch': best_idx + 1,
        })
    except Exception as e:
        print(f'{name}: 读取失败 - {e}')

print()
print('【最终Epoch结果】')
print('-' * 75)
print(f"{'模型':<20} | Epochs | mAP50  | mAP50-95 | Precision | Recall")
print('-' * 75)
for r in results:
    print(f"{r['name']:<20} | {r['epochs']:>6} | {r['final_map50']:.4f} | {r['final_map50_95']:.5f}  | {r['final_prec']:.4f}    | {r['final_rec']:.4f}")

print()
print('【最佳mAP50】')
print('-' * 75)
for r in results:
    print(f"{r['name']:<20} | Best Epoch: {r['best_epoch']:>3} | Best mAP50: {r['best_map50']:.4f}")

print()
print('=' * 75)
print('【IMCMD vs IMCMD-TS 直接对比】')
print('=' * 75)

imcmd = next(r for r in results if r['name'] == 'IMCMD Small')
imcmd_ts = next(r for r in results if r['name'] == 'IMCMD-TS Small')

map50_diff = imcmd_ts['final_map50'] - imcmd['final_map50']
map50_95_diff = imcmd_ts['final_map50_95'] - imcmd['final_map50_95']
prec_diff = imcmd_ts['final_prec'] - imcmd['final_prec']
rec_diff = imcmd_ts['final_rec'] - imcmd['final_rec']

print()
print(f"{'指标':<12} | {'IMCMD Small':>12} | {'IMCMD-TS Small':>14} | {'差异':>10}")
print('-' * 60)
print(f"{'mAP50':<12} | {imcmd['final_map50']:>12.4f} | {imcmd_ts['final_map50']:>14.4f} | {map50_diff:>+10.4f}")
print(f"{'mAP50-95':<12} | {imcmd['final_map50_95']:>12.4f} | {imcmd_ts['final_map50_95']:>14.4f} | {map50_95_diff:>+10.4f}")
print(f"{'Precision':<12} | {imcmd['final_prec']:>12.4f} | {imcmd_ts['final_prec']:>14.4f} | {prec_diff:>+10.4f}")
print(f"{'Recall':<12} | {imcmd['final_rec']:>12.4f} | {imcmd_ts['final_rec']:>14.4f} | {rec_diff:>+10.4f}")

print()
print('=' * 75)
print('【分析结论】')
print('=' * 75)

if map50_diff > 0:
    print(f'✅ IMCMD-TS mAP50 比 IMCMD 提升了 {map50_diff*100:.2f}%')
elif map50_diff < -0.01:
    print(f'⚠️  IMCMD-TS mAP50 比 IMCMD 下降了 {abs(map50_diff)*100:.2f}%')
else:
    print(f'➖ IMCMD-TS mAP50 与 IMCMD 基本持平 ({map50_diff*100:+.2f}%)')

if map50_95_diff > 0:
    print(f'✅ IMCMD-TS mAP50-95 比 IMCMD 提升了 {map50_95_diff*100:.2f}%')
elif map50_95_diff < -0.01:
    print(f'⚠️  IMCMD-TS mAP50-95 比 IMCMD 下降了 {abs(map50_95_diff)*100:.2f}%')
else:
    print(f'➖ IMCMD-TS mAP50-95 与 IMCMD 基本持平 ({map50_95_diff*100:+.2f}%)')

# 最佳结果对比
best_diff = imcmd_ts['best_map50'] - imcmd['best_map50']
print()
print(f'📊 最佳mAP50对比: IMCMD={imcmd["best_map50"]:.4f} vs IMCMD-TS={imcmd_ts["best_map50"]:.4f} (差异: {best_diff:+.4f})')
print('=' * 75)

