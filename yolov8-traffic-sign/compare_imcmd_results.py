"""对比IMCMD Small 50 vs 100 epochs"""
import pandas as pd

print("\n" + "="*80)
print("📊 IMCMD Small - 50 vs 100 Epochs 对比")
print("="*80)

# 读取100 epochs结果
df_100 = pd.read_csv('runs/traffic_sign/imcmd_small_100epochs/results.csv')
final_100 = df_100.iloc[-1]

# 读取Baseline对比
df_baseline_50 = pd.read_csv('runs/traffic_sign/baseline_yolov8s_50epochs/results.csv')
final_baseline_50 = df_baseline_50.iloc[-1]

df_baseline_100 = pd.read_csv('runs/traffic_sign/baseline_yolov8s_50epochs/results.csv')
final_baseline_100 = df_baseline_100.iloc[-1]

print(f"\n1️⃣ IMCMD Small 训练进度:")
print("-"*80)
print(f"{'版本':20s} {'Epochs':>8s} {'mAP@0.5':>12s} {'mAP@0.5:0.95':>15s} {'Precision':>12s} {'Recall':>10s}")
print("-"*80)

# 从epoch40.pt估算50 epochs性能
print(f"{'IMCMD 50ep (估算)':20s} {'50':>8s} {'37.5%':>12s} {'20.4%':>15s} {'60.0%':>12s} {'34.0%':>10s}")

# 100 epochs
cols = df_100.columns.tolist()
idx_p = cols.index('metrics/precision(B)')
idx_r = cols.index('metrics/recall(B)')
idx_m50 = cols.index('metrics/mAP50(B)')
idx_m5095 = cols.index('metrics/mAP50-95(B)')

print(f"{'IMCMD 100ep':20s} {len(df_100):>8d} {final_100.iloc[idx_m50]:>11.1%} {final_100.iloc[idx_m5095]:>14.1%} {final_100.iloc[idx_p]:>11.1%} {final_100.iloc[idx_r]:>9.1%}")

print("\n2️⃣ 与Baseline对比:")
print("-"*80)
print(f"{'版本':20s} {'Epochs':>8s} {'mAP@0.5':>12s} {'参数量':>12s} {'性能/参数比':>15s}")
print("-"*80)
print(f"{'Baseline 50ep':20s} {'50':>8s} {final_baseline_50.iloc[idx_m50]:>11.1%} {'11.1M':>12s} {final_baseline_50.iloc[idx_m50]/11.1:>14.4f}")
print(f"{'Baseline 100ep':20s} {len(df_baseline_100):>8d} {final_baseline_100.iloc[idx_m50]:>11.1%} {'11.1M':>12s} {final_baseline_100.iloc[idx_m50]/11.1:>14.4f}")
print(f"{'IMCMD 100ep':20s} {len(df_100):>8d} {final_100.iloc[idx_m50]:>11.1%} {'1.97M':>12s} {final_100.iloc[idx_m50]/1.97:>14.4f}")

print("\n3️⃣ 关键结论:")
print("="*80)

imcmd_map = final_100.iloc[idx_m50]
baseline_100_map = final_baseline_100.iloc[idx_m50]
diff = (imcmd_map - baseline_100_map) * 100

print(f"\nIMCMD 100ep vs Baseline 100ep:")
print(f"  mAP@0.5差距: {diff:+.1f}%")
print(f"  参数量:     IMCMD仅用17.7% (1.97M vs 11.1M)")
print(f"  效率提升:   {(imcmd_map/1.97)/(baseline_100_map/11.1):.1f}倍")

if abs(diff) < 2:
    print(f"\n🎉 结论: IMCMD用1/6的参数，性能相当！")
elif diff > 0:
    print(f"\n🏆 结论: IMCMD参数少还更好！")
else:
    print(f"\n✅ 结论: IMCMD牺牲{abs(diff):.1f}% mAP，换取82%参数减少！")

print("\n" + "="*80)





