"""从保存的epoch权重重建results.csv"""
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# IMCMD Small 50 epochs的epoch权重
epoch_weights = [10, 20, 30, 40]
results_data = []

print("正在从epoch权重重建数据...")
print("="*60)

for ep in epoch_weights:
    weight_path = f'runs/traffic_sign/imcmd_small_50epochs/weights/epoch{ep}.pt'
    if Path(weight_path).exists():
        print(f"评估 Epoch {ep}...")
        model = YOLO(weight_path)
        metrics = model.val(data='lisa_yolo_redistributed/data.yaml', verbose=False)
        
        results_data.append({
            'epoch': ep,
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'Precision': metrics.box.mp,
            'Recall': metrics.box.mr,
        })
        print(f"  mAP@0.5: {metrics.box.map50:.1%}")

print("="*60)

# 保存为CSV
df = pd.DataFrame(results_data)
output_path = 'imcmd_small_50epochs_reconstructed.csv'
df.to_csv(output_path, index=False)

print(f"\n✅ 数据已保存到: {output_path}")
print("\n数据预览:")
print(df.to_string(index=False))

# 也可以画图
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['mAP50']*100, 'o-', label='mAP@0.5', linewidth=2, markersize=8)
plt.plot(df['epoch'], df['mAP50-95']*100, 's-', label='mAP@0.5:0.95', linewidth=2, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('mAP (%)')
plt.title('IMCMD Small - Training Progress (Reconstructed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('imcmd_small_50epochs_reconstructed.png', dpi=150, bbox_inches='tight')
print(f"✅ 图表已保存: imcmd_small_50epochs_reconstructed.png")

