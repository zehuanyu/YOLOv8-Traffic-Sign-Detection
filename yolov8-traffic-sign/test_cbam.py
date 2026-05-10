"""测试CBAM配置"""
from cbam import CBAM
from ultralytics.nn import modules
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO

setattr(modules, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)

print("=" * 60)
print("测试YOLOv8s + CBAM")
print("=" * 60)

# 测试模块
print("\n【测试1】CBAM模块")
cbam = CBAM(256)
print(f"参数: {sum(p.numel() for p in cbam.parameters()):,}")

import torch
x = torch.randn(1, 256, 32, 32)
out = cbam(x)
print(f"Forward: {x.shape} → {out.shape} ✓")

# 测试模型
print("\n【测试2】模型构建")
try:
    model = YOLO('yolov8s_cbam.yaml')
    print("✓ 构建成功")
    
    total = sum(p.numel() for p in model.model.parameters())
    print(f"总参数: {total:,}")
    print(f"vs原生: +{total-11166560:,} (+{(total/11166560-1)*100:.1f}%)")
    
    count = 0
    for name, m in model.model.named_modules():
        if isinstance(m, CBAM):
            count += 1
            print(f"  CBAM #{count}: {name}")
    
    print(f"\n✓ 找到{count}个CBAM模块")
    print("\n可以训练！")
    print("命令: python train_cbam.py --epochs 5 --batch 16 --device 0 --name test_cbam_5epochs")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

