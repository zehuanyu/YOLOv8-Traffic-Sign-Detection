"""测试CCA_Light_v2配置"""
from cca_light_v2 import CCA_Light_v2
from ultralytics.nn import modules
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO

# 注册
setattr(modules, 'CCA_Light_v2', CCA_Light_v2)
setattr(tasks, 'CCA_Light_v2', CCA_Light_v2)

print("=" * 60)
print("测试YOLOv8s + CCA_Light_v2")
print("=" * 60)

# 测试模块
print("\n【测试1】CCA_Light_v2模块")
cca = CCA_Light_v2(c1=256, c2=256)
params = sum(p.numel() for p in cca.parameters())
print(f"参数: {params:,}")

import torch
x = torch.randn(1, 256, 32, 32)
out = cca(x)
print(f"Forward: {x.shape} → {out.shape} ✓")

# 测试模型构建
print("\n【测试2】模型构建")
try:
    model = YOLO('yolov8s_lcfe_v2.yaml')
    print("✓ 模型构建成功")
    
    total = sum(p.numel() for p in model.model.parameters())
    print(f"总参数: {total:,}")
    print(f"新增: {total - 11166560:,}")
    
    # 找CCA_Light_v2
    count = 0
    for name, m in model.model.named_modules():
        if isinstance(m, CCA_Light_v2):
            count += 1
            print(f"  找到CCA_Light_v2 #{count}: {name}")
    
    print(f"\n✓ 找到{count}个CCA_Light_v2模块")
    print("\n可以开始训练！")
    print("命令: python train_lcfe_v2.py --epochs 5 --batch 16 --device 0 --name test_v2_5epochs")
    
except Exception as e:
    print(f"✗ 构建失败: {e}")

