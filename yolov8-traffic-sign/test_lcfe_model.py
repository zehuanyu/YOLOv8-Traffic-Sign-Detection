"""测试CCA_Light模型构建"""
from cca_light import CCA_Light
from ultralytics.nn import modules
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO

# 注册
setattr(modules, 'CCA_Light', CCA_Light)
setattr(tasks, 'CCA_Light', CCA_Light)

print("=" * 60)
print("测试YOLOv8s + CCA_Light构建")
print("=" * 60)

try:
    # 构建模型
    model = YOLO('yolov8s_lcfe.yaml')
    
    print("\n✓ 模型构建成功！")
    
    # 统计
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  原生YOLOv8s: 11,166,560")
    print(f"  新增: {total_params - 11166560:,}")
    print(f"  增加比例: +{(total_params/11166560-1)*100:.1f}%")
    
    # 找CCA_Light模块
    cca_count = 0
    for name, module in model.model.named_modules():
        if isinstance(module, CCA_Light):
            cca_count += 1
            cca_params = sum(p.numel() for p in module.parameters())
            print(f"\n找到CCA_Light #{cca_count}: {name}")
            print(f"  参数: {cca_params:,}")
    
    print(f"\n✓ 共找到{cca_count}个CCA_Light模块")
    print(f"✓ 模型架构正确")
    
    # 测试forward
    import torch
    x = torch.randn(1, 3, 640, 640)
    model.model.eval()
    with torch.no_grad():
        out = model.model(x)
    print(f"\n✓ Forward测试通过")
    
    print("\n" + "=" * 60)
    print("可以开始训练！")
    print("=" * 60)
    print("\n训练命令：")
    print("python train_lcfe.py --epochs 5 --batch 16 --device 0 --name test_lcfe_5epochs")
    
except Exception as e:
    print(f"\n✗ 构建失败: {e}")
    import traceback
    traceback.print_exc()


