"""测试IMCMD模型的前向传播"""
from imcmd_model import create_imcmd_model
import torch

print("=" * 60)
print("测试 IMCMD_YOLOv8_small 前向传播")
print("=" * 60)

# 创建模型
model = create_imcmd_model('yolov8s_imcmd_small.yaml', verbose=True)
print(f"\n✅ 模型创建成功")
print(f"总参数: {sum(p.numel() for p in model.parameters()):,}\n")

# 测试前向传播
try:
    x = torch.randn(2, 3, 640, 640)  # batch=2避免BN错误
    model.eval()
    
    print("开始前向传播...")
    with torch.no_grad():
        y = model(x)
    
    print(f"✅ 前向传播成功！")
    
    if isinstance(y, (list, tuple)):
        print(f"输出数量: {len(y)}")
        for i, out in enumerate(y):
            if isinstance(out, torch.Tensor):
                print(f"  输出{i}: {out.shape}")
            else:
                print(f"  输出{i}: {type(out)}")
    else:
        print(f"输出: {y.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！可以开始训练。")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

