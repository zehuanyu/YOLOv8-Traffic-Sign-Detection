"""
训练纯YOLO-TS模型
用于消融实验，验证YOLO-TS改进的效果
架构: 标准YOLOv8s + bilinear上采样 + AGRFM增强
"""
import os
import sys
from pathlib import Path

# 注册自定义模块
from ultralytics.nn import modules, tasks
import imcmd_modules

# 注册AGRFM模块（YOLO-TS的核心模块）
setattr(modules, 'AGRFM', imcmd_modules.AGRFM)
setattr(tasks, 'AGRFM', imcmd_modules.AGRFM)

from ultralytics import YOLO

def train_ts(epochs=100, batch=16, imgsz=640, device=0):
    """训练纯YOLO-TS模型"""
    
    print("=" * 70)
    print("训练纯YOLO-TS模型")
    print("=" * 70)
    print(f"配置: epochs={epochs}, batch={batch}, imgsz={imgsz}")
    print("=" * 70)
    
    # 使用YOLO-TS配置
    yaml_file = 'yolov8s_ts.yaml'
    
    # 创建模型
    model = YOLO(yaml_file)
    
    # 训练配置
    results = model.train(
        data='lisa_yolo_redistributed/data.yaml',
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project='runs/traffic_sign',
        name=f'yolo_ts_{epochs}epochs',
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        workers=0,  # Windows兼容
    )
    
    print("=" * 70)
    print("✓ 训练完成")
    print("=" * 70)
    
    # 打印最终结果
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"📊 最终结果:")
        print(f"  mAP@0.5:       {metrics.get('metrics/mAP50(B)', 0)*100:.1f}%")
        print(f"  mAP@0.5:0.95:  {metrics.get('metrics/mAP50-95(B)', 0)*100:.1f}%")
    
    print("=" * 70)
    print(f"💾 结果保存在: runs/traffic_sign/yolo_ts_{epochs}epochs/")
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练纯YOLO-TS模型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    
    args = parser.parse_args()
    train_ts(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)

