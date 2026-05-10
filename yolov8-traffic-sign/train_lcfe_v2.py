"""
训练YOLOv8s + CCA_Light_v2
改进版训练脚本
"""

# 注册CCA_Light_v2模块
from cca_light_v2 import CCA_Light_v2
from ultralytics.nn import modules
import ultralytics.nn.tasks as tasks

setattr(modules, 'CCA_Light_v2', CCA_Light_v2)
setattr(tasks, 'CCA_Light_v2', CCA_Light_v2)
print("✓ CCA_Light_v2模块已注册\n")

from ultralytics import YOLO
from pathlib import Path
import argparse


def train_v2(
    yaml_path='yolov8s_lcfe_v2.yaml',
    data_yaml='lisa_yolo_redistributed/data.yaml',
    epochs=50,
    batch_size=16,
    device='0',
    name='yolov8s_lcfe_v2',
):
    """训练CCA_Light_v2"""
    
    print("=" * 70)
    print("YOLOv8s + CCA_Light_v2 训练")
    print("=" * 70)
    
    print(f"\n配置:")
    print(f"  模型: {yaml_path}")
    print(f"  数据集: {data_yaml}")
    print(f"  轮数: {epochs}")
    
    # 构建模型
    print(f"\n构建模型...")
    model = YOLO(yaml_path)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"✓ 总参数: {total_params:,}")
    print(f"  vs 原生: +{total_params - 11166560:,} (+{(total_params/11166560-1)*100:.1f}%)")
    
    # 训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70 + "\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        project='runs/traffic_sign',
        name=name,
        exist_ok=True,
        
        # 优化器（正常学习率）
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        
        # 其他
        patience=20,
        save_period=10,
        plots=True,
        verbose=True,
        workers=0,
    )
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    
    # 验证
    best_model_path = Path('runs/traffic_sign') / name / 'weights' / 'best.pt'
    
    print(f"\n模型: {best_model_path}")
    
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=data_yaml)
    
    print("\n" + "=" * 70)
    print("最终结果")
    print("=" * 70)
    print(f"\n整体:")
    print(f"  mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"  Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"  Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    
    print(f"\n各类别mAP50:")
    class_names = ['go', 'goForward', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    if hasattr(metrics.box, 'maps'):
        for i, name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"  {name:12s}: {metrics.box.maps[i]:.4f} ({metrics.box.maps[i]*100:.2f}%)")
    
    print("\n" + "=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8s + CCA_Light_v2')
    parser.add_argument('--yaml', type=str, default='yolov8s_lcfe_v2.yaml')
    parser.add_argument('--data', type=str, default='lisa_yolo_redistributed/data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--name', type=str, default='yolov8s_lcfe_v2')
    
    args = parser.parse_args()
    
    train_v2(
        yaml_path=args.yaml,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        name=args.name
    )


if __name__ == "__main__":
    main()

