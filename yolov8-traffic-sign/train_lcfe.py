"""
训练YOLOv8s + CCA_Light
轻量级上下文增强模块训练脚本
"""

# 注册CCA_Light模块
from cca_light import CCA_Light
from ultralytics.nn import modules
import ultralytics.nn.tasks as tasks

setattr(modules, 'CCA_Light', CCA_Light)
setattr(tasks, 'CCA_Light', CCA_Light)
print("✓ CCA_Light模块已注册到Ultralytics\n")

# 导入训练库
from ultralytics import YOLO
from pathlib import Path
import argparse


def train_with_lcfe(
    yaml_path='yolov8s_lcfe.yaml',
    data_yaml='lisa_yolo_redistributed/data.yaml',
    epochs=50,
    batch_size=16,
    img_size=640,
    device='0',
    name='yolov8s_lcfe',
):
    """使用CCA_Light训练"""
    
    print("=" * 70)
    print("YOLOv8s + CCA_Light 训练")
    print("=" * 70)
    
    print(f"\n配置:")
    print(f"  模型: {yaml_path}")
    print(f"  数据集: {data_yaml}")
    print(f"  训练轮数: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  图片尺寸: {img_size}")
    
    # 构建模型
    print(f"\n从YAML构建模型...")
    model = YOLO(yaml_path)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"✓ 模型已构建")
    print(f"  总参数: {total_params:,}")
    print(f"  vs 原生YOLOv8s: ~11.17M")
    print(f"  新增: {(total_params - 11166560):,} (+{(total_params/11166560-1)*100:.1f}%)")
    
    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70 + "\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project='runs/traffic_sign',
        name=name,
        exist_ok=True,
        
        # 优化器配置
        optimizer='AdamW',
        lr0=0.001,  # 恢复正常学习率
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        # 数据增强（适度）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        
        # 训练配置
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        workers=0,
        
        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    
    # 验证最佳模型
    best_model_path = Path('runs/traffic_sign') / name / 'weights' / 'best.pt'
    last_model_path = Path('runs/traffic_sign') / name / 'weights' / 'last.pt'
    
    print(f"\n模型保存位置:")
    print(f"  最佳模型: {best_model_path}")
    print(f"  最后模型: {last_model_path}")
    
    # 验证
    print("\n" + "=" * 70)
    print("验证最佳模型...")
    print("=" * 70)
    
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=data_yaml)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("最终验证结果")
    print("=" * 70)
    print(f"\n整体性能:")
    print(f"  mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"  Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"  Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    
    # 各类别结果
    print(f"\n各类别mAP50:")
    class_names = ['go', 'goForward', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    if hasattr(metrics.box, 'maps'):
        for i, class_name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"  {class_name:12s}: {metrics.box.maps[i]:.4f} ({metrics.box.maps[i]*100:.2f}%)")
    
    print("\n" + "=" * 70)
    
    return results, best_model_path


def main():
    parser = argparse.ArgumentParser(description='YOLOv8s + CCA_Light训练')
    parser.add_argument('--yaml', type=str, default='yolov8s_lcfe.yaml')
    parser.add_argument('--data', type=str, default='lisa_yolo_redistributed/data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--name', type=str, default='yolov8s_lcfe')
    
    args = parser.parse_args()
    
    train_with_lcfe(
        yaml_path=args.yaml,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        device=args.device,
        name=args.name
    )


if __name__ == "__main__":
    main()

