"""
YOLOv8交通标志识别训练脚本
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import argparse

def train_traffic_sign_detector(
    data_yaml='lisa_yolo/data.yaml',
    model_size='n',  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    device='',  # 留空自动选择，或指定 'cpu' 或 '0' (GPU)
    project='runs/traffic_sign',
    name='yolov8_lisa',
    resume=False
):
    """
    训练YOLOv8交通标志检测模型
    
    参数:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图片大小
        device: 设备选择 ('' 自动, 'cpu', '0' GPU)
        project: 项目保存目录
        name: 实验名称
        resume: 是否从上次中断继续训练
    """
    
    print("=" * 60)
    print("YOLOv8交通标志检测训练")
    print("=" * 60)
    
    # 检查数据集配置
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # 读取数据集信息
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n数据集信息:")
    print(f"  路径: {data_config['path']}")
    print(f"  类别数: {data_config['nc']}")
    print(f"  类别: {list(data_config['names'].values())}")
    
    # 检查设备
    if device == '':
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n训练配置:")
    print(f"  模型: YOLOv8{model_size}")
    print(f"  设备: {device} ({'GPU' if device != 'cpu' else 'CPU'})")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  图片大小: {img_size}")
    
    # 加载模型
    model_name = f'yolov8{model_size}.pt'
    print(f"\n加载预训练模型: {model_name}")
    
    if resume:
        # 从上次训练继续
        last_weights = Path(project) / name / 'weights' / 'last.pt'
        if last_weights.exists():
            print(f"从检查点继续训练: {last_weights}")
            model = YOLO(str(last_weights))
        else:
            print(f"未找到检查点，从头开始训练")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        # 优化参数
        patience=50,  # 早停耐心值
        save=True,
        save_period=10,  # 每10轮保存一次
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        # 其他参数
        verbose=True,
        plots=True,
        workers=0,  # 单线程读取，避免OneDrive冲突
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 模型保存位置
    best_model = Path(project) / name / 'weights' / 'best.pt'
    last_model = Path(project) / name / 'weights' / 'last.pt'
    
    print(f"\n模型保存位置:")
    print(f"  最佳模型: {best_model}")
    print(f"  最后模型: {last_model}")
    
    # 验证最佳模型
    print("\n" + "=" * 60)
    print("验证最佳模型...")
    print("=" * 60 + "\n")
    
    best = YOLO(str(best_model))
    metrics = best.val(data=data_yaml)
    
    print(f"\n验证结果:")
    print(f"  mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"  Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"  Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    
    # 各类别详细结果
    print(f"\n各类别mAP50:")
    class_names = ['go', 'goForward', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    if hasattr(metrics.box, 'maps'):
        for i, class_name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"  {class_name:12s}: {metrics.box.maps[i]:.4f} ({metrics.box.maps[i]*100:.2f}%)")
    
    print("\n" + "=" * 60)
    
    return results, best_model

def main():
    parser = argparse.ArgumentParser(description='YOLOv8交通标志检测训练')
    parser.add_argument('--data', type=str, default='lisa_yolo/data.yaml', help='数据集配置文件')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--img', type=int, default=640, help='图片大小')
    parser.add_argument('--device', type=str, default='', help='设备 (留空自动选择, cpu, 0)')
    parser.add_argument('--project', type=str, default='runs/traffic_sign', help='项目目录')
    parser.add_argument('--name', type=str, default='yolov8_lisa', help='实验名称')
    parser.add_argument('--resume', action='store_true', help='继续上次训练')
    
    args = parser.parse_args()
    
    train_traffic_sign_detector(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )

if __name__ == "__main__":
    main()

