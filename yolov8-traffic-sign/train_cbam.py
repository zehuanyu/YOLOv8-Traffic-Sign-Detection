"""训练YOLOv8s + CBAM"""

from cbam import CBAM
from ultralytics.nn import modules
import ultralytics.nn.tasks as tasks

setattr(modules, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)
print("✓ CBAM已注册\n")

from ultralytics import YOLO
from pathlib import Path
import argparse


def train_cbam(
    data_yaml='lisa_yolo_redistributed/data.yaml',
    epochs=50,
    batch=16,
    device='0',
    name='yolov8s_cbam',
):
    print("=" * 70)
    print("YOLOv8s + CBAM训练")
    print("=" * 70)
    
    # 构建
    model = YOLO('yolov8s_cbam.yaml')
    total = sum(p.numel() for p in model.model.parameters())
    print(f"\n总参数: {total:,} (+{total-11166560:,})")
    
    # 训练
    print("\n开始训练...\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device=device,
        project='runs/traffic_sign',
        name=name,
        exist_ok=True,
        
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        patience=20,
        save_period=10,
        plots=True,
        workers=0,
    )
    
    # 验证
    best = YOLO(f'runs/traffic_sign/{name}/weights/best.pt')
    metrics = best.val(data=data_yaml)
    
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    print(f"mAP50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='lisa_yolo_redistributed/data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--name', type=str, default='yolov8s_cbam')
    
    args = parser.parse_args()
    train_cbam(args.data, args.epochs, args.batch, args.device, args.name)


if __name__ == "__main__":
    main()

