"""
分白天和夜间评估模型性能
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def split_day_night_images(val_dir):
    """分离白天和夜间图片"""
    val_path = Path(val_dir)
    all_images = list(val_path.glob('*.jpg'))
    
    day_images = []
    night_images = []
    
    for img in all_images:
        if 'day' in img.stem.lower():
            day_images.append(str(img))
        elif 'night' in img.stem.lower():
            night_images.append(str(img))
    
    return day_images, night_images


def evaluate_on_subset(model, images, data_yaml):
    """在图片子集上评估"""
    # 创建临时数据配置
    import yaml
    import tempfile
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 临时保存图片列表
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for img in images:
        temp_file.write(f"{img}\n")
    temp_file.close()
    
    # 验证
    # 注意：ultralytics可能不支持自定义图片列表，需要用其他方法
    # 简化方案：逐张推理然后统计
    
    from ultralytics.utils.metrics import ConfusionMatrix
    import torch
    
    model.model.eval()
    
    # 这里需要实现自定义评估逻辑
    # 暂时返回占位符
    print(f"  评估{len(images)}张图片...")
    
    return {
        'map50': 0.0,  # 需要实现完整评估
        'map50_95': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='白天/夜间性能评估')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, default='lisa_yolo_redistributed/data.yaml')
    parser.add_argument('--val-dir', type=str, default='lisa_yolo_redistributed/images/val')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("白天/夜间性能评估")
    print("=" * 70)
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 分离图片
    print(f"\n分析验证集: {args.val_dir}")
    day_images, night_images = split_day_night_images(args.val_dir)
    
    print(f"  白天图片: {len(day_images)}")
    print(f"  夜间图片: {len(night_images)}")
    print(f"  总计: {len(day_images) + len(night_images)}")
    
    # 整体验证
    print("\n" + "=" * 70)
    print("整体性能")
    print("=" * 70)
    
    metrics_all = model.val(data=args.data)
    print(f"\nmAP50: {metrics_all.box.map50:.4f} ({metrics_all.box.map50*100:.2f}%)")
    print(f"mAP50-95: {metrics_all.box.map:.4f} ({metrics_all.box.map*100:.2f}%)")
    
    # TODO: 分别评估白天和夜间
    # 需要实现自定义评估逻辑，因为ultralytics不直接支持图片子集验证
    
    print("\n" + "=" * 70)
    print("注意")
    print("=" * 70)
    print("分白天/夜间评估需要自定义实现。")
    print("当前显示的是整体结果。")
    print("\n建议：")
    print("1. 手动创建day.yaml和night.yaml（只包含对应图片）")
    print("2. 分别运行model.val(data='day.yaml')和model.val(data='night.yaml')")


if __name__ == "__main__":
    main()

