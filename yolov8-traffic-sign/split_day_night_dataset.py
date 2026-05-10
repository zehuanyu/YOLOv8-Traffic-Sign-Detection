"""
物理分离白天和夜间数据集
创建val_day和val_night子集用于独立评估
"""

from pathlib import Path
import shutil
import yaml


def split_dataset():
    """分离验证集为白天和夜间"""
    
    print("=" * 70)
    print("分离白天/夜间数据集")
    print("=" * 70)
    
    # 路径配置
    val_img_dir = Path('lisa_yolo_redistributed/images/val')
    val_label_dir = Path('lisa_yolo_redistributed/labels/val')
    
    # 创建目标文件夹
    day_img_dir = Path('lisa_yolo_redistributed/images/val_day')
    night_img_dir = Path('lisa_yolo_redistributed/images/val_night')
    day_label_dir = Path('lisa_yolo_redistributed/labels/val_day')
    night_label_dir = Path('lisa_yolo_redistributed/labels/val_night')
    
    # 创建文件夹
    for dir_path in [day_img_dir, night_img_dir, day_label_dir, night_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n文件夹已创建:")
    print(f"  {day_img_dir}")
    print(f"  {night_img_dir}")
    print(f"  {day_label_dir}")
    print(f"  {night_label_dir}")
    
    # 获取所有图片
    all_images = list(val_img_dir.glob('*.jpg'))
    
    print(f"\n总验证集图片: {len(all_images)}")
    
    # 分类统计
    day_count = 0
    night_count = 0
    other_count = 0
    
    print("\n开始复制文件...")
    
    for img_path in all_images:
        img_name = img_path.name
        label_name = img_path.stem + '.txt'
        label_path = val_label_dir / label_name
        
        # 判断白天还是夜间
        if 'day' in img_name.lower() and 'night' not in img_name.lower():
            # 白天
            shutil.copy2(img_path, day_img_dir / img_name)
            if label_path.exists():
                shutil.copy2(label_path, day_label_dir / label_name)
            day_count += 1
            
        elif 'night' in img_name.lower():
            # 夜间
            shutil.copy2(img_path, night_img_dir / img_name)
            if label_path.exists():
                shutil.copy2(label_path, night_label_dir / label_name)
            night_count += 1
            
        else:
            other_count += 1
    
    print(f"\n复制完成:")
    print(f"  白天: {day_count}张")
    print(f"  夜间: {night_count}张")
    print(f"  其他: {other_count}张")
    
    # 创建YAML配置
    print("\n创建数据配置文件...")
    
    # 白天配置
    day_config = {
        'path': str(Path('lisa_yolo_redistributed').absolute()),
        'train': str(Path('lisa_yolo_redistributed/images/train').absolute()),
        'val': str(day_img_dir.absolute()),
        'nc': 7,
        'names': ['go', 'goForward', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    }
    
    with open('lisa_day.yaml', 'w') as f:
        yaml.dump(day_config, f, default_flow_style=False)
    
    # 夜间配置
    night_config = {
        'path': str(Path('lisa_yolo_redistributed').absolute()),
        'train': str(Path('lisa_yolo_redistributed/images/train').absolute()),
        'val': str(night_img_dir.absolute()),
        'nc': 7,
        'names': ['go', 'goForward', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    }
    
    with open('lisa_night.yaml', 'w') as f:
        yaml.dump(night_config, f, default_flow_style=False)
    
    print("  lisa_day.yaml - 白天数据配置")
    print("  lisa_night.yaml - 夜间数据配置")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    
    print("\n使用方法:")
    print("\n1. 评估白天性能:")
    print("   python eval_model.py --model runs/traffic_sign/imcmd_small_100epochs/weights/best.pt --data lisa_day.yaml")
    
    print("\n2. 评估夜间性能:")
    print("   python eval_model.py --model runs/traffic_sign/imcmd_small_100epochs/weights/best.pt --data lisa_night.yaml")
    
    print("\n3. 对比整体性能:")
    print("   python eval_model.py --model runs/traffic_sign/imcmd_small_100epochs/weights/best.pt --data lisa_yolo_redistributed/data.yaml")


if __name__ == "__main__":
    split_dataset()

