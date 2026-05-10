"""
创建白天和夜间的独立数据配置文件
用于分别评估模型在不同光照条件下的性能
"""

from pathlib import Path
import yaml
import shutil


def create_day_night_configs():
    """创建day.yaml和night.yaml"""
    
    print("=" * 70)
    print("创建白天/夜间数据配置")
    print("=" * 70)
    
    # 读取原始配置
    with open('lisa_yolo_redistributed/data.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 获取验证集图片
    val_dir = Path('lisa_yolo_redistributed/images/val')
    all_images = list(val_dir.glob('*.jpg'))
    
    print(f"\n总验证集图片: {len(all_images)}")
    
    # 分离白天和夜间
    day_images = [img for img in all_images if 'day' in img.stem.lower()]
    night_images = [img for img in all_images if 'night' in img.stem.lower()]
    
    print(f"  白天图片: {len(day_images)}")
    print(f"  夜间图片: {len(night_images)}")
    print(f"  其他: {len(all_images) - len(day_images) - len(night_images)}")
    
    # 创建白天配置
    day_config = base_config.copy()
    day_config['val'] = str(val_dir.absolute())  # 还是用同一个目录，但后面只验证day图片
    
    with open('lisa_yolo_redistributed/data_day.yaml', 'w') as f:
        yaml.dump(day_config, f)
    
    # 创建夜间配置  
    night_config = base_config.copy()
    night_config['val'] = str(val_dir.absolute())
    
    with open('lisa_yolo_redistributed/data_night.yaml', 'w') as f:
        yaml.dump(night_config, f)
    
    print("\n配置文件已创建:")
    print("  lisa_yolo_redistributed/data_day.yaml")
    print("  lisa_yolo_redistributed/data_night.yaml")
    
    # 创建图片列表文件（供参考）
    with open('day_images.txt', 'w') as f:
        for img in day_images:
            f.write(f"{img.name}\n")
    
    with open('night_images.txt', 'w') as f:
        for img in night_images:
            f.write(f"{img.name}\n")
    
    print("\n图片列表已保存:")
    print("  day_images.txt ({len(day_images)}张)")
    print("  night_images.txt ({len(night_images)}张)")
    
    print("\n" + "=" * 70)
    print("使用方法")
    print("=" * 70)
    print("\n评估白天性能:")
    print("  python eval_day_night_simple.py --model runs/traffic_sign/imcmd_small_100epochs/weights/best.pt --type day")
    print("\n评估夜间性能:")
    print("  python eval_day_night_simple.py --model runs/traffic_sign/imcmd_small_100epochs/weights/best.pt --type night")


if __name__ == "__main__":
    create_day_night_configs()

