"""可视化数据集样本（带标注框）"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def visualize_samples(dataset_path='lisa_yolo_redistributed', num_samples=6):
    """
    可视化数据集样本，显示原始标注框
    
    参数:
        dataset_path: 数据集路径
        num_samples: 显示样本数量
    """
    
    val_labels_dir = Path(dataset_path) / 'labels' / 'val'
    val_images_dir = Path(dataset_path) / 'images' / 'val'
    
    # 类别名称和颜色
    class_names = {
        0: 'go',
        1: 'goForward',
        2: 'goLeft',
        3: 'stop',
        4: 'stopLeft',
        5: 'warning',
        6: 'warningLeft'
    }
    
    colors = {
        0: 'green',
        1: 'blue',
        2: 'cyan',
        3: 'red',
        4: 'orange',
        5: 'yellow',
        6: 'magenta'
    }
    
    # 随机选择有标注的图片
    label_files = list(val_labels_dir.glob('*.txt'))
    valid_files = [f for f in label_files if f.stat().st_size > 0]
    
    if len(valid_files) < num_samples:
        print(f"警告: 只找到 {len(valid_files)} 个有标注的文件")
        num_samples = len(valid_files)
    
    selected = random.sample(valid_files, num_samples)
    
    print("=" * 80)
    print(f"可视化 {num_samples} 个数据集样本（带原始标注框）")
    print("=" * 80)
    
    for idx, label_file in enumerate(selected, 1):
        # 找到对应的图片
        img_name = label_file.stem + '.jpg'
        img_path = val_images_dir / img_name
        
        if not img_path.exists():
            print(f"跳过: 图片不存在 {img_name}")
            continue
        
        # 读取图片
        img = Image.open(img_path)
        w, h = img.size
        
        # 创建绘图对象
        draw = ImageDraw.Draw(img)
        
        # 读取标注并绘制边界框
        with open(label_file) as f:
            box_count = 0
            classes_in_image = set()
            
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    box_w = float(parts[3]) * w
                    box_h = float(parts[4]) * h
                    
                    # 计算边界框的左上角和右下角坐标
                    x1 = x_center - box_w / 2
                    y1 = y_center - box_h / 2
                    x2 = x_center + box_w / 2
                    y2 = y_center + box_h / 2
                    
                    # 绘制边界框
                    color = colors.get(class_id, 'white')
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # 添加类别标签
                    class_name = class_names.get(class_id, f'Class{class_id}')
                    classes_in_image.add(class_name)
                    
                    # 绘制标签背景和文本
                    text = class_name
                    try:
                        text_bbox = draw.textbbox((x1, y1-20), text)
                        draw.rectangle(text_bbox, fill=color)
                        draw.text((x1, y1-20), text, fill='black')
                    except:
                        # 如果textbbox不可用，使用简单的矩形
                        draw.rectangle([x1, y1-20, x1+80, y1], fill=color)
                        draw.text((x1+2, y1-18), text, fill='black')
                    
                    box_count += 1
        
        # 保存图片
        output_path = f'sample_{idx}.jpg'
        img.save(output_path)
        print(f"{idx}. {img_name}")
        print(f"   尺寸: {w}x{h}")
        print(f"   标注框: {box_count} 个")
        print(f"   类别: {', '.join(sorted(classes_in_image))}")
        print(f"   保存为: {output_path}\n")
    
    print("=" * 80)
    print("✓ 已生成带原始标注框的图片")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = 'lisa_yolo_redistributed'
    
    visualize_samples(dataset_path)


