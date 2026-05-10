"""列出包含goForward的图片文件"""
from pathlib import Path

def list_goforward_images():
    val_labels_dir = Path('lisa_yolo/labels/val')
    val_images_dir = Path('lisa_yolo/images/val')
    
    goforward_images = []
    
    for label_file in val_labels_dir.glob('*.txt'):
        with open(label_file) as f:
            has_goforward = False
            goforward_boxes = []
            all_boxes = []
            
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    all_boxes.append(class_id)
                    if class_id == 1:  # goForward
                        has_goforward = True
                        goforward_boxes.append(line.strip())
            
            if has_goforward:
                img_name = label_file.stem + '.jpg'
                img_path = val_images_dir / img_name
                
                if img_path.exists():
                    goforward_images.append({
                        'image': str(img_path),
                        'label': str(label_file),
                        'goforward_count': sum(1 for c in all_boxes if c == 1),
                        'total_signs': len(all_boxes),
                        'boxes': goforward_boxes
                    })
    
    print("=" * 80)
    print("包含 goForward 标志的图片")
    print("=" * 80)
    print(f"\n共找到 {len(goforward_images)} 张图片包含goForward标志\n")
    
    # 显示前10个
    class_names = {
        0: 'go', 1: 'goForward', 2: 'goLeft', 3: 'stop',
        4: 'stopLeft', 5: 'warning', 6: 'warningLeft'
    }
    
    for idx, img_info in enumerate(goforward_images[:10], 1):
        print(f"{idx}. {Path(img_info['image']).name}")
        print(f"   路径: {img_info['image']}")
        print(f"   goForward数量: {img_info['goforward_count']}")
        print(f"   总标志数: {img_info['total_signs']}")
        
        # 读取完整标注显示所有类别
        with open(img_info['label']) as f:
            classes_in_image = []
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_name = class_names.get(class_id, f'Unknown{class_id}')
                    classes_in_image.append(class_name)
        
        print(f"   包含的标志类型: {', '.join(set(classes_in_image))}")
        print()

if __name__ == "__main__":
    list_goforward_images()


