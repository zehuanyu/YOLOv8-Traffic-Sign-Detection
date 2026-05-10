"""诊断增强数据集的问题"""
from pathlib import Path
from collections import Counter
import random

def diagnose_augmented_dataset():
    print("=" * 80)
    print("诊断增强数据集")
    print("=" * 80)
    
    # 检查类别分布
    print("\n1. 类别分布检查")
    print("-" * 80)
    
    train_labels = []
    train_files = list(Path('lisa_yolo_augmented_fixed/labels/train').glob('*.txt'))
    
    for f in train_files:
        with open(f) as file:
            for line in file:
                parts = line.split()
                if parts:
                    train_labels.append(int(parts[0]))
    
    counter = Counter(train_labels)
    print("训练集类别分布:")
    for k in sorted(counter.keys()):
        count = counter[k]
        percentage = count / sum(counter.values()) * 100
        print(f"  类别 {k}: {count:6d} ({percentage:5.1f}%)")
    
    # 检查图片和标注匹配
    print("\n2. 图片和标注匹配检查")
    print("-" * 80)
    
    train_images = list(Path('lisa_yolo_augmented_fixed/images/train').glob('*.jpg'))
    print(f"训练集图片数: {len(train_images)}")
    print(f"训练集标注数: {len(train_files)}")
    
    # 检查是否有图片没有标注
    img_stems = {f.stem for f in train_images}
    label_stems = {f.stem for f in train_files}
    
    no_label = img_stems - label_stems
    no_image = label_stems - img_stems
    
    if no_label:
        print(f"⚠️ {len(no_label)} 张图片没有标注")
        print(f"   示例: {list(no_label)[:5]}")
    else:
        print("✓ 所有图片都有标注")
    
    if no_image:
        print(f"⚠️ {len(no_image)} 个标注没有图片")
        print(f"   示例: {list(no_image)[:5]}")
    else:
        print("✓ 所有标注都有图片")
    
    # 检查空标注文件
    print("\n3. 空标注文件检查")
    print("-" * 80)
    
    empty_files = []
    for f in train_files:
        if f.stat().st_size == 0 or not open(f).read().strip():
            empty_files.append(f)
    
    if empty_files:
        print(f"⚠️ 发现 {len(empty_files)} 个空标注文件")
        print(f"   示例: {[f.name for f in empty_files[:5]]}")
    else:
        print("✓ 没有空标注文件")
    
    # 随机检查几个标注文件的内容
    print("\n4. 标注格式检查（随机抽样）")
    print("-" * 80)
    
    sample_files = random.sample(train_files, min(5, len(train_files)))
    
    for f in sample_files:
        print(f"\n文件: {f.name}")
        with open(f) as file:
            lines = file.readlines()
            print(f"  标注数: {len(lines)}")
            
            # 检查格式
            for i, line in enumerate(lines[:3], 1):  # 只看前3个
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # 检查范围
                    if not (0 <= class_id <= 6):
                        print(f"    ⚠️ 行{i}: 类别ID异常 ({class_id})")
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        print(f"    ⚠️ 行{i}: 中心坐标超出范围 ({x:.3f}, {y:.3f})")
                    if not (0 < w <= 1 and 0 < h <= 1):
                        print(f"    ⚠️ 行{i}: 宽高异常 ({w:.3f}, {h:.3f})")
                    
                    if i <= 2:  # 只显示前2个
                        print(f"    行{i}: 类别{class_id} x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f} ✓")
                else:
                    print(f"    ⚠️ 行{i}: 格式错误（字段数={len(parts)}）")
    
    # 对比原始数据集
    print("\n" + "=" * 80)
    print("5. 与原始数据集对比")
    print("-" * 80)
    
    # 原始数据集
    orig_train_labels = []
    orig_files = list(Path('lisa_yolo_redistributed/labels/train').glob('*.txt'))
    for f in orig_files:
        with open(f) as file:
            for line in file:
                parts = line.split()
                if parts:
                    orig_train_labels.append(int(parts[0]))
    
    orig_counter = Counter(orig_train_labels)
    
    print("\n原始数据集:")
    for k in sorted(orig_counter.keys()):
        print(f"  类别 {k}: {orig_counter[k]:6d}")
    print(f"  总标注: {sum(orig_counter.values())}")
    print(f"  图片数: {len(orig_files)}")
    
    print("\n增强数据集:")
    for k in sorted(counter.keys()):
        print(f"  类别 {k}: {counter[k]:6d}")
    print(f"  总标注: {sum(counter.values())}")
    print(f"  图片数: {len(train_files)}")
    
    # 增强倍数
    print("\n增强倍数:")
    print(f"  图片: {len(train_files) / len(orig_files):.2f}x")
    print(f"  标注: {sum(counter.values()) / sum(orig_counter.values()):.2f}x")
    
    # 类别比例对比
    print("\n类别比例对比:")
    for k in sorted(counter.keys()):
        orig_ratio = orig_counter[k] / sum(orig_counter.values()) * 100
        aug_ratio = counter[k] / sum(counter.values()) * 100
        diff = aug_ratio - orig_ratio
        print(f"  类别 {k}: 原始{orig_ratio:5.1f}% → 增强{aug_ratio:5.1f}% (差异{diff:+5.1f}%)")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_augmented_dataset()


