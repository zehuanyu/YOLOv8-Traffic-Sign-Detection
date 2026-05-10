"""验证重新划分后的数据集"""
from pathlib import Path
from collections import Counter

def verify():
    print("=" * 80)
    print("验证重新划分后的数据集")
    print("=" * 80)
    
    # 训练集
    print("\n训练集类别分布:")
    train_labels = []
    for f in Path('lisa_yolo_redistributed/labels/train').glob('*.txt'):
        with open(f) as file:
            for line in file:
                parts = line.split()
                if parts:
                    train_labels.append(int(parts[0]))
    
    train_counter = Counter(train_labels)
    for k in sorted(train_counter.keys()):
        print(f"  类别 {k}: {train_counter[k]}")
    print(f"总标注数: {sum(train_counter.values())}")
    
    # 验证集
    print("\n验证集类别分布:")
    val_labels = []
    for f in Path('lisa_yolo_redistributed/labels/val').glob('*.txt'):
        with open(f) as file:
            for line in file:
                parts = line.split()
                if parts:
                    val_labels.append(int(parts[0]))
    
    val_counter = Counter(val_labels)
    for k in sorted(val_counter.keys()):
        print(f"  类别 {k}: {val_counter[k]}")
    print(f"总标注数: {sum(val_counter.values())}")
    
    # 检查关键类别
    print("\n" + "=" * 80)
    print("关键检查：")
    print("=" * 80)
    
    if 1 in train_counter:
        print(f"✓ 训练集现在有 {train_counter[1]} 个 goForward 标注！")
    else:
        print("✗ 训练集仍然没有 goForward")
    
    if 1 in val_counter:
        print(f"✓ 验证集保留了 {val_counter[1]} 个 goForward 标注")
    
    # 检查类别连续性
    train_classes = set(train_counter.keys())
    val_classes = set(val_counter.keys())
    
    if train_classes == val_classes:
        print(f"✓ 训练集和验证集类别一致: {sorted(train_classes)}")
    else:
        print(f"✗ 类别不一致！")
        print(f"  训练集: {sorted(train_classes)}")
        print(f"  验证集: {sorted(val_classes)}")

if __name__ == "__main__":
    verify()


