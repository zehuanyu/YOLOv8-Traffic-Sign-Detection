"""检查数据集中的类别分布"""
from pathlib import Path
from collections import Counter

def check_dataset():
    # 训练集
    print("=" * 60)
    print("训练集类别分布")
    print("=" * 60)
    train_labels = []
    train_files = list(Path('lisa_yolo/labels/train').glob('*.txt'))
    for f in train_files:
        with open(f) as file:
            for line in file:
                parts = line.split()
                if parts:
                    train_labels.append(int(parts[0]))
    
    train_counter = Counter(train_labels)
    for k in sorted(train_counter.keys()):
        print(f"  类别 {k}: {train_counter[k]}")
    
    # 验证集
    print("\n" + "=" * 60)
    print("验证集类别分布")
    print("=" * 60)
    val_labels = []
    val_files = list(Path('lisa_yolo/labels/val').glob('*.txt'))
    for f in val_files:
        with open(f) as file:
            for line in file:
                parts = line.split()
                if parts:
                    val_labels.append(int(parts[0]))
    
    val_counter = Counter(val_labels)
    for k in sorted(val_counter.keys()):
        print(f"  类别 {k}: {val_counter[k]}")
    
    # 检查类别1的文件
    if 1 in val_counter:
        print("\n" + "=" * 60)
        print("验证集中包含类别1 (goForward) 的文件")
        print("=" * 60)
        class1_files = []
        for f in val_files:
            with open(f) as file:
                for line in file:
                    if line.split() and int(line.split()[0]) == 1:
                        class1_files.append(f)
                        break
        
        print(f"共 {len(class1_files)} 个文件包含类别1")
        if class1_files:
            print(f"\n示例文件: {class1_files[0]}")
            print("内容:")
            with open(class1_files[0]) as f:
                print(f.read())
    
    if 1 in train_counter:
        print("\n训练集中也有类别1")
    else:
        print("\n⚠️ 训练集中完全没有类别1 (goForward)!")
    
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"训练集类别: {sorted(train_counter.keys())}")
    print(f"验证集类别: {sorted(val_counter.keys())}")
    
    train_set = set(train_counter.keys())
    val_set = set(val_counter.keys())
    
    if train_set != val_set:
        print("\n⚠️ 警告: 训练集和验证集的类别不一致!")
        print(f"只在验证集中: {val_set - train_set}")
        print(f"只在训练集中: {train_set - val_set}")

if __name__ == "__main__":
    check_dataset()

