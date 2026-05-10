"""修改checkpoint中的epochs参数，使其可以继续训练"""
import torch
import argparse

def fix_checkpoint(ckpt_path, new_epochs):
    """
    修改checkpoint中的epochs参数
    
    Args:
        ckpt_path: checkpoint路径
        new_epochs: 新的目标epochs数
    """
    print(f"加载checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    old_epochs = ckpt.get('train_args', {}).get('epochs', '?')
    current_epoch = ckpt.get('epoch', '?')
    
    print(f"  当前epoch: {current_epoch}")
    print(f"  原目标epochs: {old_epochs}")
    print(f"  新目标epochs: {new_epochs}")
    
    if 'train_args' in ckpt:
        ckpt['train_args']['epochs'] = new_epochs
        
        # 关键：设置epoch字段（0-indexed）
        # 如果已训练50个epoch，epoch应该是49
        if current_epoch == -1:
            # -1表示最后一个epoch，需要根据results.csv确定
            import pandas as pd
            csv_path = ckpt_path.replace('weights/last.pt', 'results.csv')
            try:
                df = pd.read_csv(csv_path)
                actual_epochs = len(df)
                ckpt['epoch'] = actual_epochs - 1  # 0-indexed
                print(f"  根据results.csv设置epoch为: {actual_epochs - 1}")
            except:
                ckpt['epoch'] = old_epochs - 1
        
        # 保存修改后的checkpoint
        torch.save(ckpt, ckpt_path)
        print(f"\n✅ Checkpoint已更新！")
        print(f"   现在可以resume训练到{new_epochs} epochs")
    else:
        print("❌ Checkpoint格式不正确")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='修复checkpoint以支持继续训练')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint路径')
    parser.add_argument('--epochs', type=int, required=True, help='新的目标epochs数')
    
    args = parser.parse_args()
    fix_checkpoint(args.ckpt, args.epochs)

