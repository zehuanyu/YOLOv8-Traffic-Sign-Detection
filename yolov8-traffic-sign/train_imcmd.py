"""训练 IMCMD YOLOv8"""

# 注册IMCMD模块
from ultralytics.nn import modules
import ultralytics.nn.tasks as tasks
import imcmd_modules

for name in ['C2f_CA', 'AMFF', 'AGRFM', 'Detect_IMCMD', 'CoordinateAttention', 'SKAttention']:
    setattr(modules, name, getattr(imcmd_modules, name))
    setattr(tasks, name, getattr(imcmd_modules, name))
print("✓ IMCMD模块已注册（含AGRFM）\n")

from ultralytics import YOLO
from imcmd_model import create_imcmd_model
from pathlib import Path
import argparse


def train_imcmd(
    model_type='small',
    variant='imcmd',  # 'imcmd' 或 'imcmd_ts'
    data_yaml='lisa_yolo_redistributed/data.yaml',
    epochs=50,
    batch=16,
    device='0',
    name=None,
    resume=False,
):
    """
    训练IMCMD或IMCMD-TS YOLOv8，支持断点续训
    
    Args:
        model_type: 'small' 或 'large'
        variant: 'imcmd' (纯IMCMD+AMFF) 或 'imcmd_ts' (IMCMD+YOLO-TS AGRFM)
        data_yaml: 数据集配置文件
        epochs: 训练轮数
        batch: batch size
        device: GPU设备
        name: 实验名称
        resume: 是否从checkpoint恢复
    """
    # 根据variant选择YAML文件
    if variant == 'imcmd_ts':
        yaml_file = f'yolov8s_imcmd_ts_{model_type}.yaml'
        variant_name = 'IMCMD-TS'
    else:
        yaml_file = f'yolov8s_imcmd_{model_type}.yaml'
        variant_name = 'IMCMD'
    
    if name is None:
        name = f'{variant}_{model_type}_{epochs}epochs'
    
    print("=" * 70)
    print(f"{variant_name} YOLOv8 训练 - {model_type.upper()}版本")
    print(f"YAML配置: {yaml_file}")
    print("=" * 70)
    
    # 检查是否有checkpoint可以恢复
    checkpoint_path = Path(f'runs/traffic_sign/{name}/weights/last.pt')
    
    if resume and checkpoint_path.exists():
        print(f"\n✓ 发现checkpoint: {checkpoint_path}")
        print(f"  从上次训练继续...\n")
        # 加载checkpoint中的模型
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # 重新创建模型并加载权重
        model_imcmd = create_imcmd_model(yaml_file, verbose=False)
        # 加载权重
        if 'model' in ckpt:
            model_imcmd.load_state_dict(ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model'])
            print(f"  ✅ 已加载权重")
    else:
        print(f"\n构建模型...")
        model_imcmd = create_imcmd_model(yaml_file, verbose=False)
    
    total = sum(p.numel() for p in model_imcmd.parameters())
    print(f"✓ 总参数: {total:,}")
    print(f"  vs YOLOv8s: {total-11166560:+,} ({(total/11166560-1)*100:+.1f}%)")
    
    # 创建YOLO包装器（不重新解析YAML）
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import DEFAULT_CFG, RANK
    import yaml
    
    # 手动创建trainer
    class IMCMD_Trainer(DetectionTrainer):
        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            # 确保overrides不是None
            if overrides is None:
                overrides = {}
            
            # 如果不是resume模式，在init后设置模型
            self._imcmd_model = model_imcmd
            self._is_resume = 'resume' in overrides and overrides['resume']
            
            super().__init__(cfg, overrides, _callbacks)
            
            # 如果不是resume，强制使用我们的模型
            if not self._is_resume:
                self.model = self._imcmd_model
        
        def get_model(self, cfg=None, weights=None, verbose=True):
            """返回我们已经创建好的IMCMD模型"""
            return self._imcmd_model
    
    # 使用trainer训练
    overrides = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch,
        'imgsz': 640,
        'device': device,
        'project': 'runs/traffic_sign',
        'name': name,
        'exist_ok': True,
        'model': yaml_file,
        'amp': False,  # 禁用AMP，避免AMFF lazy init的类型冲突
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'patience': 50,
        'save_period': 10,
        'plots': True,
        'workers': 8,
    }
    
    # 如果是resume模式，添加resume参数
    if resume and checkpoint_path.exists():
        overrides['resume'] = str(checkpoint_path)
    
    trainer = IMCMD_Trainer(cfg=DEFAULT_CFG, overrides=overrides)
    
    # 训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70 + "\n")
    
    results = trainer.train()
    
    print("\n" + "=" * 70)
    print("✓ 训练完成")
    print("=" * 70)
    
    # 读取最终结果
    import pandas as pd
    csv_path = Path(f'runs/traffic_sign/{name}/results.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        final = df.iloc[-1]
        
        print(f"\n📊 最终结果 (Epoch {len(df)}):")
        print("=" * 70)
        print(f"  Precision:     {final['metrics/precision(B)']:.1%}")
        print(f"  Recall:        {final['metrics/recall(B)']:.1%}")
        print(f"  mAP@0.5:       {final['metrics/mAP50(B)']:.1%}")
        print(f"  mAP@0.5:0.95:  {final['metrics/mAP50-95(B)']:.1%}")
        print("=" * 70)
        
        print(f"\n💾 结果保存在: runs/traffic_sign/{name}/")
        print(f"   - best.pt (最佳权重)")
        print(f"   - results.png (训练曲线)")
        print(f"   - confusion_matrix.png (混淆矩阵)")
        print("=" * 70 + "\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMCMD / IMCMD-TS YOLOv8训练')
    parser.add_argument('--model', type=str, default='small', choices=['small', 'large'],
                        help='模型大小: small或large')
    parser.add_argument('--variant', type=str, default='imcmd', choices=['imcmd', 'imcmd_ts'],
                        help='模型变体: imcmd(纯IMCMD+AMFF) 或 imcmd_ts(IMCMD+YOLO-TS AGRFM)')
    parser.add_argument('--data', type=str, default='lisa_yolo_redistributed/data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='从checkpoint恢复训练')
    
    args = parser.parse_args()
    
    train_imcmd(
        model_type=args.model,
        variant=args.variant,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
        resume=args.resume,
    )

