"""
简单的模型评估脚本
支持不同数据配置（全部/白天/夜间）
"""

from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--model', type=str, required=True, help='模型路径(.pt)')
    parser.add_argument('--data', type=str, required=True, help='数据配置(.yaml)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("模型评估")
    print("=" * 70)
    
    print(f"\n模型: {args.model}")
    print(f"数据: {args.data}")
    
    # 加载模型
    model = YOLO(args.model)
    
    # 验证
    print("\n开始验证...\n")
    metrics = model.val(data=args.data)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    
    print(f"\n整体性能:")
    print(f"  mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"  Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"  Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    
    # 各类别
    print(f"\n各类别mAP50:")
    class_names = ['go', 'goForward', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    if hasattr(metrics.box, 'maps'):
        for i, name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"  {name:12s}: {metrics.box.maps[i]:.4f} ({metrics.box.maps[i]*100:.2f}%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

