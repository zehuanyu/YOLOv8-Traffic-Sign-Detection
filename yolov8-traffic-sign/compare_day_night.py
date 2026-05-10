"""对比各模型在白天/黑夜的性能"""
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def main():
    print('=' * 80)
    print('各模型 白天vs黑夜 性能对比')
    print('=' * 80)

    models = [
        ('runs/traffic_sign/baseline_yolov8s_50epochs/weights/best.pt', 'Baseline'),
        ('runs/traffic_sign/imcmd_small_100epochs/weights/best.pt', 'IMCMD'),
        ('runs/traffic_sign/imcmd_ts_small_100epochs/weights/best.pt', 'IMCMD-TS'),
        ('runs/traffic_sign/yolo_ts_100epochs/weights/best.pt', 'YOLO-TS'),
    ]

    results = []

    for model_path, name in models:
        print(f'\n>>> {name}')
        try:
            model = YOLO(model_path)
            
            print(f'  [白天] 评估中...')
            day = model.val(data='lisa_day.yaml', verbose=False, workers=0)
            
            print(f'  [黑夜] 评估中...')
            night = model.val(data='lisa_night.yaml', verbose=False, workers=0)
            
            gap = day.box.map50 - night.box.map50
            results.append({
                'name': name,
                'day_map50': day.box.map50,
                'night_map50': night.box.map50,
                'gap': gap,
                'gap_pct': gap / day.box.map50 * 100 if day.box.map50 > 0 else 0
            })
            print(f'  白天: {day.box.map50:.4f}, 黑夜: {night.box.map50:.4f}, 差距: {gap:.4f}')
        except Exception as e:
            print(f'  错误: {e}')

    print('\n' + '=' * 80)
    print('对比总结')
    print('=' * 80)
    print('')
    print(f"{'模型':<20} | {'白天mAP50':>10} | {'黑夜mAP50':>10} | {'差距':>8} | {'下降比例':>8}")
    print('-' * 70)
    for r in results:
        print(f"{r['name']:<20} | {r['day_map50']:>10.4f} | {r['night_map50']:>10.4f} | {r['gap']:>8.4f} | {r['gap_pct']:>7.1f}%")

    print('=' * 80)

    # 分析
    if len(results) >= 3:
        print('')
        print('【分析结论】')
        print('-' * 80)
        
        # 白天/黑夜差距对比
        print("白天/黑夜性能差距（越小越鲁棒）:")
        for r in results:
            print(f"  {r['name']:<10}: {r['gap_pct']:.1f}%")
        
        print('')
        print("夜间性能对比:")
        baseline_night = results[0]['night_map50']
        for r in results:
            diff = r['night_map50'] - baseline_night
            if r == results[0]:
                print(f"  {r['name']:<10}: {r['night_map50']:.4f}")
            else:
                print(f"  {r['name']:<10}: {r['night_map50']:.4f} ({diff:+.4f})")
        
        print('')
        # 找出最鲁棒和夜间最佳的模型
        best_robust = min(results, key=lambda x: x['gap_pct'])
        best_night = max(results, key=lambda x: x['night_map50'])
        best_overall = max(results, key=lambda x: (x['day_map50'] + x['night_map50']) / 2)
        
        print(f"🏆 最鲁棒（白天/黑夜差距最小）: {best_robust['name']} ({best_robust['gap_pct']:.1f}%)")
        print(f"🌙 夜间最佳: {best_night['name']} ({best_night['night_map50']:.4f})")
        print(f"📊 综合最佳: {best_overall['name']}")

    elif len(results) >= 2:
        baseline = results[0]
        imcmd_ts = results[1]
        
        print('')
        print('【分析结论】')
        print('-' * 80)
        print(f"Baseline 白天/黑夜差距: {baseline['gap_pct']:.1f}%")
        print(f"IMCMD-TS 白天/黑夜差距: {imcmd_ts['gap_pct']:.1f}%")

    print('=' * 80)

if __name__ == '__main__':
    main()
