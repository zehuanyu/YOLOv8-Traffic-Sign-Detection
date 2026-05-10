"""
自定义DetectionModel，支持IMCMD模块
不需要修改ultralytics源码
"""

from ultralytics.nn.tasks import DetectionModel as BaseDetectionModel
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.nn.modules import Detect
from copy import deepcopy
import torch.nn as nn
import imcmd_modules


class IMCMD_DetectionModel(BaseDetectionModel):
    """
    支持IMCMD自定义模块的DetectionModel
    自动识别C2f_CA为repeat模块
    """
    
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        """初始化IMCMD检测模型"""
        # 先调用nn.Module的init
        nn.Module.__init__(self)
        
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        
        # 定义nc
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        
        # 解析模型 - 使用我们自定义的parse_model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.model, self.save = self._parse_model_imcmd(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)
        
        # 初始化Detect head的stride
        import torch
        m = self.model[-1]
        if isinstance(m, (Detect, imcmd_modules.Detect_IMCMD)):
            # 直接使用P2和P3的stride
            # P2在160x160 → 640/160 = 4
            # P3在80x80 → 640/80 = 8
            m.stride = torch.tensor([4.0, 8.0])
            self.stride = m.stride
            # 初始化bias
            try:
                m.bias_init()
            except:
                pass
        else:
            self.stride = torch.tensor([8.0, 16.0, 32.0])
    
    def _parse_model_imcmd(self, d, ch=3, verbose=True):
        """
        自定义的parse_model，支持C2f_CA
        基于ultralytics.nn.tasks.parse_model，添加C2f_CA支持
        """
        from ultralytics.nn.modules import (
            Conv, DFL, HGBlock, HGStem, SPP, SPPF, C1, C2, C3, C2f,
            Bottleneck, BottleneckCSP, C3x, C3TR, C3Ghost,
            GhostBottleneck, Detect, Segment, Pose, OBB, Concat
        )
        from ultralytics.utils import LOGGER
        from ultralytics.utils.ops import make_divisible
        import ast
        import contextlib
        
        # 🔥 关键：在这里定义repeat_modules和base_modules，包含我们的模块
        base_modules = frozenset({
            Conv, DFL, HGBlock, HGStem, SPP, SPPF, C1, C2, C3, C2f,
            imcmd_modules.C2f_CA,  # 🔥 添加C2f_CA
            Bottleneck, BottleneckCSP, C3x, C3TR, C3Ghost,
            GhostBottleneck, Detect, Segment, Pose, OBB, Concat,
            # AMFF不加入base_modules，因为它的参数不需要自动注入c1
            # Detect_IMCMD已经在Detect集合中特殊处理
        })
        
        repeat_modules = frozenset({
            BottleneckCSP, C1, C2, C2f,
            imcmd_modules.C2f_CA,  # 🔥 添加C2f_CA到repeat模块
            C3, C3TR, C3Ghost, C3x, Bottleneck
        })
        
        # 以下是标准parse_model逻辑的简化版
        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        
        nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
        
        if scales:
            scale = d.get('scale')
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]
        else:
            depth = depth or 1.0
            width = width or 1.0
            max_channels = float('inf')
        
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            # 获取模块
            m_str = m
            m = getattr(imcmd_modules, m, None) or eval(m) if isinstance(m, str) else m
            
            # 处理args中的字符串
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            
            n = n_ = max(round(n * depth), 1) if n > 1 else n
            
            # 🔥 处理我们的自定义模块
            # 注意：特殊模块要先判断
            if m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is imcmd_modules.AMFF:
                # AMFF: 输出通道数由args[0]决定，需要应用width scaling
                c2 = args[0]  # 输出通道数
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                # 将经过scaling的c2传给AMFF
                args = [c2]  # AMFF只需要输出通道数
            elif m is imcmd_modules.AGRFM:
                # AGRFM: 输入通道从上一层推断，输出通道应用width scaling
                c1 = ch[f] if isinstance(f, int) else ch[f[0]]
                c2 = args[0]  # 输出通道数 (c_out)
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                # AGRFM需要 (c_in, c_out) 两个参数
                args = [c1, c2]
            elif m in {Detect, Segment, Pose, OBB, imcmd_modules.Detect_IMCMD}:
                args.append([ch[x] for x in f])
            elif m is nn.BatchNorm2d:
                args = [ch[f] if isinstance(f, int) else ch[f[0]]]
            elif m in base_modules:
                c1 = ch[f] if isinstance(f, int) else ch[f[0]]
                c2 = args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                
                
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)  # 插入n参数
                    n = 1
            else:
                c2 = ch[f] if isinstance(f, int) else ch[f[0]]
            
            # 创建模块
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            
            # 记录信息
            t = str(m)[8:-2].replace('__main__.', '')
            m.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            
            if verbose:
                LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')
            
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        
        return nn.Sequential(*layers), sorted(save)


def create_imcmd_model(yaml_path, verbose=True):
    """
    创建IMCMD模型的便捷函数
    
    Args:
        yaml_path: YAML配置文件路径
        verbose: 是否打印详细信息
    
    Returns:
        model: IMCMD检测模型
    """
    from ultralytics.nn import modules
    import ultralytics.nn.tasks as tasks
    
    # 注册模块
    for name in ['C2f_CA', 'AMFF', 'AGRFM', 'Detect_IMCMD', 'CoordinateAttention', 'SKAttention']:
        setattr(modules, name, getattr(imcmd_modules, name))
        setattr(tasks, name, getattr(imcmd_modules, name))
    
    # 创建模型
    model = IMCMD_DetectionModel(yaml_path, verbose=verbose)
    
    return model


if __name__ == '__main__':
    # 测试
    print("测试IMCMD_DetectionModel")
    print("=" * 60)
    
    try:
        model = create_imcmd_model('yolov8s_imcmd_small.yaml')
        print("✅ 模型创建成功！")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数: {total_params:,}")
        
        # 测试前向传播
        import torch
        x = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"✅ 前向传播成功！")
        print(f"输出: {len(y)} tensors")
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()

