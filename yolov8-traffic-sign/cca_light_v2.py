"""
CCA_Light_v2 - 改进版轻量级上下文增强模块
修改点：
1. BN → GroupNorm（提升稳定性）
2. LCFE简化为单卷积（减少复杂度）
3. 残差系数0.1（降低CCA影响，防止破坏原特征）
4. SE reduction=8（降低压缩比）
5. 添加post_norm（输出归一化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    SE（Squeeze-and-Excitation）通道注意力
    reduction=8（降低压缩比，保留更多信息）
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LCFE_Lite(nn.Module):
    """
    LCFE简化版 - 单卷积版本
    去掉多分支softmax融合，简化为标准卷积
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.GroupNorm(32, c_out),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.conv(x)


class CCA_Light_v2(nn.Module):
    """
    CCA_Light v2 - 改进版
    
    改进：
    1. BN → GroupNorm
    2. LCFE简化
    3. 残差系数0.1
    4. 输出归一化
    5. SE reduction=8
    
    Args:
        c1: 输入通道数
        c2: 输出通道数（默认等于c1）
        use_se: 是否使用SE（默认True）
    """
    def __init__(self, c1, c2=None, use_se=True):
        super().__init__()
        
        if c2 is None:
            c2 = c1
        
        self.c1 = c1
        self.c2 = c2
        self.use_se = use_se
        
        print(f"  [CCA_Light_v2] c1={c1}, c2={c2}, use_se={use_se}")
        
        # LCFE简化版
        self.lcfe = LCFE_Lite(c1, c2)
        
        # SE通道注意力（reduction=8）
        if use_se:
            self.se = SELayer(c2, reduction=8)
        else:
            self.se = nn.Identity()
        
        # 1x1混合卷积
        self.mix = nn.Conv2d(c2, c2, 1, 1, bias=False)
        
        # 输出归一化（GroupNorm）
        self.post_norm = nn.GroupNorm(32, c2)
        
        # 残差适配层
        if c1 != c2:
            self.shortcut = nn.Conv2d(c1, c2, 1, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # LCFE提取局部上下文
        x_context = self.lcfe(x)
        
        # SE通道注意力
        x_context = self.se(x_context)
        
        # 1x1混合
        x_context = self.mix(x_context)
        
        # 残差连接（系数0.1，降低CCA影响）
        out = self.shortcut(x) + 0.1 * x_context
        
        # 输出归一化
        out = self.post_norm(out)
        
        return out


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试CCA_Light_v2模块")
    print("=" * 60)
    
    # 创建模块
    cca_v2 = CCA_Light_v2(c1=256, c2=256, use_se=True)
    
    # 统计参数
    total_params = sum(p.numel() for p in cca_v2.parameters())
    print(f"\nCCA_Light_v2参数量: {total_params:,}")
    
    # 测试forward
    x = torch.randn(2, 256, 32, 32)
    print(f"\n输入: {x.shape}")
    
    cca_v2.eval()
    with torch.no_grad():
        out = cca_v2(x)
    
    print(f"输出: {out.shape}")
    print(f"✓ 测试通过")
    
    print("\n改进点:")
    print("  ✓ BN → GroupNorm（更稳定）")
    print("  ✓ LCFE简化（单卷积）")
    print("  ✓ 残差系数0.1（降低干扰）")
    print("  ✓ 输出归一化")
    print("  ✓ SE reduction=8")

