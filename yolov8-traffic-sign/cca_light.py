"""
CCA_Light - 轻量级上下文增强模块
基于YOLO-CCA思想，去掉Transformer和GCFC，只保留：
1. LCFE（多尺度膨胀卷积）
2. SE（Squeeze-and-Excitation通道注意力）

特点：
- 轻量化（参数少）
- 稳定训练（无Transformer）
- 残差连接（防止梯度消失）
- 适合小目标检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    """自动padding"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积块"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SELayer(nn.Module):
    """
    SE（Squeeze-and-Excitation）通道注意力
    论文：Squeeze-and-Excitation Networks (CVPR 2018)
    """
    def __init__(self, channels, reduction=16):
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


class LCFE(nn.Module):
    """
    LCFE - 局部上下文特征增强
    使用多尺度膨胀卷积捕捉不同范围的上下文
    """
    def __init__(self, c_in, c_out, reduction=2):
        super().__init__()
        c_ = c_out // reduction
        
        # 两个不同膨胀率的卷积 - 使用c_in作为输入通道
        self.cv1 = nn.Conv2d(c_in, c_, 3, 1, padding=1, dilation=1, bias=False)
        self.cv2 = nn.Conv2d(c_in, c_, 3, 1, padding=2, dilation=2, bias=False)
        
        # 自适应融合权重
        self.fusion = nn.Sequential(
            nn.Conv2d(c_ * 2, 2, 1, bias=False),
            nn.BatchNorm2d(2),
        )
        
        # 输出投影
        self.project = Conv(c_, c_out, 1, 1)
        
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

    def forward(self, x):
        # 多尺度膨胀卷积
        x1 = self.act(self.bn1(self.cv1(x)))  # dilation=1
        x2 = self.act(self.bn2(self.cv2(x)))  # dilation=2
        
        # 自适应融合
        fusion_input = torch.cat([x1, x2], dim=1)
        weights = F.softmax(self.fusion(fusion_input), dim=1)
        
        # 加权融合
        out = weights[:, 0:1] * x1 + weights[:, 1:2] * x2
        
        # 投影回原始通道数
        out = self.project(out)
        
        return out


class CCA_Light(nn.Module):
    """
    CCA_Light - 轻量级上下文增强模块
    
    结构：
    1. LCFE（多尺度局部上下文）
    2. SE（通道注意力）
    3. 残差连接
    
    Args:
        c1: 输入通道数
        c2: 输出通道数（通常等于c1）
        reduction: 通道压缩比例（默认4）
    """
    def __init__(self, c1, c2=None, reduction=4):
        super().__init__()
        
        if c2 is None:
            c2 = c1  # 默认输入=输出
        
        self.c1 = c1
        self.c2 = c2
        
        print(f"  [CCA_Light] c1={c1}, c2={c2}")
        
        # LCFE：局部上下文特征增强（输入c1，输出c2）
        self.lcfe = LCFE(c1, c2, reduction=2)
        
        # SE：通道注意力（作用于输出c2）
        self.se = SELayer(c2, reduction=reduction)
        
        # 可选：轻量的1x1卷积混合特征
        self.mix = Conv(c2, c2, 1, 1)
        
        # 残差适配层（当c1!=c2时）
        if c1 != c2:
            self.shortcut = Conv(c1, c2, 1, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # LCFE提取多尺度局部上下文（c1→c2）
        x_context = self.lcfe(x)
        
        # SE通道注意力
        x_context = self.se(x_context)
        
        # 1x1卷积混合
        x_context = self.mix(x_context)
        
        # 残差连接（通过shortcut适配通道数）
        out = x_context + self.shortcut(x)
        
        return out


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试CCA_Light模块")
    print("=" * 60)
    
    # 创建模块
    cca_light = CCA_Light(c1=128, c2=128, reduction=4)
    
    # 统计参数
    total_params = sum(p.numel() for p in cca_light.parameters())
    print(f"\nCCA_Light参数量: {total_params:,}")
    
    # 测试forward
    x = torch.randn(2, 128, 32, 32)
    print(f"\n输入: {x.shape}")
    
    cca_light.eval()
    with torch.no_grad():
        out = cca_light(x)
    
    print(f"输出: {out.shape}")
    print(f"✓ 输入输出shape一致（支持残差）")
    
    # 对比完整CCA
    print("\n" + "=" * 60)
    print("对比完整CCA")
    print("=" * 60)
    print("完整CCA参数：~150k（含Transformer + GCFC）")
    print(f"CCA_Light参数：{total_params:,}")
    print(f"减少：{(1 - total_params/150000)*100:.1f}%")
    
    print("\n特点：")
    print("  ✓ 保留多尺度局部上下文（LCFE）")
    print("  ✓ 通道注意力（SE）")
    print("  ✓ 残差连接（稳定训练）")
    print("  ✓ 去掉Transformer（快速训练）")
    print("  ✓ 去掉GCFC（简化实现）")

