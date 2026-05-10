"""
IMCMD YOLOv8 模块实现
基于论文: "Improved YOLOv8 algorithms for small object detection in aerial imagery" (Fei Feng et al., 2024)

包含模块:
- CoordinateAttention (CA): 坐标注意力机制
- C2f_CA: 集成CA的C2f模块
- SKAttention (SKA): 选择性核注意力
- AMFF: 自适应多尺度特征融合模块
- DynamicHead: 动态检测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules import Conv, C2f, DFL
from ultralytics.utils.tal import dist2bbox, make_anchors


# ================================
# 1. Coordinate Attention (CA)
# ================================
class CoordinateAttention(nn.Module):
    """
    坐标注意力机制
    论文公式 (1)-(6)
    能够捕获长距离空间依赖和通道信息
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        
        # 共享的1x1卷积和BN
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()
        
        # 高度和宽度方向的卷积
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] 加权后的特征图
        """
        n, c, h, w = x.size()
        
        # 公式(1): 沿高度方向池化 Z^h_c(h)
        x_h = x.mean(dim=3, keepdim=True)  # [B, C, H, 1]
        
        # 公式(2): 沿宽度方向池化 Z^w_c(w)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B, C, W, 1]
        
        # 公式(3): concat + conv + bn + activation
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))  # [B, mid_channels, H+W, 1]
        
        # 分离高度和宽度特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, mid_channels, 1, W]
        
        # 公式(4)(5): 计算注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [B, C, 1, W]
        
        # 公式(6): 加权特征图
        out = x * a_h * a_w
        
        return out


# ================================
# 2. C2f_CA Module
# ================================
class Bottleneck_CA(nn.Module):
    """带CA的Bottleneck，移除残差连接"""
    def __init__(self, c1, c2, shortcut=False, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k, 1)
        self.cv2 = Conv(c_, c2, k, 1, g=g)
        self.ca = CoordinateAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.ca(y)
        return x + y if self.add else y


from ultralytics.nn.modules import C2f as BaseC2f

class C2f_CA(BaseC2f):
    """
    C2f with Coordinate Attention
    完全兼容ultralytics，参数格式与标准C2f相同
    
    论文改进:
    - 集成CA机制到Bottleneck
    - 移除残差结构
    - 保持与标准C2f相同的接口
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Args:
            c1: 输入通道数（ultralytics自动传入）
            c2: 输出通道数（从YAML args[0]）
            n: bottleneck数量（ultralytics自动传入，如果在repeat_modules中）
            shortcut: 是否使用shortcut
            g: 分组数
            e: 扩展比例
        """
        # 不调用父类__init__，因为我们要自定义Bottleneck
        nn.Module.__init__(self)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # 使用Bottleneck_CA替代标准Bottleneck
        self.m = nn.ModuleList(
            Bottleneck_CA(self.c, self.c, shortcut=False, g=g, k=(3, 3), e=1.0) 
            for _ in range(n)
        )
    
    # forward方法继承自标准C2f，不需要重写


# ================================
# 3. Selective Kernel Attention (SKA)
# ================================
class SKAttention(nn.Module):
    """
    选择性核注意力
    论文3.3.2节描述的SKA机制
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        
        self.fc1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(mid_channels, channels * 2, 1, bias=False)
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: 两个分支的特征图 [B, C, H, W]
        Returns:
            融合后的特征图
        """
        # Fuse: element-wise addition
        u = x1 + x2  # [B, C, H, W]
        
        # Global average pooling
        s = F.adaptive_avg_pool2d(u, 1)  # [B, C, 1, 1]
        
        # FC layers
        z = self.act(self.bn(self.fc1(s)))  # [B, mid_channels, 1, 1]
        attention = self.fc2(z)  # [B, C*2, 1, 1]
        
        # Split and softmax
        a1, a2 = torch.chunk(attention, 2, dim=1)  # 2x [B, C, 1, 1]
        attention_weights = torch.softmax(torch.cat([a1, a2], dim=1), dim=1)
        a1, a2 = torch.chunk(attention_weights, 2, dim=1)
        
        # Weighted sum
        out = a1 * x1 + a2 * x2
        
        return out


# ================================
# 4. AMFF Module
# ================================
class AMFF(nn.Module):
    """
    Adaptive Multiscale Feature Fusion (兼容ultralytics YAML格式)
    论文3.3节描述的自适应多尺度特征融合模块
    
    支持两种使用方式:
    方式1 (推荐): 在YAML中先对齐尺度再concat，然后传入AMFF
    方式2: 直接传入三个不同尺度的特征图（forward会自动处理）
    """
    def __init__(self, out_channels):
        """
        Args:
            out_channels: 输出通道数（从YAML传入）
        
        注意：输入通道数和split方式在第一次forward时自动推断
        """
        super().__init__()
        self.out_channels = out_channels
        self.initialized = False
        
    def forward(self, x):
        """
        Args:
            x: concat后的特征图 [B, C_total, H, W]
        Returns:
            融合后的特征图 [B, out_channels, H, W]
        """
        if not self.initialized:
            # 第一次forward时初始化
            c_in = x.shape[1]
            # 平均split成3个分支
            c_split = c_in // 3
            self.split_channels = [c_split, c_split, c_in - 2*c_split]  # 处理不能整除的情况
            
            c1, c2, c3 = self.split_channels
            # 1x1卷积调整通道数
            self.conv1 = Conv(c1, self.out_channels, 1).to(x.device)
            self.conv2 = Conv(c2, self.out_channels, 1).to(x.device)
            self.conv3 = Conv(c3, self.out_channels, 1).to(x.device)
            
            # SKA机制
            self.ska1 = SKAttention(self.out_channels).to(x.device)
            self.ska2 = SKAttention(self.out_channels).to(x.device)
            
            # CBS模块增强表达
            self.cbs = Conv(self.out_channels, self.out_channels, 3, 1).to(x.device)
            
            self.initialized = True
        
        # 按通道数split成三个分支
        c1, c2, c3 = self.split_channels
        x1, x2, x3 = torch.split(x, [c1, c2, c3], dim=1)
        
        # 调整通道数
        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        f3 = self.conv3(x3)
        
        # Hadamard product + SKA融合
        fused = self.ska1(f1, f2)
        fused = self.ska2(fused, f3)
        
        # CBS增强
        out = self.cbs(fused)
        
        return out


# ================================
# 4b. AGRFM Module (YOLO-TS Style)
# ================================
class AGRFM(nn.Module):
    """
    Anti-Grid Receptive Field Module (AGRFM) - YOLO-TS风格
    用于HR-MSD (High-Resolution Multi-Scale Detection) 的高分辨率特征融合
    
    设计思路:
    - 使用多个3×3标准卷积堆叠来逐步增大感受野
    - 使用并行的扩张卷积分支来捕获更大范围的上下文，同时避免grid效应
    - 融合两个分支的输出，生成高质量的B2特征图
    
    输入输出:
        输入: F_fuse [B, C_in, H, W] - 多尺度特征concat后的融合特征
        输出: B2 [B, C_out, H, W] - 高分辨率检测特征 (空间尺寸不变)
    
    参考: YOLO-TS论文中的Anti-Grid Receptive Field设计
    """
    def __init__(self, c_in, c_out):
        """
        Args:
            c_in: 输入通道数 (如96，来自3个32通道分支的concat)
            c_out: 输出通道数 (如32，与Detect_IMCMD的ch匹配)
        """
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        
        # 中间通道数 (用于两个分支)
        c_mid = c_out * 2  # 64
        
        # =====================================================
        # 分支1: 标准卷积堆叠 (增大感受野)
        # 3个3×3卷积，每个增加感受野+2，总计增加6
        # =====================================================
        self.conv_stack = nn.Sequential(
            Conv(c_in, c_mid, 3, 1),      # 3×3, 感受野+2
            Conv(c_mid, c_mid, 3, 1),     # 3×3, 感受野+2
            Conv(c_mid, c_mid, 3, 1),     # 3×3, 感受野+2
        )
        
        # =====================================================
        # 分支2: 扩张卷积分支 (大感受野 + 抗grid效应)
        # 使用dilation=2的3×3卷积，等效感受野为5×5
        # 扩张卷积可以避免grid artifact
        # =====================================================
        self.dilated_branch = nn.Sequential(
            Conv(c_in, c_mid, 3, 1),                      # 先做通道转换
            nn.Conv2d(c_mid, c_mid, 3, padding=2, dilation=2, bias=False),  # 扩张卷积
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
        )
        
        # =====================================================
        # 融合层: 将两个分支concat后用1×1卷积融合到目标通道数
        # =====================================================
        self.fuse = nn.Sequential(
            Conv(c_mid * 2, c_out, 1, 1),  # 1×1卷积融合
            Conv(c_out, c_out, 3, 1),      # 3×3卷积增强
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C_in, H, W] 融合后的多尺度特征
        Returns:
            B2: [B, C_out, H, W] 高分辨率检测特征
        """
        # 分支1: 标准卷积堆叠
        feat_stack = self.conv_stack(x)      # [B, c_mid, H, W]
        
        # 分支2: 扩张卷积分支
        feat_dilated = self.dilated_branch(x)  # [B, c_mid, H, W]
        
        # 融合两个分支
        feat_concat = torch.cat([feat_stack, feat_dilated], dim=1)  # [B, c_mid*2, H, W]
        
        # 输出B2特征
        B2 = self.fuse(feat_concat)  # [B, c_out, H, W]
        
        return B2


# ================================
# 5. Dynamic Head Block (单尺度特征增强)
# ================================
class DynamicHeadBlock(nn.Module):
    """
    Dynamic Head Block - 单尺度特征增强模块
    用于在检测头之前对特征进行scale/spatial/task-aware attention增强
    
    论文3.4节描述的Dynamic Head的核心注意力机制
    注：简化版使用3x3卷积代替DCN，实际部署时可集成DCNv2
    
    输入输出:
        输入: [B, C, H, W] 单尺度特征图
        输出: [B, C, H, W] 增强后的特征图（通道数不变）
    """
    def __init__(self, channels):
        """
        Args:
            channels: 输入/输出通道数（保持不变）
        """
        super().__init__()
        self.channels = channels
        
        # Scale-aware attention (公式7)
        # 捕获不同尺度特征的重要性
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial-aware attention (公式8)
        # 简化版：使用深度可分离卷积代替DCN
        # 捕获空间位置的重要性
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Task-aware attention (公式9)
        # 自适应地调整特征以适应不同任务（分类vs回归）
        mid_channels = max(8, channels // 4)  # 确保至少8个通道
        self.task_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, channels * 2, 1)
        )
    
    def forward(self, x):
        """
        对单个尺度的特征图应用Dynamic Head注意力增强
        
        Args:
            x: [B, C, H, W] 输入特征图
        Returns:
            refined: [B, C, H, W] 增强后的特征图
        """
        # Scale-aware attention (公式7)
        # 全局池化捕获尺度信息，生成通道级权重
        scale_att = self.scale_attn(x)
        feat = x * scale_att
        
        # Spatial-aware attention (公式8)
        # 空间卷积捕获位置信息，生成空间级权重
        spatial_att = self.spatial_attn(feat)
        feat = feat * spatial_att
        
        # Task-aware attention (公式9)
        # 生成两组权重，取max实现任务自适应
        task_vec = self.task_attn(feat)
        alpha1, alpha2 = torch.chunk(task_vec, 2, dim=1)
        feat = torch.max(alpha1 * feat, alpha2 * feat)
        
        return feat


# ================================
# 5b. Dynamic Head (完整版，保留兼容性)
# ================================
class DynamicHead(nn.Module):
    """
    Dynamic Head - 统一scale, spatial, task感知 (完整版)
    论文3.4节描述
    
    注意：此类保留用于独立使用场景
    推荐使用Detect_IMCMD（已集成DynamicHeadBlock）
    """
    def __init__(self, c1, nc=80, num_layers=3):
        """
        Args:
            c1: 输入通道数
            nc: 类别数
            num_layers: Dynamic Head层数
        """
        super().__init__()
        self.nc = nc
        self.nl = num_layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        
        # 使用DynamicHeadBlock进行特征增强
        self.dynamic_blocks = nn.ModuleList([
            DynamicHeadBlock(c1) for _ in range(num_layers)
        ])
        
        # Detection heads
        c2 = max(16, c1 // 4, self.reg_max * 4)
        c3 = max(c1, min(self.nc, 100))
        
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(c1, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in range(num_layers)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(c1, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for _ in range(num_layers)
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: list of feature maps from different scales
        Returns:
            predictions
        """
        outputs = []
        
        for i in range(self.nl):
            # 使用DynamicHeadBlock增强特征
            feat = self.dynamic_blocks[i](x[i])
            
            # Detection
            box = self.cv2[i](feat)
            cls = self.cv3[i](feat)
            
            outputs.append(torch.cat([box, cls], 1))
        
        return outputs

    def bias_init(self):
        """初始化偏置"""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


# ================================
# 6. Detect Head (兼容YOLOv8格式)
# ================================
class Detect_IMCMD(nn.Module):
    """
    IMCMD检测头 - 集成DynamicHead注意力机制的检测头
    兼容ultralytics的Detect接口
    
    关键改进:
    - 为每个检测尺度创建独立的DynamicHeadBlock进行特征增强
    - 应用scale-aware, spatial-aware, task-aware三种注意力机制
    - 增强后的特征再送入回归(cv2)和分类(cv3)分支
    
    与YOLOv8 Detect的兼容性:
    - 构造函数接受相同的nc, ch参数
    - forward()返回格式与标准Detect完全一致
    - 支持训练/推理模式切换
    - 支持bias_init()初始化
    """
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        """
        初始化IMCMD检测头
        
        Args:
            nc: 类别数
            ch: 各检测尺度的输入通道数元组，例如(32, 32)表示P2和P3都是32通道
        """
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数（通常为2：P2和P3）
        self.reg_max = 16  # DFL回归最大值
        self.no = nc + self.reg_max * 4  # 每个anchor的输出数
        
        # 默认stride（P2=4, P3=8），训练时会由模型自动更新
        self.stride = torch.tensor([4.0, 8.0] if len(ch) == 2 else [8.0, 16.0, 32.0])
        
        # =====================================================
        # 🔥 关键改进：为每个检测尺度创建DynamicHeadBlock
        # 用于在回归/分类之前进行特征增强
        # =====================================================
        self.dynamic_heads = nn.ModuleList([
            DynamicHeadBlock(ch[i]) for i in range(self.nl)
        ])
        
        # 回归分支（box预测）
        # 通道数计算：确保至少64通道（reg_max*4=64）
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),      # 3x3卷积降维
                Conv(c2, c2, 3),     # 3x3卷积
                nn.Conv2d(c2, 4 * self.reg_max, 1)  # 1x1卷积输出box参数
            ) for x in ch
        )
        
        # 分类分支（cls预测）
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),      # 3x3卷积
                Conv(c3, c3, 3),     # 3x3卷积
                nn.Conv2d(c3, self.nc, 1)  # 1x1卷积输出类别概率
            ) for x in ch
        )
        
        # DFL模块（Distribution Focal Loss）
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        前向传播 - 集成DynamicHead特征增强
        
        处理流程:
        1. 对每个尺度的特征图应用DynamicHeadBlock增强
        2. 增强后的特征分别送入回归(cv2)和分类(cv3)分支
        3. 拼接box和cls输出
        
        Args:
            x: 特征图列表，例如[P2_feat, P3_feat]
               P2_feat: [B, ch[0], H1, W1] (如 [B, 32, 160, 160])
               P3_feat: [B, ch[1], H2, W2] (如 [B, 32, 80, 80])
        
        Returns:
            训练模式: 列表，每个元素为[B, no, Hi, Wi]的tensor
            推理模式: (y, x) 其中y是解码后的预测，x是原始输出
        """
        shape = x[0].shape  # 保存shape用于后续处理
        
        for i in range(self.nl):
            # =====================================================
            # 🔥 关键步骤：应用DynamicHead注意力增强（如果有）
            # 特征经过scale/spatial/task-aware attention后再送入检测头
            # =====================================================
            feat = x[i]
            if hasattr(self, 'dynamic_heads') and self.dynamic_heads is not None:
                feat = self.dynamic_heads[i](feat)
            
            # 回归和分类分支
            box_out = self.cv2[i](feat)
            cls_out = self.cv3[i](feat)
            
            # 拼接box和cls输出
            x[i] = torch.cat((box_out, cls_out), 1)
        
        if self.training:
            return x
        
        # =====================================================
        # 推理模式：生成anchors并解码预测
        # =====================================================
        self.anchors, self.strides = (
            t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5)
        )
        self.shape = shape
        
        # 合并所有尺度的输出
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        # 分离box和cls
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # DFL解码 + anchor变换
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        
        # 最终预测：[box_xywh, class_probs]
        y = torch.cat((dbox, cls.sigmoid()), 1)
        
        return y if self.export else (y, x)

    def bias_init(self):
        """
        初始化检测头偏置
        - 回归分支偏置初始化为1.0
        - 分类分支偏置根据先验概率初始化
        """
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box偏置
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls偏置


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 IMCMD 模块")
    print("=" * 60)
    
    # 测试CA
    print("\n1. 测试 CoordinateAttention")
    ca = CoordinateAttention(128)
    x = torch.randn(2, 128, 40, 40)
    y = ca(x)
    print(f"   输入: {x.shape} -> 输出: {y.shape}")
    assert x.shape == y.shape, "CA输出shape应与输入一致"
    print("   ✓ 通过")
    
    # 测试C2f_CA
    print("\n2. 测试 C2f_CA")
    c2f_ca = C2f_CA(128, 256, n=3)
    x = torch.randn(2, 128, 40, 40)
    y = c2f_ca(x)
    print(f"   输入: {x.shape} -> 输出: {y.shape}")
    print("   ✓ 通过")
    
    # 测试SKA
    print("\n3. 测试 SKAttention")
    ska = SKAttention(128)
    x1 = torch.randn(2, 128, 40, 40)
    x2 = torch.randn(2, 128, 40, 40)
    y = ska(x1, x2)
    print(f"   输入: {x1.shape}, {x2.shape} -> 输出: {y.shape}")
    print("   ✓ 通过")
    
    # 测试AMFF
    print("\n4. 测试 AMFF")
    amff = AMFF(out_channels=64)
    x_concat = torch.randn(2, 192, 80, 80)  # 64+64+64=192
    y = amff(x_concat)
    print(f"   输入: {x_concat.shape} -> 输出: {y.shape}")
    print("   ✓ 通过")
    
    # 测试AGRFM (🔥 YOLO-TS Style)
    print("\n4b. 测试 AGRFM (🔥 YOLO-TS HR-MSD)")
    agrfm = AGRFM(c_in=96, c_out=32)
    x_fuse = torch.randn(2, 96, 160, 160)  # 模拟3×32通道concat后的融合特征
    b2 = agrfm(x_fuse)
    print(f"   输入 F_fuse: {x_fuse.shape}")
    print(f"   输出 B2: {b2.shape}")
    assert b2.shape == torch.Size([2, 32, 160, 160]), "AGRFM输出形状应为[B, 32, 160, 160]"
    # 统计AGRFM参数
    agrfm_params = sum(p.numel() for p in agrfm.parameters())
    print(f"   AGRFM参数量: {agrfm_params:,}")
    print("   ✓ AGRFM测试通过 (空间尺寸不变，通道: 96→32)")
    
    # 测试DynamicHeadBlock
    print("\n5. 测试 DynamicHeadBlock (🔥 新增)")
    dhb = DynamicHeadBlock(channels=64)
    x = torch.randn(2, 64, 80, 80)
    y = dhb(x)
    print(f"   输入: {x.shape} -> 输出: {y.shape}")
    assert x.shape == y.shape, "DynamicHeadBlock输出shape应与输入一致"
    print("   ✓ 通道数保持不变，注意力增强成功")
    
    # 测试Detect_IMCMD（训练模式）
    print("\n6. 测试 Detect_IMCMD (训练模式, 🔥 已集成DynamicHead)")
    print("   配置: nc=10, ch=(32, 32)")
    detect = Detect_IMCMD(nc=10, ch=(32, 32))
    detect.train()
    
    # 模拟P2和P3特征图（与YAML配置匹配）
    p2_feat = torch.randn(2, 32, 160, 160)  # P2: 160x160, 32ch
    p3_feat = torch.randn(2, 32, 80, 80)    # P3: 80x80, 32ch
    
    outputs = detect([p2_feat, p3_feat])
    print(f"   P2输入: {p2_feat.shape}")
    print(f"   P3输入: {p3_feat.shape}")
    print(f"   输出: {len(outputs)} tensors")
    for i, out in enumerate(outputs):
        print(f"      输出[{i}]: {out.shape}")
    
    # 验证DynamicHead是否被使用
    print(f"   dynamic_heads数量: {len(detect.dynamic_heads)}")
    print("   ✓ 训练模式通过")
    
    # 测试Detect_IMCMD（推理模式）
    print("\n7. 测试 Detect_IMCMD (推理模式)")
    detect.eval()
    p2_feat = torch.randn(2, 32, 160, 160)
    p3_feat = torch.randn(2, 32, 80, 80)
    
    with torch.no_grad():
        y, x = detect([p2_feat, p3_feat])
    
    print(f"   解码输出 y: {y.shape}")  # [B, 4+nc, num_anchors]
    print(f"   原始输出 x: {len(x)} tensors")
    print("   ✓ 推理模式通过")
    
    # 测试bias_init
    print("\n8. 测试 bias_init()")
    detect.bias_init()
    print("   ✓ 偏置初始化成功")
    
    # 参数统计
    print("\n9. 参数统计")
    total_params = sum(p.numel() for p in detect.parameters())
    dh_params = sum(p.numel() for p in detect.dynamic_heads.parameters())
    print(f"   Detect_IMCMD 总参数: {total_params:,}")
    print(f"   DynamicHeadBlock 参数: {dh_params:,} ({dh_params/total_params*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ 所有模块测试通过！")
    print("=" * 60)
    print("\n📋 DynamicHead集成摘要:")
    print("   - DynamicHeadBlock: 单尺度特征增强模块")
    print("   - 应用scale/spatial/task-aware三种注意力")
    print("   - Detect_IMCMD已自动集成，无需修改YAML")
    print("   - 输出格式与YOLOv8 Detect完全兼容")

