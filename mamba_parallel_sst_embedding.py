import torch
from torch import nn
import torch.nn.functional as F
import math

# ==================== MODEL ARCHITECTURE ====================
# 改进的时间编码类
class TimeEncoding(nn.Module):
    def __init__(self, num_phases=23, embed_dim=64):
        super().__init__()
        self.num_phases = num_phases
        self.embed_dim = embed_dim

        # 创建可学习的时间编码参数
        self.time_embed = nn.Parameter(torch.randn(1, num_phases, embed_dim))

        # 位置编码（正弦/余弦）作为初始化
        position = torch.arange(num_phases).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, num_phases, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # 用正弦/余弦位置编码初始化时间嵌入
        with torch.no_grad():
            self.time_embed.copy_(pe)

    def forward(self, x):
        # x: [B, 138, H, W]，其中138 = 23个时相 × 6个波段
        B, C, H, W = x.shape

        # 重塑输入，使时相维度明确
        x_reshaped = x.view(B, self.num_phases, C // self.num_phases, H, W)

        # 扩展时间编码以匹配特征维度
        time_encoding = self.time_embed.unsqueeze(-1).unsqueeze(-1)  # [1, 23, embed_dim, 1, 1]

        # 创建波段投影以匹配时间编码维度
        band_proj = nn.Conv2d(C // self.num_phases, self.embed_dim, kernel_size=1).to(x.device)

        # 将时间编码应用于每个时相
        encoded_features = []
        for t in range(self.num_phases):
            # 提取当前时相的所有波段
            phase_data = x_reshaped[:, t]  # [B, 6, H, W]

            # 将波段投影到与时间编码相同的维度
            proj_data = band_proj(phase_data)  # [B, embed_dim, H, W]

            # 添加时间编码
            encoded_phase = proj_data + time_encoding[:, t]  # [B, embed_dim, H, W]

            # 投影回原始维度
            back_proj = nn.Conv2d(self.embed_dim, C // self.num_phases, kernel_size=1).to(x.device)
            encoded_phase = back_proj(encoded_phase)  # [B, 6, H, W]

            encoded_features.append(encoded_phase)

        # 重新组合所有编码后的时相
        encoded_x = torch.stack(encoded_features, dim=1)  # [B, 23, 6, H, W]
        encoded_x = encoded_x.view(B, C, H, W)  # [B, 138, H, W]

        return encoded_x


# 季节性时间编码，考虑一年中的周期性
class SeasonalTimeEncoding(nn.Module):
    def __init__(self, num_phases=23, embed_dim=64):
        super().__init__()
        self.num_phases = num_phases
        self.embed_dim = embed_dim

        # 生成季节性位置编码（正弦/余弦）
        self.register_buffer('season_encoding', self._generate_seasonal_encoding(num_phases, embed_dim))

        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1))

        # 可学习的时间偏置
        self.time_bias = nn.Parameter(torch.zeros(1, num_phases, embed_dim, 1, 1))

    def _generate_seasonal_encoding(self, num_phases, embed_dim):
        # 生成反映年度周期性的编码
        encoding = torch.zeros(1, num_phases, embed_dim, 1, 1)

        # 假设23个时相均匀分布在一年中（约15天一次）
        for t in range(num_phases):
            # 将时相转换为一年中的位置（0到2π）
            year_pos = 2.0 * math.pi * t / num_phases

            for i in range(0, embed_dim, 2):
                div_term = 10000.0 ** (i / embed_dim)
                encoding[0, t, i, 0, 0] = math.sin(year_pos / div_term)
                if i + 1 < embed_dim:
                    encoding[0, t, i + 1, 0, 0] = math.cos(year_pos / div_term)

        return encoding

    def forward(self, x):
        # x: [B, 138, H, W]，其中138 = 23个时相 × 6个波段
        B, C, H, W = x.shape
        bands_per_phase = C // self.num_phases  # 6

        # 重塑输入以分离时相和波段
        x_reshaped = x.view(B, self.num_phases, bands_per_phase, H, W)

        # 为每个时相的每个波段应用时间编码
        encoded_features = []

        for t in range(self.num_phases):
            # 提取当前时相的所有波段
            phase_data = x_reshaped[:, t]  # [B, 6, H, W]

            # 创建与波段数匹配的编码
            phase_encoding = self.season_encoding[:, t].expand(-1, bands_per_phase, -1, -1)  # [1, 6, H, W]
            phase_bias = self.time_bias[:, t].expand(-1, bands_per_phase, -1, -1)  # [1, 6, H, W]

            # 应用缩放和偏置后的时间编码
            encoded_phase = phase_data + self.scale * phase_encoding + phase_bias

            encoded_features.append(encoded_phase)

        # 重新组合所有编码后的时相
        encoded_x = torch.stack(encoded_features, dim=1)  # [B, 23, 6, H, W]
        encoded_x = encoded_x.view(B, C, H, W)  # [B, 138, H, W]

        return encoded_x


# 简化的时间编码实现，适合直接集成到现有模型中
class EfficientTimeEncoding(nn.Module):
    def __init__(self, num_phases=23, bands_per_phase=6, embed_dim=64):
        super().__init__()
        self.num_phases = num_phases
        self.bands_per_phase = bands_per_phase
        self.embed_dim = embed_dim

        # 创建可学习的时间编码
        self.time_embed = nn.Parameter(torch.randn(1, num_phases, embed_dim))

        # 初始化为正弦/余弦位置编码
        position = torch.arange(num_phases).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, num_phases, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        with torch.no_grad():
            self.time_embed.copy_(pe)

        # 将时间编码转换为特征通道的映射
        self.time_to_feature = nn.Linear(embed_dim, bands_per_phase)

    def forward(self, x):
        # x: [B, 138, H, W]
        B, C, H, W = x.shape

        # 从编码生成时间权重
        time_weights = self.time_to_feature(self.time_embed)  # [1, 23, 6]
        time_weights = time_weights.view(1, self.num_phases * self.bands_per_phase, 1, 1)  # [1, 138, 1, 1]

        # 应用时间权重
        encoded_x = x * (1 + time_weights)

        return encoded_x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms_x + self.eps))


class ChannelWiseAttention(nn.Module):
    def __init__(self, d_model, num_heads=6, sparsity_ratio=0.1, use_sampling=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.use_sampling = use_sampling  # 新增参数控制是否使用sampling方法

        self.to_qkv = nn.Linear(d_model, d_model * 3)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x):
        B, C, X = x.shape
        qkv = self.to_qkv(x.reshape(B * C, X)).reshape(B * C, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*C, num_heads, head_dim]

        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B*C, num_heads, num_heads]

        # 稀疏化注意力
        k_heads = max(1, int(self.num_heads * self.sparsity_ratio))
        topk_scores, topk_indices = torch.topk(attn_scores, k=k_heads, dim=-1)
        sparse_attn = torch.zeros_like(attn_scores).scatter_(-1, topk_indices, torch.softmax(topk_scores, dim=-1))

        # 应用注意力
        out = (sparse_attn @ v).transpose(1, 2).reshape(B, C, self.d_model)

        # 计算通道重要性
        channel_importance = out.norm(p=2, dim=-1)  # [B, C]

        # 选择top-k通道
        k_channels = max(1, int(C * self.sparsity_ratio))

        if self.use_sampling:
            # 使用sampling方法
            probs = F.softmax(channel_importance, dim=-1)
            cdf = torch.cumsum(probs, dim=-1)  # [B, C]

            # 生成采样点
            k_points = (torch.arange(k_channels, device=x.device).float() + 0.5) / k_channels  # [k]
            k_points = k_points.unsqueeze(0).expand(B, -1)  # [B, k]

            # 逐batch采样
            topk_indices = torch.zeros(B, k_channels, dtype=torch.long, device=x.device)
            for i in range(B):
                topk_indices[i] = torch.searchsorted(cdf[i], k_points[i])

            # 去重
            topk_indices = torch.unique(topk_indices, dim=1)
        else:
            # 原始topk方法
            _, topk_indices = torch.topk(channel_importance, k=k_channels, dim=-1)

        # 收集选中的特征
        sparse_feat = torch.stack([
            out[i, topk_indices[i]] for i in range(B)
        ], dim=0)

        return sparse_feat, topk_indices, topk_indices.shape[1]


class SparseDeformableChannelMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.2, drop_rate=0.3, use_sampling=True):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = 336
        self.sparsity_ratio = sparsity_ratio
        self.use_sampling = use_sampling  # 新增参数控制是否使用sampling方法

        self.norm = DyT(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.A = nn.Parameter(torch.zeros(d_state, d_state))

        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

        self.channel_attention = ChannelWiseAttention(self.expanded_dim,
                                                      sparsity_ratio=sparsity_ratio,
                                                      use_sampling=use_sampling)

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        residual = x

        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]
        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)

        x_sparse, topk_idx, k = self.channel_attention(x_proj_norm)

        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)

        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(k):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B.to(x.device)) * x_t
            h = torch.matmul(h, self.A.to(x.device).T) + Bx
            out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)

        return output + residual


class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, drop_rate=0.3, topk=True):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = DyT(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.A = nn.Parameter(torch.zeros(d_state, d_state))

        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=self.expanded_dim,
            bias=False
        )

        self.topk = topk

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n-1):
            A[i, i+1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        residual = x

        # Normalize and project
        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # Token selection: 改为概率采样
        center_idx = L // 2
        center = x_proj[:, center_idx:center_idx + 1, :]
        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)
        center_norm = F.normalize(center, p=2, dim=-1)
        sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)  # [B, L]
        probs = F.softmax(sim, dim=-1)  # 归一化为概率

        # 概率采样 所有batch使用相同分布
        k = max(1, int(L * self.sparsity_ratio))

        if self.topk:
            k = max(1, int(L * self.sparsity_ratio))

            _, topk_idx = torch.topk(sim, k=k, dim=-1)
        else:
            cdf = torch.cumsum(probs, dim=-1)  # [B, L]
            B = cdf.size(0)  # 获取batch size
            k_points = torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=x.device)  # [k]
            k_points = k_points.unsqueeze(0).expand(B, -1).contiguous()  # [B, k]
            topk_idx = torch.searchsorted(cdf, k_points)  # 不再有警告

            topk_idx = torch.unique(topk_idx, dim=1)  # 去重

        # 后续处理（保持不变）
        x_sparse = batched_index_select(x_proj, 1, topk_idx)
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)

        # SSM 处理（保持不变）
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []
        for t in range(topk_idx.shape[1]):  # 使用实际采样数 k'
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B.to(x.device)) * x_t
            h = torch.matmul(h, self.A.to(x.device).T) + Bx
            out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        # Scatter 回原始位置
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)
        return output + residual


class ChangeDetectionMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(config['model']['num_bands'], config['model']['hidden_dim'],
                      kernel_size=3, padding=1, groups=23),
            nn.GroupNorm(num_channels=config['model']['hidden_dim'], num_groups=23),
            nn.GELU(),
            nn.Conv2d(config['model']['hidden_dim'], config['model']['hidden_dim'],
                      kernel_size=3, padding=1, groups=23),
            nn.GroupNorm(num_channels=config['model']['hidden_dim'], num_groups=23),
            nn.GELU()
        )

        self.mamba_blocks = nn.Sequential(
            *[SparseDeformableMambaBlock(
                dim=config['model']['hidden_dim'],
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand'],
                sparsity_ratio=config['model']['sparsity_ratio']
            ) for _ in range(2)]
        )

        self.temporal_mamba = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(
                dim=13*13*24,
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand'],
                sparsity_ratio= 0.1 # config['model']['sparsity_ratio']
            ) for _ in range(2)]
        )

        self.spectral_mamba = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(
                dim=13*13*23,
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand'],
                sparsity_ratio= 0.3 # config['model']['sparsity_ratio']
            ) for _ in range(2)]
        )

        self.channel_mamba = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(
                dim=13*13,
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand'],
                sparsity_ratio= 0.1 # config['model']['sparsity_ratio']
            ) for _ in range(2)]
        )

        self.change_head = nn.Sequential(
            nn.Linear(2 * config['model']['hidden_dim'], config['model']['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['model']['hidden_dim'], 2)
        )

        self.class_head_2010 = nn.Sequential(
            nn.Linear(config['model']['hidden_dim'], config['model']['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['model']['hidden_dim'], config['model']['num_classes'])
        )

        self.class_head_2015 = nn.Sequential(
            nn.Linear(config['model']['hidden_dim'], config['model']['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['model']['hidden_dim'], config['model']['num_classes'])
        )

        self.input_proj = nn.Linear(13*13, 13*13)
        self.ffn = nn.Sequential(
            nn.Linear(13*13, 13*13),
            nn.GELU()
        )

        # 添加时间编码模块
        self.time_encoding = SeasonalTimeEncoding(
            num_phases=23,
            embed_dim=config['model'].get('time_embed_dim', 6)
        )

    def forward(self, x_2010, x_2015):
        x_2010 = self.time_encoding(x_2010)
        x_2015 = self.time_encoding(x_2015)

        feats_2010 = self.feature_extractor(x_2010)
        feats_2015 = self.feature_extractor(x_2015)

        B, C, H, W = feats_2010.shape
        feats_2010_t = feats_2010.reshape(B, C, H*W).reshape(B, 23, C//23, H*W).reshape(B, 23, C//23*H*W)
        feats_2015_t = feats_2015.reshape(B, C, H*W).reshape(B, 23, C//23, H*W).reshape(B, 23, C//23*H*W)

        feats_2010_t_out = self.temporal_mamba(feats_2010_t).reshape(B, 23, C//23, H*W).reshape(B, C, H*W)
        feats_2015_t_out = self.temporal_mamba(feats_2015_t).reshape(B, 23, C//23, H*W).reshape(B, C, H*W)

        feats_2010_s = feats_2010.reshape(B, 23, C//23, H * W).permute(0, 2, 1, 3).reshape(B, C//23, 23*H*W)
        feats_2015_s = feats_2015.reshape(B, 23, C//23, H * W).permute(0, 2, 1, 3).reshape(B, C//23, 23*H*W)

        feats_2010_s_out = self.spectral_mamba(feats_2010_s).reshape(B, C//23, 23, H*W).permute(0, 2, 1, 3).reshape(B, C, H*W)
        feats_2015_s_out = self.spectral_mamba(feats_2015_s).reshape(B, C//23, 23, H*W).permute(0, 2, 1, 3).reshape(B, C, H*W)

        feats_2010_spa_out = self.mamba_blocks(feats_2010.reshape(B, C, H*W).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H*W)
        feats_2015_spa_out = self.mamba_blocks(feats_2015.reshape(B, C, H*W).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H*W)

        # feature fusion
        feats_2010_out = feats_2010_t_out + feats_2010_s_out + feats_2010_spa_out
        feats_2015_out = feats_2015_t_out + feats_2015_s_out + feats_2015_spa_out

        feats_2010_out = feats_2010_out.permute(0, 2, 1)
        feats_2015_out = feats_2015_out.permute(0, 2, 1)

        pooled_2010 = feats_2010_out.mean(dim=1)
        pooled_2015 = feats_2015_out.mean(dim=1)

        change_feats = torch.cat([pooled_2010, pooled_2015], dim=1)
        change_logits = self.change_head(change_feats)

        class_logits_2010 = self.class_head_2010(pooled_2010)
        class_logits_2015 = self.class_head_2015(pooled_2015)

        return {
            'change': change_logits,
            'class_2010': class_logits_2010,
            'class_2015': class_logits_2015
        }

if __name__ == "__main__":
    config = {
        'model': {
            'num_bands': 138,
            'hidden_dim': 552,
            'num_classes': 11,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'sparsity_ratio': 0.3
        },
        'training': {
            'batch_size': 128,
            'batch_size_predict_on_whole_image': 4096,
            'learning_rate': 0.001,
            'weight_decay': 1e-6,
            'epochs': 50,
            'early_stopping_patience': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'to_predict_on_entire_image': True,
            'loss_weights': {
                'change': 1.0,  # Base weight
                'landcover': 0.5,  # LC loss multiplier
                'transition': 2  # Trans loss multiplier
            }

        }
    }
    dummy_input_2010 = torch.randn(512, 138, 13, 13).cuda()
    dummy_input_2015 = torch.randn(512, 138, 13, 13).cuda()
    model = ChangeDetectionMamba(config).cuda()
    prediction = model(dummy_input_2010, dummy_input_2015)
    print(prediction['change'].shape, prediction['class_2010'].shape, prediction['class_2015'].shape)