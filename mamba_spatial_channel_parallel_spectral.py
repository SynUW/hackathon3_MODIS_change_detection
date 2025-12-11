import torch
from torch import nn
import torch.nn.functional as F

# ==================== MODEL ARCHITECTURE ====================
class GroupedTimeEncoding(nn.Module):
    def __init__(self, total_channels=138, num_phases=23, embed_dim=64):
        super().__init__()
        self.num_phases = num_phases
        self.embed_dim = embed_dim
        # 每个时相分配一个编码（共23个）
        self.time_embed = nn.Parameter(torch.randn(1, num_phases, embed_dim))

    def forward(self, x):
        # x: [B, 138, H, W]
        B, C, H, W = x.shape
        # 扩展时间编码到所有波段 [1, 138, embed_dim]
        expanded_embed = self.time_embed.repeat_interleave(6, dim=1)  # 23*6=138
        # 将编码拼接到特征（或相加）
        x = x.reshape(B, C, H * W) + expanded_embed.permute(0, 2, 1)  # [B, embed_dim, 138]
        return x.reshape(B, C, H, W)

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
    def __init__(self, d_model, num_heads=6, sparsity_ratio=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sparsity_ratio = sparsity_ratio

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

        # 计算通道重要性 - 修改后的部分
        channel_importance = out.norm(p=2, dim=-1)  # [B, C]

        # 选择top-k通道
        k_channels = max(1, int(C * self.sparsity_ratio))
        _, topk_indices = torch.topk(channel_importance, k=k_channels, dim=-1)  # 使用dim=-1

        # 收集选中的特征
        sparse_feat = torch.stack([
            out[i, topk_indices[i]] for i in range(B)
        ], dim=0)

        return sparse_feat, topk_indices, k_channels

class SparseDeformableChannelMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, drop_rate=0.3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = 336  # dim * expand
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
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

        self.channel_attention = ChannelWiseAttention(self.expanded_dim, sparsity_ratio=sparsity_ratio)

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        # L = H * W
        residual = x

        # Flatten spatial dimensions
        x_flat = x  # x.reshape(B, L, C)

        # Normalize and project
        x_norm = self.norm(x_flat)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)  # [B, L, D]

        x_sparse, topk_idx, k = self.channel_attention(x_proj_norm)

        # Conv processing
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)

        # SSM processing
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

        # Combine with residual
        # x_processed = x_processed + batched_index_select(residual.reshape(B, L, C), 1, topk_idx)

        # Scatter back to original positions
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)

        # return output.reshape(B, H, W, C) + x
        return output + residual


class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

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

    def batched_index_select(self, input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, x):
        B, L, C = x.shape
        residual = x

        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)

        center_idx = L // 2
        center = x_proj[:, center_idx:center_idx + 1, :]

        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)
        center_norm = F.normalize(center, p=2, dim=-1)
        sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)
        sim = torch.softmax(sim, dim=-1)

        k = max(1, int(L * self.sparsity_ratio))
        _, topk_idx = torch.topk(sim, k=k, dim=-1)
        x_sparse = self.batched_index_select(x_proj, 1, topk_idx)

        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :k]
        x_conv = x_conv.transpose(1, 2)

        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(k):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B) * x_t
            h = torch.matmul(h, self.A.T) + Bx
            out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

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

        self.feature_extractor_spectral = nn.Sequential(
            nn.Conv2d(config['model']['hidden_dim'], config['model']['hidden_dim'],
                      kernel_size=3, padding=1, groups=24),
            nn.GroupNorm(num_channels=config['model']['hidden_dim'], num_groups=24),
            nn.GELU(),
            nn.Conv2d(config['model']['hidden_dim'], config['model']['hidden_dim'],
                      kernel_size=3, padding=1, groups=24),
            nn.GroupNorm(num_channels=config['model']['hidden_dim'], num_groups=24),
            nn.GELU()
        )

        self.mamba_blocks = nn.Sequential(
            *[SparseDeformableMambaBlock(
                dim=config['model']['hidden_dim'],
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand'],
                sparsity_ratio= 0.1  # config['model']['sparsity_ratio']
            ) for _ in range(2)]
        )

        self.channel_mamba_blocks = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(
                dim=13*13,
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand'],
                sparsity_ratio= 0.1 # config['model']['sparsity_ratio']
            ) for _ in range(2)]
        )

        self.spectral_mamba_block = nn.Sequential(
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

    def forward(self, x_2010, x_2015):

        feats_2010 = self.feature_extractor(x_2010)
        feats_2015 = self.feature_extractor(x_2015)
        B, C, H, W = feats_2010.shape
        feats_2010 = feats_2010.reshape(B, C, H*W)
        feats_2015 = feats_2015.reshape(B, C, H*W)

        feats_2010_s = feats_2010.reshape(B, 23, C // 23, H * W).permute(0, 2, 1, 3).reshape(B, C, H, W)
        feats_2015_s = feats_2015.reshape(B, 23, C // 23, H * W).permute(0, 2, 1, 3).reshape(B, C, H, W)

        feats_2010_s = self.feature_extractor_spectral(feats_2010_s).reshape(B, C, H*W)
        feats_2015_s = self.feature_extractor_spectral(feats_2015_s).reshape(B, C, H*W)

        feats_2010_c_out = self.channel_mamba_blocks(feats_2010)
        feats_2015_c_out = self.channel_mamba_blocks(feats_2015)

        feats_2010_s_out = self.spectral_mamba_block(feats_2010_s)
        feats_2015_s_out = self.spectral_mamba_block(feats_2015_s)

        # ffn放这里是sota so far
        feats_2010 = self.ffn(feats_2010)+feats_2010
        feats_2015 = self.ffn(feats_2015)+feats_2015

        feats_2010 = self.mamba_blocks(feats_2010.permute(0, 2, 1))
        feats_2015 = self.mamba_blocks(feats_2015.permute(0, 2, 1))

        pooled_2010 = (feats_2010+(feats_2010_c_out+feats_2010_s_out).permute(0, 2, 1)).mean(dim=1)
        pooled_2015 = (feats_2015+(feats_2015_c_out+feats_2015_s_out).permute(0, 2, 1)).mean(dim=1)

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