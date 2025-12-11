import torch
from torch import nn
import torch.nn.functional as F

# ==================== MODEL ARCHITECTURE ====================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms_x + self.eps))


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

        self.mamba_blocks = nn.Sequential(
            *[SparseDeformableMambaBlock(
                dim=config['model']['hidden_dim'],
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand'],
                sparsity_ratio=config['model']['sparsity_ratio']
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

        self.conv_down = nn.Sequential(
            nn.Conv2d(config['model']['hidden_dim'], config['model']['hidden_dim'], stride=2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 最终降为1x1
        )

    def forward(self, x_2010, x_2015):
        feats_2010 = self.feature_extractor(x_2010)
        feats_2015 = self.feature_extractor(x_2015)

        B, C, H, W = feats_2010.shape
        feats_2010 = feats_2010.permute(0, 2, 3, 1).reshape(B, H * W, C)  # channel 552
        feats_2015 = feats_2015.permute(0, 2, 3, 1).reshape(B, H * W, C)

        feats_2010 = self.mamba_blocks(feats_2010).permute(0, 2, 1).reshape(B, C, H, W)
        feats_2015 = self.mamba_blocks(feats_2015).permute(0, 2, 1).reshape(B, C, H, W)

        pooled_2010 = torch.squeeze(self.conv_down(feats_2010))
        pooled_2015 = torch.squeeze(self.conv_down(feats_2015))

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