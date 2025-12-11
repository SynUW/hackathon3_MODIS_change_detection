import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
import h5py
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
from osgeo import gdal
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from torch.nn import AvgPool2d, MaxPool2d
from einops import rearrange, repeat
from torch import einsum
import pdb


# ==================== RANDOM SEED ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)
# ==================== CONFIGURATION ====================
config = {
    'dataset': {
        'hdf5_path': '/mnt/storage/benchmark_datasets/MODIS/sask/patches_13_by_13_change_detection/change_detection_samples.h5',
        'patch_size': 13,
        'result_dir': './change_detection_results',
        'image_2010_file': '/mnt/storage/benchmark_datasets/MODIS/sask/MOD13Q1_sask_2010.tiff',
        'image_2015_file': '/mnt/storage/benchmark_datasets/MODIS/sask/MOD13Q1_sask_2015.tiff',
        'gt_change_file': '/mnt/storage/benchmark_datasets/MODIS/sask/sask_binary_change_2010_2015_gt.tif',
        'gt_2010LC_file': '/mnt/storage/benchmark_datasets/MODIS/sask/sask_2010_gt_new.tif',
        'gt_2015LC_file': '/mnt/storage/benchmark_datasets/MODIS/sask/sask_2015_gt.tif'
    },
    'model': {
        'num_bands': 138,
        'hidden_dim': 138 * 2,
        'num_classes': 11,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'sparsity_ratio': 0.3
    },
    'training': {
        'batch_size': 256,
        'batch_size_predict_on_whole_image': 4096,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'epochs': 100,
        'early_stopping_patience': 100,
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
        'to_predict_on_entire_image': True,
        'loss_weights': {
            'change': 1.0,  # Base weight
            'landcover': 0.5,  # LC loss multiplier
            'transition': 2  # Trans loss multiplier
        }

    }
}

# Class names for change detection (binary)
CLASS_NAMES = ["No Change", "Change"]
CLASS_NAMES_LC = [
    "1-Temp-needleleaf",
    "2-Taiga-needleleaf",
    "5-Broadleaf-deciduous",
    "6-Mixed-forest",
    "8-Shrubland",
    "10-Polar-shrubland",
    "14-Wetland",
    "15-Cropland",
    "16-Barren",
    "17-Urban",
    "18-Water"
]

TRANSITION_MATRIX = np.array([
    [0.000000, 0.028626, 0.018044, 0.004525, 0.197418, 0.596563, 0.094067, 0.000538, 0.002304, 0.000513, 0.057403],
    [0.051739, 0.000000, 0.000041, 0.000000, 0.028935, 0.864481, 0.009175, 0.000000, 0.029743, 0.000000, 0.015886],
    [0.254227, 0.000000, 0.000000, 0.043791, 0.235667, 0.038309, 0.118164, 0.215918, 0.000462, 0.009974, 0.083487],
    [0.352306, 0.001736, 0.204708, 0.000000, 0.232105, 0.022574, 0.091646, 0.046305, 0.000193, 0.005981, 0.042446],
    [0.241213, 0.002005, 0.019377, 0.007484, 0.000000, 0.575170, 0.076440, 0.027128, 0.001203, 0.004143, 0.045837],
    [0.299910, 0.019994, 0.010749, 0.001503, 0.409426, 0.000000, 0.038635, 0.123948, 0.011575, 0.005863, 0.078397],
    [0.266878, 0.006822, 0.115379, 0.002123, 0.111671, 0.272341, 0.000000, 0.111360, 0.001642, 0.006143, 0.105642],
    [0.013661, 0.000000, 0.233938, 0.005123, 0.087513, 0.196158, 0.078549, 0.000000, 0.001067, 0.094557, 0.289434],
    [0.121093, 0.001503, 0.000658, 0.000027, 0.175008, 0.665104, 0.010740, 0.000320, 0.000000, 0.000137, 0.025409],
    [0.042836, 0.000000, 0.082718, 0.014771, 0.064993, 0.081241, 0.053176, 0.522895, 0.004431, 0.000000, 0.132939],
    [0.186011, 0.009663, 0.042385, 0.003843, 0.132206, 0.421983, 0.061162, 0.110794, 0.002416, 0.029538, 0.000000],
], dtype=np.float32)


# ==================== UTILITY FUNCTIONS ====================
def load_band(band_num, image_path):
    """Load single band with error handling"""
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(band_num)
    arr = band.ReadAsArray()
    ds = None
    return arr


def parallel_load_bands(image_path, n_workers=4):
    """Parallel band loading using multiprocessing"""
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    num_bands = ds.RasterCount

    with Pool(n_workers) as pool:
        bands = list(tqdm(
            pool.imap(partial(load_band, image_path=image_path), range(1, num_bands + 1)),
            total=num_bands,
            desc="Loading bands"
        ))

    img_data = np.moveaxis(np.stack(bands), 0, -1)
    return img_data


def save_geotiff_with_metadata(pred_array, gt_file_path, output_path):
    """Save prediction array as GeoTIFF with metadata from reference file, including colormap"""
    gt_ds = gdal.Open(gt_file_path)
    driver = gdal.GetDriverByName('GTiff')

    creation_options = [
        'COMPRESS=DEFLATE',  # Good compression ratio
        'PREDICTOR=2',  # Better compression for integer data
        'ZLEVEL=9',  # Maximum compression level
        'TILED=YES',  # Better compression with tiling
        'BLOCKXSIZE=256',  # Tile size
        'BLOCKYSIZE=256',  # Tile size
        'NUM_THREADS=ALL_CPUS'  # Use all CPUs for compression
    ]

    # Create output file with same dimensions and data type
    out_ds = driver.Create(
        output_path,
        gt_ds.RasterXSize,
        gt_ds.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=creation_options
    )

    # Copy geospatial metadata
    out_ds.SetGeoTransform(gt_ds.GetGeoTransform())
    out_ds.SetProjection(gt_ds.GetProjection())

    # Get the output band and write data
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(pred_array)

    # Copy color table from ground truth if it exists
    gt_band = gt_ds.GetRasterBand(1)
    color_table = gt_band.GetColorTable()
    if color_table is not None:
        out_band.SetColorTable(color_table)

    # Set NoData value if present in ground truth
    # nodata_value = gt_band.GetNoDataValue()
    # if nodata_value is not None:
    #    out_band.SetNoDataValue(nodata_value)

    # Flush cache and close datasets
    out_band.FlushCache()
    out_ds.FlushCache()
    out_ds = None
    gt_ds = None


def calculate_metrics(preds, targets, class_names):
    """Calculate comprehensive metrics including class-wise accuracy"""
    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets

    accuracy = np.mean(preds == targets)
    cm = confusion_matrix(targets, preds)

    # Generate both dict and str versions of classification report
    report_dict = classification_report(
        targets, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        targets, preds,
        target_names=class_names,
        output_dict=False,
        zero_division=0
    )

    kappa = cohen_kappa_score(targets, preds)

    # Calculate class-wise accuracy
    class_accuracies = []
    for i in range(cm.shape[0]):
        if cm[i, :].sum() > 0:
            class_accuracies.append(cm[i, i] / cm[i, :].sum())
        else:
            class_accuracies.append(0.0)

    avg_class_accuracy = np.mean(class_accuracies)

    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'average_class_accuracy': avg_class_accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'report': report_dict,  # Dictionary version
        'report_str': report_str  # String version
    }


def save_metrics_report(metrics, class_names, filename_prefix, result_dir):
    """Save comprehensive metrics report to files"""
    os.makedirs(result_dir, exist_ok=True)

    # Save confusion matrix
    cm_df = pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(result_dir, f'{filename_prefix}_confusion_matrix.csv'))

    # Save normalized confusion matrix
    cm_normalized = metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]
    cm_normalized_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    cm_normalized_df.to_csv(os.path.join(result_dir, f'{filename_prefix}_confusion_matrix_normalized.csv'))

    # Save visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized_df, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"Normalized Confusion Matrix ({filename_prefix.replace('_', ' ')})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{filename_prefix}_confusion_matrix.png'))
    plt.close()

    # Save metrics summary
    with open(os.path.join(result_dir, f'{filename_prefix}_metrics.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"Average Class Accuracy: {metrics['average_class_accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['kappa']:.4f}\n\n")

        f.write("Class-wise Accuracies:\n")
        for i, acc in enumerate(metrics['class_accuracies']):
            f.write(f"{class_names[i]}: {acc:.4f}\n")

        # Handle classification report
        if 'report' in metrics:
            f.write("\nClassification Report:\n")
            if isinstance(metrics['report'], dict):
                # Convert dict report to string
                report_df = pd.DataFrame(metrics['report']).transpose()
                f.write(report_df.to_string())
            elif isinstance(metrics['report'], str):
                f.write(metrics['report'])

        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))


# ==================== DATASET CLASSES ====================
class ChangeDetectionDataset(Dataset):
    def __init__(self, hdf5_path, set_name='train', transform=None, normalize=False):
        self.hdf5_path = hdf5_path
        self.set_name = set_name
        self.transform = transform
        self.normalize = normalize

        with h5py.File(hdf5_path, 'r') as f:
            self.group = f[set_name]
            self.num_samples = len(self.group['patches'])

            if normalize and 'mean_2010' in self.group.attrs:
                self.mean_2010 = torch.from_numpy(self.group.attrs['mean_2010'])
                self.std_2010 = torch.from_numpy(self.group.attrs['std_2010'])
                self.mean_2015 = torch.from_numpy(self.group.attrs['mean_2015'])
                self.std_2015 = torch.from_numpy(self.group.attrs['std_2015'])
            else:
                self.mean_2010 = self.std_2010 = None
                self.mean_2015 = self.std_2015 = None

            self.labels_2010 = self.group['classes_2010'][:]
            self.labels_2015 = self.group['classes_2015'][:]
            self.change_status = self.group['change_status'][:]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            patch = f[self.set_name]['patches'][idx]
            num_channels = patch.shape[-1] // 2
            img_2010 = patch[..., :num_channels]
            img_2015 = patch[..., num_channels:]

            img_2010 = torch.from_numpy(img_2010).float()
            img_2015 = torch.from_numpy(img_2015).float()

            if self.normalize and self.mean_2010 is not None:
                img_2010 = (img_2010 - self.mean_2010) / self.std_2010
                img_2015 = (img_2015 - self.mean_2015) / self.std_2015

            img_2010 = img_2010.permute(2, 0, 1)
            img_2015 = img_2015.permute(2, 0, 1)

            label_2010 = torch.tensor(self.labels_2010[idx], dtype=torch.long)
            label_2015 = torch.tensor(self.labels_2015[idx], dtype=torch.long)
            change_label = torch.tensor(self.change_status[idx], dtype=torch.long)

            if self.transform:
                stacked = torch.cat([img_2010.unsqueeze(0), img_2015.unsqueeze(0)], dim=0)
                stacked = self.transform(stacked)
                img_2010, img_2015 = stacked[0], stacked[1]

            return {
                'image_2010': img_2010,
                'image_2015': img_2015,
                'label_2010': label_2010,
                'label_2015': label_2015,
                'change_label': change_label
            }


class FullImageChangeDetectionDataset(Dataset):
    def __init__(self, image_2010_path, image_2015_path,
                 gt_change_path, gt_2010LC_path, gt_2015LC_path,
                 patch_size=13, n_workers=8):
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

        print("Loading 2010 image...")
        self.image_2010 = parallel_load_bands(image_2010_path, n_workers)
        print("Loading 2015 image...")
        self.image_2015 = parallel_load_bands(image_2015_path, n_workers)

        # Load ground truth data
        print("Loading ground truth data...")
        self.gt_change = gdal.Open(gt_change_path).ReadAsArray()
        self.gt_2010LC = gdal.Open(gt_2010LC_path).ReadAsArray()
        self.gt_2015LC = gdal.Open(gt_2015LC_path).ReadAsArray()

        # Define valid classes for land cover
        self.valid_lc_classes = [1, 2, 5, 6, 8, 10, 14, 15, 16, 17, 18]
        self.class_mapping = {cls: idx for idx, cls in enumerate(self.valid_lc_classes)}

        # Create mask for valid pixels (change: 1 or 2, land cover: valid classes)
        self.valid_mask = (
                (self.gt_change > 0) &
                np.isin(self.gt_2010LC, self.valid_lc_classes) &
                np.isin(self.gt_2015LC, self.valid_lc_classes)
        )

        # Get coordinates of valid pixels
        self.valid_locations = list(zip(*np.where(self.valid_mask)))

        # Pad images
        self.padded_2010 = np.pad(
            self.image_2010,
            ((self.half_patch, self.half_patch),
             (self.half_patch, self.half_patch),
             (0, 0)),
            mode='constant'
        )
        self.padded_2015 = np.pad(
            self.image_2015,
            ((self.half_patch, self.half_patch),
             (self.half_patch, self.half_patch),
             (0, 0)),
            mode='constant'
        )

    def __len__(self):
        return len(self.valid_locations)

    def __getitem__(self, idx):
        i, j = self.valid_locations[idx]

        # Extract patches
        patch_2010 = self.padded_2010[
                     i:i + 2 * self.half_patch + 1,
                     j:j + 2 * self.half_patch + 1,
                     :
                     ]
        patch_2015 = self.padded_2015[
                     i:i + 2 * self.half_patch + 1,
                     j:j + 2 * self.half_patch + 1,
                     :
                     ]

        # Get corresponding labels
        change_label = self.gt_change[i, j]
        lc2010_label = self.gt_2010LC[i, j]
        lc2015_label = self.gt_2015LC[i, j]

        # Convert change label: 1→0 (no change), 2→1 (change)
        change_label = 0 if change_label == 1 else 1

        # Convert land cover labels to contiguous indices
        lc2010_label = self.class_mapping[lc2010_label]
        lc2015_label = self.class_mapping[lc2015_label]

        # Convert to tensors
        patch_2010 = torch.tensor(patch_2010, dtype=torch.float32).permute(2, 0, 1)
        patch_2015 = torch.tensor(patch_2015, dtype=torch.float32).permute(2, 0, 1)

        return {
            'image_2010': patch_2010,
            'image_2015': patch_2015,
            'change_label': torch.tensor(change_label, dtype=torch.long),
            'lc2010_label': torch.tensor(lc2010_label, dtype=torch.long),
            'lc2015_label': torch.tensor(lc2015_label, dtype=torch.long),
            'location': (i, j)
        }


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


#
# class SparseDeformableMambaBlock(nn.Module):
#     def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3):
#         super().__init__()
#         self.dim = dim
#         self.d_state = d_state
#         self.expand = expand
#         self.expanded_dim = dim * expand
#         self.sparsity_ratio = sparsity_ratio
#
#         self.norm = RMSNorm(dim)
#         self.proj_in = nn.Linear(dim, self.expanded_dim)
#         self.proj_out = nn.Linear(self.expanded_dim, dim)
#
#         self.A = nn.Parameter(torch.zeros(d_state, d_state))
#         self.B = nn.Parameter(torch.zeros(1, 1, d_state))
#         self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
#
#         self.conv = nn.Conv1d(
#             in_channels=self.expanded_dim,
#             out_channels=self.expanded_dim,
#             kernel_size=d_conv,
#             padding=d_conv-1,
#             groups=self.expanded_dim,
#             bias=False
#         )
#
#     def batched_index_select(self, input, dim, index):
#         for ii in range(1, len(input.shape)):
#             if ii != dim:
#                 index = index.unsqueeze(ii)
#         expanse = list(input.shape)
#         expanse[0] = -1
#         expanse[dim] = -1
#         index = index.expand(expanse)
#         return torch.gather(input, dim, index)
#
#     def forward(self, x):
#         B, L, C = x.shape
#         residual = x
#
#         x_norm = self.norm(x)
#         x_proj = self.proj_in(x_norm)
#
#         center_idx = L // 2
#         center = x_proj[:, center_idx:center_idx+1, :]
#
#         x_proj_norm = F.normalize(x_proj, p=2, dim=-1)
#         center_norm = F.normalize(center, p=2, dim=-1)
#         sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)
#         sim = torch.softmax(sim, dim=-1)
#
#         k = max(1, int(L * self.sparsity_ratio))
#         _, topk_idx = torch.topk(sim, k=k, dim=-1)
#         x_sparse = self.batched_index_select(x_proj, 1, topk_idx)
#
#         x_conv = x_sparse.transpose(1, 2)
#         x_conv = self.conv(x_conv)[..., :k]
#         x_conv = x_conv.transpose(1, 2)
#
#         h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
#         outputs = []
#
#         for t in range(k):
#             x_t = x_conv[:, t].unsqueeze(-1)
#             Bx = torch.sigmoid(self.B) * x_t
#             h = torch.matmul(h, self.A.T) + Bx
#             out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(-1)
#             outputs.append(out_t)
#
#         x_processed = torch.stack(outputs, dim=1)
#         x_processed = self.proj_out(x_processed)
#
#         output = torch.zeros(B, L, C, device=x.device)
#         output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)
#
#         return output + residual

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # plt.clf()
        # plt.imshow(attn.detach().cpu().numpy().mean(axis=0), cmap="jet")
        # plt.savefig("Q.png")
        # plt.clf()
        k1 = max(1, int(N * 0.3))
        row_variance = attn.mean(dim=1, keepdim=True)
        _, idx2 = torch.topk(row_variance, k=k1, dim=-1)
        expanded_indices = idx2.squeeze(1).unsqueeze(1).expand(-1, attn.shape[2], -1)

        row_variance1 = attn.mean(dim=-1, keepdim=True)
        _, idx = torch.topk(row_variance1, k=k1, dim=1)
        expanded_indices1 = idx.squeeze(-1).unsqueeze(-1).expand(-1, -1, attn.shape[2])

        # [1024, 6, 23]
        # src = torch.ones_like(expanded_indices)
        mask1 = torch.zeros_like(attn)
        # 3. 使用 scatter_ 填充 mask
        # _, topk_idx1 = torch.topk(attn, k=k1, dim=2)

        # mask = torch.zeros_like(attn)
        mask1.scatter_(-1, expanded_indices, 1.0)
        mask2 = torch.zeros_like(attn)
        mask2.scatter_(1, expanded_indices1, 1.0)
        # row_variance = attn.mean(dim=-1, keepdim=True)
        # _, idx2 = torch.topk(row_variance, k=k1, dim=1)
        # expanded_indices = idx2.squeeze(-1).unsqueeze(-1).expand(-1, -1, attn.shape[2])  # [1024, 6, 23]
        # #src = torch.ones_like(expanded_indices)
        # mask1 = torch.zeros_like(attn)
        # # 3. 使用 scatter_ 填充 mask
        # #_, topk_idx1 = torch.topk(attn, k=k1, dim=2)
        #
        # #mask = torch.zeros_like(attn)
        # mask1.scatter_(1, expanded_indices, 1.0)
        M = mask1 + mask2
        M[M == 2] = 1
        assert len(torch.unique(mask1)) == 2
        # loc = torch.where(M == 1)

        # 将稠密矩阵转为稀疏格式（COO/CSR）
        # sparse_attn = attn.to_sparse()  # 非零元素会被压缩存储
        # output = torch.sparse.mm(attn, v)  # 实际计算量 ≈ O(nnz * d_model)

        sparse_attn = attn * M
        reshape_attn = sparse_attn.reshape(B, -1, N, N)
        A = reshape_attn.mean(dim=1)
        A_mean = A.mean(dim=1)
        _, loc = torch.topk(A_mean, k=k1, dim=-1)
        out = einsum('b i j, b j d -> b i d', sparse_attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        feature = self.to_out(out).permute(0, 2, 1)

        return feature, loc


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, drop_rate=0.3, spa=True, spe=True,
                 temp=False):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = DyT(dim)

        self.A = nn.Parameter(torch.zeros(d_state, d_state))

        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.pool = nn.MaxPool1d(kernel_size=6, stride=6)  # 138 / 6 = 23

        # self.season = SeasonalModel()
        if spa == True and spe == False and temp == False:
            self.conv = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                groups=dim,
                bias=False
            )
            self.proj_in = nn.Linear(dim, dim)
            self.proj_out = nn.Linear(dim, dim)
            self.C = nn.Parameter(torch.zeros(dim, d_state))
        if spe == True and spa == False and temp == False:
            self.conv = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                groups=dim,
                bias=False
            )
            # if block_idx == 0:
            self.down = nn.Conv1d(
                in_channels=138 * 2,
                out_channels=23,
                kernel_size=1,  # 1x1卷积
                groups=23,  # 分23组，每组独立处理
                bias=False  # 可选偏置
            )
            self.attn = Attention(query_dim=dim, context_dim=dim,
                                  heads=16, dim_head=64, dropout=0.)
            self.proj_in = nn.Linear(dim, dim)
            self.proj_out = nn.Linear(dim, dim)
            self.C = nn.Parameter(torch.zeros(dim, d_state))
        if spe == False and spa == False and temp == True:
            self.conv = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                groups=dim,
                bias=False
            )

            self.attn = Attention(query_dim=dim, context_dim=dim,
                                  heads=16, dim_head=64, dropout=0.)
            self.proj_in = nn.Linear(dim, dim)
            self.proj_out = nn.Linear(dim, dim)
            self.C = nn.Parameter(torch.zeros(dim, d_state))

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x, spa=True, spe=True, temp=False):
        if spa is True and spe is False and temp is False:
            B, L, C = x.shape
            # L = H * W
            residual = x

            # Flatten spatial dimensions
            x_flat = x  # x.reshape(B, L, C)

            # Normalize and project
            x_norm = self.norm(x_flat)
            x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

            # Token selection

            center_idx = L // 2
            center = x_proj[:, center_idx:center_idx + 1, :]

            x_proj_norm = F.normalize(x_proj, p=2, dim=-1)  # [B, L, D]
            center_norm = F.normalize(center, p=2, dim=-1)  # [B, 1, D]

            sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)

            # im = torch.matmul(x_proj, center.transpose(-1, -2)).squeeze(-1)  # [B, L]
            sim = torch.softmax(sim, dim=-1)  # Normalized probabilities

            k = max(1, int(L * self.sparsity_ratio))

            _, topk_idx = torch.topk(sim, k=k, dim=-1)

            x_sparse = batched_index_select(x_proj, 1, topk_idx)  # [B, k, expanded_dim]

            # Conv processing
            x_conv = x_sparse.transpose(1, 2)
            x_conv = self.conv(x_conv)[..., :k]
            x_conv = x_conv.transpose(1, 2)

            # SSM processing
            h = torch.zeros(B, self.dim, self.d_state, device=x.device)
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
            return output + residual, sim

        if spa is False and spe is True and temp is False:

            B, L, C = x.shape
            # L = H * W
            residual = x

            # Flatten spatial dimensions
            x_flat = x  # x.reshape(B, L, C)

            # Normalize and project
            x_norm = self.norm(x_flat)
            x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]
            # if block_idx == 0:
            x_proj = self.down(x_proj)
            x_sparse, loc = self.attn(x_proj)

            x_sparse = x_sparse.permute(0, 2, 1)  # [1024, 23, 169]

            # Token selection

            x_sparse = batched_index_select(x_sparse, 1, loc)  # [B, k, expanded_dim]
            # Conv processing
            x_conv = x_sparse.transpose(1, 2)
            x_conv = self.conv(x_conv)[..., :L]
            x_conv = x_conv.transpose(1, 2)
            B, num, _ = x_conv.shape
            # SSM processing
            h = torch.zeros(B, self.dim, self.d_state, device=x.device)
            outputs = []

            for t in range(num):
                x_t = x_conv[:, t].unsqueeze(-1)
                Bx = torch.sigmoid(self.B.to(x.device)) * x_t
                h = torch.matmul(h, self.A.to(x.device).T) + Bx
                out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
                outputs.append(out_t)

            x_processed = torch.stack(outputs, dim=1)
            x_processed = self.proj_out(x_processed)

            output = torch.zeros(B, L, C, device=x.device)
            output.scatter_(1, loc.unsqueeze(-1).expand(-1, -1, C), x_processed)

            # return output.reshape(B, H, W, C) + x
            return output + residual, loc

        if spa is False and spe is False and temp is True:

            B, L, C = x.shape
            # L = H * W
            residual = x

            # Flatten spatial dimensions
            x_flat = x  # x.reshape(B, L, C)

            # Normalize and project
            x_norm = self.norm(x_flat)
            x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]
            # if block_idx == 0:
            x_sparse, loc = self.attn(x_proj)

            x_sparse = x_sparse.permute(0, 2, 1)  # [1024, 23, 169]

            # #
            x_sparse = batched_index_select(x_sparse, 1, loc)  # [B, k, expanded_dim]
            # Conv processing
            x_conv = x_sparse.transpose(1, 2)
            x_conv = self.conv(x_conv)[..., :L]
            x_conv = x_conv.transpose(1, 2)
            B, num, _ = x_conv.shape
            # SSM processing
            h = torch.zeros(B, self.dim, self.d_state, device=x.device)
            outputs = []

            for t in range(num):
                x_t = x_conv[:, t].unsqueeze(-1)
                Bx = torch.sigmoid(self.B.to(x.device)) * x_t
                h = torch.matmul(h, self.A.to(x.device).T) + Bx
                out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
                outputs.append(out_t)

            x_processed = torch.stack(outputs, dim=1)
            x_processed = self.proj_out(x_processed)

            # Scatter back to original positions
            output = torch.zeros(B, L, C, device=x.device)
            output.scatter_(1, loc.unsqueeze(-1).expand(-1, -1, C), x_processed)

            return output + residual, loc


class SimpleChangeDetectionHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128):
        super().__init__()

        # Convolutional layers to process concatenated features
        self.conv1 = nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # self.conv3 = nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # self.conv1 = nn.Linear(in_channels*2, hidden_channels)
        # self.conv2 = nn.Linear(hidden_channels, hidden_channels)

        # Final 1x1 conv to get binary change mask (logits)
        self.change_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2 * hidden_channels, 2)
        )
        self.act = nn.GELU()
        # self.change_pred = nn.Sequential(
        #     nn.Linear(2*hidden_channels, hidden_channels),
        #     nn.GELU(),
        #     nn.Linear(hidden_channels, 2)
        # )
        # self.change_pred = nn.Conv2d(2*hidden_channels, 2, kernel_size=1)

        # Optional: BatchNorm or Dropout for regularization
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

    def forward(self, feat1, feat2, feat3, feat4):
        """
        Args:
            feat1 (Tensor): Features from Image 1 [B, C, H, W]
            feat2 (Tensor): Features from Image 2 [B, C, H, W]
        Returns:
            change_logits (Tensor): Predicted change mask [B, 1, H, W]
        """
        # Concatenate features along channel dimension
        x = torch.cat([feat1, feat2], dim=1)  # [B, 2C, H, W]
        xx = torch.cat([feat3, feat4], dim=1)
        # Process features
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        xx = self.act(self.bn1(self.conv1(xx)))
        xx = self.act(self.bn2(self.conv2(xx)))

        feature = torch.cat((x, xx), dim=1)
        # Predict change logits (use `torch.sigmoid` later for probabilities)
        change_logits = self.change_pred(feature)  # [B, 1, H, W]

        return change_logits


class SpatialSpectralMambaBlock(nn.Module):
    def __init__(self, dim: int, patch_size: int, num_heads=8, block_idx=0):
        super().__init__()
        head_dim = dim // num_heads
        self.block_idx = block_idx
        self.spatial_mamba = SparseDeformableMambaBlock(dim=dim, spa=True, spe=False, temp=False)

        self.temporal_mamba = SparseDeformableMambaBlock(dim=int(patch_size ** 2), spa=False, spe=True, temp=False)

        self.spectral_mamba = SparseDeformableMambaBlock(dim=23, spa=False, spe=False, temp=True)

        self.norm1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.GELU())

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.stem = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=23, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=patch_size, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape  # = 138*2
        residual = x
        num_spe = 6
        T = 23
        x = self.norm1(x)
        x_spatial = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_spatial, sim = self.spatial_mamba(x_spatial, spa=True, spe=False, temp=False)
        x_spatial = x_spatial.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_spatial1 = self.stem(x_spatial)

        x_spectral, idx = self.maxpool(x_spatial1)  # [B, C, 1, 1]
        x_spectral = x_spectral.reshape(B, -1, T, 1, 1).reshape(B, -1, T)
        x_spectral, _ = self.spectral_mamba(x_spectral, spa=False, spe=False, temp=True)
        x_spectral = x_spectral.reshape(B, -1, T, 1, 1).reshape(B, -1, 1, 1)
        x_spectral = self.unpool(x_spectral, idx, output_size=x_spatial1.shape) + x_spatial1
        x_spectral1 = self.stem(x_spectral)

        x_temporal = x_spectral1.reshape(B, C, H * W)
        x_temporal, _ = self.temporal_mamba(x_temporal, spa=False, spe=True, temp=False)
        x_temporal = x_temporal.reshape(B, C, H, W)
        x = self.conv(x_temporal)

        return x + residual, x_spatial, x_temporal, x_spectral


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class ChangeDetectionMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(config['model']['num_bands'], config['model']['hidden_dim'],
                      kernel_size=3, padding=1, groups=23),
            nn.BatchNorm2d(config['model']['hidden_dim']),
            nn.GELU(),
        )

        self.feature_extractor1 = nn.Sequential(
            nn.Conv2d(config['model']['num_bands'], config['model']['hidden_dim'],
                      kernel_size=3, padding=1, groups=23),
            # nn.GroupNorm(23, config['model']['hidden_dim']),
            nn.BatchNorm2d(config['model']['hidden_dim']),
            nn.GELU(),
        )

        self.spectral_blocks = nn.Sequential(
            *[SpatialSpectralMambaBlock(dim=config['model']['hidden_dim'], patch_size=13, block_idx=i) for i in
              range(1)]
        )

        self.spectral_blocks1 = nn.Sequential(
            *[SpatialSpectralMambaBlock(dim=config['model']['hidden_dim'], patch_size=13, block_idx=i) for i in
              range(1)]
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

        self.proj_head2010 = nn.Sequential(
            nn.Linear(config['model']['hidden_dim'], 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.proj_head2015 = nn.Sequential(
            nn.Linear(config['model']['hidden_dim'], 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.change_head = nn.Sequential(
            nn.Linear(3 * config['model']['hidden_dim'], config['model']['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['model']['hidden_dim'], 2)
        )

    def forward(self, x_2010, x_2015):
        feats_2010 = self.feature_extractor(x_2010)
        feats_2015 = self.feature_extractor1(x_2015)

        B, C, H, W = feats_2010.shape

        feats_2010, x_spa1, x_temp1, x_spe1 = self.spectral_blocks(feats_2010)
        feats_2015, x_spa2, x_temp2, x_spe2 = self.spectral_blocks1(feats_2015)

        feats_2010 = feats_2010.reshape(B, C, H * W).permute(0, 2, 1)
        feats_2015 = feats_2015.reshape(B, C, H * W).permute(0, 2, 1)

        pooled_2010 = feats_2010.mean(dim=1)
        pooled_2015 = feats_2015.mean(dim=1)

        spa = torch.abs(x_spa1 - x_spa2).reshape(B, C, H * W).permute(0, 2, 1).mean(dim=1)
        spe = torch.abs(x_spe1 - x_spe2).reshape(B, C, H * W).permute(0, 2, 1).mean(dim=1)
        temp = torch.abs(x_temp1 - x_temp2).reshape(B, C, H * W).permute(0, 2, 1).mean(dim=1)

        change_feats = torch.cat([spa, temp, spe], dim=1)
        change_logits = self.change_head(change_feats)

        class_logits_2010 = self.class_head_2010(pooled_2010)
        class_logits_2015 = self.class_head_2015(pooled_2015)

        return {
            'change': change_logits,
            'class_2010': class_logits_2010,
            'class_2015': class_logits_2015
        }


class AreaOfUnionTransitionLoss(nn.Module):
    def __init__(self, transition_matrix, alpha=0.5, threshold=0.5):
        super().__init__()
        self.transition_matrix = transition_matrix
        self.alpha = alpha  # Weight for ground truth vs. transition matrix
        self.threshold = threshold  # Confidence threshold for model predictions

    def forward(self, pred_2010, pred_2015, pred_change, gt_change):
        # Union mask: Model's confident predictions OR ground truth changes
        pred_mask = (pred_change[:, 1] > self.threshold) | (gt_change == 1)
        pred_mask = pred_mask.float()  # Convert to 0/1 tensor

        # Transition-based expected probability
        expected_2015 = torch.matmul(pred_2010, self.transition_matrix)
        joint_probs = expected_2015 * pred_2015
        base_prob = joint_probs.sum(dim=1)

        # Blend ground truth and transition expectations
        adjusted_prob = self.alpha * gt_change.float() + (1 - self.alpha) * base_prob

        # Masked MSE loss
        loss = F.mse_loss(adjusted_prob * pred_mask,
                          pred_change[:, 1] * pred_mask,
                          reduction='sum')
        return loss / (pred_mask.sum() + 1e-6)


# ==================== TRAINING UTILITIES ====================

def predict_full_image(model, dataset, config, device):
    """Predict change and land cover for entire image using DataLoader"""
    model.eval()
    height, width = dataset.image_2010.shape[:2]

    # Initialize prediction maps
    change_map = np.full((height, width), -1, dtype=np.int8)
    lc2010_map = np.full((height, width), -1, dtype=np.int8)
    lc2015_map = np.full((height, width), -1, dtype=np.int8)
    change_prob_map = np.full((height, width), 0, dtype=np.uint8)  # Changed to uint8

    # Initialize lists to store predictions and labels for metrics
    all_change_preds = []
    all_change_labels = []
    all_lc2010_preds = []
    all_lc2010_labels = []
    all_lc2015_preds = []
    all_lc2015_labels = []

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size_predict_on_whole_image'],
        shuffle=False,
        num_workers=4
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting full image"):
            x_2010 = batch['image_2010'].to(device)
            x_2015 = batch['image_2015'].to(device)
            change_labels = batch['change_label'].to(device)
            lc2010_labels = batch['lc2010_label'].to(device)
            lc2015_labels = batch['lc2015_label'].to(device)
            locations = batch['location']

            outputs = model(x_2010, x_2015)

            # Get change probabilities (softmax) and scale to 0-255
            change_probs = F.softmax(outputs['change'], dim=1)
            change_prob_change_class = change_probs[:, 1]  # Probability of "Change" class

            clipped = torch.clamp(change_prob_change_class, min=0.5, max=1.0)
            # Rescale from [0.5, 1] to [0, 1]
            rescaled = (clipped - 0.5) * 2  # Subtract 0.5 (new min) and divide by 0.5 (range)

            change_prob_8bit = (rescaled * 255).byte()  # Scale to 8-bit

            _, change_preds = torch.max(outputs['change'], 1)
            _, lc2010_preds = torch.max(outputs['class_2010'], 1)
            _, lc2015_preds = torch.max(outputs['class_2015'], 1)

            # Store predictions and labels
            all_change_preds.extend(change_preds.cpu().numpy())
            all_change_labels.extend(change_labels.cpu().numpy())
            all_lc2010_preds.extend(lc2010_preds.cpu().numpy())
            all_lc2010_labels.extend(lc2010_labels.cpu().numpy())
            all_lc2015_preds.extend(lc2015_preds.cpu().numpy())
            all_lc2015_labels.extend(lc2015_labels.cpu().numpy())

            # Update prediction maps
            locations_np = [t.cpu().numpy() for t in locations]
            for k in range(len(change_preds)):
                i, j = locations_np[0][k], locations_np[1][k]
                change_map[i, j] = change_preds[k].item()
                lc2010_map[i, j] = lc2010_preds[k].item()
                lc2015_map[i, j] = lc2015_preds[k].item()
                change_prob_map[i, j] = change_prob_8bit[k].item()  # Store 8-bit value

    # Calculate metrics (unchanged)
    full_change_metrics = calculate_metrics(
        np.array(all_change_preds),
        np.array(all_change_labels),
        CLASS_NAMES
    )

    full_lc2010_metrics = calculate_metrics(
        np.array(all_lc2010_preds),
        np.array(all_lc2010_labels),
        CLASS_NAMES_LC
    )

    full_lc2015_metrics = calculate_metrics(
        np.array(all_lc2015_preds),
        np.array(all_lc2015_labels),
        CLASS_NAMES_LC
    )

    return change_map, lc2010_map, lc2015_map, change_prob_map, full_change_metrics, full_lc2010_metrics, full_lc2015_metrics


def partial_alignment_loss(features1, features2, labels1, labels2, temperature=0.1):
    # 找出未变化的样本（标签一致）
    unchanged_mask = (labels1 == labels2).float()  # [batch_size]
    changed_mask = 1 - unchanged_mask

    # 计算相似度矩阵
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    similarity = torch.matmul(features1, features2.T) / temperature

    # 正样本：未变化的样本对（对角线且unchanged_mask=1）
    pos_mask = torch.eye(len(features1), device=features1.device) * unchanged_mask

    # 负样本：变化的样本或其他类别
    neg_mask = 1 - pos_mask

    # 计算对比损失
    exp_sim = torch.exp(similarity) * neg_mask  # 排除正样本
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = - (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    return loss.mean()


def change_detection_loss(features1, features2, change_labels, temperature=0.1):
    """
    change_labels: 1表示变化，0表示未变化 [batch_size]
    """
    # 归一化特征
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    # 正样本：未变化的样本对（change_labels=0）
    pos_mask = (change_labels == 0).float().view(-1, 1) * torch.eye(len(features1), device=features1.device)

    # 负样本：变化的样本或其他样本
    neg_mask = 1 - pos_mask

    # 计算对比损失
    similarity = torch.matmul(features1, features2.T) / temperature
    exp_sim = torch.exp(similarity) * neg_mask
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = - (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    return loss.mean()


def plot_metrics(train_losses, val_losses, train_accs, val_accs, filename):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    def multi_task_loss(self, loss_dic, sigma1, sigma2):
        weight_diff_loss = torch.exp(-sigma1)  # 回归
        weight_rec_loss = torch.exp(-sigma2)  # 回归
        c = sigma1 + sigma2
        loss = weight_diff_loss * loss_dic["diffusion_loss"] + weight_rec_loss * loss_dic["rec_loss"] + c

        return loss


# ==================== MAIN TRAINING FUNCTION ====================
def train_model(config):
    os.makedirs(config['dataset']['result_dir'], exist_ok=True)

    # Initialize datasets
    train_dataset = ChangeDetectionDataset(
        hdf5_path=config['dataset']['hdf5_path'],
        set_name='train'
    )

    val_dataset = ChangeDetectionDataset(
        hdf5_path=config['dataset']['hdf5_path'],
        set_name='val'
    )

    test_dataset = ChangeDetectionDataset(
        hdf5_path=config['dataset']['hdf5_path'],
        set_name='test'
    )

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Generate transition matrix
    transition_matrix = TRANSITION_MATRIX  # generate_transition_matrix(config['dataset']['hdf5_path'])
    # torch.save(trans_matrix, os.path.join(config['dataset']['result_dir'], 'transition_matrix.pt'))

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(transition_matrix, cmap='plasma')
    plt.colorbar(cax)  # 添加颜色条

    # 在每个单元格中添加数据值
    for (i, j), val in np.ndenumerate(transition_matrix):
        ax.text(j, i, f'{val:.4f}',  # 格式化显示1位小数
                ha='center', va='center',
                color='white' if val > 5 else 'black')  # 根据值调整文本颜色

    plt.axis('off')
    plt.savefig('TT.png', dpi=300)
    # Initialize model and training components
    device = torch.device(config['training']['device'])
    model = ChangeDetectionMamba(config).to(device)

    change_criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()
    transition_criterion = AreaOfUnionTransitionLoss(
        # transition_matrix=model.T,
        transition_matrix=torch.from_numpy(transition_matrix).float().to(device),
        alpha=0.5,  # Blend equally between GT and transition matrix
        threshold=0.5
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    # Training metrics storage
    train_losses = []
    val_losses = []
    train_change_accs = []
    val_change_accs = []
    train_lc2010_accs = []
    val_lc2010_accs = []
    train_lc2015_accs = []
    val_lc2015_accs = []

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct_change = 0
        epoch_train_total_change = 0
        epoch_train_correct_2010 = 0
        epoch_train_total_2010 = 0
        epoch_train_correct_2015 = 0
        epoch_train_total_2015 = 0

        batch_idx = 1
        for batch in train_loader:
            x_2010 = batch['image_2010'].to(device)
            x_2015 = batch['image_2015'].to(device)
            change_labels = batch['change_label'].to(device)
            class_labels_2010 = batch['label_2010'].to(device)
            class_labels_2015 = batch['label_2015'].to(device)

            outputs = model(x_2010, x_2015)

            change_loss = change_criterion(outputs['change'], change_labels)
            class_loss_2010 = class_criterion(outputs['class_2010'], class_labels_2010)
            class_loss_2015 = class_criterion(outputs['class_2015'], class_labels_2015)

            # New transition consistency loss
            pred_2010_probs = F.softmax(outputs['class_2010'], dim=1)
            pred_2015_probs = F.softmax(outputs['class_2015'], dim=1)
            pred_change_probs = F.softmax(outputs['change'], dim=1)

            trans_loss = transition_criterion(
                pred_2010_probs, pred_2015_probs, pred_change_probs, change_labels
            )

            # Combined loss (weight transition loss appropriately)
            change_w = config['training']['loss_weights']['change']
            landcover_w = config['training']['loss_weights']['landcover']
            trans_w = config['training']['loss_weights']['transition']
            # loss1 = change_detection_loss(outputs['class_2010'], outputs['class_2015'], change_labels, temperature=0.3)
            loss = partial_alignment_loss(outputs['class_2010'], outputs['class_2015'], class_labels_2010,
                                          class_labels_2015, temperature=0.3)
            total_loss = (
                    change_w * change_loss +
                    0.5 * (class_loss_2010 + class_loss_2015) +
                    trans_w * trans_loss + loss
            )

            # total_loss = change_loss + 0.5 * (class_loss_2010 + class_loss_2015)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()

            # Calculate accuracies
            _, predicted_change = torch.max(outputs['change'], 1)
            epoch_train_correct_change += (predicted_change == change_labels).sum().item()
            epoch_train_total_change += change_labels.size(0)

            _, predicted_2010 = torch.max(outputs['class_2010'], 1)
            epoch_train_correct_2010 += (predicted_2010 == class_labels_2010).sum().item()
            epoch_train_total_2010 += class_labels_2010.size(0)

            _, predicted_2015 = torch.max(outputs['class_2015'], 1)
            epoch_train_correct_2015 += (predicted_2015 == class_labels_2015).sum().item()
            epoch_train_total_2015 += class_labels_2015.size(0)

            if epoch % 5 == 0 and batch_idx == 1:
                print(f"\nBatch {batch_idx} Loss Magnitudes:")
                print(f"Change: {change_loss.item():.4f} ({change_w * change_loss.item() / total_loss.item():.1%})")
                print(
                    f"LC2010: {class_loss_2010.item():.4f} ({landcover_w * class_loss_2010.item() / total_loss.item():.1%})")
                print(
                    f"LC2015: {class_loss_2015.item():.4f} ({landcover_w * class_loss_2015.item() / total_loss.item():.1%})")
                print(f"Trans: {trans_loss.item():.4f} ({trans_w * trans_loss.item() / total_loss.item():.1%})")

            batch_idx += 1

        # Calculate training metrics
        train_loss = epoch_train_loss / len(train_loader)
        train_change_acc = epoch_train_correct_change / epoch_train_total_change
        train_lc2010_acc = epoch_train_correct_2010 / epoch_train_total_2010
        train_lc2015_acc = epoch_train_correct_2015 / epoch_train_total_2015

        train_losses.append(train_loss)
        train_change_accs.append(train_change_acc)
        train_lc2010_accs.append(train_lc2010_acc)
        train_lc2015_accs.append(train_lc2015_acc)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct_change = 0
        epoch_val_total_change = 0
        epoch_val_correct_2010 = 0
        epoch_val_total_2010 = 0
        epoch_val_correct_2015 = 0
        epoch_val_total_2015 = 0

        all_val_preds_change = []
        all_val_labels_change = []
        all_val_preds_2010 = []
        all_val_labels_2010 = []
        all_val_preds_2015 = []
        all_val_labels_2015 = []

        with torch.no_grad():
            for batch in val_loader:
                x_2010 = batch['image_2010'].to(device)
                x_2015 = batch['image_2015'].to(device)
                change_labels = batch['change_label'].to(device)
                class_labels_2010 = batch['label_2010'].to(device)
                class_labels_2015 = batch['label_2015'].to(device)

                outputs = model(x_2010, x_2015)

                change_loss = change_criterion(outputs['change'], change_labels)
                class_loss_2010 = class_criterion(outputs['class_2010'], class_labels_2010)
                class_loss_2015 = class_criterion(outputs['class_2015'], class_labels_2015)
                total_loss = change_loss + 0.5 * (class_loss_2010 + class_loss_2015)

                epoch_val_loss += total_loss.item()

                # Calculate accuracies
                _, predicted_change = torch.max(outputs['change'], 1)
                epoch_val_correct_change += (predicted_change == change_labels).sum().item()
                epoch_val_total_change += change_labels.size(0)

                _, predicted_2010 = torch.max(outputs['class_2010'], 1)
                epoch_val_correct_2010 += (predicted_2010 == class_labels_2010).sum().item()
                epoch_val_total_2010 += class_labels_2010.size(0)

                _, predicted_2015 = torch.max(outputs['class_2015'], 1)
                epoch_val_correct_2015 += (predicted_2015 == class_labels_2015).sum().item()
                epoch_val_total_2015 += class_labels_2015.size(0)

                # Store predictions for detailed metrics
                all_val_preds_change.extend(predicted_change.cpu().numpy())
                all_val_labels_change.extend(change_labels.cpu().numpy())
                all_val_preds_2010.extend(predicted_2010.cpu().numpy())
                all_val_labels_2010.extend(class_labels_2010.cpu().numpy())
                all_val_preds_2015.extend(predicted_2015.cpu().numpy())
                all_val_labels_2015.extend(class_labels_2015.cpu().numpy())

        # Calculate validation metrics
        val_loss = epoch_val_loss / len(val_loader)
        val_change_acc = epoch_val_correct_change / epoch_val_total_change
        val_lc2010_acc = epoch_val_correct_2010 / epoch_val_total_2010
        val_lc2015_acc = epoch_val_correct_2015 / epoch_val_total_2015

        val_losses.append(val_loss)
        val_change_accs.append(val_change_acc)
        val_lc2010_accs.append(val_lc2010_acc)
        val_lc2015_accs.append(val_lc2015_acc)

        # Calculate detailed validation metrics
        val_change_metrics = calculate_metrics(
            np.array(all_val_preds_change),
            np.array(all_val_labels_change),
            CLASS_NAMES
        )

        val_lc2010_metrics = calculate_metrics(
            np.array(all_val_preds_2010),
            np.array(all_val_labels_2010),
            CLASS_NAMES_LC
        )

        val_lc2015_metrics = calculate_metrics(
            np.array(all_val_preds_2015),
            np.array(all_val_labels_2015),
            CLASS_NAMES_LC
        )

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{config['training']['epochs']} - "
              f"TrainLoss: {train_loss:.4f}, ChangeAcc: {train_change_acc:.4f}, "
              f"2010Acc: {train_lc2010_acc:.4f}, 2015Acc: {train_lc2015_acc:.4f} | "
              f"ValLoss: {val_loss:.4f}, ChangeAcc: {val_change_acc:.4f}, "
              f"2010Acc: {val_lc2010_acc:.4f}, 2015Acc: {val_lc2015_acc:.4f}")

        scheduler.step(val_loss)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['dataset']['result_dir'], 'best_model.pth'))

            # Save validation metrics for best model
            save_metrics_report(
                val_change_metrics,
                CLASS_NAMES,
                'best_val_change_metrics',
                config['dataset']['result_dir']
            )

            save_metrics_report(
                val_lc2010_metrics,
                CLASS_NAMES_LC,
                'best_val_lc2010_metrics',
                config['dataset']['result_dir']
            )

            save_metrics_report(
                val_lc2015_metrics,
                CLASS_NAMES_LC,
                'best_val_lc2015_metrics',
                config['dataset']['result_dir']
            )
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    # Plot training curves
    plot_metrics(
        train_losses,
        val_losses,
        train_change_accs,
        val_change_accs,
        os.path.join(config['dataset']['result_dir'], 'training_metrics_change.png')
    )

    plot_metrics(
        train_losses,
        val_losses,
        train_lc2010_accs,
        val_lc2010_accs,
        os.path.join(config['dataset']['result_dir'], 'training_metrics_lc2010.png')
    )

    plot_metrics(
        train_losses,
        val_losses,
        train_lc2015_accs,
        val_lc2015_accs,
        os.path.join(config['dataset']['result_dir'], 'training_metrics_lc2015.png')
    )

    # Final evaluation on test set
    model.load_state_dict(
        torch.load(os.path.join(config['dataset']['result_dir'], 'best_model.pth'), weights_only=True))
    model.eval()

    # print("model T", model.T)

    all_test_preds_change = []
    all_test_labels_change = []
    all_test_preds_2010 = []
    all_test_labels_2010 = []
    all_test_preds_2015 = []
    all_test_labels_2015 = []

    with torch.no_grad():
        for batch in test_loader:
            x_2010 = batch['image_2010'].to(device)
            x_2015 = batch['image_2015'].to(device)
            change_labels = batch['change_label'].to(device)
            class_labels_2010 = batch['label_2010'].to(device)
            class_labels_2015 = batch['label_2015'].to(device)

            outputs = model(x_2010, x_2015)

            _, preds_change = torch.max(outputs['change'], 1)
            _, preds_2010 = torch.max(outputs['class_2010'], 1)
            _, preds_2015 = torch.max(outputs['class_2015'], 1)

            all_test_preds_change.extend(preds_change.cpu().numpy())
            all_test_labels_change.extend(change_labels.cpu().numpy())
            all_test_preds_2010.extend(preds_2010.cpu().numpy())
            all_test_labels_2010.extend(class_labels_2010.cpu().numpy())
            all_test_preds_2015.extend(preds_2015.cpu().numpy())
            all_test_labels_2015.extend(class_labels_2015.cpu().numpy())

    # Calculate test metrics
    test_change_metrics = calculate_metrics(
        np.array(all_test_preds_change),
        np.array(all_test_labels_change),
        CLASS_NAMES
    )

    test_lc2010_metrics = calculate_metrics(
        np.array(all_test_preds_2010),
        np.array(all_test_labels_2010),
        CLASS_NAMES_LC
    )

    test_lc2015_metrics = calculate_metrics(
        np.array(all_test_preds_2015),
        np.array(all_test_labels_2015),
        CLASS_NAMES_LC
    )

    # Save test metrics
    save_metrics_report(
        test_change_metrics,
        CLASS_NAMES,
        'test_change_metrics',
        config['dataset']['result_dir']
    )

    save_metrics_report(
        test_lc2010_metrics,
        CLASS_NAMES_LC,
        'test_lc2010_metrics',
        config['dataset']['result_dir']
    )

    save_metrics_report(
        test_lc2015_metrics,
        CLASS_NAMES_LC,
        'test_lc2015_metrics',
        config['dataset']['result_dir']
    )

    # Print test results
    print("\n=== Final Test Metrics ===")
    print("\n=== Change Detection ===")
    print(f"Overall Accuracy: {test_change_metrics['overall_accuracy']:.4f}")
    print(f"Average Class Accuracy: {test_change_metrics['average_class_accuracy']:.4f}")
    print(f"Cohen's Kappa: {test_change_metrics['kappa']:.4f}")

    print("\n=== Land Cover Classification 2010 ===")
    print(f"Overall Accuracy: {test_lc2010_metrics['overall_accuracy']:.4f}")
    print(f"Average Class Accuracy: {test_lc2010_metrics['average_class_accuracy']:.4f}")
    print(f"Cohen's Kappa: {test_lc2010_metrics['kappa']:.4f}")

    print("\n=== Land Cover Classification 2015 ===")
    print(f"Overall Accuracy: {test_lc2015_metrics['overall_accuracy']:.4f}")
    print(f"Average Class Accuracy: {test_lc2015_metrics['average_class_accuracy']:.4f}")
    print(f"Cohen's Kappa: {test_lc2015_metrics['kappa']:.4f}")

    if config['training']['to_predict_on_entire_image']:
        print("\n=== Predicting on entire images ===")

        model.load_state_dict(
            torch.load(os.path.join(config['dataset']['result_dir'], 'best_model.pth'), weights_only=True))
        model.eval()

        full_dataset = FullImageChangeDetectionDataset(
            image_2010_path=config['dataset']['image_2010_file'],
            image_2015_path=config['dataset']['image_2015_file'],
            gt_change_path=config['dataset']['gt_change_file'],
            gt_2010LC_path=config['dataset']['gt_2010LC_file'],
            gt_2015LC_path=config['dataset']['gt_2015LC_file'],
            patch_size=config['dataset']['patch_size']
        )

        # In the train_model function, replace the predict_full_image call with:
        change_map, lc2010_map, lc2015_map, change_prob_map, full_change_metrics, full_lc2010_metrics, full_lc2015_metrics = predict_full_image(
            model, full_dataset, config, device
        )

        # Save the probability map
        change_prob_output_path = os.path.join(config['dataset']['result_dir'], 'change_probability.tif')
        save_geotiff_with_metadata(
            change_prob_map,
            config['dataset']['gt_change_file'],
            change_prob_output_path
        )
        print(f"- Change probability: {change_prob_output_path}")

        # Save metrics
        save_metrics_report(
            full_change_metrics,
            CLASS_NAMES,
            'full_image_change_metrics',
            config['dataset']['result_dir']
        )

        save_metrics_report(
            full_lc2010_metrics,
            CLASS_NAMES_LC,
            'full_image_lc2010_metrics',
            config['dataset']['result_dir']
        )

        save_metrics_report(
            full_lc2015_metrics,
            CLASS_NAMES_LC,
            'full_image_lc2015_metrics',
            config['dataset']['result_dir']
        )

        # Save prediction maps
        change_output_path = os.path.join(config['dataset']['result_dir'], 'full_change_prediction.tif')
        lc2010_output_path = os.path.join(config['dataset']['result_dir'], 'landcover_2010_prediction.tif')
        lc2015_output_path = os.path.join(config['dataset']['result_dir'], 'landcover_2015_prediction.tif')
        # Provided class information
        valid_lc_classes = [1, 2, 5, 6, 8, 10, 14, 15, 16, 17, 18]
        class_mapping = {cls: idx for idx, cls in enumerate(valid_lc_classes)}
        # Create inverse mapping: {index: original_class}
        inverse_class_mapping = {v: k for k, v in class_mapping.items()}
        # Apply inverse mapping to prediction data
        inverse_mapped_lc2010 = np.copy(lc2010_map)
        inverse_mapped_lc2015 = np.copy(lc2015_map)

        for pred_idx, orig_class in inverse_class_mapping.items():
            inverse_mapped_lc2010[lc2010_map == pred_idx] = orig_class
            inverse_mapped_lc2015[lc2015_map == pred_idx] = orig_class

        inverse_mapped_lc2010[lc2010_map == -1] = 0
        inverse_mapped_lc2015[lc2015_map == -1] = 0
        inverse_mapped_change = change_map + 1

        save_geotiff_with_metadata(
            inverse_mapped_change,
            config['dataset']['gt_change_file'],
            change_output_path
        )

        save_geotiff_with_metadata(
            inverse_mapped_lc2010,
            config['dataset']['gt_2010LC_file'],
            lc2010_output_path
        )
        save_geotiff_with_metadata(
            inverse_mapped_lc2015,
            config['dataset']['gt_2015LC_file'],
            lc2015_output_path
        )

        print(f"\nPredictions saved to:")
        print(f"- Change detection: {change_output_path}")
        print(f"- Land cover 2010: {lc2010_output_path}")
        print(f"- Land cover 2015: {lc2015_output_path}")

        # Print full image metrics
        print("\n=== Full Image Metrics ===")
        print("\n=== Change Detection ===")
        print(f"Overall Accuracy: {full_change_metrics['overall_accuracy']:.4f}")
        print(f"Average Class Accuracy: {full_change_metrics['average_class_accuracy']:.4f}")
        print(f"Cohen's Kappa: {full_change_metrics['kappa']:.4f}")

        print("\n=== Land Cover Classification 2010 ===")
        print(f"Overall Accuracy: {full_lc2010_metrics['overall_accuracy']:.4f}")
        print(f"Average Class Accuracy: {full_lc2010_metrics['average_class_accuracy']:.4f}")
        print(f"Cohen's Kappa: {full_lc2010_metrics['kappa']:.4f}")

        print("\n=== Land Cover Classification 2015 ===")
        print(f"Overall Accuracy: {full_lc2015_metrics['overall_accuracy']:.4f}")
        print(f"Average Class Accuracy: {full_lc2015_metrics['average_class_accuracy']:.4f}")
        print(f"Cohen's Kappa: {full_lc2015_metrics['kappa']:.4f}")


if __name__ == "__main__":
    train_model(config)
