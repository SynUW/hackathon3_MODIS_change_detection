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

from comparison_models.HRSCD import ChangeMask


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed=42)

def get_model_name_from_import():
    """从import语句中提取模型名称"""
    import_line = None
    with open(__file__, 'r') as f:
        for line in f:
            if 'from comparison_models.' in line and 'import' in line:
                import_line = line.strip()
                break
    
    if import_line:
        # 从 "from comparison_models.XXX import YYY" 中提取 XXX
        model_name = import_line.split('from comparison_models.')[1].split(' import')[0]
        return model_name
    return 'unknown_model'

# ==================== CONFIGURATION ====================
config = {
    'model_name': get_model_name_from_import(),  # 自动获取模型名称
    'dataset': {
        'hdf5_path': '/mnt/raid/benchmark_datasets/MODIS/sask/patches_13_by_13_change_detection/change_detection_samples.h5',
        'patch_size': 13,
        'result_dir': './change_detection_results',
        'image_2010_file': '/mnt/raid/benchmark_datasets/MODIS/sask/MOD13Q1_sask_2010.tiff',
        'image_2015_file': '/mnt/raid/benchmark_datasets/MODIS/sask/MOD13Q1_sask_2015.tiff',
        'gt_change_file': '/mnt/raid/benchmark_datasets/MODIS/sask/sask_binary_change_2010_2015_gt.tif',
        'gt_2010LC_file': '/mnt/raid/benchmark_datasets/MODIS/sask/sask_2010_gt_new.tif',
        'gt_2015LC_file': '/mnt/raid/benchmark_datasets/MODIS/sask/sask_2015_gt.tif'
    },
    'model': {
        'num_bands': 138,
        'hidden_dim': 138*4,
        'num_classes': 11,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'sparsity_ratio': 0.1
    },
    'training': {
        'batch_size': 128,
        'batch_size_predict_on_whole_image': 4096,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'epochs': 50,
        'early_stopping_patience': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'to_predict_on_entire_image': True,
        'loss_weights': {
            'change': 1.0,  # Base weight
            'landcover': 0.5,  # LC loss multiplier
            'transition': 0  # Trans loss multiplier
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
])


# ==================== UTILITY FUNCTIONS ====================

class OHEMLoss(nn.Module):
    def __init__(self, loss_func, top_k=0.3, reduction='mean'):
        """
        Args:
            loss_func: 原始损失函数（如 CrossEntropyLoss）
            top_k: float in (0, 1]，保留前 top_k 比例的 hardest samples
            reduction: 最终损失的归约方式 ('mean' or 'sum')
        """
        super().__init__()
        self.loss_func = loss_func
        self.top_k = top_k
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] ground truth labels
        Returns:
            标量损失值
        """
        batch_size = targets.size(0)
        device = inputs.device

        # 计算每个样本的损失
        per_sample_loss = self.loss_func(inputs, targets, reduction='none')  # [B]

        # 选择 top-k 个最难样本
        k = max(1, int(batch_size * self.top_k))
        top_loss_values, top_loss_indices = torch.topk(per_sample_loss, k=k)

        # 收集 hard example 的 inputs 和 targets
        hard_inputs = inputs[top_loss_indices]
        hard_targets = targets[top_loss_indices]

        # 计算 hard examples 的损失（必须为标量）
        if self.reduction == 'mean':
            final_loss = top_loss_values.mean()
        elif self.reduction == 'sum':
            final_loss = top_loss_values.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

        return final_loss


def adversarial_loss(model, x, targets, epsilon=0.01, alpha=0.8):
    # baseline 是96.15，使用alpha=0.3得到的 test acc是96.29; 0.8-> 96.25
    # 0.01, 0.8并且使用20%的不确定性样本对抗，得到的acc是96.35%
    #  0.01, 0.8并且使用10%的不确定性样本对抗，validation acc更高，但是test低了
    # 确保输入需要梯度
    x_adv = x.clone().requires_grad_(True)  # 关键修复：显式启用梯度

    # 原始预测
    logits = model(x_adv)[0]  # 假设model返回(logits, features)
    probs = torch.softmax(logits, dim=1)
    aleatoric = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

    # 仅对高不确定性样本生成对抗样本
    # high_uncertainty_mask = aleatoric > aleatoric.median()
    # 仅选择不确定性最高的top_k个样本
    _, top_indices = torch.topk(aleatoric, k=20, largest=True)  # largest=True选择最大值
    high_uncertainty_mask = torch.zeros_like(aleatoric, dtype=torch.bool)
    high_uncertainty_mask[top_indices] = True  # 将top_k位置标记为True
    if high_uncertainty_mask.any():
        # 计算对抗损失
        loss_adv = nn.CrossEntropyLoss()(logits[high_uncertainty_mask],
                                                            targets[high_uncertainty_mask])
        grad = torch.autograd.grad(loss_adv, x_adv, retain_graph=True)[0]  # 计算梯度

        # 生成对抗样本（仅修改高不确定性样本）
        with torch.no_grad():
            x_adv[high_uncertainty_mask] = x_adv[high_uncertainty_mask] + epsilon * grad[high_uncertainty_mask].sign()

    # 组合损失
    loss_clean = nn.CrossEntropyLoss()(model(x)[0], targets)
    loss_adv = nn.CrossEntropyLoss()(model(x_adv)[0], targets)
    return alpha * loss_clean + (1 - alpha) * loss_adv


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
    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    
    # 计算原始混淆矩阵
    cm = confusion_matrix(targets, preds)
    
    # 使用原始混淆矩阵计算kappa
    kappa = cohen_kappa_score(targets, preds)
    
    # 其他计算保持不变
    accuracy = np.mean(preds == targets)
    class_accuracies = []
    for i in range(cm.shape[0]):
        if cm[i, :].sum() > 0:
            class_accuracies.append(cm[i, i] / cm[i, :].sum())
        else:
            class_accuracies.append(0.0)
    
    avg_class_accuracy = np.mean(class_accuracies)
    
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

    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'average_class_accuracy': avg_class_accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'report': report_dict,
        'report_str': report_str
    }


def save_metrics_report(metrics, class_names, filename_prefix, result_dir):
    """Save comprehensive metrics report to files"""
    os.makedirs(result_dir, exist_ok=True)

    # Save confusion matrix
    cm_df = pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(result_dir, f'{filename_prefix}_confusion_matrix.csv'))

    # Save visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"Confusion Matrix ({filename_prefix.replace('_', ' ')})")
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


# ==================== MAIN TRAINING FUNCTION ====================
def train_model(config):
    # 创建以模型名称命名的结果目录
    model_result_dir = os.path.join(config['dataset']['result_dir'], config['model_name'])
    os.makedirs(model_result_dir, exist_ok=True)
    
    # 更新config中的result_dir
    config['dataset']['result_dir'] = model_result_dir

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

    # Initialize model and training components
    device = torch.device(config['training']['device'])
    # model = ChangeDetectionMamba(config).to(device)

    model = ChangeMask(num_classes=11).to(device)

    change_criterion = nn.CrossEntropyLoss()
    change_criterion_val = nn.CrossEntropyLoss(label_smoothing=0.0)
    class_criterion_10 = nn.CrossEntropyLoss()
    class_criterion_15 = nn.CrossEntropyLoss()

    class_criterion_val = nn.CrossEntropyLoss(label_smoothing=0.0)

    transition_criterion = AreaOfUnionTransitionLoss(
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

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(T_max=30, optimizer=optimizer)

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
            class_loss_2010 = class_criterion_10(outputs['class_2010'], class_labels_2010)
            class_loss_2015 = class_criterion_15(outputs['class_2015'], class_labels_2015)

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

            total_loss = (
                    change_w * change_loss +
                    landcover_w * (class_loss_2010 + class_loss_2015) +
                    trans_w * trans_loss
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

                change_loss = change_criterion_val(outputs['change'], change_labels)
                class_loss_2010 = class_criterion_val(outputs['class_2010'], class_labels_2010)
                class_loss_2015 = class_criterion_val(outputs['class_2015'], class_labels_2015)
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

        scheduler.step(val_loss)  # val_loss

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['dataset']['result_dir'], 'best_model2.pth'))

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
        torch.load(os.path.join(config['dataset']['result_dir'], 'best_model2.pth'), weights_only=True))
    model.eval()

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
            torch.load(os.path.join(config['dataset']['result_dir'], 'best_model2.pth'), weights_only=True))
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
