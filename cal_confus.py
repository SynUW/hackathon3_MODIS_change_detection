import os
import pandas as pd
import numpy as np

def compute_metrics_from_confusion(df: pd.DataFrame):
    cm = df.values.astype(float)
    labels = df.index.to_list()

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    # 防止除以 0 的 NaN
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)

    support = cm.sum(axis=1)
    total_support = support.sum()

    per_class = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=labels,
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    tp_total = tp.sum()
    fp_total = fp.sum()
    fn_total = fn.sum()
    micro_precision = tp_total / (tp_total + fp_total)
    micro_recall = tp_total / (tp_total + fn_total)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    weighted_precision = (precision * support).sum() / total_support
    weighted_recall = (recall * support).sum() / total_support
    weighted_f1 = (f1 * support).sum() / total_support

    summary = pd.DataFrame(
        {
            "precision": [macro_precision, micro_precision, weighted_precision],
            "recall": [macro_recall, micro_recall, weighted_recall],
            "f1": [macro_f1, micro_f1, weighted_f1],
        },
        index=["macro", "micro", "weighted"],
    )

    return per_class, summary

# === 使用示例 ===
def main(csv_path):
    # 读取混淆矩阵
    df = pd.read_csv(csv_path, index_col=0)

    # 计算指标
    per_class, summary = compute_metrics_from_confusion(df)

    # 构建输出目录
    output_dir = os.path.join(os.path.dirname(csv_path), "confusion_metrics_output")
    os.makedirs(output_dir, exist_ok=True)

    # 输出 TXT 文件
    output_txt_path = os.path.join(output_dir, "classification_metrics_summary.txt")
    with open(output_txt_path, "w") as f:
        f.write("=== Per-class metrics ===\n")
        f.write(per_class.round(4).to_string())
        f.write("\n\n=== Macro / Micro / Weighted averages ===\n")
        f.write(summary.round(4).to_string())

    print(f"Results saved to: {output_txt_path}")

# === 替换为你的文件路径 ===
if __name__ == "__main__":
    csv_file_path = r"D:\OneDrive - University of Calgary\Desktop\change_detection_results\ChangeMask\full_image_lc2010_metrics_confusion_matrix.csv"  # ← 修改为你的 CSV 文件路径
    main(csv_file_path)
