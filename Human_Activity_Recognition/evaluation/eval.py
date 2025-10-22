import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

def evaluate(model, checkpoint_path, ds_test, ds_info, run_paths):

    # 确保保存路径存在
    os.makedirs(run_paths['path_model_id'], exist_ok=True)

    # 从检查点加载模型权重
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(checkpoint_path).expect_partial()
    print(f"Checkpoint successfully loaded from {checkpoint_path}")

    # 初始化混淆矩阵和真实值、预测值存储
    num_classes = ds_info["label"]["num_classes"]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    y_true = []
    y_pred = []

    # 遍历测试集
    for images, labels_true in ds_test:
        predictions = model(images, training=False)
        predicted_labels = tf.argmax(predictions, axis=1)

        y_true.extend(labels_true.numpy())
        y_pred.extend(predicted_labels.numpy())

        # 更新混淆矩阵
        for true_label, predicted_label in zip(labels_true.numpy(), predicted_labels.numpy()):
            confusion_matrix[true_label, predicted_label] += 1

    # 计算指标
    sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    specificity = calculate_specificity(confusion_matrix)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # 打印并保存评估结果
    metrics = {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    save_metrics(metrics, run_paths)

    # 可视化混淆矩阵
    labels = ds_info.get("label_names", [f"Class {i}" for i in range(num_classes)])
    plot_confusion_matrix(confusion_matrix, labels, run_paths)

def calculate_specificity(confusion_matrix):

    specificity_scores = []
    for i in range(len(confusion_matrix)):
        true_negatives = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificity_scores.append(specificity)
    return np.mean(specificity_scores)



def plot_confusion_matrix(confusion_matrix, labels, run_paths):

    # 归一化混淆矩阵
    row_sums = confusion_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 避免除以零
    normalized_cm = confusion_matrix.astype('float') / row_sums[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()

    # 调整标签从 1 开始
    labels = [label for label in labels]
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)

    # 在单元格中添加数值（保留2位小数）
    thresh = normalized_cm.max() / 2
    for i, j in np.ndindex(normalized_cm.shape):
        plt.text(j, i, f"{normalized_cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if normalized_cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # 保存混淆矩阵图片
    save_path = os.path.join(run_paths['path_model_id'], 'confusion_matrix_normalized.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def save_metrics(metrics, run_paths):
    """
    将指标保存到文本文件。
    """
    save_path = os.path.join(run_paths['path_model_id'], 'evaluation_metrics.txt')
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Metrics saved to {save_path}")