import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

def evaluate(model, checkpoint_path, ds_test, ds_info, run_paths):

    # Ensure the save path exists
    os.makedirs(run_paths['path_model_id'], exist_ok=True)

    # Load model weights from the checkpoint and repeat the above steps
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(checkpoint_path).expect_partial()
    print(f"Checkpoint successfully loaded from {checkpoint_path}")

    # Initialize the confusion matrix and storage for true values and predicted values
    num_classes = ds_info["label"]["num_classes"]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    y_true = []
    y_pred = []

    # Iterate through the test set
    for images, labels_true in ds_test:
        predictions = model(images, training=False)
        predicted_labels = tf.argmax(predictions, axis=1)

        y_true.extend(labels_true.numpy())
        y_pred.extend(predicted_labels.numpy())

        # Update the confusion matrix
        for true_label, predicted_label in zip(labels_true.numpy(), predicted_labels.numpy()):
            confusion_matrix[true_label, predicted_label] += 1

    # Calculate metrics
    sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    specificity = calculate_specificity(confusion_matrix)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Print and save the evaluation results
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

    # Visualize the confusion matrix
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

    # Normalize the confusion matrix
    row_sums = confusion_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_cm = confusion_matrix.astype('float') / row_sums[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()

    # Adjust labels to start from 1
    labels = [label for label in labels]
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)

    # Add numerical values in the cells (keep two decimal places)
    thresh = normalized_cm.max() / 2
    for i, j in np.ndindex(normalized_cm.shape):
        plt.text(j, i, f"{normalized_cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if normalized_cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save the confusion matrix image
    save_path = os.path.join(run_paths['path_model_id'], 'confusion_matrix_normalized.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def save_metrics(metrics, run_paths):
    """
    Save the metrics to a text file
    """
    save_path = os.path.join(run_paths['path_model_id'], 'evaluation_metrics.txt')
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Metrics saved to {save_path}")