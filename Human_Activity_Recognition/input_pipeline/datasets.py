import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt
import gin
from collections import Counter
from sklearn.utils import resample
import random


# Define a low-pass filter
def low_pass(signal, fs=50, fc=10):
    w = fc / (fs / 2)
    b, a = butter(5, w, "low")
    return filtfilt(b, a, signal.T).T


def create_windows(data, labels, window_length, window_shift):
    """
    Sliding window processing function, only processing windows of sufficient length

    Args:
        data (np.ndarray): Input data with a shape of [time steps, channels]
        labels (np.ndarray): Corresponding labels with a shape of [time steps]
        window_length (int): Length of the sliding window
        window_shift (int): Step size of the sliding window

    Returns:
        np.ndarray, np.ndarray: Feature and main label arrays of the sliding window
    """
    features, window_labels = [], []

    # Iterate through all possible starting points of the sliding window
    for start in range(0, len(data) - window_length + 1, window_shift):
        # Extract the current window data
        window = data[start:start + window_length]
        label_window = labels[start:start + window_length]

        # Ensure the length of the window meets the requirements
        if len(window) < window_length:
            print(f"Skipping incomplete window at index {start}, length={len(window)}")
            continue

        # Calculate the main label of the window
        label = np.argmax(np.bincount(label_window))

        # Add to the result list
        features.append(window)
        window_labels.append(label)

    # Convert to a NumPy array and return
    return np.array(features), np.array(window_labels)


# TFRecord serialization function
def serialize_example(features, labels):

    # Force convert features to float32 and labels to int64
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)

    feature = {
        "features": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(features).numpy()])),
        "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(labels).numpy()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(features, labels, filename):

    # Get the directory of the file and check if it exists
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Recursively create directories
        print(f"Created directory: {output_dir}")

    print(f"Writing to TFRecord: {filename}")
    with tf.io.TFRecordWriter(filename) as writer:
        for f, l in zip(features, labels):
            example = serialize_example(f, l)
            writer.write(example.SerializeToString())
    print(f"TFRecord saved: {filename}")


def process_data(raw_data_path, labels_path, output_path, user_splits, window_length=250, window_shift=125, fc=10,
                 max_oversample_factor=6):
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # Load the label file
    labels_df = pd.read_csv(labels_path, sep=" ", header=None, names=["exp", "user", "activity", "start", "end"])

    # Define a container to store the results
    datasets = {"train": [], "val": [], "test": []}

    # Iterate through all users
    for user_id in range(1, 31):
        # The number of experiments per user: User 10 performs 3 experiments, while all other users perform 2 experiments
        num_experiments = 3 if user_id == 10 else 2

        # Iterate through the experiments of the current user
        for exp_offset in range(num_experiments):
            experiment_id = (user_id - 1) * 2 + 1 + exp_offset

            # Locate the corresponding sensor file and gyroscope file based on the user ID and experiment ID
            acc_file = os.path.join(raw_data_path,
                                    f"acc_exp{str(experiment_id).zfill(2)}_user{str(user_id).zfill(2)}.txt")
            gyro_file = os.path.join(raw_data_path,
                                     f"gyro_exp{str(experiment_id).zfill(2)}_user{str(user_id).zfill(2)}.txt")

            # If the file does not exist, skip it directly
            if not os.path.exists(acc_file) or not os.path.exists(gyro_file):
                continue

            # Load the sensor data
            acc_data = pd.read_csv(acc_file, sep=" ", header=None).values
            gyro_data = pd.read_csv(gyro_file, sep=" ", header=None).values

            # Perform Z-score normalization on each channel of the sensor data
            def zscore_normalize(data):
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                return (data - mean) / std

            # Normalize the accelerometer and gyroscope data
            acc_data = zscore_normalize(acc_data)
            gyro_data = zscore_normalize(gyro_data)

            # Merge sensor data
            data = np.hstack([acc_data, gyro_data])

            # Find the label range corresponding to the user and experiment
            user_labels = labels_df[(labels_df["exp"] == experiment_id) & (labels_df["user"] == user_id)]

            # Obtain the start and end points of the label range
            if user_labels.empty:
                continue

            start = user_labels["start"].min()
            end = user_labels["end"].max()

            # Restrict the data range to the valid label interval
            data = data[start:end]
            labels = np.zeros(len(data), dtype=np.int64)

            # Assign labels to the data within the interval based on the label file
            for _, row in user_labels.iterrows():
                activity_start = max(0, row["start"] - start)
                activity_end = min(len(data), row["end"] - start)
                labels[activity_start:activity_end] = row["activity"] - 1

            # Sliding window processing
            features, window_labels = create_windows(data, labels, window_length, window_shift)

            # Determine the dataset type
            if user_id in user_splits["train"]:
                split = "train"
            elif user_id in user_splits["val"]:
                split = "val"
            elif user_id in user_splits["test"]:
                split = "test"
            else:
                continue

            # Validate the data before adding it
            if features.shape[0] > 0 and window_labels.shape[0] > 0:
                # print(f"Features shape: {features.shape}, Labels shape: {window_labels.shape}")
                datasets[split].append((features, window_labels))

    # Data distribution statistics and oversampling logic
    for split, data in datasets.items():
        if len(data) == 0:
            continue

        features = np.concatenate([d[0] for d in data], axis=0)
        labels = np.concatenate([d[1] for d in data], axis=0)

        # Print the original data distribution
        print(f"{split} dataset before oversampling: {Counter(labels)}")

        if split == "train":  # Perform oversampling only on the training set
            label_counts = Counter(labels)
            max_count = max(label_counts.values())

            # Oversample each class
            resampled_features, resampled_labels = [], []
            for label, count in label_counts.items():
                label_features = features[labels == label]
                label_labels = labels[labels == label]

                # Calculate the target quantity (limit oversampling to within the max_oversample_factor)
                target_count = min(count * max_oversample_factor, max_count)

                if count < target_count:
                    # data  augmentation
                    def augment_data(sample):
                        augmentations = [
                            lambda x: x + np.random.normal(0, 0.01, x.shape),  # 添加噪声
                            lambda x: x * np.random.uniform(0.9, 1.1, x.shape),  # 随机缩放
                            lambda x: np.roll(x, shift=random.randint(-5, 5), axis=0),  # 时间轴滚动
                        ]
                        return random.choice(augmentations)(sample)

                    # Augmentation and oversampling
                    additional_samples = resample(label_features, replace=True, n_samples=target_count - count,
                                                  random_state=42)
                    additional_samples = np.array([augment_data(sample) for sample in additional_samples])

                    resampled_features.append(additional_samples)
                    resampled_labels.append(np.full(additional_samples.shape[0], label))

            # Combine the original data and the oversampled data
            if resampled_features:
                features = np.concatenate([features] + resampled_features, axis=0)
                labels = np.concatenate([labels] + resampled_labels, axis=0)

            # Print the data distribution after oversampling
            print(f"{split} dataset after oversampling: {Counter(labels)}")

        else:  # Keep the original distribution for the validation and test sets
            print(f"{split} dataset remains unchanged: {Counter(labels)}")

        # Write to TFRecord files
        output_file = os.path.join(output_path, f"{split}.tfrecord")

        write_tfrecord(features, labels, output_file)
        print(f"{split} dataset: {features.shape[0]} samples written to {output_file}")


@gin.configurable
def load(data_dir, batch_size=32, window_length=250, fc=None):


    def parse_example(serialized_example):

        feature_description = {
            "features": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Parse features and labels into tensors, and forcefully convert them to tf.float32 and tf.int64
        features = tf.io.parse_tensor(example["features"], out_type=tf.float32)  # Force convert to `tf.float32`.
        labels = tf.io.parse_tensor(example["labels"], out_type=tf.int64)  # Ensure that the label type is correct
        return features, labels

    # Load training, validation, and test data
    train_files = tf.io.gfile.glob(f"{data_dir}/train.tfrecord")
    val_files = tf.io.gfile.glob(f"{data_dir}/val.tfrecord")
    test_files = tf.io.gfile.glob(f"{data_dir}/test.tfrecord")

    ds_train = tf.data.TFRecordDataset(train_files).map(parse_example).shuffle(1000).batch(batch_size).prefetch(
        tf.data.AUTOTUNE)
    ds_val = tf.data.TFRecordDataset(val_files).map(parse_example).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = tf.data.TFRecordDataset(test_files).map(parse_example).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Dataset metadata
    ds_info = {
        "features": {"shape": (window_length, 6)},  # Each sample has 6 channels (accelerometer and gyroscope).
        "label": {"num_classes": 12},
    }
    return ds_train, ds_val, ds_test, ds_info


if __name__ == "__main__":
    # Configure paths and parameters
    raw_data_path = "/home/data/HAPT_dataset/RawData"
    labels_path = "/home/data/HAPT_dataset/RawData/labels.txt"
    output_path = "/home/RUS_CIP/st191716/dl-lab-24w-team08/Human_Activity_Recognition"
    user_splits = {
        "train": list(range(1, 22)),
        "val": list(range(28, 31)),
        "test": list(range(22, 28)),
    }
    # Run data processing
    process_data(raw_data_path, labels_path, output_path, user_splits)