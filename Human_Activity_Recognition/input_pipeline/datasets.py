import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt
import gin
from collections import Counter
from sklearn.utils import resample
import random


# 定义低通滤波器
def low_pass(signal, fs=50, fc=10):
    w = fc / (fs / 2)
    b, a = butter(5, w, "low")
    return filtfilt(b, a, signal.T).T


def create_windows(data, labels, window_length, window_shift):
    """
    滑动窗口处理函数，仅处理长度足够的窗口。

    Args:
        data (np.ndarray): 输入数据，形状为 [时间步数, 通道数]。
        labels (np.ndarray): 对应的标签，形状为 [时间步数]。
        window_length (int): 滑动窗口的长度。
        window_shift (int): 滑动窗口的步长。

    Returns:
        np.ndarray, np.ndarray: 滑动窗口的特征和主标签数组。
    """
    features, window_labels = [], []

    # 遍历所有可能的滑动窗口起点
    for start in range(0, len(data) - window_length + 1, window_shift):
        # 提取当前窗口数据
        window = data[start:start + window_length]
        label_window = labels[start:start + window_length]

        # 确保窗口的长度满足要求
        if len(window) < window_length:
            print(f"Skipping incomplete window at index {start}, length={len(window)}")
            continue

        # 计算窗口的主标签
        label = np.argmax(np.bincount(label_window))

        # 添加到结果列表
        features.append(window)
        window_labels.append(label)

    # 转换为 NumPy 数组返回
    return np.array(features), np.array(window_labels)


# TFRecord 序列化函数
def serialize_example(features, labels):

    # 强制转换特征为 float32，标签为 int64
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)

    feature = {
        "features": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(features).numpy()])),
        "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(labels).numpy()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(features, labels, filename):

    # 获取文件所在目录，并检查是否存在
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 递归创建目录
        print(f"Created directory: {output_dir}")

    print(f"Writing to TFRecord: {filename}")
    with tf.io.TFRecordWriter(filename) as writer:
        for f, l in zip(features, labels):
            example = serialize_example(f, l)
            writer.write(example.SerializeToString())
    print(f"TFRecord saved: {filename}")


def process_data(raw_data_path, labels_path, output_path, user_splits, window_length=250, window_shift=125, fc=10,
                 max_oversample_factor=6):


    # 加载标签文件
    labels_df = pd.read_csv(labels_path, sep=" ", header=None, names=["exp", "user", "activity", "start", "end"])

    # 定义存储结果的容器
    datasets = {"train": [], "val": [], "test": []}

    # 遍历所有用户
    for user_id in range(1, 31):  # 用户编号从 1 到 30
        # 每个用户的实验次数，用户10执行3次实验，其余用户执行2次实验
        num_experiments = 3 if user_id == 10 else 2

        # 遍历当前用户的实验
        for exp_offset in range(num_experiments):
            experiment_id = (user_id - 1) * 2 + 1 + exp_offset

            # 根据用户编号和实验编号找到对应的传感器文件和陀螺仪文件
            acc_file = os.path.join(raw_data_path,
                                    f"acc_exp{str(experiment_id).zfill(2)}_user{str(user_id).zfill(2)}.txt")
            gyro_file = os.path.join(raw_data_path,
                                     f"gyro_exp{str(experiment_id).zfill(2)}_user{str(user_id).zfill(2)}.txt")

            # 如果文件不存在，直接跳过
            if not os.path.exists(acc_file) or not os.path.exists(gyro_file):
                continue

            # 加载传感器数据
            acc_data = pd.read_csv(acc_file, sep=" ", header=None).values
            gyro_data = pd.read_csv(gyro_file, sep=" ", header=None).values

            # 对每个传感器的通道进行Z-Score标准化
            def zscore_normalize(data):
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                return (data - mean) / std

            # 对加速度计和陀螺仪数据进行标准化
            acc_data = zscore_normalize(acc_data)
            gyro_data = zscore_normalize(gyro_data)

            # 合并传感器数据
            data = np.hstack([acc_data, gyro_data])

            # 查找该用户和实验对应的标签范围
            user_labels = labels_df[(labels_df["exp"] == experiment_id) & (labels_df["user"] == user_id)]

            # 获取标签范围的起点和终点
            if user_labels.empty:
                continue

            start = user_labels["start"].min()
            end = user_labels["end"].max()

            # 限制数据范围到有效标签区间
            data = data[start:end]
            labels = np.zeros(len(data), dtype=np.int64)

            # 根据标签文件为区间内数据分配标签
            for _, row in user_labels.iterrows():
                activity_start = max(0, row["start"] - start)
                activity_end = min(len(data), row["end"] - start)
                labels[activity_start:activity_end] = row["activity"] - 1

            # 滑动窗口处理
            features, window_labels = create_windows(data, labels, window_length, window_shift)

            # 确定数据集类型
            if user_id in user_splits["train"]:
                split = "train"
            elif user_id in user_splits["val"]:
                split = "val"
            elif user_id in user_splits["test"]:
                split = "test"
            else:
                continue

            # 检查数据有效性后再添加
            if features.shape[0] > 0 and window_labels.shape[0] > 0:
                # print(f"Features shape: {features.shape}, Labels shape: {window_labels.shape}")
                datasets[split].append((features, window_labels))

    # 数据分布统计和过采样逻辑
    for split, data in datasets.items():
        if len(data) == 0:
            continue

        features = np.concatenate([d[0] for d in data], axis=0)
        labels = np.concatenate([d[1] for d in data], axis=0)

        # 打印原始数据分布
        print(f"{split} dataset before oversampling: {Counter(labels)}")

        if split == "train":  # 仅对训练集进行过采样
            label_counts = Counter(labels)
            max_count = max(label_counts.values())

            # 对每个类别进行过采样
            resampled_features, resampled_labels = [], []
            for label, count in label_counts.items():
                label_features = features[labels == label]
                label_labels = labels[labels == label]

                # 计算目标数量（限制扩充在 max_oversample_factor 倍以内）
                target_count = min(count * max_oversample_factor, max_count)

                if count < target_count:
                    # 数据增强函数
                    def augment_data(sample):
                        augmentations = [
                            lambda x: x + np.random.normal(0, 0.01, x.shape),  # 添加噪声
                            lambda x: x * np.random.uniform(0.9, 1.1, x.shape),  # 随机缩放
                            lambda x: np.roll(x, shift=random.randint(-5, 5), axis=0),  # 时间轴滚动
                        ]
                        return random.choice(augmentations)(sample)

                    # 增强和过采样
                    additional_samples = resample(label_features, replace=True, n_samples=target_count - count,
                                                  random_state=42)
                    additional_samples = np.array([augment_data(sample) for sample in additional_samples])

                    resampled_features.append(additional_samples)
                    resampled_labels.append(np.full(additional_samples.shape[0], label))

            # 合并原始数据和过采样数据
            if resampled_features:
                features = np.concatenate([features] + resampled_features, axis=0)
                labels = np.concatenate([labels] + resampled_labels, axis=0)

            # 打印过采样后的数据分布
            print(f"{split} dataset after oversampling: {Counter(labels)}")

        else:  # 验证和测试集保持原始分布
            print(f"{split} dataset remains unchanged: {Counter(labels)}")

        # 写入 TFRecord 文件
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

        # 将 features 和 labels 解析为张量，并强制转换为 tf.float32 和 tf.int64
        features = tf.io.parse_tensor(example["features"], out_type=tf.float32)  # 强制转换为 tf.float32
        labels = tf.io.parse_tensor(example["labels"], out_type=tf.int64)  # 确保标签类型正确
        return features, labels

    # 加载训练、验证和测试数据
    train_files = tf.io.gfile.glob(f"{data_dir}/train.tfrecord")
    val_files = tf.io.gfile.glob(f"{data_dir}/val.tfrecord")
    test_files = tf.io.gfile.glob(f"{data_dir}/test.tfrecord")

    ds_train = tf.data.TFRecordDataset(train_files).map(parse_example).shuffle(1000).batch(batch_size).prefetch(
        tf.data.AUTOTUNE)
    ds_val = tf.data.TFRecordDataset(val_files).map(parse_example).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = tf.data.TFRecordDataset(test_files).map(parse_example).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 数据集元信息
    ds_info = {
        "features": {"shape": (window_length, 6)},  # 每个样本有6个通道（加速度计和陀螺仪）
        "label": {"num_classes": 12},
    }
    return ds_train, ds_val, ds_test, ds_info


if __name__ == "__main__":
    # 配置路径和参数
    raw_data_path = "D:/HAPTDataSet/RawData"
    labels_path = "D:/HAPTDataSet/RawData/labels.txt"
    output_path = "D:/HAPTDataSet/TFRecords"
    user_splits = {
        "train": list(range(1, 22)),
        "val": list(range(28, 31)),
        "test": list(range(22, 28)),
    }
    # 运行数据处理
    process_data(raw_data_path, labels_path, output_path, user_splits)