import gin
import logging
from absl import app, flags
import tensorflow as tf
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import lstm_model
from itertools import product

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

# 超参数搜索空间，包括滑动窗口长度
hyperparameter_space = {
    "lstm_units": [256],  # LSTM 隐藏单元数量
    "learning_rate": [0.004, 0.01],  # 学习率
    "dropout_rate": [0.5],  # Dropout 比例
    "batch_size": [64],  # 批量大小
    "window_length": [150, 200, 250]  # 滑动窗口长度
}


def main(argv):
    # 数据路径
    raw_data_path = "/home/data/HAPT_dataset/RawData"
    labels_path = "/home/data/HAPT_dataset/RawData/labels.txt"
    total_steps = 10000  # 总训练步数
    log_interval = 1000  # 日志间隔
    ckpt_interval = 1000  # 检查点保存间隔
    fc = 10  # 滤波器参数

    # 记录最佳模型和超参数
    best_model = None
    best_hyperparams = None
    best_val_accuracy = 0

    # 遍历所有超参数组合
    for lstm_units, learning_rate, dropout_rate, batch_size, window_length in product(
            hyperparameter_space["lstm_units"],
            hyperparameter_space["learning_rate"],
            hyperparameter_space["dropout_rate"],
            hyperparameter_space["batch_size"],
            hyperparameter_space["window_length"]):

        # 当前超参数组合
        current_hyperparams = {
            "lstm_units": lstm_units,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "window_length": window_length
        }

        # 生成运行路径
        run_paths = utils_params.gen_run_folder(hyperparams=current_hyperparams)

        # 设置日志
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # 清除并解锁 Gin 配置
        gin.clear_config()

        # 动态绑定超参数
        gin.bind_parameter('lstm_model.lstm_units', lstm_units)
        gin.bind_parameter('lstm_model.dropout_rate', dropout_rate)

        # 解析配置
        gin.parse_config_files_and_bindings(['configs/config.gin'], [])
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # 预处理数据，生成 TFRecord
        print(f"Preprocessing data with window_length={window_length}...")
        datasets.process_data(
            raw_data_path=raw_data_path,
            labels_path=labels_path,
            output_path=run_paths['path_tfrecord'],
            user_splits={
                "train": list(range(1, 22)),
                "val": list(range(28, 31)),
                "test": list(range(22, 28)),
            },
            window_length=window_length,
            window_shift=window_length // 2,
            fc=fc
        )
        print(f"TFRecord files saved to {run_paths['path_tfrecord']}")

        # 加载数据集
        ds_train, ds_val, ds_test, ds_info = datasets.load(
            data_dir=run_paths['path_tfrecord'],
            batch_size=batch_size,
            window_length=window_length,
            fc=fc
        )

        input_shape = ds_info['features']['shape']  # (window_length, 6)
        n_classes = ds_info['label']['num_classes']  # 12

        # 构建模型
        model = lstm_model(
            input_shape=input_shape,
            n_classes=n_classes,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )

        # 配置优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # 打印当前超参数组合
        print(f"Training with: {current_hyperparams}")

        # 创建 Trainer 对象
        trainer = Trainer(
            model=model,
            ds_train=ds_train,
            ds_val=ds_val,
            ds_info=ds_info,
            run_paths=run_paths,
            total_steps=total_steps,
            log_interval=log_interval,
            ckpt_interval=ckpt_interval
        )

        # 训练模型
        for val_accuracy in trainer.train():
            print(f"Validation Accuracy: {val_accuracy * 100:.4f}")

            # 更新最佳模型和超参数
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model
                best_hyperparams = current_hyperparams

    # 打印最佳超参数和验证集准确率
    print("Best Hyperparameters:", best_hyperparams)
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.4f}")

    # 保存最佳模型
    if best_model:
        best_model.save(run_paths['path_ckpts_train'] + "/best_model")

    # 使用最佳模型在测试集上评估
    if FLAGS.train is False:
        evaluate(best_model, run_paths['path_ckpts_train'] + "/best_model", ds_test, ds_info, run_paths)


if __name__ == "__main__":
    app.run(main)