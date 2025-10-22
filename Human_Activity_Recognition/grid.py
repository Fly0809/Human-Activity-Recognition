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

# Hyperparameter search space, including sliding window length
hyperparameter_space = {
    "lstm_units": [256],  # Number of LSTM hidden units
    "learning_rate": [0.004, 0.01],  # learning rate
    "dropout_rate": [0.5],  # Dropout rate
    "batch_size": [64],  # batch size
    "window_length": [150, 200, 250]  # window length
}


def main(argv):
    # Data path
    raw_data_path = "/home/data/HAPT_dataset/RawData"
    labels_path = "/home/data/HAPT_dataset/RawData/labels.txt"
    total_steps = 10000
    log_interval = 1000
    ckpt_interval = 1000
    fc = 10  # Filter parameters

    # Record the best model and hyperparameters
    best_model = None
    best_hyperparams = None
    best_val_accuracy = 0

    # Iterate over all hyperparameter combinations
    for lstm_units, learning_rate, dropout_rate, batch_size, window_length in product(
            hyperparameter_space["lstm_units"],
            hyperparameter_space["learning_rate"],
            hyperparameter_space["dropout_rate"],
            hyperparameter_space["batch_size"],
            hyperparameter_space["window_length"]):

        # Current hyperparameter combination
        current_hyperparams = {
            "lstm_units": lstm_units,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "window_length": window_length
        }

        # Generate the run path
        run_paths = utils_params.gen_run_folder(hyperparams=current_hyperparams)

        # Configure logging
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # Clear and unlock the Gin configuration
        gin.clear_config()

        # Dynamically bind hyperparameters
        gin.bind_parameter('lstm_model.lstm_units', lstm_units)
        gin.bind_parameter('lstm_model.dropout_rate', dropout_rate)

        # Parse the configuration
        gin.parse_config_files_and_bindings(['configs/config.gin'], [])
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # Preprocess the data and generate TFRecord files
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

        # Load the dataset
        ds_train, ds_val, ds_test, ds_info = datasets.load(
            data_dir=run_paths['path_tfrecord'],
            batch_size=batch_size,
            window_length=window_length,
            fc=fc
        )

        input_shape = ds_info['features']['shape']  # (window_length, 6)
        n_classes = ds_info['label']['num_classes']  # 12

        # Build the model
        model = lstm_model(
            input_shape=input_shape,
            n_classes=n_classes,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )

        # Configure the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Print the current hyperparameter combination
        print(f"Training with: {current_hyperparams}")

        # Create  Trainer
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

        # Train the model
        for val_accuracy in trainer.train():
            print(f"Validation Accuracy: {val_accuracy * 100:.4f}")

            # Update the best model and hyperparameters
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model
                best_hyperparams = current_hyperparams

    # Print the best hyperparameters and validation accuracy
    print("Best Hyperparameters:", best_hyperparams)
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.4f}")

    # Save the best model
    if best_model:
        best_model.save(run_paths['path_ckpts_train'] + "/best_model")

    # Evaluate the best model on the test set
    if FLAGS.train is False:
        evaluate(best_model, run_paths['path_ckpts_train'] + "/best_model", ds_test, ds_info, run_paths)


if __name__ == "__main__":
    app.run(main)