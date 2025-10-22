import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import lstm_model



FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    data_dir = "D:/HAPTDataSet/TFRecords"
    batch_size = 32
    window_length = 250  # Consistent with data preprocessing
    fc = 10  # Filter parameters

    ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=data_dir, batch_size=batch_size, window_length=window_length, fc=fc)

    print("The training set has been loaded")


    input_shape = ds_info['features']['shape']  # (250, 6)
    n_classes = ds_info['label']['num_classes']  # 12
    model = lstm_model(input_shape=input_shape, n_classes=n_classes)

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths,)
        for _ in trainer.train():
            continue
    else:
        # checkpoint_path = tf.train.latest_checkpoint(run_paths['path_ckpts_train'])
        checkpoint_path = f"D:\\DLLAB\\dl-lab-24w-team08\\experiments2\\default\\ckpts\\ckpt_final-41"
        if not checkpoint_path:
            raise ValueError("No trained model checkpoint found for evaluation.")
        evaluate(model, checkpoint_path, ds_test, ds_info, run_paths)


if __name__ == "__main__":
    app.run(main)