import os
import datetime

import os


def gen_run_folder(path_model_id='default', hyperparams=None):

    run_paths = {}

    if not os.path.isdir(path_model_id):
        # 根目录路径
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments2'))

        if hyperparams:
            # 如果提供了超参数字典，生成基于超参数的唯一运行路径
            run_id = "run_" + "_".join([f"{key}-{value}" for key, value in hyperparams.items()])
        else:
            # 默认路径使用 'default'
            run_id = 'default'

        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        # 如果传入的 path_model_id 是一个有效路径，直接使用它
        run_paths['path_model_id'] = path_model_id

    # 构建文件路径
    run_paths['model_id'] = run_paths['path_model_id'].split(os.sep)[-1]
    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'run.log')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    run_paths['path_tfrecord'] = os.path.join(run_paths['path_model_id'], 'tfrecord')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    # 创建目录和文件
    for k, v in run_paths.items():
        if "path_" in k and not os.path.exists(os.path.dirname(v)):
            os.makedirs(os.path.dirname(v), exist_ok=True)

    return run_paths


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)
