from . import dataset
import tensorflow as tf
from datetime import datetime
import pandas as pd
import argparse


TIMESTAMP_FORMAT = "%b-%d-%Y-%H:%M:%S"


def construct_paths_using_timestamp(model_params):
    ts = datetime.now().strftime(TIMESTAMP_FORMAT)
    hyper_str = "_".join([f"{k}-{model_params[k]}" for k in model_params])
    model_spec = f"single_res/{hyper_str}/{ts}"
    log_path = f"./logs/{model_spec}"
    checkpoint_path = f"./checkpoints/{model_spec}/"
    checkpoint_path += "epoch-{epoch:02d}-val_acc-{val_sparse_categorical_accuracy:.2f}.hdf5"

    return log_path, checkpoint_path, ts


def config_gpu_memory(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=memory_limit)])


def construct_datasets(train_patches_info_path,
                       val_patches_info_path,
                       mini_batch_counts=-1):
    train_patches_info = pd.read_csv(train_patches_info_path)
    train_ds = dataset.get_dataset(train_patches_info, is_training=True)

    val_patches_info = pd.read_csv(val_patches_info_path)
    val_ds = dataset.get_dataset(val_patches_info, is_training=False)

    assert mini_batch_counts == -1 or mini_batch_counts > 0, \
        "mini_batch_counts must be -1 or a positive number."

    print(f"Dataset Sampling: {'None' if mini_batch_counts == -1 else str(mini_batch_counts) + ' batches'}")

    if mini_batch_counts > 0:
        train_ds = train_ds.take(mini_batch_counts)
        val_ds = val_ds.take(mini_batch_counts)
    return train_ds, val_ds


def parse_for_runtime_arguments():
    parser = argparse.ArgumentParser(description='Process runtime arguments.')
    parser.add_argument('--level', default=1, type=int, help='zoom level of WSI slides')
    parser.add_argument('--lr', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--freeze', default=False, type=bool, help='freeze pretrained conv layers or not')

    args = parser.parse_args()
    return args

