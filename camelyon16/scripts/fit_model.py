from training import callbacks, model, utils
import tensorflow as tf
import os
from pathlib import Path


def fit_single_res_model(train_ds, val_ds,
                         hyper_parameters,
                         optimizer,
                         freeze_conv_layers,
                         epoch):
    assert "level" in hyper_parameters, "Must include the zoom level parameter \"level\" in hyper-parameter dict"

    single_res_model = model.construct_single_resolution_model(optimizer=optimizer,
                                                               freeze_conv_layers=freeze_conv_layers)

    log_path, checkpoint_path, ts = utils.construct_paths_using_timestamp(hyper_parameters)

    callback_list = [callbacks.get_checkpoint_callback(checkpoint_path),
                     callbacks.get_early_stopping_callback(),
                     callbacks.get_tensorboard_callback(log_path)]

    history = single_res_model.fit(train_ds,
                                   epochs=epoch,
                                   validation_data=val_ds,
                                   callbacks=callback_list)

    return history


if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    os.chdir(script_path.parent.parent.parent)  # change to the project root

    args = utils.parse_for_runtime_arguments()
    LEVEL = args.level
    initial_lr = args.lr
    freeze = args.freeze

    train_patch_info_path = f"./data/training/level_{LEVEL}/info.csv"
    val_patch_info_path = f"./data/validation/level_{LEVEL}/info.csv"

    if not freeze:
        utils.config_gpu_memory(4096)
    else:
        utils.config_gpu_memory(1024)

    train_ds, val_ds = utils.construct_datasets(train_patch_info_path, val_patch_info_path,
                                                mini_batch_counts=-1)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, 10000, 0.5, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    fit_single_res_model(train_ds,
                         val_ds,
                         {"level": LEVEL, "freeze": freeze, "lr": initial_lr},
                         optimizer=optimizer,
                         epoch=30,
                         freeze_conv_layers=freeze)
