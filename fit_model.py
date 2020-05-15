from camelyon16.training import callbacks, model, utils
import tensorflow as tf
import os
from pathlib import Path


def fit_model(train_ds,
              val_ds,
              hyper_parameters,
              optimizer,
              freeze_conv_layers,
              epoch,
              model_type):
    assert "level" in hyper_parameters, "Must include the zoom level parameter \"level\" in hyper-parameter dict"

    if model_type == "single":
        nn_model = model.construct_single_resolution_model(optimizer=optimizer,
                                                           freeze_conv_layers=freeze_conv_layers)
    else:
        nn_model = model.construct_multi_resolution_model(optimizer=optimizer,
                                                          freeze_conv_layers=freeze_conv_layers)

    log_path, checkpoint_path, ts = utils.construct_paths_using_timestamp(hyper_parameters)

    callback_list = [callbacks.get_checkpoint_callback(checkpoint_path),
                     callbacks.get_early_stopping_callback(),
                     callbacks.get_tensorboard_callback(log_path)]

    history = nn_model.fit(train_ds,
                           epochs=epoch,
                           validation_data=val_ds,
                           callbacks=callback_list)

    return history


if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    os.chdir(script_path.parent.parent.parent)  # change to the project root

    args = utils.parse_for_runtime_arguments()
    level = args.level
    initial_lr = args.lr
    freeze = args.freeze
    use_neighbor = args.use_neighbor
    zoom_level = args.zoom_level
    use_zoom_in = zoom_level > 1
    model_type = args.model_type
    memory = args.gpu_memory

    train_patch_info_path = f"./data/training/level_{level}/info.csv" if not use_neighbor else \
        f"./data/training_neighbor/level_{level}/info.csv"
    train_zoom_in_patch_info_path = f"./data/training/level_{level}/zoom_level_{zoom_level}/info.csv"

    val_patch_info_path = f"./data/validation/level_{level}/info.csv"
    val_zoom_in_patch_info_path = f"./data/validation/level_{level}/zoom_level_{zoom_level}/info.csv"

    batch_size = 32

    if model_type == "single" and use_zoom_in:
        train_patch_info_path = train_zoom_in_patch_info_path
        val_patch_info_path = val_zoom_in_patch_info_path

    if model_type == "multi":
        batch_size = 32
    else:
        train_zoom_in_patch_info_path = None
        val_zoom_in_patch_info_path = None

    utils.config_gpu_memory(memory)

    strategy = "pos_only" if level > 0 else "both"
    train_ds, val_ds = utils.construct_datasets(train_patch_info_path,
                                                val_patch_info_path,
                                                train_low_res_patches_info_path=train_zoom_in_patch_info_path,
                                                val_low_res_patches_info_path=val_zoom_in_patch_info_path,
                                                batch_size=batch_size,
                                                mini_batch_counts=-1,
                                                strategy=strategy)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, 10000, 0.5, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model_params = {
        "level": level,
        "freeze": freeze,
        "lr": initial_lr,
        "use_neighbor": use_neighbor,
        "zoom_level": zoom_level,
        "model_type": model_type,
    }

    print(model_params)

    fit_model(train_ds,
              val_ds,
              model_params,
              optimizer=optimizer,
              epoch=30,
              freeze_conv_layers=freeze,
              model_type=model_type)
