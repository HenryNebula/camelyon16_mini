import tensorflow as tf
from pathlib import Path


def get_early_stopping_callback(metric="val_sparse_categorical_accuracy",
                                patience=5,
                                save_best_weights=False):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                           patience=patience,
                                                           restore_best_weights=save_best_weights)
    return early_stop_callback


def get_checkpoint_callback(path,
                            metric="val_sparse_categorical_accuracy"):

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                             monitor=metric,
                                                             save_best_only=True)

    return checkpoint_callback


def get_tensorboard_callback(path,
                             freq=50):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(path,
                                                          write_graph=False,
                                                          update_freq=freq,
                                                          profile_batch=0)
    return tensorboard_callback
