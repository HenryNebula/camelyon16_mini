import tensorflow as tf
from tensorflow import keras


PATCH_SHAPE = (299, 299, 3)


def get_inception_model(freeze_conv_layers):
    model = keras.applications.InceptionV3(include_top=False,
                                           weights='imagenet',
                                           input_shape=PATCH_SHAPE)

    if freeze_conv_layers:
        for layer in model.layers:
            layer.trainable = False

    inception_model = keras.models.Sequential()
    inception_model.add(model)
    inception_model.add(keras.layers.GlobalAveragePooling2D())
    inception_model.add(keras.layers.Dense(128, activation='relu'))
    return inception_model


def construct_single_resolution_model(optimizer,
                                      freeze_conv_layers,
                                      loss=keras.losses.SparseCategoricalCrossentropy(),
                                      metrics=None):
    if metrics is None:
        metrics = ["sparse_categorical_accuracy", ]

    single_resolution_model = get_inception_model(freeze_conv_layers)
    single_resolution_model.add(keras.layers.Dense(2, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    single_resolution_model.summary()

    single_resolution_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return single_resolution_model


def construct_multi_resolution_model(optimizer,
                                     freeze_conv_layers,
                                     loss=keras.losses.SparseCategoricalCrossentropy(),
                                     metrics=None):
    if metrics is None:
        metrics = ["sparse_categorical_accuracy", ]

    input = keras.layers.Input(shape=(2, *PATCH_SHAPE))
    high_res_input, low_res_input = keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(input)
    high_res_input = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(high_res_input)
    low_res_input = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(low_res_input)

    high_res_inc_model = get_inception_model(freeze_conv_layers)
    low_res_inc_model = get_inception_model(freeze_conv_layers)

    high_res_dense = high_res_inc_model(high_res_input)
    low_res_dense = low_res_inc_model(low_res_input)

    concatenated = keras.layers.concatenate([high_res_dense, low_res_dense])
    output = keras.layers.Dense(2, activation="softmax")(concatenated)

    multi_resolution_model = keras.Model(inputs=input, outputs=output)
    multi_resolution_model.summary()

    multi_resolution_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return multi_resolution_model
