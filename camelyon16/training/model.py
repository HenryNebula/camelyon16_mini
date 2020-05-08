from tensorflow import keras


def construct_single_resolution_model(optimizer,
                                      freeze_conv_layers,
                                      loss=keras.losses.SparseCategoricalCrossentropy(),
                                      metrics=None):
    if metrics is None:
        metrics = ["sparse_categorical_accuracy", ]

    model = keras.applications.InceptionV3(include_top=False, weights='imagenet')

    if freeze_conv_layers:
        for layer in model.layers:
            layer.trainable = False

    single_resolution_model = keras.models.Sequential()

    single_resolution_model.add(model)
    single_resolution_model.add(keras.layers.GlobalAveragePooling2D())
    single_resolution_model.add(keras.layers.Dense(128, activation='relu'))
    single_resolution_model.add(keras.layers.Dense(2, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    single_resolution_model.summary()

    single_resolution_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return single_resolution_model
