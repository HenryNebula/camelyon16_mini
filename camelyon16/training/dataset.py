import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from random import randint


def resample_label_df(label_df, sampling_strategy):
    if sampling_strategy == "auto":
        patch_count = len(label_df)
        tumor_patch_count = label_df["has_tumor"].sum()
        pos_preserve_rate = 2 * tumor_patch_count / (patch_count - tumor_patch_count)
        print(f"Using default sampling strategy, pair two pos patches with one neg patches.")
    else:
        # TODO: add other strategy
        # return dataframe without sampling
        return label_df

    info_df = label_df.copy()
    info_df["reserve"] = np.random.binomial(1, pos_preserve_rate, size=(len(info_df),))
    info_df.loc[info_df.has_tumor, "reserve"] = 1
    info_df = info_df[info_df.reserve == 1]
    tumor_patch_count = info_df["has_tumor"].sum()
    patch_count = len(info_df)
    print(f"pos patches/ neg patches: {patch_count / tumor_patch_count - 1:.4f}.")
    print(f"total patches count: {patch_count}.")
    return info_df[["patch_path", "has_tumor"]].reset_index()


def load_img_with_preprocessing(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_bmp(img, channels=3)
    img_with_preprocessing = tf.keras.applications.inception_v3.preprocess_input(tf.cast(img, tf.float32))
    label = 1 if label else 0
    return img_with_preprocessing, label


def load_img_with_augmentation(path, label):
    image, label = load_img_with_preprocessing(path, label)
    rotation = randint(0, 3)
    image = tf.image.rot90(image, k=rotation)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 64/255)
    image = tf.image.random_saturation(image, 1-0.25, 1+0.25)
    image = tf.image.random_hue(image, 0.04)
    image = tf.image.random_contrast(image, 1-0.75, 1+0.75)
    return image, label


def display_processed_img(image):
    image = image.numpy()
    image = image / 2 + 0.5
    image[image < 0] = 0
    image[image > 1] = 1
    plt.imshow(image)


def get_dataset(label_df,
                is_training,
                sampling_strategy="auto",
                batch_size=32,
                shuffle_buffer_size=20000):

    sampled_df = resample_label_df(label_df, sampling_strategy)

    if is_training:
        sampled_df = sampled_df.sample(frac=1) # shuffle here instead of using a huge buffer size

    paths, labels = sampled_df.patch_path, sampled_df.has_tumor
    dataset_size = len(paths)

    print(f"Constructing {'training' if is_training else 'validation'} dataset with {dataset_size} patches.")
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = (dataset.shuffle(shuffle_buffer_size)
                          .map(load_img_with_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    else:
        dataset = dataset.map(load_img_with_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = (dataset
               .prefetch(tf.data.experimental.AUTOTUNE)
               .batch(batch_size))

    return dataset


def generate_next_batch(generator, return_shape=True):
    batch = next(iter(generator))
    return [b.shape for b in batch] if return_shape else batch
