import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import pandas as pd
from functools import partial


def resample_label_df(label_df, sampling_strategy):
    if sampling_strategy == "both":
        label_df = label_df.copy()
        tumor_patch_count = label_df["has_tumor"].sum()
        label_df["reserve"] = 1
        label_df.loc[label_df.has_tumor, "reserve"] = np.random.binomial(1, 0.25, size=(tumor_patch_count,))
        label_df = label_df[label_df.reserve == 1]
        print(f"Only use 25% of negative patches.")

    if sampling_strategy == "pos_only" or sampling_strategy == "both":
        patch_count = len(label_df)
        tumor_patch_count = label_df["has_tumor"].sum()
        pos_preserve_rate = 2 * tumor_patch_count / (patch_count - tumor_patch_count)
        print(f"Using default sampling strategy, pair two pos patches with one neg patches.")

    else:
        # return dataframe without sampling
        return label_df

    info_df = label_df.copy()
    info_df["reserve"] = np.random.binomial(1, min(pos_preserve_rate, 1), size=(len(info_df),))
    info_df.loc[info_df.has_tumor, "reserve"] = 1
    info_df = info_df[info_df.reserve == 1]
    tumor_patch_count = info_df["has_tumor"].sum()
    patch_count = len(info_df)
    print(f"pos patches / neg patches: {patch_count / tumor_patch_count - 1:.4f}.")
    print(f"total patches count: {patch_count}.")
    return info_df[label_df.columns].reset_index()


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
    image = tf.image.random_brightness(image, 64 / 255)
    image = tf.image.random_saturation(image, 1 - 0.25, 1 + 0.25)
    image = tf.image.random_hue(image, 0.04)
    image = tf.image.random_contrast(image, 1 - 0.75, 1 + 0.75)
    return image, label


def multi_resolution_wrapper(map_func, paths, label):
    images = [map_func(path, label)[0] for path in paths]
    return images, label


def display_processed_img(image):
    image = image.numpy()
    image = image / 2 + 0.5
    image[image < 0] = 0
    image[image > 1] = 1
    plt.imshow(image)


def create_dataset_from_merged_dataframes(high_res_df, low_res_df):
    merged_df = high_res_df.merge(low_res_df,
                                  on=["row_id", "col_id", "slide_name"],
                                  suffixes=["_high", "_low"],
                                  how="inner",
                                  validate="one_to_one")

    high_res_paths = merged_df.patch_path_high
    low_res_paths = merged_df.patch_path_low
    labels = merged_df.has_tumor_high

    img_dataset = tf.data.Dataset.from_tensor_slices((high_res_paths, low_res_paths))
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((img_dataset, label_dataset))
    return dataset


def get_dataset(label_df: pd.DataFrame,
                is_training,
                low_res_label_df: pd.DataFrame = None,
                sampling_strategy="pos_only",
                batch_size=32,
                shuffle_buffer_size=20000):
    sampled_df = resample_label_df(label_df, sampling_strategy)
    if is_training:
        sampled_df = sampled_df.sample(frac=1, replace=False)  # shuffle here instead of using a huge buffer size

    if low_res_label_df is not None:
        # construct multi-resolution dataset
        # join different resolutions
        selected_columns = ["row_id", "col_id", "patch_path", "slide_name", "has_tumor"]
        sampled_df = sampled_df[selected_columns]
        low_res_label_df = low_res_label_df[selected_columns]
        dataset = create_dataset_from_merged_dataframes(sampled_df, low_res_label_df)
    else:
        paths, labels = sampled_df.patch_path, sampled_df.has_tumor
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    print(f"Constructing {'training' if is_training else 'validation'} dataset "
          f"with {len(sampled_df)} patches.")

    mapping_func = load_img_with_preprocessing

    if is_training:
        dataset = dataset.shuffle(shuffle_buffer_size)
        mapping_func = load_img_with_augmentation

    if low_res_label_df is not None:
        mapping_func = partial(multi_resolution_wrapper, mapping_func)

    dataset = (dataset
               .map(mapping_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .prefetch(tf.data.experimental.AUTOTUNE)
               .batch(batch_size))

    return dataset


def get_eval_dataset(label_df: pd.DataFrame,
                     low_res_label_df: pd.DataFrame = None,
                     batch_size=128,
                     verbose=False):
    if verbose: print(f"Constructing evaluation dataset with {len(label_df)} patches.")

    if low_res_label_df is not None:
        dataset = create_dataset_from_merged_dataframes(label_df, low_res_label_df)
        mapping_func = partial(multi_resolution_wrapper, load_img_with_preprocessing)
    else:
        mapping_func = load_img_with_preprocessing
        dataset = tf.data.Dataset.from_tensor_slices((label_df.patch_path, label_df.has_tumor))

    dataset = (dataset
               .map(mapping_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .prefetch(tf.data.experimental.AUTOTUNE)
               .batch(batch_size))

    return dataset


def show_next_batch(generator, return_shape=True):
    batch = next(iter(generator))
    return [b.shape for b in batch] if return_shape else batch
