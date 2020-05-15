import matplotlib.cm as cm
from functools import partial
from PIL import Image
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from camelyon16.training.model import construct_multi_resolution_model, construct_single_resolution_model
from camelyon16.preprocessing.dataset import get_eval_dataset


def color_mapping_func(val, cmap, alpha=1.0):
    colors = [int(255 * c) for c in cmap(val)]
    colors[-1] = int(255 * alpha)
    return tuple(colors)


def generate_heatmap(meta_info_for_slide,
                     patches_info_for_slide,
                     input_level,
                     output_level,
                     patch_proba=None,
                     patch_size=299,
                     cmap=cm.viridis):
    if patch_proba is None:
        # use the ground truth
        patch_proba = patches_info_for_slide.has_tumor

    slide_shape = meta_info_for_slide[f"Level_{input_level}"].to_list()[0][1]

    downsample_ratio = 2 ** (output_level - input_level)
    heatmap_shape = tuple(map(lambda x: int(x / downsample_ratio), slide_shape))
    new_patch_size = int(patch_size / downsample_ratio)

    color_mapping = partial(color_mapping_func, cmap=cmap)
    heatmap = Image.new("RGBA", heatmap_shape, color_mapping(0.0))
    heatmap_greyscale = Image.new("L", heatmap_shape)

    for offset_x, offset_y, prob in zip(patches_info_for_slide.offset_x,
                                        patches_info_for_slide.offset_y,
                                        patch_proba):
        heat = color_mapping(float(prob))
        downsampled_patch = Image.new("RGBA", (new_patch_size, new_patch_size), heat)
        downsampled_grayscale_patch = Image.new("L", (new_patch_size, new_patch_size), int(prob * 255))
        offset = int(offset_x / downsample_ratio), int(offset_y / downsample_ratio)
        heatmap.paste(downsampled_patch, offset)
        heatmap_greyscale.paste(downsampled_grayscale_patch, offset)

    return heatmap, heatmap_greyscale


def init_model(model_name):
    # load model lazily
    optimizer_template = tf.keras.optimizers.Adam()
    if "multi" in model_name:
        model = construct_multi_resolution_model(optimizer_template, freeze_conv_layers=False)
    else:
        model = construct_single_resolution_model(optimizer_template, freeze_conv_layers=False)
    model_weights_path = f"./models/{model_name}.hdf5"
    assert Path(model_weights_path).exists(), f"Model weights file doesn't exists in {model_weights_path}"
    model.load_weights(model_weights_path)
    return model


def batch_generate_heatmap(meta_info,
                           label_info,
                           dataset_name,
                           model_name,
                           low_res_label_info=None,
                           input_level=1,
                           output_level=5,
                           batch_size=128,
                           overwrite=False):
    model = None

    if "multi" in model_name:
        assert low_res_label_info is not None, "need to input low resolution label dataframes"

    slide_names = label_info.slide_name.unique()
    for slide_name in tqdm(slide_names):
        result_dir = Path(f"./results/{model_name}/{dataset_name}/{slide_name}")
        result_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = result_dir / "heatmap.bmp"
        heatmap_grey_path = result_dir / "heatmap_grey.bmp"

        if not overwrite:
            if heatmap_path.exists() and heatmap_grey_path.exists():
                # use saved results
                continue

        if model is None:
            model = init_model(model_name)

        patches_info_for_slide = label_info[label_info.slide_name == slide_name]
        low_res_patches_info_for_slide = low_res_label_info[low_res_label_info.slide_name == slide_name] \
            if low_res_label_info is not None else None
        slide_info = meta_info[meta_info.id == slide_name.replace(".tif", "")]

        ds = get_eval_dataset(patches_info_for_slide, low_res_patches_info_for_slide, batch_size)
        proba = model.predict(ds)[:, 1]
        heatmap, heatmap_grey = generate_heatmap(meta_info_for_slide=slide_info,
                                                 patches_info_for_slide=patches_info_for_slide,
                                                 input_level=input_level,
                                                 output_level=output_level,
                                                 patch_proba=proba)

        heatmap.save(heatmap_path)
        heatmap_grey.save(heatmap_grey_path)
