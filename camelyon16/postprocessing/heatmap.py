import matplotlib.cm as cm
from functools import partial
from PIL import Image
from .features import extract_features


def color_mapping_func(val, cmap, alpha=1.0):
    colors = [int(255*c) for c in cmap(val)]
    colors[-1] = int(255*alpha)
    return tuple(colors)


def generate_heatmap(slide_name,
                     meta_info,
                     patches_info,
                     input_level,
                     output_level,
                     patch_proba=None,
                     patch_size=299,
                     cmap=cm.viridis):

    patches_info_for_slide = patches_info[patches_info.slide_name == slide_name + ".tif"]

    if patch_proba is None:
        # use the ground truth
        patch_proba = patches_info_for_slide.has_tumor

    slide_info = meta_info[meta_info.id == slide_name]
    slide_shape = slide_info[f"Level_{input_level}"].to_list()[0][1]

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


def extract_heatmap_features_from_ground_truth(slide_path):
    pass