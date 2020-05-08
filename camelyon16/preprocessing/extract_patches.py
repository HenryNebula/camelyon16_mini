from PIL import Image
from skimage.filters import threshold_otsu
from .slide_utils import read_slide, read_region_from_slide, read_full_slide_by_level, get_slide_and_mask_paths
import numpy as np
from itertools import product
import pandas as pd
import multiprocessing
from pathlib import Path
from functools import reduce


def crop_slide_as_patches(slide_path,
                          mask_path,
                          level,
                          pos_preserve_weight,
                          save_dir,
                          strategy="default",
                          size=299,
                          stride=128):

    slide = read_slide(slide_path)
    mask = read_slide(mask_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    thumbnail = read_full_slide_by_level(slide_path, 5).convert("HSV")  # use level 5 as thumbnail
    otsu_thresh = [threshold_otsu(np.asarray(thumbnail.getchannel(c)))
                   for c in ["H", "S", "V"]]

    level_dims = slide.level_dimensions[level]
    row_count, col_count = ((level_dims[0] - size) // stride + 1,
                            (level_dims[1] - size) // stride + 1,)

    info = []  # collect patch info

    print(f"Start cropping slide {slide_path.name}")

    for row_id, col_id in product(range(row_count), range(col_count)):

        offset_x, offset_y = row_id * stride, col_id * stride

        slide_patch = read_region_from_slide(slide, offset_x, offset_y, level,
                                             width=size, height=size)

        # detect tissue regions having pixels with H, S >= otsu_threshold
        hsv_patch = slide_patch.convert("HSV")
        channels = [np.asarray(hsv_patch.getchannel(c)) for c in ["H", "S"]]
        tissue_mask = np.bitwise_and(*[c >= otsu_thresh[i] for i, c in enumerate(channels)])

        if mask is not None:
            # for masks, only focus on the center 128 * 128 region
            diff = (size - stride) // 2
            center_offset = offset_x + diff, offset_y + diff

            mask_patch = read_region_from_slide(mask, center_offset[0], center_offset[1],
                                                level, width=stride, height=stride)
            # decide patch-level label
            has_tumor = np.any(np.asarray(mask_patch.getchannel(0)) > 0)
        else:
            has_tumor = False

        pos_preserve_flag = np.random.binomial(1, pos_preserve_weight)

        if has_tumor or pos_preserve_flag:
            # only save tumor patches or normal patches passing the random sampling
            if np.any(tissue_mask):
                patch_name = f"patch_row_{row_id:04d}_col_{col_id:04d}.bmp"
                patch_path = save_dir / patch_name
                slide_patch.save(patch_path)

                info_dict = {
                    "slide_name": slide_path.name,
                    "patch_path": patch_path,
                    "row_id": row_id, "col_id": col_id,
                    "offset_x": offset_x, "offset_y": offset_y,
                    "has_tumor": has_tumor
                }
                info.append(info_dict)

            else:
                if has_tumor:
                    # mistakenly discard a tumor region
                    print(f"[Warning] For slide {slide_path.name}, discard block {(row_id, col_id)} with tumors")

    return info


def crop_all_slides(data_dir, save_dir, level, pos_preserve_weight=1.0, workers=4):
    slide_paths, mask_paths = get_slide_and_mask_paths(data_dir)
    save_dirs = [Path(save_dir) / f"level_{level}/{str(slide_path.name).replace('.tif', '')}" for slide_path in slide_paths]
    levels = [level] * len(slide_paths)
    weights = [pos_preserve_weight] * len(slide_paths)

    with multiprocessing.Pool(workers) as p:
        info_list = p.starmap(crop_slide_as_patches,
                              zip(slide_paths, mask_paths, levels, weights, save_dirs))

    level_info = pd.DataFrame(reduce(lambda x, y: x + y, info_list))
    level_info.to_csv(Path(save_dir) / f"level_{level}/info.csv", index=False)
    return level_info


def calculate_patch_coverage(meta_info_df, level_info_df, level):
    shapes = map(lambda t: t[1], meta_info_df[f"Level_{level}"])
    size, stride = 299, 128
    total_counts = [ ((row_pixel - size) // stride + 1) * ((col_pixel - size) // stride + 1)
                    for row_pixel, col_pixel in shapes]
    actual_counts = level_info_df.groupby("slide_name")["has_tumor"].count().to_list()
    mean_coverage_by_sample = np.mean(np.array(actual_counts) / np.array(total_counts))
    total_mean = np.sum(actual_counts) / np.sum(total_counts)
    print(f"Total Mean:\t{total_mean:.4f}\t" + \
          f"Avg Mean by WSI:\t{mean_coverage_by_sample:.4f}\t" + \
          f"Total counts:\t{np.sum(total_counts)}")


def reconstruct_from_patch(meta_info_df, level_info_df, level, slide_id, resize_factor=None):
    if resize_factor is None:
        resize_factor = 10 * 4 ** (int(5 / level))

    size = meta_info_df.iloc[slide_id][f"Level_{level}"][1]
    img = Image.new('RGB', size)

    slide_name = meta_info_df.iloc[slide_id].id + ".tif"
    print(f"Visualizing slide {slide_name} under resize factor {resize_factor} ... ")
    patch_info = level_info_df[level_info_df.slide_name == slide_name]

    patch_paths = patch_info.patch_path.to_list()
    offsets = zip(patch_info.offset_x.to_list(), patch_info.offset_y.to_list())
    has_tumors = patch_info.has_tumor.to_list()
    tumor_mask = Image.new("RGBA", (299, 299), (173, 255, 47, 30))

    for patch_path, offset, has_tumor in zip(patch_paths, offsets, has_tumors):
        patch = Image.open(patch_path)
        if has_tumor:
            patch = Image.blend(patch.convert("RGBA"), tumor_mask, 0.5)
            img.paste(patch, box=offset)
        else:
            img.paste(patch, box=offset)

    return img.resize(map(lambda x: int(x / resize_factor), img.size))