import multiprocessing
from functools import reduce
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu

from .slide_utils import read_slide, read_region_from_slide, read_full_slide_by_level, get_slide_and_mask_paths, \
    get_connected_regions_from_tumor_slides

SIZE = 299
STRIDE = 128


def crop_slide_as_patches(slide_path,
                          mask_path,
                          level,
                          pos_preserve_weight,
                          save_dir,
                          strategy="default",
                          size=SIZE,
                          stride=STRIDE):
    def check_patch_include_tumor(x, y):
        # x, y indicates the left upper corner coordinates of the patch at current level
        if mask is not None:
            center_x, center_y = x + diff, y + diff
            mask_patch = read_region_from_slide(mask, center_x, center_y,
                                                level, width=stride, height=stride)
            # decide patch-level label
            has_tumor = np.any(np.asarray(mask_patch.getchannel(0)) > 0)
        else:
            has_tumor = False

        return has_tumor

    def check_patch_include_tissue(patch):
        # detect tissue regions having pixels with H, S >= otsu_threshold
        hsv_patch = patch.convert("HSV")
        channels = [np.asarray(hsv_patch.getchannel(c)) for c in ["H", "S"]]
        tissue_mask = np.bitwise_and(*[c >= otsu_thresh[i] for i, c in enumerate(channels)])
        return np.any(tissue_mask)

    def save_patch_to_disk(patch, rid, cid, has_tumor):
        patch_name = f"patch_row_{rid:04d}_col_{cid:04d}.bmp"
        x, y = rid * stride, cid * stride
        patch_path = save_dir / patch_name
        patch.save(patch_path)

        info = {
            "slide_name": slide_path.name,
            "patch_path": patch_path,
            "row_id": rid, "col_id": cid,
            "offset_x": x, "offset_y": y,
            "has_tumor": has_tumor
        }
        info_list.append(info)

    def batch_extract_patches(list_of_row_id_col_id, weight):

        for row_id, col_id in list_of_row_id_col_id:

            if (row_id, col_id) in saved_patches:
                continue

            offset_x, offset_y = row_id * stride, col_id * stride

            slide_patch = read_region_from_slide(slide, offset_x, offset_y, level,
                                                 width=size, height=size)

            tumor_flag = check_patch_include_tumor(offset_x, offset_y)
            pos_preserve_flag = np.random.binomial(1, weight) if 0 < weight < 1 else True

            if tumor_flag or pos_preserve_flag:
                # only save tumor patches or normal patches passing the random sampling
                if check_patch_include_tissue(slide_patch):
                    save_patch_to_disk(slide_patch, row_id, col_id, tumor_flag)
                    saved_patches.add((row_id, col_id))
                else:
                    if tumor_flag:
                        # mistakenly discard a tumor region
                        print(f"[Warning] For slide {slide_path.name}, "
                              f"discard block {(row_id, col_id)} with tumors")

    slide = read_slide(slide_path)
    mask = read_slide(mask_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    thumbnail = read_full_slide_by_level(slide_path, 5).convert("HSV")  # use level 5 as thumbnail
    otsu_thresh = [threshold_otsu(np.asarray(thumbnail.getchannel(c)))
                   for c in ["H", "S", "V"]]

    level_dims = slide.level_dimensions[level]
    row_count, col_count = ((level_dims[0] - size) // stride + 1,
                            (level_dims[1] - size) // stride + 1,)

    diff = (size - stride) // 2  # distance from patch bounding to corner rectangle

    info_list = []  # collect patch info
    saved_patches = set()

    print(f"Start cropping slide {slide_path.name}")

    if strategy == "default" or mask is None:
        list_of_coordinates = product(range(row_count), range(col_count))
        batch_extract_patches(list_of_coordinates, pos_preserve_weight)

    elif strategy == "neighborhood":
        tumor_bbox = get_connected_regions_from_tumor_slides(mask_path)
        zoom_factor = 2 ** (5 - level)  # todo: double check the mapping here
        for bbox in tumor_bbox:
            # save every patch within the bounding box
            min_x, min_y, max_x, max_y = [int(coord * zoom_factor) for coord in bbox]

            # use patch size as padding area
            min_x -= size
            min_y -= size
            max_x += size
            max_y += size

            # may add a inflate factor here instead of 1.0
            length, width = int((max_x - min_x) * 1.0), int((max_y - min_y) * 1.0)
            bbox_row_count, bbox_col_count = length // stride + 1, length // stride + 1
            start_row_id, start_col_id = min_x // stride, min_y // stride
            list_of_coordinates = product(range(start_row_id, start_row_id + bbox_row_count),
                                          range(start_col_id, start_col_id + bbox_col_count))

            batch_extract_patches(list_of_coordinates, weight=1)

    else:
        assert strategy in ["default", "neighborhood"], f"{strategy} is not a valid strategy."

    return info_list


def crop_all_slides(data_dir, save_dir, level,
                    pos_preserve_weight=1.0,
                    strategy="default",
                    workers=4):
    slide_paths, mask_paths = get_slide_and_mask_paths(data_dir)
    save_dirs = [Path(save_dir) / f"level_{level}/{str(slide_path.name).replace('.tif', '')}" for slide_path in
                 slide_paths]
    levels = [level] * len(slide_paths)
    weights = [pos_preserve_weight] * len(slide_paths)
    strategies = [strategy] * len(slide_paths)

    with multiprocessing.Pool(workers) as p:
        info_list = p.starmap(crop_slide_as_patches,
                              zip(slide_paths, mask_paths, levels, weights, save_dirs, strategies))

    level_info = pd.DataFrame(reduce(lambda x, y: x + y, info_list))
    level_info.to_csv(Path(save_dir) / f"level_{level}/info.csv", index=False)
    return level_info


def get_zoom_out_context(slide_path,
                         patches_info,
                         output_dir,
                         input_level,
                         output_level):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting cropping zoomed patches for {slide_path.name}: "
          f"Base level: {input_level}, Output level: {output_level}.")

    slide = read_slide(slide_path)
    zoom_patch_info = patches_info.copy()
    zoom_patch_info["patch_path"] = zoom_patch_info["patch_path"].apply(lambda p: output_dir / Path(p).name)
    factor = 2 ** (output_level - input_level)

    for patch in zoom_patch_info.itertuples():
        # top-left corner should move a certain distance to guarantee that the center region remains in the center
        dist = int((factor - 1) * SIZE / 2)
        # change to the output level coordinate system
        zoom_offset_x = int((patch.offset_x - dist) / factor)
        zoom_offset_y = int((patch.offset_y - dist) / factor)

        patch_img = read_region_from_slide(slide,
                                           x=zoom_offset_x,
                                           y=zoom_offset_y,
                                           level=output_level,
                                           width=SIZE,
                                           height=SIZE)
        patch_img.save(patch.patch_path)

    return zoom_patch_info


def zoom_all_slides(patch_data_root,
                    slide_data_root,
                    input_level,
                    output_level,
                    workers=4):
    input_dir = Path(patch_data_root) / f"level_{input_level}"
    save_dir = input_dir / f"zoom_level_{output_level}"

    slide_paths, mask_paths = get_slide_and_mask_paths(slide_data_root)
    patch_info = pd.read_csv(input_dir / "info.csv")
    patch_infos = [patch_info[patch_info.slide_name == slide_path.name] for slide_path in slide_paths]
    output_dirs = [save_dir / slide_path.name.replace('.tif', '') for slide_path in slide_paths]
    input_levels = [input_level] * len(slide_paths)
    output_levels = [output_level] * len(slide_paths)

    with multiprocessing.Pool(workers) as p:
        info_list = p.starmap(get_zoom_out_context,
                              zip(slide_paths, patch_infos, output_dirs, input_levels, output_levels))

    level_info = pd.concat(info_list).sort_index()
    level_info.to_csv(Path(save_dir) / f"info.csv", index=False)
    return level_info


def calculate_patch_coverage(meta_info_df, level_info_df, level):
    shapes = map(lambda t: t[1], meta_info_df[f"Level_{level}"])
    size, stride = 299, 128
    total_counts = [((row_pixel - size) // stride + 1) * ((col_pixel - size) // stride + 1)
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
