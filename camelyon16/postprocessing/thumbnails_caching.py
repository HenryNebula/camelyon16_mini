from skimage.filters import threshold_otsu
from tqdm import tqdm
from pathlib import Path
import numpy as np

from camelyon16.preprocessing.slide_utils import read_full_slide_by_level


def construct_tissue_mask(slide_path):
    thumbnail = read_full_slide_by_level(slide_path, 5).convert("HSV")  # use level 5 as thumbnail
    channels = [np.asarray(thumbnail.getchannel(c)) for c in ("H", "S", "V")]
    otsu_thresh = [threshold_otsu(c) for c in channels]
    tissue_mask = np.bitwise_and(*[c >= otsu_thresh[i] for i, c in enumerate(channels)])
    return tissue_mask


def save_tissue_masks(slide_names, dataset_name):
    # only need to run once to cache the tissue mask thumbnails
    for slide_name in tqdm(slide_names):
        save_root = Path("./results/tissue_thumbnails/")
        save_root.mkdir(parents=True, exist_ok=True)
        tissue_thumbnail_path = save_root / f"{slide_name}.thumbnail.npy"
        if tissue_thumbnail_path.exists():
            continue
        else:
            slide_path = Path(f"./data/{dataset_name}/samples/{slide_name}")
            tissue_mask = construct_tissue_mask(slide_path)
            np.save(save_root / f"{slide_name}.thumbnail.npy", tissue_mask)


def save_slides(slide_names, dataset_name):
    # only need to run once to cache the slide thumbnails
    for slide_name in tqdm(slide_names):
        save_root = Path("./results/thumbnails/")
        save_root.mkdir(parents=True, exist_ok=True)
        thumbnail_path = save_root / f"{slide_name}.thumbnail.npy"
        if thumbnail_path.exists():
            continue
        else:
            slide_path = Path(f"./data/{dataset_name}/samples/{slide_name}")
            thumbnail = read_full_slide_by_level(slide_path, 5)
            np.save(save_root / f"{slide_name}.thumbnail.npy", thumbnail)


def save_tumor_masks(slide_names, dataset_name):
    for slide_name in tqdm(slide_names, total=len(slide_names)):
        save_root = Path("./results/mask_thumbnails/")
        save_root.mkdir(parents=True, exist_ok=True)
        mask_thumbnail_path = save_root / f"{slide_name}.thumbnail.npy"

        if mask_thumbnail_path.exists():
            continue
        else:
            mask_fname = slide_name.replace(".tif", "_mask.tif")
            slide_path = Path(f"./data/{dataset_name}/samples/{mask_fname}")
            if not slide_path.exists():
                print(f"No mask found for {mask_fname}")
            else:
                thumbnail = read_full_slide_by_level(slide_path, 5).getchannel(0)
                np.save(save_root / f"{slide_name}.thumbnail.npy", thumbnail)