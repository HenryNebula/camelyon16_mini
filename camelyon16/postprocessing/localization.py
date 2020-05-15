import numpy as np
from itertools import product
import pandas as pd
from PIL import Image
from pathlib import Path
from multiprocessing import Pool


def circular_mask(candidate_region, center, radius):
    # for candidate_region, every row is a coordinate pair (x, y)
    distance = np.sqrt(np.sum(np.power(candidate_region - center, 2), axis=1))
    mask = distance <= radius
    return candidate_region[mask]


def argmax2d(array):
    return np.unravel_index(np.argmax(array, axis=None), array.shape)


def non_maxima_suppression(heatmap_path, threshold=0.5, radius=128 >> 4):
    heatmap = Image.open(heatmap_path)
    heatmap = np.array(heatmap) / 255
    center_list = []
    prob = []
    x_max, y_max = heatmap.shape

    while np.max(heatmap) > threshold:
        center = argmax2d(heatmap)
        center_list.append(center)
        prob.append(heatmap[center])
        candidate_x = range(max(0, center[0] - radius), min(x_max, center[0] + radius + 1))
        candidate_y = range(max(0, center[1] - radius), min(y_max, center[1] + radius + 1))
        candidate_region = list(product(candidate_x, candidate_y))
        candidate_region = np.array(candidate_region)
        masked_region = circular_mask(candidate_region, center, radius)
        heatmap[masked_region[:, 0], masked_region[:, 1]] = 0
        heatmap[center] = 0

    return pd.DataFrame({
        "x": [c[0] for c in center_list],
        "y": [c[1] for c in center_list],
        "confidence": prob
    })


def save_coordinates_as_csv(slide_names,
                            dataset_name,
                            model_name,
                            num_workers=6,
                            overwrite=True):

    result_root = Path(f"./results/{model_name}/{dataset_name}")
    coord_df_path = [result_root / f"{slide_name}/coordinates.csv" for slide_name in slide_names]

    heatmap_paths = []
    for idx, slide_name in enumerate(slide_names):

        if not overwrite:
            if coord_df_path[idx].exists():
                coord_df_path[idx] = None
                continue

        heatmap_paths.append(result_root / f"{slide_name}/heatmap_grey.bmp")

    coord_df_path = list(filter(lambda x: x is not None, coord_df_path))

    if heatmap_paths:
        with Pool(num_workers) as p:
            results = p.map(non_maxima_suppression, heatmap_paths)

        for path, coord_df in zip(coord_df_path, results):
            coord_df.to_csv(path)
