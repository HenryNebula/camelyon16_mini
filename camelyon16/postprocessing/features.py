import numpy as np
import scipy.stats.stats as st
from skimage.measure import label, regionprops
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pandas as pd


FILTER_DIM = 2
N_FEATURES = 31
feature_tuple = namedtuple("feature_tuple", "MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS")

"""
The feature extraction functions are borrowed from existing codes from
 https://github.com/arjunvekariyagithub/camelyon16-grand-challenge/blob/master/camelyon16/postprocess/extract_feature_heatmap.py
"""

def format_2f(number):
    return float(f"{number:.2f}")


def get_feature(region_props, n_region, feature_name):
    if n_region > 0:
        feature_values = [region[feature_name] for region in region_props]
        feature = feature_tuple(
            MAX=format_2f(np.max(feature_values)),
            MEAN=format_2f(np.mean(feature_values)),
            VARIANCE=format_2f(np.var(feature_values)),
            SKEWNESS=format_2f(st.skew(np.array(feature_values))),
            KURTOSIS=format_2f(st.kurtosis(np.array(feature_values)))
        )
    else:
        feature = feature_tuple(*([0] * 5))
    return feature


def get_region_props(heatmap_threshold_2d, heatmap_prob_2d):
    labeled_img = label(heatmap_threshold_2d)
    return regionprops(labeled_img, intensity_image=heatmap_prob_2d)


def get_largest_tumor_index(region_props):
    largest_tumor_index = -1

    largest_tumor_area = -1

    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area:
            largest_tumor_area = region_props[index]['area']
            largest_tumor_index = index

    return largest_tumor_index


def get_longest_axis_in_largest_tumor_region(region_props, largest_tumor_region_index):
    largest_tumor_region = region_props[largest_tumor_region_index]
    return max(largest_tumor_region['major_axis_length'], largest_tumor_region['minor_axis_length'])


def get_tumor_region_to_tissue_ratio(region_props, tissue_area):
    tissue_area = np.count_nonzero(tissue_area)
    tumor_area = 0

    n_regions = len(region_props)
    for index in range(n_regions):
        tumor_area += region_props[index]['area']

    return float(tumor_area) / tissue_area


def get_average_prediction_across_tumor_regions(region_props):
    # close 255
    region_mean_intensity = [region.mean_intensity for region in region_props]
    return np.mean(region_mean_intensity)


def extract_features(heatmap_prob, tissue_array):
    """
        Feature list:
        -> (01) given t = 0.90, total number of tumor regions
        -> (02) given t = 0.90, percentage of tumor region over the whole tissue region
        -> (03) given t = 0.50, the area of largest tumor region
        -> (04) given t = 0.50, the longest axis in the largest tumor region
        -> (05) given t = 0.90, total number pixels with probability greater than 0.90
        -> (06) given t = 0.90, average prediction across tumor region
        -> (07-11) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'area'
        -> (12-16) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'perimeter'
        -> (17-21) given t = 0.90, max, mean, variance, skewness, and kurtosis of  'compactness(eccentricity[?])'
        -> (22-26) given t = 0.50, max, mean, variance, skewness, and kurtosis of  'rectangularity(extent)'
        -> (27-31) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'solidity'
    :param heatmap_prob:
    :return:
    """

    heatmap_threshold_t90 = np.array(heatmap_prob)
    heatmap_threshold_t50 = np.array(heatmap_prob)
    heatmap_threshold_t90[heatmap_threshold_t90 < int(0.90 * 255)] = 0
    heatmap_threshold_t90[heatmap_threshold_t90 >= int(0.90 * 255)] = 255
    heatmap_threshold_t50[heatmap_threshold_t50 <= int(0.50 * 255)] = 0
    heatmap_threshold_t50[heatmap_threshold_t50 > int(0.50 * 255)] = 255

    heatmap_threshold_t90_2d = np.squeeze(heatmap_threshold_t90[:, :, 0])
    heatmap_threshold_t50_2d = np.squeeze(heatmap_threshold_t50[:, :, 0])
    heatmap_prob_2d = np.squeeze(heatmap_prob[:, :, 0])

    region_props_t90 = get_region_props(np.array(heatmap_threshold_t90_2d), heatmap_prob_2d)
    region_props_t50 = get_region_props(np.array(heatmap_threshold_t50_2d), heatmap_prob_2d)

    features = []

    f_count_tumor_region = len(region_props_t90)
    if f_count_tumor_region == 0:
        return [0.00] * N_FEATURES

    features.append(format_2f(f_count_tumor_region))

    f_percentage_tumor_over_tissue_region = get_tumor_region_to_tissue_ratio(region_props_t90, tissue_array)
    features.append(format_2f(f_percentage_tumor_over_tissue_region))

    largest_tumor_region_index_t90 = get_largest_tumor_index(region_props_t90)
    largest_tumor_region_index_t50 = get_largest_tumor_index(region_props_t50)
    f_area_largest_tumor_region_t50 = region_props_t50[largest_tumor_region_index_t50].area
    features.append(format_2f(f_area_largest_tumor_region_t50))

    f_longest_axis_largest_tumor_region_t50 = get_longest_axis_in_largest_tumor_region(region_props_t50,
                                                                                       largest_tumor_region_index_t50)
    features.append(format_2f(f_longest_axis_largest_tumor_region_t50))

    f_pixels_count_prob_gt_90 = np.count_nonzero(heatmap_threshold_t90_2d)
    features.append(format_2f(f_pixels_count_prob_gt_90))

    f_avg_prediction_across_tumor_regions = get_average_prediction_across_tumor_regions(region_props_t90)
    features.append(format_2f(f_avg_prediction_across_tumor_regions))

    f_area = get_feature(region_props_t90, f_count_tumor_region, 'area')
    features += f_area

    f_perimeter = get_feature(region_props_t90, f_count_tumor_region, 'perimeter')
    features += f_perimeter

    f_eccentricity = get_feature(region_props_t90, f_count_tumor_region, 'eccentricity')
    features += f_eccentricity

    f_extent_t50 = get_feature(region_props_t50, len(region_props_t50), 'extent')
    features += f_extent_t50

    f_solidity = get_feature(region_props_t90, f_count_tumor_region, 'solidity')
    features += f_solidity

    return features


def save_features_as_csv(slide_names,
                         tumor_flags,
                         dataset_name,
                         model_name):

    feature_list = [{"slide_name": slide_name} for slide_name in slide_names]
    result_root = Path(f"./results/{model_name}/{dataset_name}")
    for slide_id, slide_name in tqdm(enumerate(slide_names), total=len(slide_names)):
        thumbnail_path = Path(f"./results/thumbnails/{slide_name}.thumbnail.npy")
        tissue_mask = np.load(thumbnail_path)
        heatmap_path = result_root / f"{slide_name}/heatmap_grey.bmp"
        heatmap_grey = Image.open(heatmap_path)
        features = extract_features(np.array(heatmap_grey)[..., np.newaxis], tissue_mask[..., np.newaxis])
        features = {f"f{idx+1}": f for idx, f in enumerate(features)}
        feature_list[slide_id].update(features)

    feature_df = pd.DataFrame(feature_list).sort_values("slide_name")
    feature_df["label"] = tumor_flags
    feature_df.to_csv(result_root / f"../{dataset_name}_features.csv")
    return feature_df





