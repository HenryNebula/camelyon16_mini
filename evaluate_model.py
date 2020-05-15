import argparse
import pandas as pd
from camelyon16.postprocessing.heatmap import batch_generate_heatmap
from camelyon16.postprocessing.thumbnails_caching import save_tissue_masks, save_tumor_masks, save_slides
from camelyon16.postprocessing.features import save_features_as_csv
from camelyon16.postprocessing.localization import save_coordinates_as_csv
from camelyon16.postprocessing.evaluate_CLF import classify_slides
from camelyon16.postprocessing.evaluate_FROC import computeFROC


def parse_for_runtime_arguments():
    parser = argparse.ArgumentParser(description='Process runtime arguments.')
    parser.add_argument('--level', default=1, type=int,
                        help='base level of WSI slides')
    parser.add_argument('--zoom_level', default=1, type=int,
                        help='auxiliary zoom-in level of WSI slides')
    parser.add_argument('--use_neighbor', action='store_true',
                        help='whether to use neighboring patches surrounding tumors only or not')
    parser.add_argument('--model_type', default="single", type=str,
                        choices=["single", "multi"],
                        help='whether to use multi-resolution ensemble model')
    parser.add_argument('--gpu_memory', default=4096, type=int,
                        help='graphical memory reserved for the model')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size for evaluation dataset')
    parser.add_argument('--overwrite', action='store_true',
                        help="whether or not to overwrite existing results for some time consuming operations.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_for_runtime_arguments()
    input_level = args.level
    zoom_level = args.zoom_level
    use_neighbor = args.use_neighbor
    model_type = args.model_type
    overwrite = args.overwrite

    if use_neighbor:
        model_name = "single_res_neighbor"
    else:
        if model_type == "single":
            if zoom_level > 1:
                model_name = "single_res_zoom"
            else:
                model_name = "single_res"
        else:
            model_name = "multi_res"

    for dataset_name in ["training", "validation"]:
        print(f"Start evaluating model {model_name} on {dataset_name} dataset.")
        # meta information for slides
        meta_info = pd.read_json(f"./data/{dataset_name}/meta_info.json")

        # label information for patches
        if "zoom" not in model_name:
            label_info = pd.read_csv(f"./data/{dataset_name}/level_{input_level}/info.csv")
        else:
            label_info = pd.read_csv(f"./data/{dataset_name}/level_{input_level}/zoom_level_{zoom_level}/info.csv")

        if "multi" in model_name:
            low_res_label_info = pd.read_csv(f"./data/{dataset_name}/level_{input_level}/zoom_level_{zoom_level}/info.csv")
        else:
            low_res_label_info = None

        # config_gpu_memory(args.gpu_memory)

        # generate and save heatmap for every slide
        print("Start generating heatmaps ...")
        batch_generate_heatmap(meta_info,
                               label_info,
                               dataset_name,
                               model_name,
                               low_res_label_info=low_res_label_info,
                               input_level=input_level,
                               batch_size=256,
                               overwrite=overwrite)
        print("Finish generating heatmaps!")

        slide_names = label_info.slide_name.unique()

        # cache thumbnails of tissue masks and tumor masks of level 5 if not cached yet
        print("Start caching thumbnails ...")
        save_tissue_masks(slide_names=slide_names, dataset_name=dataset_name)
        save_tumor_masks(slide_names=slide_names, dataset_name=dataset_name)
        save_slides(slide_names=slide_names, dataset_name=dataset_name)
        print("Finish caching thumbnails!")

        # generate and save features for classification
        print("Start generating classification features ...")
        tumor_flags = meta_info.has_tumor
        save_features_as_csv(slide_names=slide_names,
                             tumor_flags=tumor_flags,
                             dataset_name=dataset_name,
                             model_name=model_name)
        print("Finish generating classification features!")

        # generate and save coordinates for tumor regions
        print("Start generating tumor coordinates ...")
        save_coordinates_as_csv(slide_names,
                                dataset_name,
                                model_name,
                                overwrite=overwrite)
        print("Finish generating tumor coordinates!")

        # evaluate FROC to reflect tumor localization
        print("Start computing FROC ...")
        computeFROC(meta_info=meta_info,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    overwrite=overwrite)
        print("Finish computing FROC!")

    # evaluate slide-level classification accuracy
    classify_slides(model_name=model_name)
