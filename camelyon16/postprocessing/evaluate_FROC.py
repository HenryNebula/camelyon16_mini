# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016

@author: Babak Ehteshami Bejnordi

Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""

"""
The FROC curves calculating functions are borrowed from the original Challenge website
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
from tqdm import tqdm
from pathlib import Path

EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
L0_RESOLUTION = 0.243 # pixel resolution at level 0

   
def computeEvaluationMask(mask, resolution, level):
    """Computes the evaluation mask.
    
    Args:
        mask:    numpy array of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made
        
    Returns:
        evaluation_mask
    """
    distance = nd.distance_transform_edt(1 - mask)
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2) 
    return evaluation_mask
    
    
def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    
    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object 
        should be less than 275µm to be considered as ITC (Each pixel is 
        0.243µm*0.243µm in level 0). Therefore the major axis of the object 
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.
        
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
        
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)    
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = [] 
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcoor:      list of X-coordinates of the lesions
        Ycoor:      list of Y-coordinates of the lesions
    """
    df = pd.read_csv(csvDIR)
    Probs = df["confidence"]
    Xcoor, Ycoor = df["x"], df["y"]
    return Probs, Xcoor, Ycoor
    
         
def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells):
    """Generates true positive and false positive stats for the analyzed image
    
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
         
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        
        TP_probs:   A list containing the probabilities of the True positive detections
        
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
        
        detection_summary:   A python dictionary object with keys that are the labels 
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate]. 
        Lesions that are missed by the algorithm have an empty value.
        
        FP_summary:   A python dictionary object with keys that represent the 
        false positive finding number and values that contain detection 
        details [confidence score, X-coordinate, Y-coordinate]. 
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = [] 
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}  
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []        
     
    FP_counter = 0       
    if (is_tumor):
        for i in range(0,len(Xcorr)):
            HittedLabel = evaluation_mask[Xcorr[i], Ycorr[i]]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter+=1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]                                     
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i]) 
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]] 
            FP_counter+=1
            
    num_of_tumors = max_label - len(Isolated_Tumor_Cells)
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary
 
 
def computeFROC(meta_info, dataset_name, model_name, overwrite):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    result_path = f"./results/{model_name}/{dataset_name}_features_local.json"
    if not overwrite and Path(result_path).exists():
        df = pd.read_json(result_path)
        return df.total_FPs, df.total_sensitivity

    slide_names = meta_info.id
    has_tumor = meta_info.has_tumor
    FROC_data = np.zeros((4, len(slide_names)), dtype=np.object)
    FP_summary = np.zeros((2, len(slide_names)), dtype=np.object)
    detection_summary = np.zeros((2, len(slide_names)), dtype=np.object)

    for idx, slide_name in tqdm(enumerate(slide_names), total=len(slide_names)):
        slide_name = slide_name + ".tif" if ".tif" not in slide_name else slide_name
        csvDIR = f"./results/{model_name}/{dataset_name}/{slide_name}/coordinates.csv"
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
        tumor_flag = has_tumor[idx]

        if has_tumor[idx]:
            ground_truth_mask = np.load(f"./results/mask_thumbnails/{slide_name}.thumbnail.npy")
            evaluation_mask = computeEvaluationMask(ground_truth_mask,
                                                    resolution=L0_RESOLUTION,
                                                    level=EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)

        else:
            evaluation_mask = 0
            ITC_labels = []

        FROC_data[0][idx] = slide_name
        FP_summary[0][idx] = slide_name
        detection_summary[0][idx] = slide_name
        FROC_data[1][idx], FROC_data[2][idx], \
            FROC_data[3][idx], detection_summary[1][idx], \
            FP_summary[1][idx] = \
            compute_FP_TP_Probs(Ycorr,
                                Xcorr,
                                Probs,
                                tumor_flag,
                                evaluation_mask,
                                ITC_labels)

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))

    pd.DataFrame({
        "total_FPs": total_FPs,
        "total_sensitivity": total_sensitivity
    }).to_json(result_path)

    return total_FPs, total_sensitivity

  
            
        
        
        
        
        
        