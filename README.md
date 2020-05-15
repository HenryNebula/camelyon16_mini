# Detecting Tumors on Gigapixel WSI with Deep Learning

Class project of COMS 4995: Applied Deep Learning, using Tensorflow 2 to detect tumors on pathology images.

## Report links
 1. Video demo can be found [here](https://youtu.be/qG_kbv_vZaw).
 2. Slides can be found [here](https://docs.google.com/presentation/d/1Jfn1F9yaKrh6ymrjr1H_0s_AMmb8X_oQqwiVEEpVvGs/edit?usp=sharing).
 3. Pretrained model weights can be found [here](https://console.cloud.google.com/storage/browser/camelyon16_mini/models).

## Structure of source code

* Data preprocessing code
  * [TF dataset generator](camelyon16/preprocessing/dataset.py) contains
   code for image preprocessing and code for generating tf.data.Dataset from
   pandas dataframes containing image information.
  * [Patch extractions](camelyon16/preprocessing/extract_patches.py) contains code for patch extractions using sliding
  window approach.
  * [Slide related utilities](camelyon16/preprocessing/slide_utils.py) contains code for manipulating huge WSI files.
  * [Mask conversion](camelyon16/preprocessing/conversion.ipynb) helps convert xml mask to tif mask. The `multiresolutionimageinterface` package can only be loaded under 
  Python 3.6 and users need to specify the installation location of [ASAP](https://github.com/computationalpathologygroup/ASAP).

* Model & Training
    * [Model factory](camelyon16/training/model.py) constructs models using single and multiple resolution.
    * [Callbacks](camelyon16/training/callbacks.py) creates several useful Keras callback methods.
    * [Training utilities](camelyon16/training/utils.py) contains high level code for generating datasets and parsing runtime arguments.

* Data postprocessing code
    * [Heatmap generation](camelyon16/postprocessing/heatmap.py) generates heatmap from pretrained Inception-V3 model.
    * [Morphology feature extraction](camelyon16/postprocessing/features.py) extracts morphology features from heatmap, 
 used for slide-level classification.
    * [Tumor center localization](camelyon16/postprocessing/localization.py) generate tumor centers using non-maxima-suppression 
 algorithm.
    * [Thumbnails caching](camelyon16/postprocessing/thumbnails_caching.py) contains code for caching thumbnails of masks and slides.
    * [Slide level Classification](camelyon16/postprocessing/evaluate_CLF.py) uses random forest to classify slides based on morphology heatmap features.
    * [FROC evaluation](camelyon16/postprocessing/evaluate_FROC.py) contains code to evaluate FROC based on tumor center localization.

* Scripts and demo
  * [Model fitting script](fit_model.py) fits models with different runtime parameters.
  * [Model evaluation script](evaluate_model.py) evaluates different models on slide-level classification and lesion level localization.
  * [Visualization](visualization.ipynb) visualizes generated heatmaps and FROC curves.
  * [Demo](demo.ipynb) demonstrates the pipeline of the whole model.

## How to run code locally

If you want to debug your own model or existing models using this framework, follow the following step:

1. Install all necessary libraries mentioned in [requirements](camelyon16/requirements.txt).

2. Make sure the data is saved in the data directory under root directories, can be symbolic link. The list of sampled slide names can be found [here](https://storage.googleapis.com/camelyon16_mini/train_val_slide_names). 
And the whole Camelyon16 challenge dataset can be found [here](https://camelyon17.grand-challenge.org/Data/).

    The final structure of this project should be like,

    ```bash
    .
    ├── camelyon16
    ├── checkpoints
    ├── data
    ├── demo.ipynb
    ├── evaluate_model.py
    ├── fit_model.py
    ├── logs
    ├── models
    ├── README.md
    ├── results
    ├── test
    └── visualization.ipynb
    ```
    where raw slides images are saved under `data/training/samples` or `data/validation/samples`.

3. Start by modifying the [demo notebook](demo.ipynb) is a good choice for getting familiar with the code. 

    For running [model fitting script](fit_model.py), the shell command can be like,
    ```shell script
    # fit a single resolution model based on level 1 
    python fit_model.py \
          --level=1 \
          --lr=0.002 \
          --zoom_level=1 \
          --model_type="single" \
          --gpu_memory=3300
    ``` 
   
   For running [model evaluation script](evaluate_model.py), the shell command can be like,
    ```shell script
     # fit a single resolution model based on context patches of level 2 
     python evaluate_model.py --zoom_level=2
    ``` 
     
Note that all package dependencies are specified in the [requirements](camelyon16/requirements.txt) file. The code was developed using Ubuntu 16.04 LTS, and the code 
is not guaranteed to work under other operating systems.