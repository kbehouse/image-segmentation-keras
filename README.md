# Image Segmentation Keras

Fork from https://github.com/kbehouse/image-segmentation-keras
* Switch VGGsegnet to Tensorflow version
* Switch VGGsegnet to channel 'Last' version
* Switch to Python3 

## Implememnation of various Deep Image Segmentation models in keras


<p align="center">
  <img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png" width="50%" >
</p>

## Models 

* VGG Segnet Tensorflow Ready
* Other from original fork support Theonao

## Getting Started

### Prerequisites

Set Keras to Tensorflow for VGGSegnet

* Keras 2.0
* opencv for python
* Tensorflow 

# Preparing the data for training

## Get image annotation(label)

Check one image annotation(label) by HSV

```
cd gen_semantic_data/
# for real
python3 hsv_gen_label.py cube_shadow.png
# for simulator
python3 hsv_gen_label.py sim_cube_2.png
```

Check one image annotation(label)

```
cd gen_semantic_data/

# for digit txt
python3 check_img_label_to_digit.py --image=sim_cube_2_label.png
```

## Augmentation image

* Generate the mix real image and sim image 

* You could modify target_dir, table_path, cube_path, cube_label_path

```
python3 generate_img.py
```

Visiualize the batch image and annotation(label)
```
python3 check_batch_img_label_visualize.py \
 --images="../data/dataset1/train/" \
 --annotations="../data/dataset1/trainannot/" \
 --n_classes=2 

```


## Downloading the Pretrained VGG Weights

Look Models/VGGSegnet.py 
```
VGG_16_WEIGHTS_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
```

It will auto download to 
**~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5**

## Training the Model

To train the model run the following command:

```shell
python3  train.py \
 --save_weights_path=weights/ex1 \
 --train_images="data/Redcube/train/" \
 --train_annotations="data/Redcube/trainannot/" \
 --val_images="data/Redcube/val/" \
 --val_annotations="data/Redcube/valannot/" \
 --n_classes=2 \
 --model_name="vgg_segnet" \
 --epochs=3 
```
## Getting the predictions

Predict batch image

```shell
python3  predict.py \
 --save_weights_path=weights/ex1 \
 --epoch_number=2 \
 --test_images="data/Redcube/test/" \
 --output_path="data/Redcube/test_predictions/" \
 --n_classes=2 \
 --model_name="vgg_segnet" 
```

Predict one image

```shell
python3  predict_onepic.py \
 --save_weights_path=weights/ex1 \
 --epoch_number=3 \
 --input="test.png" \
 --output="test_annot.png" \
 --n_classes=2
```