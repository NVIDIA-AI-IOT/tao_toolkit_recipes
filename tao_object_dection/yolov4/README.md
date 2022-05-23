# Train TAO YOLOV4 on COCO14 dataset
In this page, we will walk you through the steps to reproduce the best mAP on COCO14 using YOLOV4 with TAO Toolkit. In the first section, you will train a classification model with CSPDarkNet53 architecture on ImageNet2012. In the second section, you will leverage the imagenet pretrained CSPDarkNet53 as the backbone of YOLOV4 and use the techniques provided in TAO toolkit to train the YOLOV4 on COCO14 dataset.   

## Train CSPDarkNet53 backbone on ImageNet2012
In this section, you will train a classification model against the ImageNet 2012 classification dataset with CSPDarkNet53 backbone.

### Prepare the ImageNet 2012 dataset
The ImageNet 2012 dataset contains more than 1.1 million images over 1000 classes. Start by downloading the [ImageNet classification dataset](http://www.image-net.org/download-images) (choose "Download Original Images"), which contains more than 140 GB of images. There are two tarballs to download and save to the same directory:

```shell
ILSVRC2012_img_train.tar (138 GB)—Used for training.
ILSVRC2012_img_val.tar (6.3 GB) —Used for validation.
```

After the dataset has been downloaded, use the [imagenet.py](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/release/tao3.0/misc/dev_blog/SOTA/dataset_tools/imagenet.py) Python script and the [imagenet_val_maps.pklz](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/release/tao3.0/misc/dev_blog/SOTA/dataset_tools/imagenet_val_maps.pklz) validation map file to unzip the two tarballs and restructure the validation dataset. Due to copyright issues, we can’t provide the ImageNet dataset or any ImageNet-pretrained models in TAO Toolkit. Use the following command to unzip and restructure the dataset:

```shell
python3.6 imagenet.py --download-dir <tarballs_download_directory>  --target_dir <unzip_target_dir>
```

Assume that the paths from inside the TAO Toolkit container to the dataset are as follows:

```shell
/home/<username>/tao-experiments/data/imagenet2012/train
/home/<username>/tao-experiments/data/imagenet2012/val
```

The first path is a directory that contains all the training images, where each of the 1K classes has its own subdirectory. The same is assumed for the validation split as well. The structure of the classification dataset follows the [TAO Toolkit classification model training requirements](https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html#image-classification-format).

### Training specification:
For every TAO Toolkit model training, you have a configuration file (spec file) to configure some necessary parameters used to customize the model and the training process. Please refer to 
[classification_cspdarknet53.txt](specs/classification_cspdarknet53.txt) for the training spec file.

### Start Training:
Run following command to start training on 8 GPUs:

```
tao classification train --gpus 8 -e </path/to/spec_file> -r </path/to/results/> -k nvidia_tao
```
For 8 x A100 GPUs, the training will require about ~30 hours. And you might get around 78.3% of val accuracy.


## Train YOLOV4-CSPDarkNet53 on COCO14 dataset
In this section, you will train YOLOV4 on COCO14 dataset with imagenet pretrained CSPDarkNet53 backbone you create in the first section.

### Prepare COCO14 dataset
Firstly, you will prepare the COCO14 dataset for training. To compare with the SOTA model, you will do the training/testing split the same way as the original YOLOV4. And this split is different from the official split of COCO14. You could download the original COCO14 dataset and the split images list (5k.txt/trainvalno5k.txt)by running [get_coco_dataset.sh](https://raw.githubusercontent.com/AlexeyAB/darknet/master/scripts/get_coco_dataset.sh)

After the downloading dataset, you should convert the json format labels to KITTI format by using [coco2kitti.py](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/release/tao3.0/misc/dev_blog/SOTA/dataset_tools/coco2kitti.py)

```shell
# Convert instances_train2014.json to KITTI format
python3 ./coco2kitti.py <path to COCO2014 root dir> train2014
mv ./labels ./train2014_KITTI

# Convert instances_val2014.json to KITTI format
python3 ./coco2kitti.py <path to COCO2014 root dir> val2014
mv ./labels ./val2014_KITTI
```
Once you get the images and KITTI format labels, you could re-split them according to 5k.txt / trainvalno5k.txt. 

### Training specification:
Before we start the training, there are 3 more steps to do to get a better results on the dataset.

#### Generate anchor setting:
TAO Toolkit YOLOV4 supports the ground truth bboxes clustering to find suitable anchor setting on a specific dataset:

```shell
tao yolo_v4 kmeans -l </path/to/labels> \
                   -i </path/to/images> \
                   -x <network_input_width> \
                   -y <network_input_height>
```

You can replace the `small_anchor_shape`, `mid_anchor_shape`, `big_anchor_shape` in `yolov4_config` with the generated anchors shapes. 

#### Enable model EMA
TAO Toolkit YOLOV4 supports the model exponential moving average (EMA) during the training. Enable it by setting `model_ema: true` in the `train_config`.

You can also do some hyperparameters (learning rate, learning rate schduler, regularization factor) search by using part of dataset or less epochs. Here we provide a [spec](specs/yolov4_416_coco14.txt) to train YOLOV4-416-Leaky on COCO14 YOLO split. In this spec, we use raw KITTI-style labels for training.

### Start Training:
Run following command to start training on 8 GPUs:

```
tao yolo_v4 train --gpus=8 -e </path/to/spec_file> -r </path/to/results/> -k nvidia_tao
```
The training will require ~130 hours on 8 V100 16G. And you might get around mAP@0.5 60.9% using COCO-style metrics. 
