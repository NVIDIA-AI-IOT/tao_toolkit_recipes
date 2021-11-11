# TensorRT inference sample for TAO ActionRecognitionNet

## Introduction
This is a TensorRT inference sample with TAO ActionRecognitionNet deployable model. This sample will consume TensorRT engine and sequence of images and predict the people's action in those images.

## Prequisites
`TensorRT`, `numpy`, `PIL` is needed for this sample. You can try TensorRT docker image on [NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt) for easily building environment. 

You also need to download `tao-converter` from [TAO toolkit](https://developer.nvidia.com/tao-toolkit-get-started) to convert the encrypted tao model to TensorRT engine.

## Steps to run inference:

```sh
# Download the deployable action recognition model from NGC
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/actionrecognitionnet/versions/deployable_v1.0/zip -O actionrecognitionnet_deployable_v1.0.zip

# Generate TensorRT engine of action recognition model
# generate engine of 2D model:
tao-converter resnet18_2d_rgb_hmdb5_32.etlt -k nvidia_tao -p input_rgb,1x96x224x224,1x96x224x224,1x96x224x224 -e trt2d.engine -t fp16
# generate engine of 3D model:
tao-converter resnet18_3d_rgb_hmdb5_32.etlt -k nvidia_tao -p input_rgb,1x96x224x224,1x96x224x224,1x96x224x224 -e trt3d.engine -t fp16

# run inference:
# run inference with 2D engine:
python ar_trt_inference.py --input_images_folder=/path/to/images --trt_engine=./trt2d.engine --input_2d
# run inference with 3D engine:
python ar_trt_inference.py --input_images_folder=/path/to/images --trt_engine=./trt3d.engine
```
 