# TensorRT inference sample for TAO key points estimation

## Introduction
This is a TensorRT inference sample for TAO key points estimation. This sample will consume TensorRT engine and json format input generated in FPENet notebook.

## Prequisites
`TensorRT`, `numpy`, `cv2` is needed for this sample. You can try TensorRT docker image on [NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt) for easily building environment. 


## Steps to run inference:

```sh
# Generate TensorRT engine of fpenet model
tao fpenet export -m <Trained TAO Model Path> -k <Encode Key> -o <Output file .etlt> --engine_file trt_fpenet.engine

# run inference:
python3 fpenet_trt_inference.py --input_json=<Input json file> --trt_engine=<trt fpenet engine> --output_img_dir=<Path to output images>
```