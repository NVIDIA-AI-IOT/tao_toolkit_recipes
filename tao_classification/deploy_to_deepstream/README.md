# Deploy Classification model to Deepstream
Some tips to deploy TAO Classification model to Deepstream.

# Deploy Classification model as primary tensorrt engine
There are two ways of deploying classification model in deepstream.
One is working as primary tensorrt engine, anohter is working as secondary tensorrt engine.


## Detailed Steps

* Create ds_classification_as_primary_gie.txt. Refer to [link](https://forums.developer.nvidia.com/t/issue-with-image-classification-tutorial-and-testing-with-deepstream-app/165835/12?u=morganh)

Below is a snippet of the config file.

```
# config-file property is mandatory for any gie section.
# Other properties are optional and if set will override the properties set in
# the infer config file.
[primary-gie]
enable=1
gpu-id=0
#model-engine-file=your_classification.engine
batch-size=1
#Required by the app for OSD, not a plugin property
bbox-border-color0=1;0;0;1
bbox-border-color1=0;1;1;1
bbox-border-color2=0;0;1;1
bbox-border-color3=0;1;0;1
interval=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_as_primary_gie.txt
```

* Create config_as_primary_gie.txt.
```
[property]
gpu-id=0
net-scale-factor=1.0
#below offsets=b,g,r  which can be also changed according to the "image_mean" in your training spec file.
offsets=123.67;116.28;103.53
model-color-format=1
batch-size= 30

tlt-model-key=yourkey
tlt-encoded-model=your_unpruned_or_pruned_model.etlt
labelfile-path=labels.txt
#int8-calib-file=cal.bin
#model-engine-file=your_classification.engine
#input-dims=c;h;w;0. Can be also changed according to the "input_image_size" in your training spec file.
input-dims=3;224;224;0
uff-input-blob-name=input_1
output-blob-names=predictions/Softmax

# process-mode: 2 - inferences on crops from primary detector, 1 - inferences on whole frame
process-mode=1
## 0=FP32, 1=INT8, 2=FP16 mode
network-mode=0

network-type=1 # defines that the model is a classifier.
num-detected-classes=2
interval=0
gie-unique-id=1
#threshold=0.05
classifier-async-mode=1
classifier-threshold=0.2
operate-on-gie-id=1
#operate-on-class-ids=0
```

* Run deepstream-app
```
$ deepstream-app -c ds_classification_as_primary_gie.txt
```

# Deploy Classification model as secondary tensorrt engine

## Detailed Steps

* Create ds_classification_as_secondary_gie.txt. Refer to [link](https://forums.developer.nvidia.com/t/issue-with-image-classification-tutorial-and-testing-with-deepstream-app/165835/12?u=morganh)

Below is a snippet of the config file.

```
[secondary-gie3]
enable=1
#model-engine-file=your_classification.engine
batch-size=4
gpu-id=0
gie-unique-id=7
operate-on-gie-id=1
#operate-on-class-ids=0;
config-file=config_as_secondary_gie.txt
```

* Create config_as_secondary_gie.txt.

```
[property]
gpu-id=0
net-scale-factor=1.0
#below offsets=b,g,r  which can be also changed according to the "image_mean" in your training spec file.
offsets=123.67;116.28;103.53
model-color-format=1
batch-size= 30

tlt-model-key=yourkey
tlt-encoded-model=your_unpruned_or_pruned_model.etlt
labelfile-path=labels.txt
#int8-calib-file=cal.bin
#model-engine-file=your_classification.engine
#input-dims=c;h;w;0. Can be also changed according to the "input_image_size" in your training spec file.
input-dims=3;224;224;0
uff-input-blob-name=input_1
output-blob-names=predictions/Softmax

# process-mode: 2 - inferences on crops from primary detector, 1 - inferences on whole frame
process-mode=2
## 0=FP32, 1=INT8, 2=FP16 mode
network-mode=0

network-type=1 # defines that the model is a classifier.
num-detected-classes=2
interval=0
gie-unique-id=1
#threshold=0.05
classifier-async-mode=1
classifier-threshold=0.2
operate-on-gie-id=1
#operate-on-class-ids=0
```

* Run deepstream-app
```
$ deepstream-app -c ds_classification_as_secondary_gie.txt
```

# Other tips
## Generate avi video file as input test file. It is better than mp4 input file.
```
gst-launch-1.0 multifilesrc location="/tmp/%d.jpg" caps=“image/jpeg,framerate=30/1” ! jpegdec ! x264enc ! avimux ! filesink location=“out.avi”
```

## Change "scaling-filter". More info can be found in [DeepStream Gst-nvinfer Plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications)
```
scaling-filter=5
```
