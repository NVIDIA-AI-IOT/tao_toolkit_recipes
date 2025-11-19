# FAQ

## FPENet
1. *Why is the bounding box recalculated just using the key points when I have also supplied the face bbox ground truth in the annotation file ? What is the purpose of the bbox in the ground truth file?*

The annotation file just provide all the keypoints. FPEnet will find the xmin, ymin, xmax, ymax of the points and then calculate a square face bounding box based on the key points. And then crop bounding box from image and scale the Keypoints to target resolution

## Emotionnet
1. *How to find the input name of EmotionNet?*
```
tao-converter model.etlt
-k nvidia_tlt
-t fp32
-p input_landmarks:0,1x1x136x1,1x1x136x1,2x1x136x1
-e model.engine
```

## tlt or etlt
1. *How to decode etlt file?*
```
$ docker run --runtime=nvidia -it --rm -v /home/morganh:/home/morganh nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5 /bin/bash
# wget --content-disposition ‘https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/fpenet/deployable_v1.0/files?redirect=true&path=model.etlt’ -O fpenet_model_v1.0.etlt
```

Generate deocde_etlt.py file as below.
```
import argparse
import struct
from nvidia_tao_tf1.encoding import encoding

def parse_command_line(args):
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description='ETLT Decode Tool')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Path to the etlt file.')
    parser.add_argument('-o',
                        '--uff',
                        required=True,
                        type=str,
                        help='The path to the uff file.')
    parser.add_argument('-k',
                        '--key',
                        required=True,
                        type=str,
                        help='encryption key.')
    return parser.parse_args(args)


def decode(tmp_etlt_model, tmp_uff_model, key):
    with open(tmp_uff_model, 'wb') as temp_file, open(tmp_etlt_model, 'rb') as encoded_file:
        size = encoded_file.read(4)
        size = struct.unpack("<i", size)[0]
        input_node_name = encoded_file.read(size)
        encoding.decode(encoded_file, temp_file, key.encode())

def main(args=None):
    args = parse_command_line(args)
    decode(args.model, args.uff, args.key)
    print("Decode successfully.")
```

Then, decode the etlt file with following command.

```
# python decode_etlt.py -m fpenet_model_v1.0.etlt -o fpenet_model_v1.0.onnx -k nvidia_tlt
```

## Segformer
1. *How to train multiclass?*
Refence: https://forums.developer.nvidia.com/t/training-segformer-cradiov2-with-multiple-classes-not-working-only-learning-1-class.

Please change 1-channel mask png file to 3-channel mask png file.

```
# cat change_1_channle_to_3_channel_green.py
# pip install pillow numpy
import os, glob
import numpy as np
from PIL import Image

#in_dir  = "xxx/data/masks/train"     # 1-channel 0/1 mask folder
#out_dir = "xxx/data/masks_3channel/train"      # output RGB mask folder
in_dir  = "xxx/data/masks/val"     # 1-channel 0/1 mask folder
out_dir = "xxx/data/masks_3channel/val"      # output RGB mask folder
os.makedirs(out_dir, exist_ok=True)

for p in glob.glob(os.path.join(in_dir, "*.png")):
    g = np.array(Image.open(p))                   # 8-bit 1-channel
    assert g.ndim == 2, f"Not single channel: {p}"
    rgb = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.uint8)
    rgb[g == 1] = (0, 255, 0)                     # set to green color
    Image.fromarray(rgb, mode="RGB").save(
        os.path.join(out_dir, os.path.basename(p)), format="PNG"
```

Snippet of training spec file for training 3 classes:
```
dataset:
  segment:
    dataset: "SFDataset"
    root_dir: /path/to/dataset_rgb
    num_classes: 3            # background + 2 foreground classes
    img_size: 224
    train_split: "train"
    validation_split: "val"
    test_split: "val"
    predict_split: "test"
    label_transform: None     # palette uses 0–255 range; disable normalization on labels
    palette:
      - label_id: 0
        mapping_class: background
        rgb: [0, 0, 0]
        seg_class: background
      - label_id: 1
        mapping_class: class_1
        rgb: [0, 255, 0]
        seg_class: class_1
      - label_id: 2
        mapping_class: class_2
        rgb: [255, 0, 0]
        seg_class: class_2
    # masks must contain ONLY these exact RGB colors; labels will be mapped to indices {0,1,2} accordingly
```
