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
