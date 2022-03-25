# Load I3D Kinetics pretrained weights in TAO and finetune on HMDB51

I3D is a 3D inception architecture proposed in paper *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*. In this paper, the authors show us the enormous benefit of pretrained weights on Kinetics400 of I3D architecture for the downstream dataset --- We can get much higher accuracy on other action recognition datasets with Kinetics pretrained weights:

|Model type|Dataset|Pretrained|Acc|
|:---:|:---:|:---:|:---:|
|I3D RGB-Only|HMDB51|ImageNet|49.8%|
|I3D OF-Only|HMDB51|ImageNet|61.9%|
|I3D RGB-Only|HMDB51|Kinetics|74.3%|
|I3D OF-Only|HMDB51|Kinetics|77.3%|
 
In TAO Toolkit, we support to use I3D architecture for action recognition and it could alos load the pytorch version of Kinect400 pretrained I3D model to help improve the accuracy of the downstream dataset.

## Load I3D Kinetics pretrained weights and finetune on HMDB51

The I3D architecture in TAO Toolkit is following the public [repo](https://github.com/piergiaj/pytorch-i3d). And this repo also contains the [RGB](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and the [Optical flow](https://github.com/piergiaj/pytorch-i3d/blob/master/models/flow_imagenet.pt) pretrained weights converted from DeepMind.

To load these models, some config options should be set. Take RGB models as an example, the following are the `model_config` in the training config yaml file to load pretrained I3D RGB pretrained weights. 

```yaml
model_config:
  model_type: rgb
  input_type: 3d
  backbone: i3d
  rgb_seq_length: 64
  rgb_pretrained_model_path: /workspace/action_recognition/i3d_pretrained/rgb_imagenet_kinetics.pt
  rgb_pretrained_num_classes: 400
```

In the above config, the `backbone` is set to `i3d`, `rgb_pretrained_model_path` is set to the path of pretrained pytorch weights and the `rgb_pretrained_num_classes` is set to 400 to match with Kinetics-400 classes. 

We provide the [spec](https://github.com/NVIDIA-AI-IOT/tao_toolkit_recipes/blob/main/tao_action_recognition/specs/train_rgb_3d_64_i3d.yaml) to finetune I3D model on HMDB51 dataset. You might get ~75% accuracy after the training with following command.

```shell
tao action_recognition train -e /path/to/train_rgb_3d_64_i3d.yaml -k your_key -r /path/to/results 
```

## Export the I3D model
The exported I3D model could be consumed by TensorRT 8.2.3 and above. We provide the [spec](https://github.com/NVIDIA-AI-IOT/tao_toolkit_recipes/blob/main/tao_action_recognition/specs/i3d_rgb_3d_64_export.yaml) to export TAO Toolkit trained I3D model. And you could use the following command to export the model to etlt format:

```shell
tao action_recognition export -k your_key -e /path/to/i3d_rgb_3d_64_export.yaml 
```

## Reference
- [I3D models trained on Kinetics - pytorch version](https://github.com/piergiaj/pytorch-i3d)
- [I3D models trained on Kinetics](https://github.com/piergiaj/pytorch-i3d)