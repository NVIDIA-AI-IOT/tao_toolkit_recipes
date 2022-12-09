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
