# Sample to estimate best scales and aspect ratio values for TAO retinanet

This is an experimental sample to estimate best scales and aspect ratio values for TAO retinanet:

```
retinanet_config {
  aspect_ratios_global: "[1.0, 2.0, 0.5]"
  scales: "[0.05, 0.2, 0.35, 0.5, 0.65, 0.8]"
```

Please do try more parameters for best model performance.


## Detailed Steps

### Download kitti dataset

Assume link below has the txt label files:
```
/home/user/tlt-experiments/data/training/label_2/
```



### Prepare parameters for sample to estimate optimal values

#### Change tao_retinanet_scales_aspect_ratio_estimate.py to point to correct folder for labels
```
folder="/home/user/tlt-experiments/data/training/label_2/"
```


#### Change tao_retinanet_scales_aspect_ratio_estimate.py to set shorten value of image width and image height
```
shorter_length_of_image = 375
```


##### Change tao_retinanet_scales_aspect_ratio_estimate.py to remove outliers for aspect ratios
```
limit_max_ar=4
```



### Run sample to estimate optimal values

```
python tao_retinanet_scales_aspect_ratio_estimate.py
```



### Running log with kitti dataset


```
scales:  [0.0691874  0.13098365 0.21473368 0.33218772 0.48606437 0.82403735]
aspect ratios:  [0.52757496 0.95228595 2.57963582]
```

