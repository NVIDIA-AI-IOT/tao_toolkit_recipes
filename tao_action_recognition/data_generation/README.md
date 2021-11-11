# Data generation sample for TAO ActionRecognitionNet

## Introduction
This projects contains the sample scripts to generate dataset to proper format used by TAO ActionRecognitionNet

- `convert_dataset.py` : Convert the video to RGB frames.
- `convert_of.py` : Convert the optical flow vectors to grayscale images.  
- `split_dataset.py` : Script to split the HMDB51 dataset.
- `load_tracks.py` / `save_tracks_shad.py` : Scripts to process SHAD dataset's annotation


## Prequisites
 - xmltodict
 - cv2

```
pip install xmltodict opencv-python
```

And we use the sample application `AppOFCuda` in Nvidia optical flow [SDK]() to generate optical flow of frames. You could get this app by compiling by yourself or download the compiled binary in on [NGC](). 

## Steps to generate dataset for TAO ActionRecognitionNet
We provide 3 all_in_one scripts:

- `preprocess_HMDB_RGB.sh`: Generate RGB dataset of HMDB51
- `preprocess_SHAD_RGB.sh`: Generate RGB dataset of SHAD
- `preprocess_SHAD.sh`: Generate RGB+OF dataset of SHAD

### SHAD dataset

Dataset [URL](https://best.sjtu.edu.cn/Data/View/990)

```sh
# make directory to contain 
mkdir -p train_raw

# Download the dataset you need and unrar:
wget -P ./ https://best.sjtu.edu.cn/Assets/userfiles/sys_eb538c1c-65ff-4e82-8e6a-a1ef01127fed/files/ZIP/Bend-train.rar
unrar x Bend-train.rar train_raw
...

# Generate RGB dataset with all_in_one script:
./preprocess_SHAD_RGB.sh train_raw train
# Or you can generate RGB+OF dataset:
# ./preprocess_SHAD.sh train_raw train

``` 

### HMDB51 dataset

Dataset [URL](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

```sh
# download the dataset and unrar:
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51_org.rar video_rar

# unrar the videos packages:
unrar x ./video_rar/climb.rar ./HMDB51_videos/
unrar x ./video_rar/run.rar ./HMDB51_videos/
...

# run all_in_one script:
./preprocess_HMDB_RGB.sh ./HMDB51_videos ./HMDB51

# split the dataset if needed:
# python split_dataset.py

```