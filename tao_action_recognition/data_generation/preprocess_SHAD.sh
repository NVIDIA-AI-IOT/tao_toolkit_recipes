# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

MKDIR(){
    if [ ! -d $1 ]; then
        mkdir -p $1
    fi 
}

WORKER_CNT=4
VIDEO_LIST=("NULL" "NULL" "NULL" "NULL")
RGB_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
OF_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
OF_IMG_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
ANNO_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
TEMP_VIDEO_PATH_LIST=("NULL" "NULL" "NULL" "NULL")


RUN_WORKERS(){
    for((i=0;i<$WORKER_CNT;i++)); do
        if [ ${VIDEO_LIST[i]} != "NULL" ]; then 
            python3 ./convert_dataset.py --input_video ${VIDEO_LIST[i]} --output_folder ${RGB_PATH_LIST[i]} \
            && ./AppOFCuda --input=${RGB_PATH_LIST[i]}/"*.png" --output=${OF_PATH_LIST[i]}/"flow" --preset=slow --gridSize=1 \
            && python3 ./convert_of.py --input_flow_folder=${OF_PATH_LIST[i]} --output_folder=${OF_IMG_PATH_LIST[i]} \
            && python3 ./save_tracks_shad.py --anno_folder ${ANNO_PATH_LIST[i]} --image_folder ${RGB_PATH_LIST[i]} \
            --of_folder ${OF_IMG_PATH_LIST[i]} --output_folder $TEMP_DIR_ & 
        fi
    done
    wait
    for((i=0;i<$WORKER_CNT;i++)); do
        if [ ${VIDEO_LIST[i]} != "NULL" ]; then 
            rm -r ${TEMP_VIDEO_PATH_LIST[i]} 
        fi
    done
}

if [ $# -ne 2 ]; then
    echo "USAGE:./preprocess_SHAD.sh [shad_dataset_top_dir] [output_top_dir]"
    exit 1
else
    SHAD_TOP_DIR=$1
    OUTPUT_TOP_DIR=$2
    echo $SHAD_TOP_DIR
    echo $OUTPUT_TOP_DIR
    TEMP_DIR="./tmp"
    TEMP_DIR_="./tmp_"
    MKDIR $TEMP_DIR
    MKDIR $TEMP_DIR_
    MKDIR $OUTPUT_TOP_DIR
fi

# 1st stage: Clip video and generate optical flow out of it 
for class in $SHAD_TOP_DIR/*; do 
    if [ ! -d $class/"video"/ ]; then
        echo "Please use original SHAD dataset"
        exit 1
    fi
    echo "Preprocess $class"
    CLASS_NAME=$(basename $class)
    cnt=0 
    for video in $class/"video"/*; do
        VIDEO_NAME=$(echo $(basename $video) | cut -d . -f1) 
        ANNO_PATH=$class/"Annotations"/$VIDEO_NAME
        RGB_PATH=$TEMP_DIR/$CLASS_NAME/$VIDEO_NAME/"rgb"
        OF_PATH=$TEMP_DIR/$CLASS_NAME/$VIDEO_NAME/"of"
        OF_IMG_PATH=$TEMP_DIR/$CLASS_NAME/$VIDEO_NAME/"of_img"
        TEMP_VIDEO_PATH=$TEMP_DIR/$CLASS_NAME/$VIDEO_NAME
        MKDIR $RGB_PATH
        MKDIR $OF_PATH
        MKDIR $OF_IMG_PATH
        VIDEO_LIST[$cnt]=$video
        ANNO_PATH_LIST[$cnt]=$ANNO_PATH
        RGB_PATH_LIST[$cnt]=$RGB_PATH
        OF_PATH_LIST[$cnt]=$OF_PATH
        OF_IMG_PATH_LIST[$cnt]=$OF_IMG_PATH
        TEMP_VIDEO_PATH_LIST[$cnt]=$TEMP_VIDEO_PATH

        cnt=$((cnt + 1))
        if [ $cnt -eq $WORKER_CNT ]; then
            cnt=0
            RUN_WORKERS
            VIDEO_LIST=("NULL" "NULL" "NULL" "NULL")
            RGB_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
            OF_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
            OF_IMG_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
            ANNO_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
            TEMP_VIDEO_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
        fi 
    done
    if [ $cnt -ne 0 ]; then
        RUN_WORKERS
    fi
done

rm -r $TEMP_DIR

python generate_new_dataset_format.py $TEMP_DIR_ $OUTPUT_TOP_DIR

rm -r $TEMP_DIR_
