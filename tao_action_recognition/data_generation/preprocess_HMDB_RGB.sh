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


RUN_WORKERS(){
    for((i=0;i<$WORKER_CNT;i++)); do
        if [ ${VIDEO_LIST[i]} != "NULL" ]; then 
            python3 ./convert_dataset.py --input_video ${VIDEO_LIST[i]} --output_folder ${RGB_PATH_LIST[i]}
        fi
    done
    wait
}

if [ $# -ne 2 ]; then
    echo "USAGE:./preprocess_HMDB_RGB.sh [hmdb_dir] [output_top_dir]"
    exit 1
else
    HMDB_TOP_DIR=$1
    OUTPUT_TOP_DIR=$2
    echo $HMDB_TOP_DIR
    echo $OUTPUT_TOP_DIR
    #TEMP_DIR="./tmp"
    #MKDIR $TEMP_DIR
    MKDIR $OUTPUT_TOP_DIR
fi

# 1st stage: unrar rar package:
# for class in $HMDB_TOP_DIR/*; do 
#     unrar x $class $TEMP_DIR > /dev/null &
# done

# 2nd stage: Clip video and generate optical flow out of it 
for class in $HMDB_TOP_DIR/*; do 
    CLASS_NAME=$(echo $(basename $class) | cut -d . -f1)
    echo "Preprocess $CLASS_NAME"
    cnt=0 
    # extract the frames 
    for video in $HMDB_TOP_DIR/$CLASS_NAME/*; do
        VIDEO_NAME=$(echo $(basename $video) | cut -d . -f1) 
        RGB_PATH=$OUTPUT_TOP_DIR/$CLASS_NAME/$VIDEO_NAME/"rgb"
        MKDIR $RGB_PATH
        VIDEO_LIST[$cnt]=$video
        RGB_PATH_LIST[$cnt]=$RGB_PATH

        cnt=$((cnt + 1))
        if [ $cnt -eq $WORKER_CNT ]; then
            cnt=0
            RUN_WORKERS
            VIDEO_LIST=("NULL" "NULL" "NULL" "NULL")
            RGB_PATH_LIST=("NULL" "NULL" "NULL" "NULL")
        fi 
    done
    if [ $cnt -ne 0 ]; then
        RUN_WORKERS
    fi
done

# rm -r $TEMP_DIR
