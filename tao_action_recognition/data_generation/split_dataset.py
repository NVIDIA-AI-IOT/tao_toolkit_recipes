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

import os
import shutil
import sys

root_path = sys.argv[1]
split_files_path = sys.argv[2]
target_train_path = sys.argv[3]
target_test_path = sys.argv[4]

if not os.path.exists(target_train_path):
    os.makedirs(target_train_path)
if not os.path.exists(target_test_path):
    os.makedirs(target_test_path)

train_cnt = 0
test_cnt = 0
for class_name in os.listdir(root_path):
    split_files = os.path.join(split_files_path, class_name + "_test_split1.txt")
    cls_train_path = os.path.join(target_train_path, class_name)
    cls_test_path = os.path.join(target_test_path, class_name)
    if not os.path.exists(cls_train_path):
        os.makedirs(cls_train_path)
    if not os.path.exists(cls_test_path):
        os.makedirs(cls_test_path)

    with open(split_files, "r") as f:
        split_list = f.readlines()

    for line in split_list:
        video_name, label = line.split()
        video_name = video_name.split(".")[0]
        cur_path = os.path.join(root_path, class_name, video_name)
        if int(label) == 1:
            train_cnt += 1
            des_path = os.path.join(target_train_path, class_name, video_name)
            shutil.move(cur_path, des_path)
        elif int(label) == 2:
            test_cnt += 1
            des_path = os.path.join(target_test_path, class_name, video_name)
            shutil.move(cur_path, des_path)


print("Split 1: \n Train: {}\n Test: {}".format(train_cnt, test_cnt))
