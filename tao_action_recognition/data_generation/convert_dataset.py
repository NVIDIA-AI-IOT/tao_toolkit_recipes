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

import argparse
import os
import cv2


def clip_video(input_video_path, output_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("f cnt: {}".format(frame_cnt))
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    img_id = 1
    while cap.isOpened():
        ret, frame = cap.read()
        img_name = os.path.join(output_path, str(img_id).zfill(6)+".png")
        if ret:
            cv2.imwrite(img_name, frame)
        else:
            break
        img_id += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clip video to RGB frames')
    parser.add_argument('--input_video', type=str, help='input video path')
    parser.add_argument('--output_folder', type=str, help='output images path')
    args = parser.parse_args()
    clip_video(args.input_video, args.output_folder)
