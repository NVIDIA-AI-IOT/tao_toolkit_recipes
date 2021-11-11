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
import cv2
import numpy as np
import os


def parse_flow(flow_file):
    """Parse the optical flow vector generated from NVOF SDK."""

    with open(flow_file, "rb") as f:
        _ = f.read(4)
        width = int.from_bytes(f.read(4), byteorder="little", signed=False) 
        height = int.from_bytes(f.read(4), byteorder="little", signed=False)
        data = f.read()
        of_flatten = np.frombuffer(data, dtype=np.float32)
        of_array = np.reshape(of_flatten, (height, width, 2))
        of_array = of_array.transpose((2, 0, 1))
        flow_x = np.squeeze(of_array[0, :, :])
        flow_y = np.squeeze(of_array[1, :, :])

    return flow_x, flow_y


def minmax_grayscale(flow_x, flow_y):
    """Map the flow to grayscale images. The map method follows I3D"""
    higher_end = 20.0
    lower_end = -20.0
    flow_x = np.maximum(np.minimum(255.0, 255.0 * ((flow_x - lower_end) / (higher_end - lower_end))), 0.0)
    flow_y = np.maximum(np.minimum(255.0, 255.0 * ((flow_y - lower_end) / (higher_end - lower_end))), 0.0)

    img_x = np.array(np.around(flow_x), dtype=np.uint8)
    img_y = np.array(np.around(flow_y), dtype=np.uint8)

    return img_x, img_y


def max_rad_grayscale(flow_x, flow_y):
    """Map the flow to grayscale images. Normalize vector using max_rad"""
    max_rad = 1.0
    rad = np.sqrt(flow_x * flow_x + flow_y * flow_y)
    max_rad = max(max_rad, rad.max())

    img_x = np.array((flow_x / max_rad) * 127.999 + 128, dtype=np.uint8)
    img_y = np.array((flow_y / max_rad) * 127.999 + 128, dtype=np.uint8)

    return img_x, img_y


def convert(input_flow_folder, output_folder):
    """Convert the flow in input_flow floder to grayscale images"""

    u_img_root = os.path.join(output_folder, "u")
    v_img_root = os.path.join(output_folder, "v")
    if not os.path.exists(u_img_root):
        os.makedirs(u_img_root)
    if not os.path.exists(v_img_root):
        os.makedirs(v_img_root)

    for flow_name in os.listdir(input_flow_folder):
        frame_id = str(int(flow_name.split("_")[1]) + 1).zfill(6)
        flow_file_path = os.path.join(input_flow_folder, flow_name)
        flow_x, flow_y = parse_flow(flow_file_path)
        img_x, img_y = max_rad_grayscale(flow_x, flow_y)

        img_x_path = os.path.join(u_img_root, frame_id+".jpg")
        img_y_path = os.path.join(v_img_root, frame_id+".jpg")

        cv2.imwrite(img_x_path, img_x)
        cv2.imwrite(img_y_path, img_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert raw optical flow vectors to grayscale images')
    parser.add_argument('--input_flow_folder', type=str, help='input optical flow path', required=True)
    parser.add_argument('--output_folder', type=str, help='output images path', required=True)
    args = parser.parse_args()

    convert(args.input_flow_folder, args.output_folder)