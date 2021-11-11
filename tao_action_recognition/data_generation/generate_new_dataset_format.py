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
import sys

root_dir = sys.argv[1]
target_dir = sys.argv[2]

for class_name in os.listdir(root_dir):
    root_class_path = os.path.join(root_dir, class_name)
    target_class_path = os.path.join(target_dir, class_name)
    if not os.path.exists(target_class_path):
        os.makedirs(target_class_path)
    for video_name in os.listdir(root_class_path):
        video_path = os.path.join(root_class_path, video_name)
        target_video_path = os.path.join(target_class_path, video_name)
        target_rgb_path = os.path.join(target_video_path, "rgb")
        target_u_path = os.path.join(target_video_path, "u")
        target_v_path = os.path.join(target_video_path, "v")

        if not os.path.exists(target_rgb_path):
            os.makedirs(target_rgb_path)
        if not os.path.exists(target_u_path):
            os.makedirs(target_u_path)
        if not os.path.exists(target_v_path):
            os.makedirs(target_v_path)

        img_idx = 0
        for video_clip_name in sorted(os.listdir(video_path)):
            video_clip_path = os.path.join(video_path, video_clip_name)
            rgb_path = os.path.join(video_clip_path, "rgb")
            u_path = os.path.join(video_clip_path, "u")
            v_path = os.path.join(video_clip_path, "v")

            assert len(os.listdir(u_path)) == \
                len(os.listdir(v_path)), "video clip mismatch. {}".format(video_clip_path)

            for file_name in sorted(os.listdir(rgb_path)):
                ext = file_name.split(".")[-1]
                rgb_file = os.path.join(rgb_path, file_name)
                u_file = os.path.join(u_path, file_name)
                v_file = os.path.join(v_path, file_name)

                target_file_name = str(img_idx).zfill(6) + "." + ext
                img_idx += 1
                target_rgb_file = os.path.join(target_rgb_path, target_file_name)
                target_u_file = os.path.join(target_u_path, target_file_name)
                target_v_file = os.path.join(target_v_path, target_file_name)

                os.rename(rgb_file, target_rgb_file)
                if os.path.exists(u_file):
                    os.rename(u_file, target_u_file)
                if os.path.exists(v_file):
                    os.rename(v_file, target_v_file)

