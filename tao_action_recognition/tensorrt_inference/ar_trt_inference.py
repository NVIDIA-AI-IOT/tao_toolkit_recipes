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
from trt_inference import allocate_buffers, do_inference, load_tensorrt_engine
import numpy as np
import PIL
from PIL import Image
import os

SEQ = 32
CENTER_CROP = False
INPUT_2D = False


def preprocess_ds_ncdhw(batch_img):
    batch_img_array = np.array([np.array(img) for img in batch_img], dtype=np.float32)
    batch_img_array = ((batch_img_array / 255.0) - 0.5) / 0.5
    batch_transpose = np.transpose(batch_img_array, (3, 0, 1, 2))
    if INPUT_2D:
        batch_reshape = np.reshape(batch_transpose, (3*SEQ, 224, 224))
    else:
        batch_reshape = batch_transpose

    return batch_reshape


def test_consecutive_sample(max_sample_cnt, seq_length, sample_rate=1):
    """Choose the middle consecutive frames of each video."""
    total_frames_req = seq_length * sample_rate
    average_duration = max_sample_cnt - total_frames_req + 1
    if average_duration > 0:
        start_idx = int(average_duration/2.0)
    else:
        start_idx = 0

    img_ids = start_idx + np.arange(seq_length) * sample_rate
    # # loop the video to form sequence:
    img_ids = np.mod(img_ids, max_sample_cnt)

    return img_ids


def sample_patch(img_root_path, seq_len=SEQ):
    img_list = sorted(os.listdir(img_root_path))
    img_id_list = []
    if len(img_list) < seq_len:
        img_ids = np.arange(seq_len)
        img_ids = np.mod(img_ids, len(img_list))
        img_id_list.append(img_ids)
    else:
        end_index = len(img_list) - seq_len + 1
        for idx in range(end_index):
            img_ids = idx + np.arange(seq_len)
            img_id_list.append(img_ids)
    return img_id_list


def resize_and_center_crop(img):
    # resize the short side to 224
    w, h = img.size
    if h <= w:
        target_w = int((224.0 / float(h)) * w)
        resized_img = img.resize((target_w, 224), resample=PIL.Image.BILINEAR)
    else:
        target_h = int((224.0 / float(w)) * h)
        resized_img = img.resize((224, target_h), resample=PIL.Image.BILINEAR)

    # center crop to 224x224
    resized_w, resized_h = resized_img.size
    center_x = (resized_w - 224) / 2
    center_y = (resized_h - 224) / 2
    crop_img = resized_img.crop((center_x, center_y, center_x + 224, center_y + 224))

    return crop_img


def load_images(img_ids, img_root_path):
    img_list = sorted(os.listdir(img_root_path))

    raw_imgs = []
    for img_id in img_ids:
        img_path = os.path.join(img_root_path, img_list[img_id])
        img = Image.open(img_path)
        if CENTER_CROP:
            img = resize_and_center_crop(img)
        else:
            img = img.resize((224, 224), resample=PIL.Image.BILINEAR)

        raw_imgs.append(img)

    images = preprocess_ds_ncdhw(raw_imgs)

    return images


def get_prob(pred):

    pred = pred - pred.max()
    pred_exp = np.exp(pred)

    return pred_exp.max()/pred_exp.sum()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Do AR inference using TRT')
    parser.add_argument('--input_images_folder', type=str, help='input images path', required=True)
    parser.add_argument('--trt_engine', type=str, help='trt engine file path', required=True)
    parser.add_argument('--center_crop', action="store_true", help='resize the short side of image to 224 and center crop 224x224')
    parser.add_argument('--input_2d', action="store_true", help='set if it is a 2d model')

    args = parser.parse_args()

    if args.center_crop:
        CENTER_CROP = True

    if args.input_2d:
        INPUT_2D = True

    batch_size = 1
    engine_file = args.trt_engine
    label_map = ["push", "fall_floor", "walk", "run", "ride_bike"]
    img_root = args.input_images_folder
    batch_cnt = 1

    total_cnt = 0
    ac_cnt = 0

    with load_tensorrt_engine(engine_file) as engine:
        with engine.create_execution_context() as context:
            if INPUT_2D:
                context.set_binding_shape(0, (1, 3*SEQ, 224, 224))
            else:
                context.set_binding_shape(0, (1, 3, SEQ, 224, 224))
            inputs, outputs, bindings, stream = allocate_buffers(engine, context)
            img_ids_list = sample_patch(img_root)
            for img_ids in img_ids_list:
                images = load_images(img_ids, img_root)
                for sample_id in range(batch_size):
                    batch_images = images
                    # Hard Coded For explicit_batch and the ONNX model's batch_size = 1
                    batch_images = batch_images[np.newaxis, :, :, :]
                    outputs_shape, outputs_data = do_inference(batch=batch_images, context=context,
                                                               bindings=bindings, inputs=inputs,
                                                               outputs=outputs, stream=stream)

                    pred_data = np.squeeze(outputs_data['fc_pred'])
                    label = label_map[np.argmax(pred_data)]
                    prob = get_prob(pred_data)
                    print("{} : {} {}".format(img_ids, label, prob))
