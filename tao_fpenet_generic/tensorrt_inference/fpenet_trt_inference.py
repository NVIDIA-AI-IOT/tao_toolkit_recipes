# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import cv2
from trt_inference import allocate_buffers, do_inference, load_tensorrt_engine
import json
import numpy as np
import os
import tqdm

INPUT_CHANNEL=1
INPUT_WIDTH=80
INPUT_HEIGHT=80
NUM_KEYPOINTS=6


def preprocess(sample):
    fname = str(sample['filename'])

    for chunk in sample['annotations']:
        if 'facebbox' not in chunk['class'].lower():
            continue

        bbox_data = (entry for entry in chunk if ('class' not in entry and
                                                    'version' not in entry))
        for entry in bbox_data:
            if 'face_tight_bboxheight' in str(entry).lower():
                height = int(float(chunk[entry]))
            if 'face_tight_bboxwidth' in str(entry).lower():
                width = int(float(chunk[entry]))
            if 'face_tight_bboxx' in str(entry).lower():
                x = int(float(chunk[entry]))
            if 'face_tight_bboxy' in str(entry).lower():
                y = int(float(chunk[entry]))

        image = cv2.imread(os.path.join(fname))

        image_shape = image.shape
        image_height = image_shape[0]
        image_width = image_shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)

        # transform it into a square bbox wrt the longer side
        longer_side = max(width, height)
        new_width = longer_side
        new_height = longer_side
        x = int(x - (new_width - width) / 2)
        y = int(y - (new_height - height) / 2)
        x = min(max(x, 0), image_width)
        y = min(max(y, 0), image_height)
        new_width = min(new_width, image_width - x)
        new_height = min(new_height, image_height - y)
        new_width = min(new_width, new_height)
        new_height = new_width  # make it a square bbox
        crop_bbox = [x, y, new_width, new_height]

        # crop the face bounding box
        img_crop = image[y:y + new_height, x:x + new_width, :]  # pylint:disable=E1136
        image_resized = cv2.resize(img_crop,
                                   (INPUT_HEIGHT, INPUT_WIDTH),
                                    interpolation=cv2.INTER_CUBIC)
        if INPUT_CHANNEL == 1:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            image_resized = np.expand_dims(image_resized, 2)
        # make it channel first (channel, height, width)
        image_resized = np.transpose(image_resized, (2, 0, 1))
        image_resized = np.expand_dims(image_resized, 0).astype(np.float32)  # add batch dimension
        
        return crop_bbox, image_resized


def postprocess(outputs, crop_bbox):

    keypoints = outputs['softargmax/strided_slice:0']
    scale = float(crop_bbox[2]) / INPUT_HEIGHT
    shift = np.tile(np.array((crop_bbox[0], crop_bbox[1])),
                    (NUM_KEYPOINTS, 1))
    result = (keypoints[0, :, :] * scale) + shift
    
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Do FPENet inference using TRT')
    parser.add_argument('--input_json', type=str, help='input json path', required=True)
    parser.add_argument('--trt_engine', type=str, help='trt engine file path', required=True)
    parser.add_argument('--output_img_dir', type=str, help='output imgs save path')

    args = parser.parse_args()

    batch_size = 1
    engine_file = args.trt_engine
    input_json = args.input_json
    output_dir = args.output_img_dir


    with load_tensorrt_engine(engine_file) as engine:
        with engine.create_execution_context() as context:
            context.set_binding_shape(0, (1, INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH))
            inputs, outputs, bindings, stream = allocate_buffers(engine, context)
            json_data = json.loads(open(input_json , 'r').read())
            results = []
            for sample in tqdm.tqdm(json_data):
                fname = str(sample['filename'])
                crop_bbox, img = preprocess(sample)
                outputs_shape, outputs_data = do_inference(batch=img, context=context,
                                                           bindings=bindings, inputs=inputs,
                                                           outputs=outputs, stream=stream)
                keypoints = postprocess(outputs_data, crop_bbox)
                keypoints = keypoints[0]
                img = cv2.imread(fname)
                for idx, kp in enumerate(keypoints):
                    x = kp[0]
                    y = kp[1]
                    cv2.circle(img,(int(x), int(y)), 1, (0,255,0), 2)
                    cv2.putText(img, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                cv2.imwrite(os.path.join(output_dir, fname.split("/")[-1]), img)
