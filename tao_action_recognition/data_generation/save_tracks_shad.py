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
import os
from load_tracks import *

parser = argparse.ArgumentParser(
    description='Save trajectory crops of a video to disk')
parser.add_argument('--anno_folder', dest='anno_folder',
                    help="input folder containing annotations for building track class")
parser.add_argument('--image_folder', dest='image_folder',
                    help="input folder containing images for building track class")
parser.add_argument('--of_folder', dest='of_folder',
                    help="input folder containing of for building track class")
parser.add_argument('--output_folder', dest='output_folder', default="./",
                    help="folder to store crops")


Class_Labels = ["walk", "fall", "sits", "squa", "bend"]


def get_generator_for_tracks_for_video(annotation_folder, image_folder, of_folder, num_channels):
    output_image_annotation_folder = './dummy'
    all_tracks = Tracks(annotation_folder, image_folder, of_folder, output_image_annotation_folder)
    track_generator = Data_generator_from_track(
        num_channels, all_tracks, add_color=True)
    return track_generator


def prepare_generator(args):
    trk_length = 10
    trk_generator = get_generator_for_tracks_for_video(
        args.anno_folder, args.image_folder, args.of_folder, trk_length)
    _basename = os.path.basename(args.anno_folder)
    if args.anno_folder[-1] == "/":
        _basename = args.anno_folder[:-1]
        _basename = os.path.basename(_basename)
    return trk_generator, _basename


def prepare_folders(output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # make directories for classes
    for i in range(len(Class_Labels)):
        _tmp = os.path.join(output_folder, Class_Labels[i])
        if not(os.path.isdir(_tmp)):  # can be made by other video
            os.makedirs(_tmp)


def save_images_to_disk(images, folder, ext=".jpg"):
    for idx, img in enumerate(images):
        _tmp = "%06d" % idx + ext
        _tmp = os.path.join(folder, _tmp)
        assert(cv2.imwrite(_tmp, img))


def save_crops(track_generator, video_name, args):
    sample_counter_per_class = {"walk": 0,
                                "fall": 0, "sits": 0, "squa": 0, "bend": 0}
    while True:
        try:
            gen_val = track_generator.next()
        except NoTrajectoryLeft:
            break
        except NotEnoughPointsInTrack:
            continue
        else:
            # do actual crop save to disk
            action_label = gen_val[-1]
            # sometime this will return NONE - TODO Investigate - one reason could be this part of track is not labeled
            print(action_label)
            if (action_label in sample_counter_per_class.keys()):
                sample_counter_per_class[action_label] += 1
                counter = sample_counter_per_class[action_label]
                str_counter = "%06d" % counter
                opt_folder = os.path.join(args.output_folder, action_label)
                opt_folder = os.path.join(opt_folder, video_name)
                if not os.path.isdir(opt_folder):
                    os.makedirs(opt_folder)
                _opf_folder_u = os.path.join(
                    opt_folder, str_counter + "/u/")
                _opf_folder_v = os.path.join(
                    opt_folder, str_counter + "/v/")
                _rgb_folder = os.path.join(
                    opt_folder, str_counter + "/rgb/")
                # above folders should NOT exist
                print(_opf_folder_u)
                assert not(os.path.isdir(_opf_folder_u))
                assert not(os.path.isdir(_opf_folder_v))
                assert not(os.path.isdir(_rgb_folder))
                os.makedirs(_opf_folder_u)
                os.makedirs(_opf_folder_v)
                os.makedirs(_rgb_folder)
                save_images_to_disk(gen_val[0], _opf_folder_u)
                save_images_to_disk(gen_val[1], _opf_folder_v)
                save_images_to_disk(gen_val[-2], _rgb_folder)


def main(args):
    trk_generator, video_name = prepare_generator(args)
    prepare_folders(args.output_folder)
    save_crops(trk_generator, video_name, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
