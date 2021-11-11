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
from __future__ import division
import collections
import csv
import glob
import io
import os
import re
import cv2
import xmltodict
import numpy as np  # for confusion_matrix

# make one trajectory structure


class track(object):
    def __init__(self, ts, te, label, person_id):
        self.ts = int(ts)
        self.te = int(te)
        label = label.strip()
        self.action_label = {}
        self.action_label[label] = [(int(ts), int(te))]
        self.person_id = int(person_id)  # id in gtruth
        self.bboxes = {}
        self.interp_boxes = {}
        self.sorted_frame_nums = []
        self.all_boxes = {}

    def add_action(self, a, ts, te):
        if not(a in self.action_label.keys()):
            self.action_label[a] = [(int(ts), int(te))]
        else:
            self.action_label[a].append((int(ts), int(te)))

    def fill_box(self, bbox, t):
        # bbox should be in [x1, y1, x2, y2]
        self.bboxes[t] = bbox

    def print_track_initialization(self):
        print("trajectory is for person: {}, with Ts: {}, Te: {}, Action: {}"
              .format(self.person_id, self.ts, self.te, self.action_label))

    def print_track_details(self):
        self.print_track_initialization()
        print("printing details for bboxes below:")
        print("number of bboxes: {}".format(len(self.bboxes)))
        for key, value in self.bboxes.items():
            print("bbox for frame: {}, are x1, y1, x2, y2: {}".
                  format(key, value))
        self.interpolate_track()

    def get_interpolated_boxes(self, t1, t2):
        assert(t1-t2 != 0)
        del_t = (t2 - t1)
        bbox1 = self.bboxes[t1]
        bbox2 = self.bboxes[t2]
        del_x1 = (bbox2[0] - bbox1[0])/del_t
        del_y1 = (bbox2[1] - bbox1[1])/del_t
        del_x2 = (bbox2[2] - bbox1[2])/del_t
        del_y2 = (bbox2[3] - bbox1[3])/del_t
        pos = t1
        for i in range(del_t - 1):
            x1 = (i+1)*del_x1 + bbox1[0]
            y1 = (i+1)*del_y1 + bbox1[1]
            x2 = (i+1)*del_x2 + bbox1[2]
            y2 = (i+1)*del_y2 + bbox1[3]
            pos += 1
            self.interp_boxes[pos] = [x1, y1, x2, y2]

    def find_next_key(self, cur_key):
        assert(isinstance(cur_key, int))
        ret_key = cur_key + 1
        found = False
        while found is False:
            if (ret_key in sorted(self.bboxes)):
                found = True
            else:
                ret_key += 1
        return ret_key

    def interpolate_track(self):
        # find key + 1 in the bboxes
        # if it is not found - then find the next successive key
        # interpolate - make a list which needs to be inserted later
        # add a sanity check to see everything is continuous
        len_dict = len(self.bboxes.keys())
        len_counter = 0
        for key in sorted(self.bboxes.keys()):
            len_counter += 1
            if (len_counter == len_dict):
                break
            next_key = self.find_next_key(key)
            if (next_key != key):
                assert(next_key > key)
                self.get_interpolated_boxes(key, next_key)
        self.sanity_check_continuity()  # TODO - this needs to be without visual
        # also compute sorted_frame_nums
        self.all_boxes = self.get_all_boxes()
        self.sorted_frame_nums = sorted(
            self.all_boxes.items(), key=lambda x: x[0])

    def _merge_two_dicts(self, x, y):
        """
        needed for python 2.7
        """
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z

    def sanity_check_continuity(self):
        # print("printing sanity checks! WITH increasing frame number for boxes")
        # z= {**self.bboxes, **self.interp_boxes} this is for python 3.5+
        # z = self._merge_two_dicts(self.bboxes, self.interp_boxes)
        """
        for key in sorted(z):
            print (key)
            print(z[key])
        """
        # add checks on unique keys
        for key in self.bboxes.keys():
            assert not(key in self.interp_boxes)

    def get_all_boxes(self):
        """
        returns all boxes with interp
        """
        # return {**self.bboxes, **self.interp_boxes} this is for python 3.5+
        return self._merge_two_dicts(self.bboxes, self.interp_boxes)

    def get_action_label_for_a_frame(self, frame):
        for key, value in self.action_label.items():
            for val in value:
                if frame >= val[0] and frame <= val[1]:
                    return key
        # how to assert and check this ?


class Tracks(object):
    # make class of all trajectories from a video
    def __init__(self, annotation_folder, image_folder, of_folder, output_test_folder):
        if annotation_folder[-1] == "/":
            annotation_folder = annotation_folder[:-1]  # getting rid of last /
        if image_folder[-1] == "/":
            image_folder = image_folder[:-1]  # getting rid of last /
        if of_folder[-1] == "/":
            of_folder = of_folder[:-1]  # getting rid of last /
        self.top_folder = annotation_folder
        self.tracks = {}  # key is track id, i.e. person id
        _basename = os.path.basename(self.top_folder)
        self.txt_fname = os.path.join(self.top_folder,
                                      _basename + "-multi.txt")
        print("base text file is: {}".format(self.txt_fname))
        self.read_text_file()
        self.images_folder = image_folder
        self.optical_flow_folder = of_folder
        self.image_fnames = []
        self.opf_u_fnames = []
        self.opf_v_fnames = []
        self._load_image_filenames()
        self.images = []  # for this small dataset, loading image files
        self.u_images = []
        self.v_images = []
        self.output_folder = os.path.join(output_test_folder, _basename)
        # assuming the w,h stays constant across xmls for the same dataset
        self.width = -1
        self.height = -1
        self._find_and_read_xml_files()

    def make_a_track(self, a, ts, te, pid):
        if pid in self.tracks.keys():
            self.add_to_an_existing_track(
                a, ts, te, pid)
        else:
            self.tracks[pid] = track(ts, te, a, pid)

    def add_to_an_existing_track(self, a, ts, te, pid):
        if self.tracks[pid].ts > ts:
            self.tracks[pid].ts = ts
        if self.tracks[pid].te < te:
            self.tracks[pid].te = te
        self.tracks[pid].add_action(a, ts, te)

    def read_text_file(self):
        fname = self.txt_fname
        with io.open(fname, encoding="latin-1") as fp:
            # each line is a separate track, with some lines are special
            count = 0
            for line in fp.readlines():
                count += 1
                csv_split = line.split(',')
                for i in range(len(csv_split)):
                    # remove special chars
                    tmp = csv_split[i]
                    tmp = re.sub(r'\W+', ' ', tmp)
                    csv_split[i] = tmp
                if len(csv_split) == 3:
                    tmp = csv_split[0].strip(" ")
                    tmp = tmp.split(" ")
                    if len(tmp) == 2:
                        s1 = csv_split[0]
                        s1 = s1.strip()
                        s2 = csv_split[1]
                        s3 = csv_split[2]
                        action_label = s1.split(" ")[1]
                        action_label = action_label.strip()
                        action_label = action_label.lower()
                        person_id = int(s1.split(" ")[0])
                        ts = int(s2)
                        te = int(s3)
                        self.make_a_track(
                            action_label, ts, te, person_id)
                    if len(tmp) == 1:
                        s1 = csv_split[0]
                        s2 = csv_split[1]
                        s2 = s2.strip()
                        s3 = csv_split[2]
                        if len(s2) == 2:
                            action_label = s2.split(" ")[0]
                            action_label = action_label.strip()
                            action_label = action_label.lower()
                            person_id = int(s1)
                            ts = int(s2.split(" ")[1])
                            te = int(s3)
                            self.make_a_track(
                                action_label, ts, te, person_id)
                        elif len(s2) == 1:
                            s3 = s3.strip()
                            action_label = s2.split(" ")[0]
                            action_label = action_label.strip()
                            action_label = action_label.lower()
                            person_id = int(s1)
                            ts = int(s3.split(" ")[0])
                            te = int(s3.split(" ")[1])
                            self.make_a_track(
                                action_label, ts, te, person_id)

                elif len(csv_split) == 2:
                    if len(csv_split[0].split(" ")) == 2:
                        s1 = csv_split[0]
                        s2 = csv_split[1]
                        s1 = s1.strip()
                        s2 = s2.strip()
                        action_label = s1.split(" ")[1]
                        action_label = action_label.strip()
                        action_label = action_label.lower()
                        person_id = int(s1.split(" ")[0])
                        ts = int(s2.split(" ")[0])
                        te = int(s2.split(" ")[1])
                        self.make_a_track(
                            action_label, ts, te, person_id)
                    elif len(csv_split[0].split(" ")) == 1:
                        s1 = csv_split[0]
                        s2 = csv_split[1]
                        s1 = s1.strip()
                        s2 = s2.strip()
                        action_label = s2.split(" ")[0]
                        action_label = action_label.strip()
                        action_label = action_label.lower()
                        ts = int(s2.split(" ")[1])
                        te = int(s2.split(" ")[2])
                        person_id = int(s1)
                        self.make_a_track(
                            action_label, ts, te, person_id)
                    else:
                        # : > 2
                        s1 = csv_split[0]
                        s2 = csv_split[1]
                        s1 = s1.strip()
                        s2 = s2.strip()
                        action_label = s1.split(" ")[1]
                        action_label = action_label.strip()
                        action_label = action_label.lower()
                        person_id = int(s1.split(" ")[0])
                        ts = int(s1.split(" ")[2])
                        te = int(s2)
                        self.make_a_track(
                            action_label, ts, te, person_id)

                elif len(csv_split) == 1:
                    s1 = csv_split[0]
                    s1 = s1.strip()
                    s1 = s1.split(" ")
                    action_label = s1[1]
                    action_label = action_label.strip()
                    action_label = action_label.lower()
                    person_id = int(s1[0])
                    ts = int(s1[2])
                    te = int(s1[3])
                    self.make_a_track(
                        action_label, ts, te, person_id)
                else:
                    action_label = csv_split[1]
                    action_label = action_label.strip()
                    action_label = action_label.lower()
                    person_id = int(csv_split[0])
                    ts = int(csv_split[2])
                    te = int(csv_split[3])
                    self.make_a_track(
                        action_label, ts, te, person_id)

    def print_all_tracks(self):
        print("total tracks: {}".format(len(self.tracks)))
        for key, val in self.tracks.items():
            print("printing initialize information for person id: {}"
                  .format(key))
            val.print_track_details()

    def _find_and_read_xml_files(self):
        fnames = glob.glob(self.top_folder+"/*.xml")
        self.read_all_xmls(fnames)

    def _load_image_filenames(self):
        """
        png for rgb and jpg for opf
        """
        self.image_fnames = sorted(glob.glob(self.images_folder+"/*.png"))
        self.opf_u_fnames = sorted(
            glob.glob(self.optical_flow_folder+"/u/*.jpg"))
        self.opf_v_fnames = sorted(
            glob.glob(self.optical_flow_folder+"/v/*.jpg"))
        print("loaded filenames for {} images, {} u-opf, {} v-opf".format(
            len(self.image_fnames), len(self.opf_u_fnames),
            len(self.opf_v_fnames)))

    def interpolate_all_tracks(self):
        for key, val in self.tracks.items():
            val.interpolate_track()

    def read_one_xml(self, fname):
        document_file = open(fname, "r")
        original_doc = document_file.read()
        document = xmltodict.parse(original_doc)
        # filename = document['xml']['filename']
        framenum = int(document['xml']['source']['framenum'])
        verify_object = document['xml']['pedestriandescription']['ID']
        self.width = int(document['xml']['size']['width'])
        self.height = int(document['xml']['size']['height'])
        if (isinstance(verify_object, collections.OrderedDict)):
            all_ids = verify_object['item']
            num_ids = len(all_ids)
            for i in range(num_ids):
                id = int(all_ids[i])
                x1 = int(document['xml']['pedestriandescription']
                         ['bndbox']['item'][i]['xmin'])
                y1 = int(document['xml']['pedestriandescription']
                         ['bndbox']['item'][i]['ymin'])
                x2 = int(document['xml']['pedestriandescription']
                         ['bndbox']['item'][i]['xmax'])
                y2 = int(document['xml']['pedestriandescription']
                         ['bndbox']['item'][i]['ymax'])
                if (id in self.tracks.keys()):
                    self.tracks[id].bboxes[framenum] = [x1, y1, x2, y2]
        else:
            id = int(document['xml']['pedestriandescription']['ID'])
            x1 = int(document['xml']['pedestriandescription']
                     ['bndbox']['xmin'])
            y1 = int(document['xml']['pedestriandescription']
                     ['bndbox']['ymin'])
            x2 = int(document['xml']['pedestriandescription']
                     ['bndbox']['xmax'])
            y2 = int(document['xml']['pedestriandescription']
                     ['bndbox']['ymax'])
            # outrageous : Sometimes the ID itself does not exist in multi.txt file
            if id in self.tracks:
                self.tracks[id].bboxes[framenum] = [x1, y1, x2, y2]

    def read_one_xml_with_only_one_person(self, fname):
        document_file = open(fname, "r")
        original_doc = document_file.read()
        document = xmltodict.parse(original_doc)
        # filename = document['xml']['filename']
        framenum = int(document['xml']['source']['framenum'])
        print(document['xml']['pedestriandescription']['ID'])
        print(fname)
        id = int(document['xml']['pedestriandescription']['ID'])
        x1 = int(document['xml']['pedestriandescription']['bndbox']['xmin'])
        y1 = int(document['xml']['pedestriandescription']['bndbox']['ymin'])
        x2 = int(document['xml']['pedestriandescription']['bndbox']['xmax'])
        y2 = int(document['xml']['pedestriandescription']['bndbox']['ymax'])
        self.tracks[id].bboxes[framenum] = [x1, y1, x2, y2]

    def read_all_xmls(self, fnames):
        for fname in fnames:
            self.read_one_xml(fname)

    def _draw_box(self, bbox, _image):
        cv2.rectangle(_image, (int(bbox[0]), int(
            bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 5)

    def _putTextOnImage(self, text, bbox, image, isPred=False):
        borderColor = (0, 0, 0)
        text_pos = (int(bbox[0]), int(bbox[1]) - 4)
        if isPred:
            # change location of prediction text
            text_pos = (int(bbox[2]), int(bbox[3]) - 4)
        text_background_pos = [(int(bbox[0]), int(bbox[1])),
                               (int(bbox[0]), int(bbox[1]) - 22)]
        if isPred:
            text_background_pos = [(int(bbox[2]), int(bbox[3])),
                                   (int(bbox[2]), int(bbox[3]) - 22)]

        if text_background_pos[1][1] <= 0:
            text_background_pos[1] = (
                text_background_pos[1][0], int(bbox[1]) + 22)
            text_pos = (text_pos[0], text_pos[1] + 22)
        textWidth = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
        text_background_pos[1] = (text_background_pos[1]
                                  [0] + textWidth + 1,
                                  text_background_pos[1][1])
        cv2.rectangle(image, text_background_pos[0],
                      text_background_pos[1], borderColor, -1)
        cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), thickness=1)

    def _annotate_tracks(self, _images, prediction_label_map, prediction_output_file=""):
        assert(len(_images) != 0)
        # if prediction output file is present, then we will have to color the output
        if not prediction_output_file:
            assert not (len(prediction_label_map.keys()) == 0)
        num_classes = len(prediction_label_map.keys())
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        grepd_struct = 0
        if prediction_output_file:
            grepd_struct = get_prediction(prediction_output_file)
        for key, val in self.tracks.items():
            cum_box_dict = val.get_all_boxes()
            cum_action_label = val.action_label
            print("Annotating this person (trackID): {}, and action label {}".
                  format(
                      key, cum_action_label))
            for key in sorted(cum_box_dict):
                bbox = cum_box_dict[key]
                framenum = key
                # Outrageous labeling - Some te are more than number of frames: TODO: remove this are read time
                if (framenum >= len(_images)):
                    continue
                self._draw_box(bbox, _images[framenum])
                gtruth_action_label = val.get_action_label_for_a_frame(
                    framenum)
                self._putTextOnImage(gtruth_action_label,
                                     bbox, _images[framenum], isPred=False)
                if grepd_struct is not 0:
                    pred_action_label = grepd_struct.get_action_label_for_a_frame_for_person(
                        framenum, val.person_id)
                    # check if gtruth id exists in map
                    if gtruth_action_label in prediction_label_map.keys() and pred_action_label in prediction_label_map.keys():
                        gtruth_action_id = int(
                            prediction_label_map[gtruth_action_label])
                        pred_action_id = int(
                            prediction_label_map[pred_action_label])
                        confusion_matrix[gtruth_action_id][pred_action_id] += 1
                    self._putTextOnImage(pred_action_label,
                                         bbox, _images[framenum], isPred=True)
        return confusion_matrix

    def _save_images(self, images, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for idx, img in enumerate(images):
            fname = "%06d" % idx
            fname = os.path.join(output_folder, fname+".jpg")
            retval = cv2.imwrite(fname, img)
            assert(retval)  # else writing to a file is not correct

    def annotate_tracks_and_save_images(self, prediction_label_map, prediction_output_file=""):
        self.images = self._load_images_from_filenames(self.image_fnames)
        confusion_matrix = self._annotate_tracks(
            self.images, prediction_label_map, prediction_output_file)
        assert(len(self.images) != 0)
        self._save_images(self.images, self.output_folder)
        return confusion_matrix

    def save_trajectory_info(self):
        output_fname = os.path.join(self.top_folder, "trajectory_action.txt")
        print('printing to this file: {}'.format(output_fname))
        with open(output_fname, "w") as f:
            writer = csv.writer(f)
            for key, val in self.tracks.items():
                print("Annotating this person (trackID): {} ".format(key))
                cum_box_dict = val.get_all_boxes()
                action_label = val.action_label
                for key in sorted(cum_box_dict):
                    bbox = cum_box_dict[key]
                    framenum = key
                    row = (int(bbox[0]), int(bbox[1]), int(
                        bbox[2]), int(bbox[3]), framenum, action_label)
                    writer.writerow(row)

    def _load_images_from_filenames(self, filenames, load_type=1):
        """
        load type is for reading grayscale (0), native or rgb (1)
        """
        _images = []
        for f in filenames:
            img = cv2.imread(f, load_type)
            img = cv2.resize(img, (self.width, self.height))
            if img is not None:
                _images.append(img)
        print("total images loaded: {}, out of {} number of files".format(
            len(_images), len(filenames)))
        return _images

    def save_annotated_optical_flow_images(self):
        """ This is mainly for sanity checks: """
        u_image_fnames = sorted(glob.glob(self.optical_flow_folder+"/u/*.jpg"))
        v_image_fnames = sorted(glob.glob(self.optical_flow_folder+"/v/*.jpg"))
        u_images = self._load_images_from_filenames(u_image_fnames, 0)
        v_images = self._load_images_from_filenames(v_image_fnames, 0)
        self._annotate_tracks(u_images)
        self._annotate_tracks(v_images)
        opf_u_folder = os.path.join(self.output_folder, 'u')
        opf_v_folder = os.path.join(self.output_folder, 'v')
        self._save_images(u_images, opf_u_folder)
        self._save_images(v_images, opf_v_folder)

    def prepare_data_for_generator(self, add_color=False):
        self.u_images = self._load_images_from_filenames(self.opf_u_fnames, 0)
        self.v_images = self._load_images_from_filenames(self.opf_v_fnames, 0)
        assert(len(self.u_images) == len(self.v_images))
        self.interpolate_all_tracks()
        self.images = []
        if add_color is True:
            self.images = self._load_images_from_filenames(self.image_fnames)
        return len(self.images)

    def _extend_context(self, bbox):
        """
        extend 10% of width on top and bottom
        """
        ext_bbox = [0, 0, 0, 0]
        context_increment_factor = 0.1
        dw = context_increment_factor*float(bbox[2] - bbox[0])
        ext_bbox[0] = bbox[0] - dw
        if (ext_bbox[0] < 0):
            ext_bbox[0] = bbox[0]
        ext_bbox[1] = bbox[1] - dw
        if (ext_bbox[1] < 0):
            ext_bbox[1] = bbox[1]
        ext_bbox[2] = bbox[2] + dw
        if (ext_bbox[2] >= self.width):
            ext_bbox[2] = bbox[2]
        ext_bbox[3] = bbox[3] + dw
        if (ext_bbox[3] >= self.height):
            ext_bbox[3] = bbox[3]
        # ext_bbox[ext_bbox<0] = 0
        ext_bbox = [k if k > 0 else 0 for k in ext_bbox]
        return ext_bbox

    def get_crops(self, trk_id, ts, te, add_color=False):
        u_crops = []
        v_crops = []
        rgb_crops = []
        for t in range(ts, te):  # maybe I am missing 1 ?? TODO
            assert(t in self.tracks[trk_id].all_boxes)  # I need not check
            bbox = self.tracks[trk_id].all_boxes[t]
            bbox = self._extend_context(bbox)
            if (len(self.u_images) != 0) and (len(self.v_images) != 0):
                t_crop_u = self.u_images[t][int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])]
                t_crop_v = self.v_images[t][int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])]
                u_crops.append(t_crop_u)
                v_crops.append(t_crop_v)
            if add_color is True:
                t_rgb_crops = self.images[t][int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])]
                rgb_crops.append(t_rgb_crops)
        return (u_crops, v_crops, rgb_crops)


class NoFetch(Exception):
    """ Base class for trajectory feeder error """
    pass


class NotEnoughPointsInTrack(NoFetch):
    """ Raise an error if not much point left to compute a data tensor"""
    pass


class NoTrajectoryLeft(NoFetch):
    """No trajectory from the list is left"""
    pass


class Data_generator_from_track(object):
    def __init__(self, trk_length, d_tracks, add_color=False):
        self.rgb = add_color
        self.output_data_length = trk_length
        self.total_images = d_tracks.prepare_data_for_generator(self.rgb)
        # d_tracks.print_all_tracks()
        self.tracks_struct = d_tracks
        self.all_tracks = d_tracks.tracks
        self.trk_last_served_frame = -1
        # this list is for all tracks: each track has key - person id
        self.lst_person_ids = self._build_key_list()
        assert (len(self.lst_person_ids) != 0)
        self.person_id = self.lst_person_ids.pop(0)
        self.trk_last_served_frame = self.all_tracks[self.person_id].sorted_frame_nums[0][0]

    def _build_key_list(self):
        lst_keys = []
        for key in self.all_tracks:
            lst_keys.append(key)
        return lst_keys

    def next(self):
        k = self.build_data()
        if isinstance(k, NotEnoughPointsInTrack):
            check = self._move_to_next_track()
            if isinstance(check, NoTrajectoryLeft):
                raise check
            else:
                raise k
        return k

    def build_data(self):
        end_frame = self.trk_last_served_frame + self.output_data_length
        if end_frame in self.all_tracks[self.person_id].all_boxes.keys() and end_frame < self.total_images:
            # serve this track
            """
            print("first and last frame for serving: {} and {} with person_id {}".
                  format(self.trk_last_served_frame, end_frame, self.person_id))
            """
            u, v, rgb = self.tracks_struct.get_crops(
                self.person_id, self.trk_last_served_frame, end_frame, self.rgb)
            _start_frame = self.trk_last_served_frame
            _mid_frame = _start_frame + \
                (end_frame - _start_frame)/2
            _action_label = self.all_tracks[self.person_id].get_action_label_for_a_frame(
                _mid_frame)
            # update self.trk_last_served_frame for next iteration
            self.trk_last_served_frame = end_frame
            return u, v, self.person_id, _start_frame, end_frame-1, rgb, _action_label
        else:
            return NotEnoughPointsInTrack()

    def _move_to_next_track(self):
        if (len(self.lst_person_ids) >= 1):
            self.person_id = self.lst_person_ids.pop(0)
            self.trk_last_served_frame = self.all_tracks[self.person_id].sorted_frame_nums[0][0]
        else:
            return NoTrajectoryLeft()


def save_a_cropped_track(u_imgs, v_imgs, counter, opt_folder):
    """
    cropped in both spatial and temporal dimensions
    """
    out = os.path.join(opt_folder, str(counter))
    if not os.path.exists(out):
        os.makedirs(out)
    assert(len(u_imgs) == len(v_imgs))
    for i in range(len(u_imgs)):
        fname = out + "/u-" + "%06d" % i + ".jpg"
        assert(cv2.imwrite(fname, u_imgs[i]))
        fname = out + "/v-" + "%06d" % i + ".jpg"
        assert(cv2.imwrite(fname, v_imgs[i]))


class get_prediction(object):
    def __init__(self, input_filename):
         # val for this is triplet: ts, te, pred_label
        # key would be pid itself
        self._label_dict = {}
        self._file = input_filename
        self._load_output_prediction()

    def _add_new_prediction(self, pred, ts, te, pid):
        if not(pid in self._label_dict.keys()):
            self._label_dict[pid] = [[int(ts), int(te), pred]]
        else:
            self._label_dict[pid].append([int(ts), int(te), pred])

    def _load_output_prediction(self):
        f = open(self._file)
        counter = 0  # hack for not reading header
        while True:
            line = f.readline()
            if counter == 0:
                counter += 1
                continue
            if not line:
                break
            tmp = line.split(",")
            assert(len(tmp) == 4)
            pid = int(tmp[0])
            ts = int(tmp[1])
            te = int(tmp[2])
            pred = tmp[3]
            pred = pred.rstrip()
            self._add_new_prediction(pred, ts, te, pid)

    def get_action_label_for_a_frame_for_person(self, frame, pid):
        assert(pid in self._label_dict.keys())
        vector_action_spans = self._label_dict[pid]
        for action_span in vector_action_spans:
            if frame >= action_span[0] and frame < action_span[1]:
                return action_span[2]

    def sanity_check(self):
        print("printing action dict naively: ")
        print(self._label_dict)


if __name__ == '__main__':
    # test above infrastructure
    # tmp_folder="/home/scratch.nvdrivenet/actionrecognition/data/shad_action/test_videos/Squa-test/Annotations/SJTU-SEIEE-170_Squa_0001
    # tmp_folder="/home/scratch.nvdrivenet/actionrecognition/data/shad_action/test_videos/Fall-test/Annotations/SJTU-SEIEE-135_Fall_0001
    # tmp_folder = "/home/scratch.nvdrivenet/actionrecognition/data/shad_action/test_videos/Sits-test/Annotations/SJTU-SEIEE-135_Sits_0008"
    tmp_folder = "/home/scratch.nvdrivenet/actionrecognition/data/shad_action/test_videos/Fall-test/Annotations/SJTU-SEIEE-135_Fall_0016"
    gpred = get_prediction('./SJTU-SEIEE-135_Fall_0016.txt')
    print(gpred.sanity_check())
    tmp_output = "./tmp-1check-10-predictions"
    tracks_t = Tracks(tmp_folder, tmp_output)
    # tracks_t.find_and_read_xml_files()  # made this private
    # tmp_xml = os.path.join(tmp_folder, "SJTU-SEIEE-135_Bend_0029-0053.xml")
    # tracks_t.read_one_xml(tmp_xml)
    tracks_t.interpolate_all_tracks()
    # tracks_t.save_trajectory_info()  # saves into a text file
    # tracks_t.print_all_tracks()
    tracks_t.annotate_tracks_and_save_images(
        prediction_output_file='./SJTU-SEIEE-135_Fall_0016.txt')
    don_check = Data_generator_from_track(10, tracks_t)
    tmp_output_gen = os.path.join(tmp_output, "tracks")
    if not os.path.exists(tmp_output_gen):
        os.makedirs(tmp_output_gen)
    counter = 0
    while True:
        try:
            gen_val = don_check.next()
        except NoTrajectoryLeft:
            break
        except NotEnoughPointsInTrack:
            # "lets move to next track"
            # print("lets move to next track")
            continue
        else:
            save_a_cropped_track(
                gen_val[0], gen_val[1], counter, tmp_output_gen)
            counter += 1
    # In this case we need to read image as rgb
    tracks_t.save_annotated_optical_flow_images()
