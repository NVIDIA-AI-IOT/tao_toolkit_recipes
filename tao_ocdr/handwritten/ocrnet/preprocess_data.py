# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import cv2
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess_data", add_help=True, description="Preprocess IAMDATA to TAO Toolkit OCRNet format")
    parser.add_argument(
        "--images_dir",
        help="Path to original images",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--labels_dir",
        help="Path to original label txt files",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--output_images_dir",
        help="Path to pre-processed images",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--gt_file_path",
        help="Path to ground truth list",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--character_list_path",
        help="Path to character list",
        default=None,
        required=True,
    )

    args, _ = parser.parse_known_args()
    root_dir = args.images_dir
    gt_file_dir = args.labels_dir
    target_dir = args.output_images_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    p_gt_file = open(args.gt_file_path, "w")

    gt_file_list = os.listdir(gt_file_dir)
    character_set = set()

    for gt_file_name in tqdm(gt_file_list):
        img_id = gt_file_name.split(".")[0].replace("gt_", "")
        f = open(os.path.join(gt_file_dir, gt_file_name), "r")
        reader = f.readlines()
        
        img_path = os.path.join(root_dir, img_id+".png")
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        for idx, ann in enumerate(reader):
            ann = ann.split(",")
            vs = ann[:8]
            text = ann[8:]
            if len(text) == 1:
                text = text[0].strip()
                if text.count("\"") == 4:
                    text = "\""
            elif len(text) == 2:
                text = ","
            else:
                # for label like: "163,000,000"
                # ignore the " " at the begin and end
                text = ",".join(text)
                text = text.replace("\"", "")
                text = text.strip()

            # Skip the words which length > 25 or non-word-level label
            if len(text) > 25 or (" " in text):
                continue
            # Lower-case:
            text = text.lower()
            
            for c in text:
                character_set.add(c)
                
            xs = [int(vs[idx]) for idx in range(0, len(vs), 2)]
            ys = [int(vs[idx]) for idx in range(1, len(vs), 2)]
            xmin = max(0, min(xs))
            ymin = max(0, min(ys))
            xmax = min(width, max(xs))
            ymax = min(height, max(ys))

            try:
                crop_img = img[ymin:ymax, xmin:xmax, :]
                target_img_path = f"{img_id}_{idx}.jpg"
                p_gt_file.write(target_img_path + "\t" + text + "\n")
                cv2.imwrite(target_img_path, crop_img)
            except Exception as err:
                print(err)
                print(f"img_id: {img_id} bbox: {vs} img_shape: {img.shape}")
                exit()

    p_gt_file.close()
    with open(args.character_list_path, "w") as f:
        character_set = sorted(list(character_set))
        for c in character_set:
            f.write(f"{c}\n")