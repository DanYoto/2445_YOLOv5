
from __future__ import print_function
import argparse
import glob
import os
import json
 
if __name__ == '__main__':
    # input parameter in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help='Directory of json files containing annotations')  # path of json file
    parser.add_argument(
        "output_path", help='Output directory for image.txt files')  # the path to save the txt file
    args = parser.parse_args()
 
    # glob.glob obtain all of the matching paths
    json_files = sorted(glob.glob(os.path.join(args.path, '*.json')))  # get pathes of all json files under the specific folder
 
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)  # transfer json file to dictionary
            images = data['images']
            annotations = data['annotations']
 
            # In flir, w and h is fixed
            width = 640.0
            height = 512.0
 
            for i in range(0, len(images)):
                converted_results = []
                for ann in annotations:
                    if ann['image_id'] == images[i]['id'] and ann['category_id'] <= 100:
                        # 100 could be changed due to categories number, this could be changed to study the specific category
                        cat_id = int(ann['category_id'])
 
                        # letf top stands for the lower left coordinate, bbox_width bbox_height are the fixed 
                        left, top, bbox_width, bbox_height = map(float, ann['bbox'])  
 
                        # index of Yolo starts from 0 while Flir it starts from 1
                        cat_id -= 1
 
                        # find the center
                        x_center, y_center = (
                            left + bbox_width / 2, top + bbox_height / 2)
 
                        # normalize
                        x_rel, y_rel = (x_center / width, y_center / height)
                        w_rel, h_rel = (bbox_width / width, bbox_height / height)
                        converted_results.append(
                            (cat_id, x_rel, y_rel, w_rel, h_rel))
 
                image_name = images[i]['file_name']
 
                image_name = image_name[5:-4]
 
                print(image_name)  # name verification
 
                file = open(args.output_path + str(image_name) + '.txt', 'w+')
                file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))
                file.close()