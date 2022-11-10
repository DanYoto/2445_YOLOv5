from __future__ import print_function
import argparse
import os
import cv2
#from google.colab.patches import cv2_imshow


if __name__ == '__main__':
    # input parameter in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "label_path", help='Directory of labels.txt')  # path of label file
    parser.add_argument(
        "image_path", help='Directory of image folder')  # the path to save the txt file
    args = parser.parse_args()

    path_root_labels = args.label_path  #'/content/drive/MyDrive/EQ2445/images_thermal_val/labels'
    path_root_imgs = args.image_path  #'/content/drive/MyDrive/EQ2445/images_thermal_val/images'
    type_object = '.txt'

    cnt = 0
    for ii in os.walk(path_root_imgs):
        for j in ii[2]:
            tmp_type = j.split(".")[1]
            if tmp_type != 'jpg':
                continue
            path_img = os.path.join(path_root_imgs, j)
            #print(path_img)
            label_name = j[:-4]+type_object
            path_label = os.path.join(path_root_labels, label_name)
            #print(path_label)
            f = open(path_label, 'r+', encoding='utf-8')
            # print(os.path.exists(path_label))
            if os.path.exists(path_label) == True:
                img = cv2.imread(path_img)
                w = img.shape[1]
                h = img.shape[0]
                new_lines = []
                img_tmp = img.copy()
                while True:
                    line = f.readline()
                    #print(line)
                    if line:
                        msg = line.split(" ")
                        #print(msg)
                        #return
                        # print(x_center,",",y_center,",",width,",",height)
                        x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                        y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                        x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                        y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                        #print(x1,",",y1,",",x2,",",y2)
                        cv2.rectangle(img_tmp,(x1,y1),(x2,y2),(0,0,255),1)
                    else :
                        break
            #cv2_imshow(img_tmp)
            cv2.imshow('show', img_tmp)
            c = cv2.waitKey(0)
            cnt += 1
            if cnt == 3:
                #This is set not to show all of the labeled image, or plenty of time is needed
                break