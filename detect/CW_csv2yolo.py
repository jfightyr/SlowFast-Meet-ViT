import pandas as pd
import random
import cv2
import os
import csv
import shutil

videos_root = r"/remote-home/ChaoticWorld/frames_320_180"
labels_root = r"/remote-home/ChaoticWorld/YOLO_format/train_640_640/labels"
images_root = r"/remote-home/ChaoticWorld/YOLO_format/train_640_640/images"
tags_path = r"/remote-home/ChaoticWorld/AR_ava_format/chaos_train_1fps_woXGXJVBCQ.csv"

if __name__ == '__main__':

    total = sum(1 for line in open(tags_path))
    with open(tags_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        count = 1
        for row in csv_reader:

            # got img_name and bbox
            video_uid = row[0]  
            frame_id = int(row[1])  
            img_name = '%s_%06d.png' % (video_uid, frame_id)
            x0, y0, x1, y1 = float(row[2]), float(row[3]), float(row[4]), float(row[5])

            img_root = os.path.join(videos_root, video_uid)
            ori_path = os.path.join(img_root, img_name)
            dis_path = os.path.join(images_root, img_name)

            if os.path.exists(dis_path) == False:
                img_ori = cv2.imread(ori_path)
                img_res = cv2.resize(img_ori, (320, 180), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(dis_path, img_res)

            # [x0, y0, x1, y1] -> [center_x, center_y, bbox_w, bbox_h]
            bbox_w = x1 - x0
            bbox_h = y1 - y0
            center_x = x0 + bbox_w/2
            center_y = y0 + bbox_h/2

            # got label_path and label for write
            label_path = os.path.join(labels_root, "%s_%06d.txt"%(video_uid, frame_id))
            label_new = "0 %.6f %.6f %.6f %.6f" % (center_x, center_y, bbox_w, bbox_h)
            with open(label_path, 'a') as file:
                file.write(label_new + '\n')

        
            print("proccessing: %d / %d" % (count, total))
            count += 1
