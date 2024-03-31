from ultralytics import YOLO
import argparse
import pickle
import os
import pandas as pd
import csv


labels = [
    {'name': 'aiming weapon', 'id': 1,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'arguing', 'id': 2,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'arresting', 'id': 3,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'burning/setting fire', 'id': 4,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'carrying', 'id': 5,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'carrying casualty', 'id': 6,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'chanting/cheering', 'id': 7,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'chasing', 'id': 8,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'clapping', 'id': 9,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'creating barricade', 'id': 10,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'dancing', 'id': 11,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'destroying', 'id': 12,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'extinguishing fire', 'id': 13,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'fighting', 'id': 14,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'guarding', 'id': 15,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'hitting an object/smashing', 'id': 16,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'holding a burning stick', 'id': 17,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'holding flag', 'id': 18,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'holding hands', 'id': 19,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'holding signage', 'id': 20,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'holding weapon', 'id': 21,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'hugging', 'id': 22,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'injured', 'id': 23,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'kneeling', 'id': 24,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'painting', 'id': 25,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'pinning', 'id': 26,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'playing instrument', 'id': 27,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'praying', 'id': 28,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'pulling barricade', 'id': 29,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'punching', 'id': 30,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'pushing', 'id': 31,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'raising fist', 'id': 32,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'raising hands', 'id': 33,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'recording', 'id': 34,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'reporting live', 'id': 35,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'retreating', 'id': 36,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'running/escaping', 'id': 37,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'shooting/firing', 'id': 38,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'shouting', 'id': 39,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'singing', 'id': 40,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'speaking on stage', 'id': 41,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'speaking/talking', 'id': 42,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'spraying', 'id': 43,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'stealing/looting', 'id': 44,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'throwing object', 'id': 45,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'walking', 'id': 46,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'watching', 'id': 47,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'waving flag', 'id': 48,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'waving flarestick', 'id': 49,'label_type': 'PERSON_MOVEMENT'},
    {'name': 'waving hands', 'id': 50,'label_type': 'PERSON_MOVEMENT'},
]


# please config your file root
ckpt_path = r'/remote-home/wjj/YOLOv8/train_w_train/yolo8x_stop79/weights/best.pt'
csv_path = r'/remote-home/ChaoticWorld/AR_ava_format/chaos_test_1fps.csv'
img_root = r'/remote-home/ChaoticWorld/YOLO_format/test_640_640/images'
pkl_path = r"test.pkl"



# default config
img_w, img_h = 320, 180
threshold = 0.3
device = 'cpu'
total = sum(1 for line in open(csv_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch YOLO detector for .pkl annotation')
    parser.add_argument('--ckpt_path', type=str, default=ckpt_path)
    parser.add_argument('--csv_path', type=str, default=csv_path, help="")
    parser.add_argument('--img_root', type=str, default=img_root, help="img folder, 640*640")
    parser.add_argument('--pkl_path', type=str, default=pkl_path, help="output file path in pkl format")
    args = parser.parse_args()

    csv_path = args.csv_path
    img_root = args.img_root
    pkl_path = args.pkl_path

    # load model and ckpt
    ckpt_path = args.ckpt_path
    model = YOLO(ckpt_path)

    elements = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)

        for idx, row in enumerate(csv_reader):
            print("proccessing: %d / %d" % (idx, total))

            video_uid = row[0]  
            time_current = float(row[-3])  
            frame_start = int(row[-5])   
            frame_end = int(row[-4])  
            frame = row[1]  

            # create element dict
            element = {
                'video': video_uid,
                'time': int(time_current),
                'start_frame': int(frame_start),
                'n_frames': int(frame_end) - int(frame_start),
                'mid_frame': int(frame),
                'format_str': video_uid+'_%06d.png',
                'frame_rate': 25,  
                'labels': []  
            }
            
            img_path = os.path.join(img_root, video_uid + "_%06d.png"%int(frame)) 

            # infering by yolov8, attention: imgsz should be (h, w), or you will get a terrible res. 
            inf_res = model.predict(img_path, imgsz=(img_h,img_w), conf=threshold, device=device)
            inf_boxes = inf_res[0].boxes

            for box in inf_boxes:
                x0, y0, x1, y1 = float(box.xyxy[0,0]), float(box.xyxy[0,1]), \
                                 float(box.xyxy[0,2]), float(box.xyxy[0,3])

                bounding_box = [x0/img_w, y0/img_h, x1/img_w, y1/img_h]
                label_info = {
                    'bounding_box': bounding_box,
                    'label': [0],                   # unused
                    'person_id': [0]                # unused
                }

                element['labels'].append(label_info)

            elements.append(element)

    data = [elements,labels]


    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
