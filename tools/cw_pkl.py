import pickle
import os
import pandas as pd

import csv

csv_path = 'AR_ava_format/chaos_train_1fps_woXGXJVBCQ.csv'

elements = []

with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    headers = next(csv_reader)  

    for row in csv_reader:
        video_uid = row[0]  
        time_current = float(row[-3])  
        frame_start = int(row[-5])   
        frame_end = int(row[-4])  
        frame = row[1]  

        element = {
            'video': video_uid,
            'time': int(time_current),
            'start_frame': int(frame_start),
            'n_frames': int(frame_end) - int(frame_start),
            'mid_frame': int(frame),
            'format_str': video_uid+'_%06d.png',
            'frame_rate': 25,  # Choatic FPS=25
            'labels': [] 
        }
        
        label_info = {
            'bounding_box': [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
            'label': [int(row[6])],
            'person_id': [int(row[7])]
        }
        element['labels'].append(label_info)
        

        elements.append(element)
        print("done 1")


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

data = [elements,labels]

# Save data as a pkl file
with open('AR_ava_format/cw_data_train1fps_woXGXJVBCQ.pkl', 'wb') as f:
    pickle.dump(data, f)