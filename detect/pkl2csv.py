import pickle
import csv

def pkl_to_csv(input_file, output_file):
    # 打开.pkl文件并加载数据
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # 创建CSV文件并写入数据
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video', 'time', 'bbox', 'label', 'person_id'])  # 写入CSV文件的标题行
        for item in data[0]:
            video = item['video']
            # time = item['time']
            time = '%04d' % item['time'] 
            labels = item['labels']
            for label in labels:
                bounding_boxes = label['bounding_box']
                x1 = round(bounding_boxes[0],4)
                y1 = round(bounding_boxes[1],4)
                x2 = round(bounding_boxes[2],4)
                y2 = round(bounding_boxes[3],4)
                label_id = label['label'][0]
                person_id = label['person_id'][0]
                # for bbox, label, person_id in zip(bounding_boxes, label_list, person_id_list):
                writer.writerow([video, time, x1, y1,x2,y2,label_id, person_id])  # 将每个bbox,label,person_id写入一行

pkl_pave = r'/remote-home/wjj/detect/annotation_pkl/ava_val_v2.2_fair_0.85.pkl'
csv_pave = r'/remote-home/wjj/detect/annotation_csv/ava_val_v2.2_fair_0.85.csv'

if __name__ == '__main__':
    # 使用示例
    pkl_to_csv(pkl_pave, csv_pave)