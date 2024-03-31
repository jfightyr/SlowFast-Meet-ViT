import sys
# sys.pth.append("")
from calc_mAP import *         
import csv
from multiprocessing import Pool
import os
# import partial
from functools import partial

def read_pbtxt(pbtxt_file):
    names = {}
    with open(pbtxt_file, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if 'label' in lines[i]:
                name_line = lines[i+1].strip()
                name = name_line.split(':')[-1].strip().strip('"')
                label_id_line = lines[i+2].strip()
                label_id = int(label_id_line.split(':')[-1].strip())
                names[label_id] = name
                i += 4  # Skip label_type line
            else:
                i += 1
    return names


def get_name(label_id, names):
    name = names.get(label_id, f"label_{label_id}")
    # Replace spaces and slashes with underscores
    name = name.replace(' ', '_').replace('/', '_')
    return name

def write_scores_and_labels_to_csv(scores, tp_fp_labels, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入CSV文件的标题行
        writer.writerow(['scores', 'tp_fp_labels'])
        # 遍历每个得分和标签，并将其写入CSV文件
        for score, label in zip(scores, tp_fp_labels):
            writer.writerow([score, label])


def run_evaluation(labelmap, groundtruth, detections, exclusions,cap):
    """Runs evaluations given input files.

    Args:
      labelmap: file object containing map of labels to consider, in pbtxt format
      groundtruth: file object
      detections: file object
      exclusions: file object or None.
    """
    categories, class_whitelist = read_labelmap(labelmap)
    # print("CATEGORIES (%d):\n%s", len(categories),
    #              pprint.pformat(categories, indent=2))
    excluded_keys = read_exclusions(exclusions)

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)

    # Reads the ground truth data.
    boxes, labels, _ = read_csv(groundtruth, class_whitelist, 0)
    start = time.time()
    for image_key in boxes:
        if image_key in excluded_keys:
            print(("Found excluded timestamp in ground truth: %s. "
                          "It will be ignored."), image_key)
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key, {
                standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(boxes[image_key], dtype=float),
                standard_fields.InputDataFields.groundtruth_classes:
                    np.array(labels[image_key], dtype=int),
                standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(boxes[image_key]), dtype=bool)
            })
    print_time("convert groundtruth", start)

    # Reads detections data.
    boxes, labels, scores = read_csv(detections, class_whitelist, cap)  # 50
    start = time.time()
    for image_key in boxes:
        if image_key in excluded_keys:
            print(("Found excluded timestamp in detections: %s. "
                          "It will be ignored."), image_key)
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key, {
                standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores[image_key], dtype=float)
            })
    print_time("convert detections", start)

    start = time.time()
    metrics,scores,tp_fp_labels = pascal_evaluator.evaluate()
    # write_scores_and_labels_to_csv(scores,tp_fp_labels,"test.csv")
    print_time("run_evaluator", start)
    # print(pprint.pformat(metrics, indent=2))

    return metrics


# predict_csv = r'/remote-home/wjj/experiments/ACAR-SF-VIT/VITSF_gpu5_bz3_epoch12_giant_0330/predict_epoch12.csv'
# result_txt = r"predict_epoch12.txt"


def evaluate_cap(cap, predict):
    metrics = run_evaluation(
        open("/remote-home/ChaoticWorld/AR_ava_format/list_action.pbtxt", 'r'), 
        open("/remote-home/ChaoticWorld/AR_ava_format/self/chaos_test_t_+1.csv", 'r'),
        open(predict, 'r'),
        None,
        cap=cap,
    )
    mAP = metrics['PascalBoxes_Precision/mAP@0.5IOU']
    return cap, mAP


# if __name__ == '__main__':
def cal_cap_map(predict_csv,result_txt,cap_range):
    mAPs = []
    caps = []
    caps = [i for i in cap_range]
    predict = predict_csv
    partial_cap = partial(evaluate_cap, predict=predict)

    print("Num of CPU: ", os.cpu_count())
    print("Please be patient and wait for about 10 minutes as mAP is being calculated...")
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(partial_cap,caps)

    for res in results:
        cap, mAP = res[0], res[1]
        caps.append(cap)
        mAPs.append(mAP)

        with open(result_txt, "a") as file:
            file.write(f"cap: {cap}, mAP: {mAP}\n")
            print("cap: %d | mAP: %f" % (cap, mAP))

    max_mAP = max(mAPs)
    max_idx = mAPs.index(max_mAP)
    max_caps = caps[max_idx]
    
    with open(result_txt, "a") as file:
        file.write(f"cap:{max_caps},max_mAP:{max_mAP}\n")
        print(f"cap:{max_caps},mAP:{max_mAP}")
    return max_mAP,max_caps