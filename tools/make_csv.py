'''
Purpose: As the label range of our model is from 1 to 50,
the label range provided by the official chaos_train/test_1fps.csv is 0-49.
Here, make a match and create chaos_train/test_1fps_t_+1csv.

Input: chaos_train_1fps_woXGXJVBCQ.csv or chaos_test_1fps.csv

Output: chaos_train_t_+1.csv or chaos_test_t_+1.csv
'''
import csv

input_file = "AR_ava_format/chaos_train_1fps_woXGXJVBCQ.csv"
output_file = "AR_ava_format/self/chaos_train_t_+1.csv"
selected_columns = [2,3,4,5]

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        new_row = [str(row[0])]  # video_id
        new_row.append('%04d' % int(float(row[-3])))    # Time (in seconds)
        for col_index in selected_columns:  # x1y1x2y2
            new_row.append(float(row[col_index]))
        new_row.append(int(row[6])+1)   # action_label_id，attention +1 !
        new_row.append(int(row[7]))     # person_id
        writer.writerow(new_row)
