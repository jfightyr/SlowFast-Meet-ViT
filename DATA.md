# Data Preparation
## Chaotic World dataset
### Download the official dataset

   **1.Documents processing**：

   ```
   AR_ava_format
   ├── annotation
   │   ├── video1.csv
   │   ├── video2.csv
   │   ├── ...
   │   ├── videon.csv
   ├── bbox_test_1fps.json
   ├── bbox_train_1fps.json
   ├── chaos_test_1fps.csv
   ├── chaos_test_exclude.csv
   ├── chaos_train_1fps.csv
   ├── chaos_train_1fps_woXGXJVBCQ.csv
   ├── chaos_train_exclude.csv
   └── list_action.pbtxt
   ```

   Note that since the 'XGXJVBCQ' video could not be found, the `chaos_train_1fps_woXGXJVBCQ.csv` was created by **removing the XGXJVBCQ data** from the **chaos_train_1fps.csv**.The csv is also provided in the  [`Google Drive`](https://drive.google.com/drive/folders/1ktWZzT6eU83IodbxMksu1R6FW619zB--?usp=sharing). All other files are officially provided.

   **2.Video Processing**: Download the video as per official code `code_preprocessing/01preprocessing.py` ( we also provide it in [tools/01preprocessing.py](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/tools/01preprocessing.py))  and pump the frames by size(320, 180) and fps = 25. However, using the official code download link may lead to some video download failures due to URL access permissions issues. We have manually downloaded these videos for you. To facilitate reproduction, we have combined the missing videos provided by the official source into a collection of 628 (320, 180) videos. Finally get like:

   ```
   frames
   ├── video1
   │   ├── video1_000001.png
   │   ├── video1_000002.png
   │   ├── ...
   │   ├── video1_00000n.png
   ├── video2
   ├── ...
   └── videon
   ```

### Preparation of our model

Our model also requires some additional files, as shown in the table below. All our annotations are provided in the  [`Google Drive`](https://drive.google.com/drive/folders/1ktWZzT6eU83IodbxMksu1R6FW619zB--?usp=sharing).

Name | Dataset | Split | Ground Truth | Detection
--- | :---: | :---: | :---: | :---:
`cw_data_train1fps_woXGXJVBCQ.pkl` | Chaotic World | train | YES | None
`chaos_train_t_+1.csv`| Chaotic World | train | YES | None
`chaos_test_t_+1.csv`| Chaotic World | val | YES | None
`cw_data_test1fps.pkl` | Chaotic World | val | YES | None
`cw_test_yolov8_on_cw_0.3.pkl` | Chaotic World | val | NO | yolov8n

How to make:

   - Make annotation pkl: follow [`tools/cw_pkl.py`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/tools/cw_pkl.py) to make `cw_data_train1fps_woXGXJVBCQ.pkl` and `cw_data_test1fps.pkl`. These two folders will be used in the `annotation_path` in [`config`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs). 

   - Groundtruth: make `chaos_train_t_+1.csv` and `chaos_test_t_+1.csv` according to [`tools/make_csv.py`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/tools/make_csv.py). They will be used in `groundtruth` in [`config`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs).

     In particular, note that in order to match our model output label range (1~50), we performed a **label+1** when making the csv for groundtruth.
   - Yolov8 as detector: make `cw_test_yolov8_on_cw_0.3.pkl`.Please refer to [`detect/README.md`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/detect/README.md) for details.

   - Finally, we get the dataset as a whole as follows:

     ```
     AR_ava_format
     ├── [official Chaotic World files]
     ├── cw_test_yolov8_on_cw_0.3.pkl
     ├── cw_data_test1fps.pkl
     ├── chaos_test_t_+1.csv
     ├── chaos_train_1fps_woXGXJVBCQ.csv
     ├── cw_data_train1fps_woXGXJVBCQ.pkl
     └── chaos_train_t_+1.csv
     frames
     ├── video1
     │   ├── video1_000001.png
     │   ├── video1_000002.png
     │   ├── ...
     │   ├── video1_00000n.png
     ├── video2
     ├── ...
     └── videon
     ```

Once this is done, replace the file path in config to run the code.


