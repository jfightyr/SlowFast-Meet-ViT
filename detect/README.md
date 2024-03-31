## Detection
We decompose the STAL task into two stages: object detection and action recognition. In the object detection stage, we use YOLOv8 as our object detector to detect human targets in video keyframes. Specifically, we use the official YOLOv8n as the pre-trained weights and train it on the ChaoticWorld training set. For details on data format conversion, please see the next part. The training configuration we use is as follows:

- pretrained model: [YOLOv8n](https://github.com/ultralytics/assets/releases)
- epoch: 200
- devices: 2 GeForce RTX 3090
- batch: Automatic batch processing
- imgsz: [640, 640]
- data: provided in [`DATA.md`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/DATA.md)


You can easily configure the training environment through [ultralytics](https://docs.ultralytics.com/) or directly load our pre-trained weights [yolov8n_cw_epoch200.pt](https://drive.google.com/drive/folders/12JQYCU9fPKJvqqgFGpSm8egXygCwCh01?usp=sharing ) for further detection.

 

## Chaotic World dataset in YOLO Format
To adapt YOLOv8 pretrained weights on the Chaotic World dataset, we need to organize the Chaotic World dataset into the YOLO data structure for training. The YOLO format dataset is as follows, and you can obtain the corresponding dataset using the `cw_csv2yolo.py` file. Finally, organize your data paths according to the [`data.yaml`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/detect/data.yaml).
```
├── ChaoticWorld in YOLO format 
    ├── images
        ├── train
            ├── 1111.jpg
            ├── 2222.jpg
        ├── test
            ├── 1111.jpg
            ├── 2222.jpg
    ├── labels
        ├── train
            ├── 1111.txt
            ├── 2222.txt
        ├── test
            ├── 1111.txt
            ├── 2222.txt
```
## Infer to get annotation file
It needs to be emphasised that the annotation file `.pkl` is necessary for our final results. You can get the `.pkl` file format annotation from the trained YOLOv8 weights [yolov8n_cw_epoch200.pt](https://drive.google.com/drive/folders/12JQYCU9fPKJvqqgFGpSm8egXygCwCh01?usp=sharing). Or you can directly use the **cw_test_yolov8_on_cw_0.3.pkl** we provided as object detection result detected from this YOLOv8n weight. It is provided in [`Google Drive`](https://drive.google.com/drive/folders/1ktWZzT6eU83IodbxMksu1R6FW619zB--?usp=sharing).

When using detect_cw_yolov8.py for inference, several key parameters need to be provided. Here is an explanation of the parameters:

- **ckpt_path**: The trained weights of YOLO series models, or you can directly used our YOLOv8n weight trained on the Chaotic World training dataset.
- **csv_path**: We use this file to find the paths of key images for inference. The format of this file should follow the official `chaos_test_1fps.csv` provided.
- **img_root**: Ensure that this folder contains all the images needed for inference, and also ensure that the dimensions of the images stored here are consistent with the dimensions of our training models (640, 640). You can use `cw_csv2yolo.py` to get this folder.
- **pkl_path**:  The output path for the `.pkl` file.


