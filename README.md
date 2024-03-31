# [2024 ICME Grand challenge]:Spatiotemporal-Action-Localization-on-Chaotic-World-dataset 
We have implemented Track # 1 for ICME 2024: Spatial Action Localization on Chaotic World dataset. We decompose the STAL task into two stages: object detection and action recognition. Our mAP on the validation set reaches **26.62%**, and if we directly use officially provided  **chaos_test_1fps.csv** as the results of object detection, the mAP reaches **42.28%**.


## Requirements<a id="Requirements"></a>

Some key dependencies are listed below, while others are given in [`requirements.txt`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/requirements.txt).

- Python >= 3.7
- PyTorch >= 1.3, and a corresponding version of torchvision
- ultralytics >= 8.0 (used for object detection, refer to [Ultralytics](https://github.com/ultralytics) for details.)
- ffmpeg (used in data preparation)
- Download checkpoints for inference, which are listed in [`checkpoints/README.md`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/checkpoints/README.md), to the `checkpoints` folder.
- Download pre-trained models for training, which are listed in [`pretrained/README.md`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/pretrained/README.md), to the `pretrained` folder.
- Prepare data. Please refer to [`DATA.md`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/DATA.md).


## How to reproduce

**RESULTS**: 
- We got up to **26.62%** mAP on the validation set. We trained 12 epochs on 4 GeForce RTX 3090 with batch_size=3/GPU to get this result. Our best mAP results are saved in [our_result/final_result](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/our_result/final_result).
- If we directly use **chaos_test_1fps.csv** as the results of object detection, the mAP reaches **42.28%**. We trained 7 epochs on 4 GeForce RTX 3090 with batch_size=3/GPU to get this result. Our best mAP results are saved in [our_result/gt_result](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/our_result/gt_result).
- The CSV we predicted can be downloaded from [`Google Drive`](https://drive.google.com/drive/folders/12JQYCU9fPKJvqqgFGpSm8egXygCwCh01?usp=sharing ). `predict_epoch_final.csv` corresponds to the result of [our_result/final_result](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/our_result/final_result), and `predict_epoch_gt.csv` corresponds to the result of [our_result/gt_result](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/our_result/gt_result).


**Realization steps**:

- First, Complete **all the contents** in [Requirements](#Requirements), including environment configuration, checkpoints, data preparation, etc.

- Inference：
  - If you want to quickly reproduce our optimal results directly, please download our [predict.csv](https://drive.google.com/drive/folders/12JQYCU9fPKJvqqgFGpSm8egXygCwCh01?usp=sharing ) first,replace the path in [ACAR-SF-VIT/multi_cap_mAP.py](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/ACAR-SF-VIT/multi_cap_mAP.py) to obtain.

  - Run the following code at the command line. Default values for arguments `nproc_per_node`, `backend` and `master_port` are `4`, `nccl` and `31114` respectively.

    ```
    python main.py --config ./configs/eval_VITSF_ACAR_HR2O_CW.yaml [--nproc_per_node N_PROCESSES] [--backend BACKEND] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]
    ```
- Train：
  - **Training from scratch**, see [preteained](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/pretrained/README.md) for details:

    Run the following code at the command line. Default values for arguments `nproc_per_node`, `backend` and `master_port` are `4`, `nccl` and `31114` respectively.

    ```
    python main.py --config ./configs/VITSF_ACAR_HR2O_CW_train.yaml [--nproc_per_node N_PROCESSES] [--backend BACKEND] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]
    ```
  - **Continue training based on our checkpoint**, see [checkpoints](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/checkpoints/README.md) for details:

    Run the following code at the command line. Default values for arguments `nproc_per_node`, `backend` and `master_port` are `4`, `nccl` and `31114` respectively.

    ```
    python main.py --config ./configs/VITSF_ACAR_HR2O_CW_train_resume.yaml [--nproc_per_node N_PROCESSES] [--backend BACKEND] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]
    ```
- Train/Inference on ground truth：
  If we directly use officially provided **chaos_test_1fps.csv** as the results of object detection instead of detector , please **follow the comments we wrote** in [`config`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs) and replace `annotation_path`. All other operations are consistent.

## Model Introduction
- We use [ACAR](https://github.com/Siyu-C/ACAR-Net) as the baseline.
- Our backbone is a dual stream backbone of SlowFast-R50 and VIT-Giant.
- When calculating mAP, `capacity` is introduced, which sets the upper limit of bbox to capacity for a key of {img, time}, thereby retaining the best score of capacity bbox. By iteratively enumerating `capacity`, we can obtain the optimal mAP.

## Detection
We decompose the STAL task into two stages: object detection and action recognition. First, we need to use the object detection model to generate annotations in `.pkl` format. It needs to be emphasised that the annotation file `.pkl` is necessary for our final results. This file will be fed into our main model to assist in obtaining the final results. For detailed information, please refer to  [`detect/README.md`](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/detect/README.md).


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
- [ACAR](https://github.com/Siyu-C/ACAR-Net)
- [YOLOv8n](https://github.com/ultralytics/assets/releases)
