# Checkpoints

After training on the Chaotic World dataset, we present our optimal checkpoint which can be directly used for inference/testing.

Checkpoints are released in [`Google Drive`](https://drive.google.com/drive/folders/1ktWZzT6eU83IodbxMksu1R6FW619zB--?usp=sharing ), and should be downloaded to this local `checkpoints` folder by default for evaluation.

Name | Architecture | Pre-train | Train Dataset | Train Config | Eval Config | Val mAP | Ground Truth mAP
--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
`SFR50_VITG_ACAR_CWtrain_ckpt_11.pth.tar` | ViT Giant and SlowFast-R50 | AVA v2.2 | Chaotic World | [config](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs/VITSF_ACAR_HR2O_CW_train_resume.yaml) | [config](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs/eval_VITSF_ACAR_HR2O_CW.yaml) | **26.62%**| 41.07%
`SFR50-VITG-CW-gt.pth.tar` | ViT Giant and SlowFast-R50 | AVA v2.2 | Chaotic World | [config](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs/VITSF_ACAR_HR2O_CW_train_resume.yaml) | [config](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs/eval_VITSF_ACAR_HR2O_CW.yaml) | /| **42.28%**

Additional notes:
- Val mAP refers to the results on the Chaotic World validation set with our detector.
- Ground Truth mAP means the results on the Chaotic World validation set when we directly use officially provided **chaos_test_1fps.csv** as the results of object detection.
- You can continue training on checkpoint `SFR50_VITG_ACAR_CWtrain_ckpt_11.pth.tar` using the **train config** in the table. If you want to train from scratch, please see [pretrained](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/pretrained/README.md) .
- You can use the **val config** in the table to test based on `SFR50_VITG_ACAR_CWtrain_ckpt_11.pth.tar`. Using this checkpoint can achieve our best performance on **the validation set**, with val mAP=**26.62%**.
- You can use the **val config** in the table to test based on `SFR50-VITG-CW-gt.pth.tar`. Using this checkpoint can achieve our best performance on the validation set while using officially provided **chaos_test_1fps.csv** as the results of object detection, with Ground Truth mAP=**42.28%**.
