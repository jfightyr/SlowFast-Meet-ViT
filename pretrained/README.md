# Pre-trained Models


`vit_g_ps14_ak_ft_ckpt_7_clean.pth`  and `SLOWFAST_R50_K400.pth.tar` are used as a pre-training for our model. They are provided in [`Google Drive`](https://drive.google.com/drive/folders/1ktWZzT6eU83IodbxMksu1R6FW619zB--?usp=sharing).Please place it in this local `pretrained` folder by default.

Name | Architecture | Dataset | Source| Objective
--- | :---: | :---: |:---: | :---:
`vit_g_ps14_ak_ft_ckpt_7_clean.pth` | ViT Giant | AVA v2.2| [VideoMAE](https://github.com/MCG-NJU/VideoMAE)| Train 
`SLOWFAST_R50_K400.pth.tar`| SlowFast_R50 | AVA v2.2| [ACAR](https://github.com/Siyu-C/ACAR-Net)| Train 

- You can use [train config](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/configs/VITSF_ACAR_HR2O_CW_train.yaml) to load the weights of `vit_g_ps14_ak_ft_ckpt_7_clean.pth`  and `SLOWFAST_R50_K400.pth.tar` and train our model from scratch. We trained 12 epochs on 4 GeForce RTX 3090 with batch_size=2/GPU to get the [provided checkpoint](https://github.com/jfightyr/Spatiotemporal-Action-Localization-on-Chaotic-World-dataset/blob/main/checkpoints/README.md).