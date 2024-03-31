import torch.nn as nn
import torch

from timm.models import create_model
from .VideoMAEv2 import *

from .slowfast import *
from utils import load_pretrain


def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])


class AVA_backbone(nn.Module):
    def __init__(self, config):
        super(AVA_backbone, self).__init__()
        
        # get slowfast module and Vit module
        self.config = config
        self.module = model_entry(config)
        self.submodule = create_model(
                            "vit_giant_patch14_224",
                            img_size=224,
                            pretrained=False,
                            num_classes=80,
                            all_frames=32,
                            tubelet_size=2,
                            drop_path_rate=0.3,
                            use_mean_pooling=True)
        
        # load ckpt for slowfast
        if config.get('pretrain', None) is not None:
            load_pretrain(config.pretrain, self.module)

        # load ckpt for VideoMaev2
        if config.get('subbackbone', None) is not None:
            print("loading ckpt for subbackbone.")
            ckpt_path = config['subbackbone']['path']
            ckpt = torch.load(ckpt_path, map_location='cpu')
            for model_key in ['model', 'module']:
                if model_key in ckpt:
                    ckpt = ckpt[model_key]
                    break
            ckpt['fc_norm.weight'] = ckpt['norm.weight']
            ckpt['fc_norm.bias'] = ckpt['norm.bias']
            ckpt.pop("norm.weight")
            ckpt.pop("norm.bias")

            ckpt_keys = set(ckpt.keys())
            own_keys = set(self.submodule.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            ignore_keys = ckpt_keys - own_keys
            loaded_keys = own_keys - missing_keys

            print("missing_keys: ",missing_keys )
            print("ignore_keys: ",ignore_keys )
            # print("loaded_keys: ",loaded_keys )

            self.submodule.load_state_dict(ckpt)

        # config trainable
        self.submodule.train(False)
        self.submodule.requires_grad_(False)
        self.submodule.cuda()
            
        if not config.get('learnable', True):
            self.module.requires_grad_(False)
            

    # data: clips
    # returns: features
    def forward(self, data):
        inputs = data['clips']
        inputs = inputs.cuda()
        features = self.module(inputs)  
        
        subfeature = self.submodule(inputs)
        features.append(subfeature)

        return {'features': features}

        # slow torch.Size([B, 2048, 8, 14, 14])
        # fast torch.Size([B, 256, 32, 14, 14])
        # vit  torch.size([B, 1408, 16, 16, 16])
