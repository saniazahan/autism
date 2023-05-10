#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:34:16 2023
https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
@author: 22905553
"""

import torchvision
import torch.nn as nn
import torch
from opts import *

import hyptorch.nn as hypnn

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
class Vit_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.imageModel = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        self.imageModel.heads = Identity()
        self.feature = nn.Sequential( nn.Linear(768, 384),
                                     nn.ReLU(inplace=True)
                                     )
        if hyp_c > 0 and class_embed_type=='hyperbolic':
                self.hpy_layer = hypnn.ToPoincare(
                    c=hyp_c,
                    ball_dim=384,
                    riemannian=False,
                    clip_r=2.3 , # clip_r: float = 2.3  # feature clipping radius
                )
        self.fc_image = nn.Linear(384, num_class)
    def forward(self, x):
        x = self.feature(self.imageModel(x))
        out = self.fc_image(x)
        if cluster_distance_loss:
            if hyp_c > 0 and class_embed_type=='hyperbolic':
                return self.hpy_layer(x)
            else:
                return x
        else:
                
            return out

if __name__ == "__main__":
#    imageModel.heads = Identity()
    x = torch.rand(1,3,224,224)
    Vit_b = Vit_b()
    pred = Vit_b(x)
