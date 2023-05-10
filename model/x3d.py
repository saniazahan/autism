#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 20:09:14 2023
cd /home/22905553/phd_codes/Autism_Detection_Works/Autism_classification/asd/
@author: 22905553@student.uwa.edu.au
"""
import torch
import torch.nn as nn
from torchvision import models
import thop
from thop import clever_format

from opts import *


def init_param_fc(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
class VideoModel(nn.Module):
            def __init__(self):
                super(VideoModel, self).__init__()
                model_name = 'x3d_l'#'mvit_base_32x3'   X3D_L - 77.44% and mvit_base_32x3 - 80.30 on kinetics400 
                #model_name = 'mvit_base_32x3'#'mvit_base_32x3'   X3D_L - 77.44% and mvit_base_32x3 - 80.30 on kinetics400 
                original_model =  torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
                self.x3d_feat = nn.Sequential(
                    # stop at conv4
                    #*list(original_model.children())[:] #mvit
                    *list(original_model.blocks.children())[:-1] #x3d
                )
                
                self.asd_mlp = nn.Sequential(
                    nn.Linear(12288, 2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, 384),
                    nn.ReLU(inplace=True)                    
                    )
                self.fc_asd = nn.Linear(384, num_class)
                
                
            def forward(self, x):
                N,T,C,H,W = x.size()
                x = x.permute(0,2,1,3,4)
                x = self.x3d_feat(x)
                x = x.view(N,192,T, -1).mean(-1).view(N,-1)
                x = self.asd_mlp(x)
                x = self.fc_asd(x)
                return x
                            
if __name__ == "__main__":
# cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    import thop
    from thop import clever_format
      
    
    model = VideoModel()#.cuda()
    model.eval()
    
    x = torch.randn(1, 64,3, 224,224)#.cuda()
    
    out = model(x)
    out.shape    
    
    macs, params = thop.profile(model, inputs=(x,), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    print( macs, params)    
    
    for name, params in model.named_parameters():
        if 'x3d_feat.4' in name or 'asd' in name:     
            print(name)
            params.requires_grad = True
        else:
            params.requires_grad = False

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    