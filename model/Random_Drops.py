#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:36:12 2021

@author: 22905553
"""

import torch
import torch.nn.functional as F
from torch import nn

import warnings

class Randomized_DropBlock_Ske(nn.Module):
    def __init__(self, keep_prob, block_size=7):
        super(Randomized_DropBlock_Ske, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input, A, num_point=25):  # n,c,t,v
        
        self.num_point = num_point
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        #print(input.shape)
        input_abs = torch.mean(torch.mean(
            torch.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        if self.num_point == 25:  # Kinect V2
            gamma = (1. - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:  # Kinect V1
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:
            gamma = (1. - self.keep_prob) / (1 + 1.92)
            warnings.warn('undefined skeleton graph')
        M_seed = torch.bernoulli(torch.clamp(
            input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        #print(M_seed.shape)
        #print(A.shape)
        M = torch.matmul(M_seed, A)
        #M = torch.einsum('nv,cvw->nv', (M_seed, A)).contiguous()
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        #print(M.shape)
        mask = (1 - M).view(n, 1, 1, self.num_point)
        #print(M.shape)
        return input * mask * mask.numel() / mask.sum()
    
    
class Randomized_DropBlockT_1d(nn.Module):
    def __init__(self, keep_prob, block_size=7):
        super(Randomized_DropBlockT_1d, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.size()

        input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1).detach()
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n,1,t)
        gamma = (1. - self.keep_prob) / self.block_size
        input1 = input.permute(0,1,3,2).contiguous().view(n,c*v,t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        idx = torch.randperm(Msum.shape[2])
        RMsum = Msum[:,:,idx].view(Msum.size()) ## shuffles MSum to drop random frames instead of dropping a block of frames
        mask = (1 - RMsum).to(device=input.device, dtype=input.dtype)
        #print(mask.shape)
        return (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)
    
    
if __name__ == "__main__":
    
    
    #dropS = DropBlock_Ske()
    
    x = torch.randn(12,96,50,6)
    A = torch.randn(3,6,25)
    
    #out = dropS(x, 0.9, A, x.shape[3])
    
    dropT = DropBlockT_1d(block_size=41)
    mask, out = dropT(x, 0.9)
    