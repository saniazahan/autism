#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:06:14 2023
cd /home/uniwa/students3/students/22905553/linux/phd_codes/Autism_Detection_Works/Autism_classification/asd/
@author: 22905553
"""

#!/usr/bin/env python
from __future__ import print_function
import os
import sys

sys.path.append('../')
from itertools import chain
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
#import apex

from dataset import VideoDataset as VideoDataset
from opts import *
from utils import count_params, import_class
from model.VitB import Vit_b

from model.x3d import VideoModel

from model.msg3d import Model


from hyptorch.pmath import dist_matrix
import torch.nn.functional as F
from functools import partial

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    
    parser.add_argument(
        '--dataset-seed',
        type=int,
        default=0,
        help='dataset seed for cross validation')
    
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--amp-opt-level',
        type=int,
        default=1,
        help='NVIDIA Apex AMP optimization level')

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')
    
    parser.add_argument(
        '--finetune',
        type=str2bool,
        default=False,
        help='finetune mode; default false')

    return parser


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'train_debug'), 'train_debug')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val_debug'), 'val_debug')
        self.global_step = 0
        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.arg.amp_opt_level}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        if GCN_stream:
            Model = import_class(self.arg.model)
            self.model = Model(**self.arg.model_args).cuda(output_device)
            shutil.copy2('./model/msg3d.py', self.arg.work_dir)
        # Copy model file and main
        #shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        #shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)
        
        if video_stream:
            self.VideoModel = VideoModel().cuda(output_device)
            if x3d_state_path:
                weights = torch.load(x3d_state_path)
                weights = OrderedDict(
                    [[k.split('module.')[-1],
                      v.cuda(output_device)] for k, v in weights.items()])
                #print(weights)
                
                try:
                    self.VideoModel.load_state_dict(weights)
                except:
                    state = self.VideoModel.state_dict()
                    diff = list(set(state.keys()).difference(set(weights.keys())))
                    self.print_log('Can not find these weights:')
                    for d in diff:
                        self.print_log('  ' + d)
                    state.update(weights)
                    self.VideoModel.load_state_dict(state)
                    
            #print(self.VideoModel)
            for name, params in self.VideoModel.named_parameters():
                #if 'x3d_feat.4' in name or 'asd' in name:     
                if 'asd' in name:     
                    #print(name)
                    params.requires_grad = True
                else:
                    params.requires_grad = False
                    

            
            
        if GaitEnergy:
            self.Vit_b = Vit_b().cuda(output_device)
            #print(Vit_b)
            if states_path:
                #states = torch.load(states_path)
                #Vit_b_states = states['Vit_b']
                #self.Vit_b.load_state_dict(Vit_b_states)
                self.print_log(f'Loading weights from {states_path}')
                if '.pkl' in states_path:
                    with open(states_path, 'r') as f:
                        weights = pickle.load(f)
                else:
                    weights = torch.load(states_path)

                weights = OrderedDict(
                    [[k.split('module.')[-1],
                      v.cuda(output_device)] for k, v in weights.items()])
                #print(weights)
                for w in self.arg.ignore_weights:
                    if weights.pop(w, None) is not None:
                        self.print_log(f'Sucessfully Remove Weights: {w}')
                    else:
                        self.print_log(f'Can Not Remove Weights: {w}')

                try:
                    self.Vit_b.load_state_dict(weights)
                except:
                    state = self.Vit_b.state_dict()
                    diff = list(set(state.keys()).difference(set(weights.keys())))
                    self.print_log('Can not find these weights:')
                    for d in diff:
                        self.print_log('  ' + d)
                    state.update(weights)
                    self.Vit_b.load_state_dict(state)

                
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.loss_pred = nn.MSELoss().cuda(output_device)
        self.w_pred = 0.01
        #self.print_log(f'Model total number of params: {count_params(self.model)}')

        if self.arg.weights and GCN_stream:
            if not self.arg.weights.endswith('.pt'):
                self.arg.weights = self.arg.weights+'/'+str(seed)+'/weights/weights.pt'
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            #print(weights)
            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        if GCN_stream:
            self.param_groups = defaultdict(list)

            for name, params in self.model.named_parameters():
                self.param_groups['other'].append(params)

            self.optim_param_groups = {
                'other': {'params': self.param_groups['other']}
                }
        
        if GaitEnergy:
            
            self.param_groups_Vit_b = defaultdict(list)

            for name, params in self.Vit_b.named_parameters():
                self.param_groups_Vit_b['other'].append(params)

            self.optim_param_groups_Vit_b = {
                'other': {'params': self.param_groups_Vit_b['other']}
            }
        
        if video_stream:            
            self.param_groups_x3d = defaultdict(list)

            for name, params in self.VideoModel.named_parameters():
                self.param_groups_x3d['other'].append(params)

            self.optim_param_groups_x3d = {
                'other': {'params': self.param_groups_x3d['other']}
            }
        
        
        

    def load_optimizer(self):
        if GaitEnergy:
            params = list(self.optim_param_groups_Vit_b.values())
        if GCN_stream:
            params = list(self.optim_param_groups.values())
        if video_stream:
            params = list(self.optim_param_groups_x3d.values())
        #if two_stream:
        #    params = list(self.optim_param_groups.values())+list(self.optim_param_groups_Vit_b.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self):
        #Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)
        
        if self.arg.phase == 'train':
            train_dataset = VideoDataset('train', self.arg)
            #train_sampler = train_dataset.__sampler__()
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=True,
                #sampler = train_sampler,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=VideoDataset('test', self.arg),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)
        

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        #if not os.path.exists(self.arg.work_dir+'/confusion_matrix_figs'):
        #    os.makedirs(self.arg.work_dir+'/confusion_matrix_figs')
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        #checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        checkpoint_name = f'checkpoint.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        if GCN_stream:
            state_dict = self.model.state_dict()
            weights = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict.items()
            ])

            #weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
            weights_name = f'weights.pt'
            self.save_states(epoch, weights, out_folder, weights_name)
            
        if video_stream:
            state_dict = self.VideoModel.state_dict()
            weights = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict.items()
            ])

            #weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
            weights_name = f'x3d_weights.pt'
            self.save_states(epoch, weights, out_folder, weights_name)
        
        if GaitEnergy:
            state_dict = self.Vit_b.state_dict()
            weights = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict.items()
            ])

            #weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
            weights_name = f'Vit_b_weights.pt'
            self.save_states(epoch, weights, out_folder, weights_name)
            
    def euclidean_loss(self, gcn_feat, vit_feat):
        #EuclideanDistance
        #x = gcn_feat.cpu().detach().numpy()
        #y = vit_feat.cpu().detach().numpy()
        #cluster_loss = np.linalg.norm(x - y)
        #print(cluster_loss)
        
        cluster_loss = (gcn_feat - vit_feat).pow(2).sum(1).sqrt().mean()

        #print(cluster_loss)
        return cluster_loss
    
    def hyperbolic_loss(self, x0, x1, tau, hyp_c):
        # x0 and x1 - positive pair
        # tau - temperature
        # hyp_c - hyperbolic curvature, "0" enables sphere mode

        if hyp_c == 0:
            dist_f = lambda x, y: x @ y.t()
        else:
            dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = dist_f(x0, x0) / tau - eye_mask
        logits01 = dist_f(x0, x1) / tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        '''
        stats = {
            "logits/min": logits01.min().item(),
            "logits/mean": logits01.mean().item(),
            "logits/max": logits01.max().item(),
            "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
        }
        '''
        return loss#, stats
    
    def train(self, epoch, save_model=False):
        
        torch.autograd.detect_anomaly(True)
        
        if GCN_stream:
            self.model.train()
        
        if video_stream:
            self.VideoModel.train()
            
        if GaitEnergy:
            if cluster_distance_loss and use_vit_for_cluster_loss_only:
                self.Vit_b.eval()
            else:
                self.Vit_b.train()
                
        loader = self.data_loader['train']
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')  
    
        process = tqdm(loader, dynamic_ncols=True)
        #torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, _) in enumerate(process):
            '''
            print( "batch index {}, 0/1: {}/{}".format(
                batch_idx, len(np.where(label.numpy()==0)[0]), len(np.where(label.numpy()==1)[0])))
            '''
            self.global_step += 1
            # get data
            with torch.no_grad():              
                
                label = data['label'].long().cuda(self.output_device)
                if GaitEnergy: 
                    GE = data['GaitEnergy'].float().cuda(self.output_device)
                if GCN_stream:
                    data = data['skel'].float().cuda(self.output_device)
                if video_stream:
                    vid_data = data['video'].float().cuda(self.output_device)
                
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            if GCN_stream:
                splits = len(data) // real_batch_size
                assert len(data) % real_batch_size == 0, \
                    'Real batch size should be a factor of arg.batch_size!'
            if video_stream:
                splits = len(vid_data) // real_batch_size
                assert len(vid_data) % real_batch_size == 0, \
                    'Real batch size should be a factor of arg.batch_size!'
            
            

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_label = label[left:right]
                if GCN_stream:
                    batch_data = data[left:right]
                if GaitEnergy: 
                    batch_gait = GE[left:right]
                if video_stream:
                    batch_vid_data = vid_data[left:right]
                    
                    
                # forward
                if GaitEnergy: 
                    #print(batch_gait.shape)
                    #output_skel = self.model(batch_data)
                    output_gait = self.Vit_b(batch_gait)
                    #print(output_gait.shape)
                    if cluster_distance_loss and use_vit_for_cluster_loss_only:
                        vit_feat = output_gait
                    else:
                        output = output_gait
                if GCN_stream:
                    output_gcn = self.model(batch_data)
                    if cluster_distance_loss:
                        output = output_gcn[0]
                        gcn_feat = output_gcn[1]
                    else:                            
                        output = output_gcn
                if video_stream:
                    output_video = self.VideoModel(batch_vid_data)
                    if cluster_distance_loss:
                        output = output_video[0]
                        vid_feat = output_video[1]
                    else:                            
                        output = output_video
                #if two_stream:
                #    output = output_gait + output_gcn
                    
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                
                
                loss = self.loss(output, batch_label) / splits
                
                if cluster_distance_loss:
                    if distance_type == 'euclidean':
                        distance_loss  = self.euclidean_loss(gcn_feat, vit_feat)
                    elif distance_type == 'hyperbolic':
                        distance_loss = self.hyperbolic_loss(gcn_feat, vit_feat, tau, hyp_c)
                    loss = loss + 0.01*distance_loss
                    
                    
                loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())

                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step)

                self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            #####################################
            if video_stream:
                torch.nn.utils.clip_grad_norm_(self.VideoModel.parameters(), 2)
            self.optimizer.step()
            '''
            for name, param in self.model.named_parameters():
                print(name, torch.isfinite(param.grad).all(), torch.isnan(param.grad).all())
            '''
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        self.lr_scheduler.step()

        #if save_model:
            # save training checkpoint & weights
            #self.save_weights(epoch + 1)
            #self.save_checkpoint(epoch + 1)
          
        # Empty cache after evaluation
        torch.cuda.empty_cache()
        
    def perf_measure(self, y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
    
        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1
    
        return(TP, FP, TN, FN)
        
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        
        ## inference time 
        # start time
        torch.cuda.synchronize()
        tsince = int(round(time.time()*1000))
        
        with torch.no_grad():
            if GCN_stream:
                #self.model = self.model.cuda(self.output_device)
                self.model.eval()
            if video_stream:
                self.VideoModel.eval()                
            if GaitEnergy and not use_vit_for_cluster_loss_only:
                self.Vit_b.eval()
                
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                score_batches = []
                preds = []
                targets = []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, index) in enumerate(process):
                    # print('batch', index)
                    # print((data['sample']))
                    # print(len(data['skel']))
                    # print(len(data['GaitEnergy']))
                    # print((data['label']))
                    # print('++++')
                    label = data['label'].long().cuda(self.output_device)
                    if GCN_stream:
                        skel = data['skel'].float().cuda(self.output_device)
                    if GaitEnergy: 
                        GE = data['GaitEnergy'].float().cuda(self.output_device)
                    if video_stream:
                        video = data['video'].float().cuda(self.output_device)
                        
                    if GaitEnergy and not use_vit_for_cluster_loss_only: 
                        #print(batch_gait.shape)
                        #output_skel = self.model(batch_data)
                        output_gait = self.Vit_b(GE)
                        #print(output_gait.shape)
                        output = output_gait
                        del output_gait
                        
                    if GCN_stream:
                        output_gcn = self.model(skel)
                        output = output_gcn    
                        del output_gcn
                        
                    if video_stream:
                        out_video = self.VideoModel(video)
                        output = out_video
                        del out_video
                        
                    #if two_stream:
                    #    output = output_gait + output_gcn
                        
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1
                    
                    preds.append(predict_label.cpu().numpy())   ## SZ
                    targets.append(label.data.cpu().numpy())    ## SZ
                    #print(data['sample'])
                    
                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        # print(len(predict))
                        # print(len(true))
                        # print(len(data['sample']))
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + ',' + str(data['sample'][i]) + '\n')
                                
                    del output
                    torch.cuda.empty_cache()
                    
            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                self.save_weights(epoch + 1)
                self.save_checkpoint(epoch + 1)
           
            if self.arg.phase == 'test':
                #print(len(targets), targets,'\n')
                #print(len(preds), preds)
                targets = list(chain(*targets))
                
                preds = list(chain(*preds))
                
                TP, FP, TN, FN = self.perf_measure(targets, preds)
                #fp_rate = FP/(FP+TN)
                specificity = TN/(TN+FP)
                precision = TP/(TP+FP)
                sensitivity = TP/(TP+FN) #recall 
                F1_score = 2*(precision*sensitivity)/(precision+sensitivity)
                accuracy= (TP+TN)/(TP+TN+FP+FN)
                self.print_log(f'\tAccuracy: {accuracy} \tF1_score: {F1_score} \nSpecificity: {specificity} \tSensitivity: {sensitivity} \tPrecision: {precision}')
                
            ## Confusion matrix  SZ ====
            #preds = np.concatenate(preds)   
            #targets = np.concatenate(targets)
            #conf_mat = confusion_matrix(preds, targets)
            #df_cm = pd.DataFrame(conf_mat, index = range(2), columns=range(2))
            #fig_ = sns.heatmap(df_cm,cbar=False).get_figure()
            #fig_.savefig('./'+self.arg.work_dir+'/confusion_matrix_figs/Conf_mat_'+str(epoch+1)+'.png', bbox_inches='tight')
            #====
            
            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)
            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
            
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_path, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
          
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')
            
            torch.cuda.synchronize()
            ttime_elapsed = int(round(time.time()*1000)) - tsince
            print ('test time elapsed {}ms'.format(ttime_elapsed))
            print(len(score))
            
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
          
        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            #self.print_log(f'Model total number of params: {count_params(self.model)}')
            if self.arg.start_epoch == 0:
                self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            else:
                self.global_step = self.global_step
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
            
            if GCN_stream:
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if video_stream:
                num_params = sum(p.numel() for p in self.VideoModel.parameters() if p.requires_grad)
            else:
                num_params = sum(p.numel() for p in self.Vit_b.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(seed = 0):
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    arg.dataset_seed = seed
    if cross_validation:
        arg.work_dir = arg.work_dir+'/'+str(seed)
    #print(arg.dataset_seed)
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    if cross_validation:
        
        for seed in range(folds):
            if seed_type == 'block': # block  random
                seed = seed+1
            main(seed)
    else:
        main(seed)
                