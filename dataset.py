# Adapted from the code for paper 'What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment'.
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
from opts import *
from scipy import stats
import pickle
import random
import cv2


def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

class VideoDataset(Dataset):

    def __init__(self, mode, args=None):
        super(VideoDataset, self).__init__()
        self.mode = mode  # train or test
        # loading annotations
        self.args = args
        #self.annotations = pkl.load(open(os.path.join(info_dir, 'augmented_final_annotations_dict.pkl'), 'rb'))
        #self.keys = pkl.load(open(os.path.join(info_dir, f'{self.mode}_split_0.pkl'), 'rb'))
        
            
        if stream=='skel_rgb':
            if mode == 'train':
                self.path = train_label_dir
            else:
                self.path = test_label_dir
            with open(self.path, 'rb') as f:
                    self.label, self.sample_path = pickle.load(f, encoding='latin1')
        else:
            if mode == 'train':
                if cross_validation:
                    if seed_type == 'block': # block  random
                        self.label_path = './data/skeleton/train_labels_block_'+str(self.args.dataset_seed)+'.pkl'
                        self.data_path = './data/skeleton/train_data_block_'+str(self.args.dataset_seed)+'.npy'
                    else:
                        self.label_path = './data/skeleton/train_labels_skel_'+str(self.args.dataset_seed)+'.pkl'
                        self.data_path = './data/skeleton/train_data_skel_'+str(self.args.dataset_seed)+'.npy'
                    #print(self.label_path)
                    print(self.data_path)
                else:
                    self.label_path = train_label_dir
                    self.data_path = train_data_dir
            else:
                if cross_validation:
                    if seed_type == 'block': # block  random
                        if test_type == 'NoAug':
                            self.label_path = './data/skeleton/test_labels_NoAug_block_'+str(self.args.dataset_seed)+'.pkl'
                            self.data_path = './data/skeleton/test_data_NoAug_block_'+str(self.args.dataset_seed)+'.npy'
                        else:
                            self.label_path = './data/skeleton/test_labels_block_'+str(self.args.dataset_seed)+'.pkl'
                            self.data_path = './data/skeleton/test_data_block_'+str(self.args.dataset_seed)+'.npy'
                    else:
                        if test_type == 'NoAug':
                            self.label_path = './data/skeleton/test_labels_NoAug_skel_'+str(self.args.dataset_seed)+'.pkl'
                            self.data_path = './data/skeleton/test_data_NoAug_skel_'+str(self.args.dataset_seed)+'.npy'
                        else:
                            self.label_path = './data/skeleton/test_labels_skel_'+str(self.args.dataset_seed)+'.pkl'
                            self.data_path = './data/skeleton/test_data_skel_'+str(self.args.dataset_seed)+'.npy'
                else:
                    self.label_path = test_label_dir
                    self.data_path = test_data_dir
            # print(self.data_path)   
            # print(self.label_path)  
            
            with open(self.label_path, 'rb') as f:
                self.label, self.sample_path = pickle.load(f, encoding='latin1')
            if GCN_stream:
                self.data = np.load(self.data_path)        
            # print(len(self.data))
            # print(len(self.label))
            # print(len(self.sample_path))
            if normalization:
                self.get_mean_map()
                
    def get_mean_map(self):
        data = self.data
        #print(data.shape)
        #N, C, T, V, M = data.shape
        N, T, V, C = data.shape
        #data = data.permute(0,3,1,2)
        self.mean_map = data.mean(axis=1, keepdims=True).mean(axis=0)
        #print(self.mean_map.shape)
        self.std_map = data.transpose((0, 1, 3, 2)).reshape((N * T, C * V)).std(axis=0).reshape((1,V,C))
        self.std_map[self.std_map==0.0] = 0.0001
        #print(self.std_map.shape)

    def get_skelimgs(self, key):
        if '_v' in self.path:
            frames_dir = self.sample_path[key]+'/'+self.sample_path[key].split('/')[-1]+'_frames/'
        else:
            sample = self.sample_path[key][0]
            frames_dir = sample+self.sample_path[key][1][-1]+'/'
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_list = sorted((glob.glob(os.path.join(frames_dir,'*.jpg'))))
        #sample_range = np.arange(0, 103)
        fr_idx = np.arange(1,len(image_list)-7)
        #print(fr_idx)
        if len(fr_idx)>num_frames:
            sample_range = np.array(sorted(np.random.choice(fr_idx, num_frames-7, replace=False)))
        else:
            sample_range = fr_idx
        
        
        # temporal augmentation
        if self.mode == 'train':
            temporal_aug_shift = random.randint(0, self.args.temporal_aug)
            #print(sample_range)
            #print(temporal_aug_shift)
            sample_range += temporal_aug_shift
        # spatial augmentation
        if self.mode == 'train':
            hori_flip = random.randint(0, 1)
        images = torch.zeros(num_frames, C, H, W)
        for j, i in enumerate(sample_range):
            #break
            if self.mode == 'train':
                try:
                    images[j] = load_image_train(image_list[i], hori_flip, transform)
                except:
                    print(len(image_list))    
                    print(i)
            if self.mode == 'test':
                images[j] = load_image(image_list[i], transform)
        return images
    
    def get_video(self, key, sample_path):
        #print(sample_path)
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        
        # print(sample_path.split('Dataset/'))
        # print(sample_path.split('Dataset/')[2].split('/'))
        
        path = sample_path.split('Dataset/')[2].split('/')#+'.video/video_frames/'
        path = './data/raw_data/Gait-Dataset/'+path[0]+'/'+path[1]+'/video/video_frames/'
        # print(path)
        #frames = sorted(os.listdir(path))
        #print(frames)
        image_list = sorted((glob.glob(os.path.join(path,'*.jpg'))))
        #print(len(image_list))
        fr_idx = np.arange(1,len(image_list)-7)
        #print(fr_idx)
        if len(fr_idx)>num_frames:
            sample_range = np.array(sorted(np.random.choice(fr_idx, num_frames-7, replace=False)))
        else:
            sample_range = fr_idx
        
        
        # temporal augmentation
        if self.mode == 'train':
            temporal_aug_shift = random.randint(0, temporal_aug)
            sample_range += temporal_aug_shift
        # spatial augmentation
        if self.mode == 'train':
            hori_flip = random.randint(0, 1)
        images = torch.zeros(num_frames, C, H, W)
        for j, i in enumerate(sample_range):
            #print(image_list[i])
            if self.mode == 'train':
                try:
                    images[j] = load_image_train(image_list[i], hori_flip, transform)
                except:
                    print(len(image_list))    
                    print(i)
            if self.mode == 'test':
                images[j] = load_image(image_list[i], transform)
            break
        #print(images[0][0])
        return images, path
        
        
        
    def create_GaitEnergy(self, x):
        #GE = np.zeros((50,64,3))
        GE = np.zeros((25,64,3))
        for i in range(25):
            inception = x[0][i]
            for j in range(64):
                #new = np.linalg.norm(np.abs(inception-x[j,i]))
                new = np.abs(inception-x[j,i])
                GE[i,j,:] = new
                #m = 25
                #for n in range(25):
                #    new = np.linalg.norm(np.abs(inception-x[j,n]))
                #    GE[m,j] = new
                #    m = m+1
        return GE
    
    def super_pixel(self, skel_frame, random_seed):
        random.seed(random_seed)
        joints_order = np.reshape(random.sample(range(25), 25),(5,5))
        #print(skel_frame.shape)
        skel_spixel = skel_frame[joints_order]
        return skel_spixel

    def create_skepxel(self, data):
        # https://github.com/liujianee/SKEPXEL-Skeleton_Pixels_for_Action_Recognition
        SPIXEL = 5
        SPATIAL_DIM = 64-5
        TEMPORAL_DIM = 64-5
        STRIDE = 1	# decide how many pseudo images to be created
        SKIP = 1	# decide how dense/sparse the skeleton frames are sampled, to build one pseudo image
        skel_arr = np.zeros((SPATIAL_DIM*SPIXEL,TEMPORAL_DIM*SPIXEL,3), dtype=float)
        for frame_ix in range(TEMPORAL_DIM):
            #print(data.shape)
            current_frame = data[(STRIDE + frame_ix*SKIP)]
            #print(current_frame.shape)
            for order_ix in range(SPATIAL_DIM):
                skel_arr[order_ix*SPIXEL : (order_ix+1)*SPIXEL, frame_ix*SPIXEL : (frame_ix+1)*SPIXEL] = self.super_pixel(current_frame, order_ix)
    
        skel_img = cv2.normalize(skel_arr, skel_arr, 0, 1, cv2.NORM_MINMAX)
        skel_img = np.array(skel_img * 255, dtype = np.uint8)
        
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((H,W)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image = transform(skel_img)#.unsqueeze(0)
        return image
    
    def __getitem__(self, ix):
        data = {}
        #print(ix)
        data['sample'] = self.sample_path[ix]
        if stream=='skel_rgb':
            data['video'] = self.get_skelimgs(ix)
        elif 'video' in stream:
            data['video'], data['path'] = self.get_video(ix, data['sample'])
            
            #print(data['video'].shape)
        if 'skel' in stream:
            data_numpy = self.data[ix]            
            if normalization:
                data_numpy = (data_numpy - self.mean_map) / self.std_map
            data['skel'] = data_numpy
            #print(data['skel'].shape)
            #print(self.label)
            #print(self.label[ix])
        if GaitEnergy:
            data['GaitEnergy']  = self.create_skepxel(data['skel'])
            #data['GaitEnergy']  = self.create_GaitEnergy(data['skel'])
        data['label'] = self.label[ix]
        
        #print(ix, ' ' , data['sample'])
        #print('inside', len(self.sample_path))
        return data, ix

    def __len__(self):
        sample_pool = len(self.label)
        return sample_pool
    
    def top_k(self, score, top_k=1):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
