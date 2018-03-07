#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' data extractor module for EVVE dataset, works with python3.5'

__author__ = 'fyang'

import numpy as np
import os, glob, platform, joblib
from settings import *
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import Bottleneck
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo


class VideoFrames(Dataset):
    '''
    Frames of a video
    '''
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.frame_paths = glob.glob('{0}/*.jpeg'.format(video_dir))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        image = Image.open(path)
        sample = self._transform(image)
        return sample

    def _transform(self, image):
        return transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])(image)


class Extractor(object):
    '''
    Extract feature vectors (MVLAD, alexnet, resnet) from EVVE dataset
    '''
    def __init__(self):
        '''
        Initialize instance.
        '''
        if not os.path.exists(infos_dir):
            os.makedirs(infos_dir)
        self.infos_dir = infos_dir
        self.feature_type = feature_type
        self.__annots = sorted(glob.glob('{0}/*.dat'.format(annot_dir)))


    def extract_evve(self, events = range(1, 14)):
        '''
        Extract query videos' information for EVVE dataset
        '''
        for event in events:
            annot = self.__annots[event - 1]
            self.extract_event(event, annot, 'query')
            self.extract_event(event, annot, 'database')


    def extract_event(self, event, annot, tag):
        '''
        Extract video information
        '''
        if tag is 'query':
            dir = '{0}/{1}_qr'.format(self.infos_dir, event)
        else:
            dir = '{0}/{1}_db'.format(self.infos_dir, event)
        if not os.path.exists(dir):
            os.makedirs(dir)
        annot_name = os.path.split(annot)[-1]
        event_name = os.path.splitext(annot_name)[0]
        with open(annot, 'r') as file:
            content = file.readlines()
        lines = [line.strip() for line in content]
        for line in lines:
            # video name, ground truth, category = query or database
            video_name, gt, category = line.split()
            if category != tag:
                continue
            # remove two doublets in EVVE dataset, will not affect the results
            if event == 1 and tag is 'database':
                if video_name in ['fLIOVqAMeMo', 'UyiGRY8zMOg']:
                    continue
            path = '{0}/{1}.pkl'.format(dir, video_name)
            #if os.path.exists(path):
            #    continue
            info = {'event': event, 'name': video_name}
            if self.feature_type == 'mvlad':
                fvecs_path = '{0}/{1}/{2}.fvecs'.format(fvecs_dir, event_name, video_name)
                self.save_mvlads(info, fvecs_path, path)
            elif self.feature_type == 'alexnet':
                video_frames_dir = '{0}/{1}/{2}'.format(frames_dir, event_name, video_name)
                model = AlexNet().cuda().eval()
                model.extract_features(video_frames_dir, info, path)
            elif self.feature_type == 'resnet50':
                video_frames_dir = '{0}/{1}/{2}'.format(frames_dir, event_name, video_name)
                model = ResNet50().cuda().eval()
                model.extract_features(video_frames_dir, info, path)


    def save_mvlads(self, info, fvecs_path, info_path):
        '''
        Read frame feature vectors
        '''
        with open(fvecs_path, 'rb') as f:
            info['fvecs']=np.fromfile(f, np.float32).reshape(-1, 1025)[:, 1:1025]
        info['length'] = info['fvecs'].shape[0]
        print('saving video {0}\'s info in event {1}'
                .format(info['name'], info['event']))
        with open(info_path, 'wb') as info_file:
            joblib.dump(info, info_file, compress=True)


class AlexNet(nn.Module):
    '''
    Extract feature vectors from last convolutional layer
    '''
    def __init__(self):
        super(AlexNet, self).__init__()
        alexnet_model = models.alexnet(pretrained=True)
        self.features = alexnet_model.features
        self.avgpool = nn.AvgPool2d(6)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


    def extract_features(self, video_frames_dir, info, path):
        '''
        Extract features for given video
        '''
        video_frames = VideoFrames(video_frames_dir)
        frame_loader = DataLoader(video_frames, batch_size =
                self.batch_size, num_workers = 4)
        frame_len = len(video_frames)
        fvecs = np.zeros((frame_len, 2048, 1, 1))
        for i, sample_batched in enumerate(frame_loader): # i = i_batch
            result = self(Variable(sample_batched).cuda())
            features = result.data.cpu().numpy()
            if (i+1) * self.batch_size < frame_len:
                fvecs[i*self.batch_size:(i+1)*self.batch_size] = features
            else:
                fvecs[i*self.batch_size:] = features
        fvecs[~np.isfinite(fvecs)] = 0
        info['length'] = len(video_frames)
        info['fvecs'] = fvecs[:,:,0,0]
        print('saving video {0}\'s info in event {1}'
                .format(info['name'], info['event']))
        with open(path, 'wb') as info_file:
            joblib.dump(info, info_file, compress=True)


class ResNet50(models.ResNet):
    '''
    Extract feature vectors from last convolutional layer
    '''
    def __init__(self):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        # self.avgpool = nn.AvgPool2d(3, padding = 1)
        self.load_state_dict(torch.load(resnet_model_path))
        self.batch_size = 60

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

    def extract_features(self, video_frames_dir, info, path):
        '''
        Extract features for given video
        '''
        video_frames = VideoFrames(video_frames_dir)
        frame_loader = DataLoader(video_frames, batch_size =
                self.batch_size, num_workers = 4)
        frame_len = len(video_frames)
        fvecs = np.zeros((frame_len, 2048, 1, 1))
        size = self.batch_size
        for i, sample_batched in enumerate(frame_loader): # i = i_batch
            result = self(Variable(sample_batched).cuda())
            features = result.data.cpu().numpy()
            if (i+1) * size < fvecs.shape[0]:
                fvecs[i*size:(i+1)*size] = features
            else:
                fvecs[i*size:] = features
        fvecs[~np.isfinite(fvecs)] = 0
        info['length'] = len(video_frames)
        info['fvecs'] = fvecs[:,:,0,0]
        print('saving video {0}\'s info in event {1}'
                .format(info['name'], info['event']))
        with open(path, 'wb') as info_file:
            joblib.dump(info, info_file, compress=True)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    extractor = Extractor()
    extractor.extract_evve()
