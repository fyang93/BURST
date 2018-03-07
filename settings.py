#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' TE setting module '

__author__ = 'fyang'


feature_type = 'resnet50'   # 'mvlad', 'alexnet', 'resnet50'
annot_dir = 'dataset/annots'
fvecs_dir = '/net/dl380g7a/export/ddn11a3/satoh-lab/spoullot/anh/EVVE/descs'
videos_dir = 'dataset/videos'
fps = '5fps'
frames_dir = 'dataset/frames_{0}'.format(fps)
infos_dir = 'dataset/{0}_infos_{1}'.format(feature_type, fps)
data_dir = 'data/{0}_data_{1}'.format(feature_type, fps)
result_dir = 'result/{0}_result_{1}'.format(feature_type, fps)

alexnet_model_path = '../models/alexnet-owt-4df8aa71.pth'
resnet_model_path = '../models/resnet50-19c8e357.pth'

max_freq_num = 16   # parament for Fourier series
frame_desc_dim = 1024

fourier_coefs_path = '{0}/fourier.pkl'.format(data_dir)
combine_result_path = '{0}/result.dat'.format(result_dir)

# parameters for query expansion
short_list_length = 10
far_list_length = 2000
epsilon = 50


# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
#  610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657]
periods = [144,233,377,610]
print('periods are', periods)
