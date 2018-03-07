#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' TE mebed module '

__author__ = 'fyang'

import numpy as np
import sys, os, time, fourier, joblib
from loader import Loader
from functools import reduce
from joblib import Parallel, delayed
from settings import *


timestamps_dict = fourier.get_timestamps()
with open('{0}_pca.pkl'.format(feature_type), 'rb') as pca_file:
    pca = joblib.load(pca_file)


def embed_evve(events = range(1, 14)):
    '''
    Extract video descriptors for full EVVE dataset, 
    events should be a iterable value like [7], range(1, 14)
    '''
    for event in events:
        embed_event(event, 'qr')
        embed_event(event, 'db')


def embed_event(event, tag):
    '''
    Extract video descriptors for given events, 
    event should be a integer in the range of [1, 13]
    '''
    loader = Loader(event, tag)
    # embed
    start = time.time()
    data = get_data(loader, tag)
    print('embedding {0} for event {1} cost {2:.2f}s'.format(
            tag, event, time.time() - start))
    path = '{0}/{1}_{2}.pkl'.format(data_dir, event, tag)
    with open(path, 'wb') as file:
        joblib.dump(data, file)


def get_data(loader, tag):
    '''
    Get the data of videos by using their information
    '''
    data = Parallel(n_jobs=-1)([delayed(get_datum)(info, tag) for info in loader])
    return data


def get_datum(info, tag):
    '''
    Get the data of an video by using its information
    '''
    datum = {
            'event': info['event'],
            'name': info['name'],
            'length': info['length'],
    }
    fvecs = normalize(info['fvecs'])
    fvecs = pca.transform(fvecs)
    fvecs = normalize(fvecs)
    for period in periods:
        datum[period] = get_burst_descriptor(fvecs, period)
    return datum


def get_burst_descriptor(descs, period):
    descs[np.isnan(descs)] = 0
    timestamps = timestamps_dict[period]
    period, stamp_len = timestamps.shape
    frame_len, d = descs.shape
    shuffle_idx = np.random.permutation(np.arange(frame_len))
    new_descs = descs - descs[shuffle_idx]
    dc_comp = np.sum(descs, axis = 0) / frame_len
    dc_comp /= np.linalg.norm(dc_comp)
    ac_comp = reduce(np.add, (np.kron(timestamps[i % period][1:],
                     new_descs[i]) for i in range(frame_len)))
    ac_comp /= np.linalg.norm(ac_comp)
    new_descriptor = np.append(0.4 * dc_comp, ac_comp)
    return new_descriptor.reshape(stamp_len, d)


def normalize(vectors):
    '''
    Conduct L2 normalization on vectors
    '''
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        norms = 1 / (np.linalg.norm(vectors, 2, 1))
    norms[~np.isfinite(norms)] = 0
    vectors = np.multiply(norms[:, np.newaxis], vectors)
    return vectors


if __name__ == '__main__':
    embed_evve()
