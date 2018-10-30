#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' TE mebed module '

__author__ = 'fyang'

import os
import time
import argparse
import numpy as np
from fourier import Fourier
from loader import Loader
from functools import reduce
from joblib import Parallel, delayed
import utils


def embed_video(info, tag):
    '''
    Get the data of an video by using its information
    '''
    result = {
        'event': info['event'],
        'name': info['name'],
        'length': info['length'],
    }
    fvecs = utils.normalize(info['fvecs'])
    fvecs[np.isnan(fvecs)] = 0
    if use_mean:
        result['mean'] = get_mean(fvecs)
        if apply_pca:
            result['mean'] = pca.transform(result['mean'].reshape(1,-1))[0]
            result['mean'] = utils.normalize(result['mean'])
    else:
        if apply_pca:
            fvecs = pca.transform(fvecs)
            fvecs = utils.normalize(fvecs)
        for period in periods:
            result[period] = get_burst_descriptor(fvecs, period)
    return result


def get_burst_descriptor(descs, period):
    # descs[np.isnan(descs)] = 0
    timestamps = timestamps_dict[period]
    period, stamp_len = timestamps.shape
    frame_len, d = descs.shape
    shuffle_idx = np.random.permutation(np.arange(frame_len))
    new_descs = descs - descs[shuffle_idx]
    dc_comp = np.sum(descs, axis=0) / frame_len
    dc_comp /= np.linalg.norm(dc_comp)
    ac_comp = reduce(np.add, (np.kron(timestamps[i % period][1:],
                                      new_descs[i]) for i in range(frame_len)))
    ac_comp /= np.linalg.norm(ac_comp)
    new_descriptor = np.append(0.4 * dc_comp, ac_comp)
    return new_descriptor.reshape(stamp_len, d)


def get_mean(descs):
    '''
    mean baseline
    '''
    descriptor = np.mean(descs, axis=0)
    descriptor /= np.linalg.norm(descriptor)
    return descriptor


class Embed(object):
    def __init__(self, infos_dir, embed_dir):
        self.infos_dir = infos_dir
        self.embed_dir = embed_dir

    def __call__(self, events=range(1, 14)):
        for event in events:
            self.embed_event(event, 'qr')
            self.embed_event(event, 'db')

    def embed_event(self, event, tag):
        '''
        Extract video descriptors for given events,
        event should be a integer in the range of [1, 13]
        '''
        loader = Loader(self.infos_dir, event, tag)
        t0 = time.time()
        path = '{0}/{1}_{2}.jbl'.format(self.embed_dir, event, tag)
        embedded = Parallel(n_jobs=-1)([delayed(embed_video)(info, tag)
                                        for info in loader])
        print('embedding {0} for event {1} cost {2:.2f}s'.format(
            tag, event, time.time() - t0))
        utils.save(embedded, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infos_dir', type=str,
                        help="directory to embeddings")
    parser.add_argument('--embed_dir', type=str,
                        help="directory to embeddings")
    parser.add_argument('--pca', action='store_true',
                        help="apply pca or not")
    parser.add_argument('--pca_path', type=str,
                        help="path to pca file")
    parser.add_argument('--periods', type=int, nargs='+',
                        help="list of periods")
    parser.add_argument('--mean', action='store_true',
                        help="use mean descriptor or not")
    parser.add_argument('--events', type=int, nargs='+', default=range(1, 14),
                        help="list of events")
    args = parser.parse_args()
    if not os.path.exists(args.embed_dir):
        os.makedirs(args.embed_dir)
    apply_pca = args.pca
    use_mean = args.mean
    if apply_pca:
        assert os.path.exists(args.pca_path)
        pca = utils.load(args.pca_path)
    if not use_mean:
        periods = args.periods
        fourier = Fourier(periods)
        timestamps_dict = fourier.get_timestamps()
    embed = Embed(args.infos_dir, args.embed_dir)
    embed(args.events)
