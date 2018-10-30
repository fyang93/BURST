#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' TE retrieve module '

__author__ = 'fyang'

import os
import time
import argparse
from fourier import Fourier
import numpy as np
from joblib import Parallel, delayed
import utils


# global data kept for parallel processes
global_qr_data = ''
global_db_data = ''


def get_event_results(qr_data, db_data):
    '''
    Conduct retrieve task with given query data and database data
    '''
    global global_qr_data, global_db_data
    global_qr_data = qr_data
    global_db_data = db_data
    qr_data_num = len(qr_data)
    db_data_num = len(db_data)
    results = Parallel(n_jobs=-1)([delayed(get_query_result)(i, j)
                                   for i in range(qr_data_num)
                                   for j in range(db_data_num)])
    return results


def get_query_result(qr_datum_index, db_datum_index):
    '''
    Get query result with a given query datum index and a database datum index
    '''
    qr_datum = global_qr_data[qr_datum_index]
    db_datum = global_db_data[db_datum_index]
    if use_mean:
        result = {
            'qrevent': qr_datum['event'],
            'dbevent': db_datum['event'],
            'qrname': qr_datum['name'],
            'dbname': db_datum['name'],
            'qrlen': qr_datum['length'],
            'dblen': db_datum['length'],
            'score': qr_datum['mean'] @ db_datum['mean']
        }
    else:
        result = get_result(qr_datum, db_datum)
    return result


def get_result(qr_datum, db_datum):
    '''
    Get query result with a given query datum and a database datum
    '''
    qr_len = qr_datum['length']
    db_len = db_datum['length']
    # all possible offsets
    all_offsets = np.arange(-qr_len+1, db_len)
    # number of all possible offsets
    all_offset_num = qr_len+db_len-1
    # video similarities at all possible offsets
    all_video_sims = np.zeros(all_offset_num)

    # multiple periods strategy
    for period in periods:
        qr_desc = qr_datum[period]
        db_desc = db_datum[period]
        indices = (np.arange(period) - qr_len + 1) % period
        video_sims = compare_two_videos(qr_desc, db_desc, period)[indices]
        reps = int(all_offset_num / period) + 1
        all_video_sims += np.tile(video_sims, reps)[0:all_offset_num]

    index = np.argmax(all_video_sims)
    result = {
        'qrevent': qr_datum['event'],       # video's event number
        'dbevent': db_datum['event'],       # video's event number
        'qrname': qr_datum['name'],         # video's name
        'dbname': db_datum['name'],         # video's name
        'qrlen': qr_len,                    # video's length
        'dblen': db_len,                    # video's length
        'offset': all_offsets[index],
        'score': all_video_sims[index]
    }
    return result


def compare_two_videos(desc1, desc2, period):
    '''
    Compare two videos by given period and return the similarity scores
    '''
    freq_num = fourier_coefs[period]['freqnum']
    cos_mat = offset_mat_dicts[period]['cos']
    sin_mat = offset_mat_dicts[period]['sin']
    dc_comp = np.dot(desc1[0], desc2[0])        # DC component
    cos1 = desc1[1:freq_num+1]
    cos2 = desc2[1:freq_num+1]
    sin1 = desc1[freq_num+1:]
    sin2 = desc2[freq_num+1:]
    cos_comps = np.einsum('ij,ij->i', cos1, cos2) + \
        np.einsum('ij,ij->i', sin1, sin2)
    sin_comps = np.einsum('ij,ij->i', cos1, sin2) - \
        np.einsum('ij,ij->i', sin1, cos2)
    return dc_comp + cos_mat.dot(cos_comps) + sin_mat.dot(sin_comps)


class Retrieve(object):
    def __init__(self, embed_dir, results_dir):
        self.embed_dir = embed_dir
        self.results_dir = results_dir

    def __call__(self, qr_events=range(1, 14), db_events=range(1, 14)):
        for qr_event in qr_events:
            qr_data = utils.load('{0}/{1}_qr.jbl'.format(self.embed_dir, qr_event))
            self.retrieve_event(qr_event, db_events, qr_data)

    def retrieve_event(self, qr_event, db_events, qr_data, iteration=0):
        '''
        Conduct CBVR task with a given event and query data
        '''
        results = []
        search_times = []
        print('start searching, iteration = {0}'.format(iteration))
        # search
        for db_event in db_events:
            db_data = utils.load('{}/{}_db.jbl'.format(self.embed_dir, db_event))
            t0 = time.time()
            results.extend(get_event_results(qr_data, db_data))
            search_time = time.time() - t0
            search_times.append(search_time)
            print('searching for event {0} vs event {1} cost {2:.2f}s'.format(
                qr_event, db_event, search_time))
        print('searching for event {0} cost {1:.2f}s overall'.format(
            qr_event, sum(search_times)))
        # save results
        t0 = time.time()
        utils.save(results, '{}/{}_{}.jbl'.format(self.results_dir, qr_event, iteration))
        print('saving results for event {0} cost {1:.2f}s'.format(
            qr_event, time.time() - t0))
        print('--------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dir', type=str,
                        help="directory to embeddings")
    parser.add_argument('--results_dir', type=str,
                        help="directory to results")
    parser.add_argument('--periods', type=int, nargs='+',
                        help="list of periods")
    parser.add_argument('--mean', action='store_true',
                        help="use mean descriptor or not")
    parser.add_argument('--events', type=int, nargs='+', default=range(1, 14),
                        help="list of events")
    args = parser.parse_args()
    use_mean = args.mean
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if not use_mean:
        periods = args.periods
        fourier = Fourier(periods)
        fourier_coefs = fourier.get_fourier_coefs()
        offset_mat_dicts = fourier.get_offset_mats()
    retrieve = Retrieve(args.embed_dir, args.results_dir)
    retrieve()
