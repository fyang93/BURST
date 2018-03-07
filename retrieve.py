#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' TE retrieve module '

__author__ = 'fyang'

import numpy as np
import sys, os, time, fourier, joblib
from joblib import Parallel, delayed
from settings import *


if not os.path.exists(result_dir):
    os.makedirs(result_dir)

fourier_coefs = fourier.load_fourier_coefs()
offset_mat_dicts = fourier.get_offset_mats()

# global data kept for parallel processes
global_qr_data = ''
global_db_data = ''


def retrieve_evve(qr_events = range(1,14), db_events = range(1,14)):
    '''
    Conduct CBVR task on EVVE dataset
    '''
    for qr_event in qr_events:
        # load query data
        path = '{0}/{1}_qr.pkl'.format(data_dir, qr_event)
        with open(path, 'rb') as qr_file:
            qr_data = joblib.load(qr_file)
        retrieve_event(qr_event, db_events, qr_data)


def retrieve_event(qr_event, db_events, qr_data, iteration = 0):
    '''
    Conduct CBVR task with a given event and query data
    '''
    results = []
    search_times = []
    print('start searching, iteration = {0}'.format(iteration))
    for db_event in db_events:
        # load database data
        path = '{0}/{1}_db.pkl'.format(data_dir, db_event)
        with open(path, 'rb') as db_file:
            db_data = joblib.load(db_file)
        # search
        start = time.time()
        results.extend(get_event_results(qr_data, db_data))
        search_time = time.time() - start
        search_times.append(search_time)
        print('searching for event {0} vs event {1} cost {2:.2f}s'.format(qr_event, db_event, search_time))
    print('searching for event {0} cost {1:.2f}s overall'.format(qr_event, sum(search_times)))
    # save results
    start = time.time()
    path = '{0}/{1}_{2}.pkl'.format(result_dir, qr_event, iteration)
    with open(path, 'wb') as result_file:
        joblib.dump(results, result_file)
    print('saving results for event {0} cost {1:.2f}s'.format(qr_event, time.time() - start))
    print('--------------------------------------')


def get_event_results(qr_data, db_data):
    '''
    Conduct retrieve task with given query data and database data
    '''
    global global_qr_data, global_db_data
    global_qr_data = qr_data
    global_db_data = db_data
    qr_data_num = len(qr_data)
    db_data_num = len(db_data)
    results = Parallel(n_jobs=-1) ([delayed(get_query_result)(i,j) for i in range(qr_data_num) for j in range(db_data_num)])
    return results


def get_query_result(qr_datum_index, db_datum_index):
    '''
    Get query result with a given query datum index and a database datum index
    '''
    qr_datum = global_qr_data[qr_datum_index]
    db_datum = global_db_data[db_datum_index]
    return get_result(qr_datum, db_datum)


def get_result(qr_datum, db_datum):
    '''
    Get query result with a given query datum and a database datum
    '''
    qr_len = qr_datum['length']
    db_len = db_datum['length']
    all_offsets = np.arange(-qr_len + 1, db_len)    # all possible offsets
    all_offset_num = qr_len + db_len - 1            # number of all possible offsets
    all_video_sims = np.zeros(all_offset_num)       # video similarities at all possible offsets

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
        'qrevent': qr_datum['event'],               # query video event number
        'dbevent': db_datum['event'],               # database video event number
        'qrname': qr_datum['name'],                 # query video name
        'dbname': db_datum['name'],                 # database video name
        'qrlen': qr_len,                            # query video length
        'dblen': db_len,                            # database video length
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
    cos_comps = np.einsum('ij,ij->i', cos1, cos2) + np.einsum('ij,ij->i', sin1, sin2)
    sin_comps = np.einsum('ij,ij->i', cos1, sin2) - np.einsum('ij,ij->i', sin1, cos2)
    return dc_comp + cos_mat.dot(cos_comps) + sin_mat.dot(sin_comps)


if __name__ == '__main__':
    retrieve_evve()
