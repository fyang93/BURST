#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' EVVE combine module '

__author__ = 'fyang'

import numpy as np
import os, sys, time, glob, joblib
from settings import *


if os.path.exists(combine_result_path):
    os.remove(combine_result_path)

annots = sorted(glob.glob('{0}/*.dat'.format(annot_dir)))


def combine_evve(events = range(1,14), iteration = 0):
    '''
    Combine results for given events into a .dat file for evaluation
    '''
    for qr_event in events:
        results = load_event_results(qr_event, iteration)
        annot_path = annots[qr_event - 1]
        with open(annot_path, 'r') as file:
            content = file.readlines()
        lines = [line.strip() for line in content]
        for line in lines:
            qr_name, gt, tag = line.split()
            if (tag != 'query'):
                continue
            qr_results = (_ for _ in results if _['qrname'] == qr_name)
            qr_results = sorted(qr_results,
                                key=lambda _: _['score'], reverse=True)
            db_names = (_['dbname'] for _ in qr_results)
            with open(combine_result_path, 'a') as file:
                file.write('{0} {1}\n'.format(qr_name, ' '.join(db_names)))


def load_event_results(qr_event, iteration):
    '''
    Load all retrieval results for given event
    '''
    path = '{0}/{1}_{2}.pkl'.format(result_dir, qr_event, iteration)
    with open(path, 'rb') as result_file:
        return joblib.load(result_file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        combine_evve()
    else:
        combine_evve(iteration=sys.argv[1])
    os.system('python eval_evve.py {0} -annotdir {1}'.format(combine_result_path, annot_dir))
