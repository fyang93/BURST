#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' EVVE combine module '

__author__ = 'fyang'

import os
import argparse
import glob
import utils


def combine_evve(results_dir, output_path, iteration=0, events=range(1, 14)):
    '''
    Combine results for given events into a .dat file for evaluation
    '''
    for qr_event in events:
        results = utils.load('{}/{}_{}.jbl'.format(results_dir,
                                                   qr_event, iteration))
        annot_path = annots[qr_event-1]
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
            with open(output_path, 'a') as f:
                f.write('{0} {1}\n'.format(qr_name, ' '.join(db_names)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_dir', type=str,
                        help="directory to annotation file")
    parser.add_argument('--results_dir', type=str,
                        help="directory to retrieval results")
    parser.add_argument('--output_path', type=str,
                        help="path to output file")
    parser.add_argument('--iter', type=int, default=0,
                        help="order of query expansion iteration")
    args = parser.parse_args()
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
    annots = sorted(glob.glob('{0}/*.dat'.format(args.annot_dir)))
    combine_evve(args.results_dir, args.output_path, iteration=args.iter)
    os.system('python eval_evve.py ' +
              '{} -annotdir {}'.format(args.output_path, args.annot_dir))
