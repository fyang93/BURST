#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' TE Fourier series module '

__author__ = 'fyang'

import numpy as np
import os
import joblib
from settings import *


def get_fourier_coefs():
    '''
    Get Fourier series coefficients
    '''
    save_fourier_coefs()
    return load_fourier_coefs()


def load_fourier_coefs():
    '''
    Load Fourier series coefficients
    '''
    with open(fourier_coefs_path, 'rb') as fourier_coefs_file:
        return joblib.load(fourier_coefs_file)


def save_fourier_coefs():
    '''
    Save Fourier series coefficients into data directory
    '''
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    coefs = compute_fourier_series()
    with open(fourier_coefs_path, 'wb') as fourier_coefs_file:
        joblib.dump(coefs, fourier_coefs_file)


def compute_fourier_series():
    '''
    Get Fourier series coefficients
    '''
    coefs = {}
    for period in periods:
        an = _fourier_series(period)
        coefs[period] = {'coef' : an, 'freqnum' : len(an) - 1}
    return coefs


def get_timestamps():
    '''
    Make timestamps for every period
    '''
    fourier_coefs = get_fourier_coefs()
    timestamps_dict = {}
    for period in periods:
        sqrt_an = np.sqrt(fourier_coefs[period]['coef']).astype(np.float32)
        freq_num = fourier_coefs[period]['freqnum']
        theta = np.arange(period, dtype = np.float32) / period * 2 * np.pi
        timestamps = np.zeros((2*freq_num+1, period))
        timestamps[0] = sqrt_an[0]
        for i in range(1, freq_num + 1):
            timestamps[i] = sqrt_an[i] * np.cos(i*theta)
            timestamps[freq_num + i] = sqrt_an[i] * np.sin(i*theta)
        # timestamps shape: period * (2 * frequency number + 1)
        timestamps_dict[period] = timestamps.T
    return timestamps_dict


def get_offset_mats():
    '''
    Make offset matrix for every period
    '''
    fourier_coefs = load_fourier_coefs()
    offset_mat_dicts = {}
    for period in periods:
        freq_num = fourier_coefs[period]['freqnum']
        shifts = np.arange(period) / period * 2 * np.pi
        shift_mat = np.outer(shifts, np.arange(1, freq_num + 1))
        cos_mat = np.cos(shift_mat)
        sin_mat = np.sin(shift_mat)
        offset_mat_dicts[period] = {'cos': cos_mat, 'sin': sin_mat}
    return offset_mat_dicts


def _fourier_series(period):
    '''
    Compute Fourier series with given period
    '''
    dx = 0.001              # dx for integral
    x = np.arange(-np.pi + dx, np.pi, dx)
    y = __delta_like(x, period)
    an = np.zeros(max_freq_num + 1)
    an[0] = np.mean(y)      # dc component
    for i in range(1, max_freq_num + 1):
        an[i] = np.dot(np.cos(i * x), y) * dx / np.pi
    threshold = 0.9
    y0 = 0                  # approximated y at x=0
    for i in range(0, max_freq_num + 1):
        y0 += an[i]
        if y0 > threshold: break
    freq_num = i
    print('need {0} frequencies when period is {1}'.format(freq_num, period))
    return an[0:freq_num+1]


def __delta_like(x, period):
    '''
    Function used to approximate Dirac-delta
    '''
    # the higher the param (0,1] is, the function is more like
    # a Dirac-delta function
    param = 0.8
    coef = (param * period / np.pi)**2 / 8
    return np.exp(-coef * (x ** 2))


if __name__ == '__main__':
    save_fourier_series()
