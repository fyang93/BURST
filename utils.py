#!/usr/bin/env python3
# -*- coding: utf-8 -*-

" utils module "

import os
import time
import numpy as np
import joblib
import _pickle as pickle


def cache(path):
    """Decorator to cache results
    """
    def decorator(func):
        def wrapper(*args, **kw):
            time0 = time.time()
            if os.path.exists(path):
                result = load(path)
                cost = time.time() - time0
                print('[cache] loading {} costs {:.2f}s'.format(path, cost))
            result = func(*args, **kw)
            cost = time.time() - time0
            print('[cache] obtaining {} costs {:.2f}s'.format(path, cost))
            save(result, path)
            return result
        return wrapper
    return decorator


def load(path):
    if not os.path.exists(path):
        raise Exception("{} does not exist".format(path))
    ext = os.path.splitext(path)[-1]
    if ext == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    return {'.npy': np, '.jbl': joblib}[ext].load(path)


def save(tosave, path):
    ext = os.path.splitext(path)[-1]
    if ext == '.npy':
        np.save(path, tosave)
    elif ext == '.jbl':
        joblib.dump(tosave, path)


def _normalize(vectors):
    """Conduct L2 normalization on vectors
    Args:
      vectors: an array of vectors
    Returns:
      vectors: normalized vectors
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        norms = 1 / (np.linalg.norm(vectors, 2, 1))
    norms[~np.isfinite(norms)] = 0
    vectors = np.multiply(norms[:, np.newaxis], vectors)
    return vectors


def normalize(x):
    """Conduct L2 normalization
    Args:
       x: a single vector or an array of vectors
    Returns: normalized vector or vectors
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(_normalize(x.reshape(1, -1)))
    else:
        return _normalize(x)

