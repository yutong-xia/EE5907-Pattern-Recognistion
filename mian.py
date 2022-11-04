import argparse
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import glob
import os

from model.PCA import PCA

parser = argparse.ArgumentParser()

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser.add_argument('--model', type=str, default='PCA',help='which model/method you want to run [PCA, LDA, GMM, SVM]')
parser.add_argument('--use_cache', type=str_to_bool, default=True, help='if use cache dataset')
parser.add_argument('--save_fig', type=str_to_bool, default=True)
# PCA
parser.add_argument('--n_components', type=int, default=2)

args = parser.parse_args()

def gen_dataset(path = './PIE', n_objects = 28, use_cache = True):
    '''
    Randomly sample 28 objects from the CMU PIE set
    '''
    if use_cache and os.path.exists('{}/cache/samples.npy'.format(path)):
        samples = np.load('{}/cache/samples.npy'.format(path))
        labels = np.load('{}/cache/labels.npy'.format(path))
    else:
        if not os.path.exists('{}/cache/'.format(path)):
            os.mkdir('{}/cache/'.format(path))
        idx = random.sample([i for i in range(0, 69)],n_objects)
        samples = []
        labels = []
        for i in idx:
            path_tmp = glob.glob('{}/{}/*.jpg'.format(path,i))
            for p in path_tmp:
                sample = mpimg.imread(p)
                samples.append(sample)
                labels.append(i)
        samples = np.stack(samples, 0)
        np.save('{}/cache/samples.npy'.format(path), samples)
        np.save('{}/cache/labels.npy'.format(path), labels)
    return samples,labels

samples, labels = gen_dataset(n_objects = 28, use_cache = args.use_cache)

if args.model == 'PCA':
    x_pca = PCA(samples, num_components = args.n_components, save_fig = args.save_fig)
    x_pca.plot_raw()
    x_pca.reduce_dim()
    x_pca.recon()
    x_pca.plot_pc()
    x_pca.plot_recon()

print('Finished. The results are saved in "./results"')