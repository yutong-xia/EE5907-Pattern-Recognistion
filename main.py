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
from model.classifier import kNN, get_metrics
from model.PCA import PCA
from model.LDA import LDA
from model.GMM import GMM

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
# parser.add_argument('--use_cache', type=str_to_bool, default=True, help='if use cache dataset')
parser.add_argument('--save_fig', type=str_to_bool, default=True)
# kNN
parser.add_argument('--k', type=int, default=10)

args = parser.parse_args()


def gen_dataset():
    '''
    choose the first 25 subjects in the PIE dataset and 10 self photo
    for each subject, use 70% for training set and use the remaining 30% for testing set
    save them in ./data/
    '''
    if not os.path.exists('./data/'): os.mkdir('./data')
    
    train_x, train_y = [], []
    test_x, test_y = [], []
    
    for sub_idx in range(1,26):
        path_samples = glob.glob('PIE/{}/*.jpg'.format(sub_idx))
        n_train_samples = round(len(path_samples) * 0.7)
        for p in path_samples[:n_train_samples]:
            sample = mpimg.imread(p)
            train_x.append(sample)
            train_y.append(sub_idx)
        for p in path_samples[n_train_samples:]:
            sample = mpimg.imread(p)
            test_x.append(sample)
            test_y.append(sub_idx)
    
    ######### self data #########
    for idx in range(1,8):
        sample = mpimg.imread('PIE/self/{}.jpg'.format(idx))
        if len(sample.shape) == 3:
            sample = sample[...,0]
        train_x.append(sample)
        train_y.append(26)
    for idx in range(8,11):
        sample = mpimg.imread('PIE/self/{}.jpg'.format(idx))
        if len(sample.shape) == 3:
            sample = sample[...,0]
        test_x.append(sample)
        test_y.append(26)
    
    train_x = np.stack(train_x, 0)
    test_x = np.stack(test_x, 0)
    np.save('./data/train_x.npy', train_x)
    np.save('./data/train_y.npy', train_y)
    np.save('./data/test_x.npy', test_x)
    np.save('./data/test_y.npy', test_y)
    
    print('Train: {}, Test: {}'.format(train_x.shape, test_x.shape))
    
# gen_dataset()
train_x = np.load('./data/train_x.npy')
train_y = np.load('./data/train_y.npy')
test_x = np.load('./data/test_x.npy')
test_y = np.load('./data/test_y.npy')

if args.model == 'PCA':
    # randomly choose 493 samples from PIE and 7 self samples
    idx = random.sample([i for i in range(len(train_x)-7)], 493)
    data_pca = np.concatenate([train_x[idx],train_x[-7:] ], 0)
    label_pca = np.concatenate([train_y[idx], [26 for i in range(7)]], 0)
    # apply PCA
    x_pca = PCA(data_pca, save_fig = args.save_fig)
    # x_pca.plot_raw()
    # reduce the dim to 2d and 3d
    x_reduced_2d = x_pca.reduce_dim(n_component = 2)
    x_reduced_3d = x_pca.reduce_dim(n_component = 3)
    x_pca.plot_data(x_reduced_2d)
    x_pca.plot_data(x_reduced_3d)
    # plot 3 eigenfaces
    x_pca.plot_pc()
    
    # x_pca.recon()
    # x_pca.plot_recon()

    # reduce the dim to 40 80 and 200
    results = pd.DataFrame(columns=['Dimension','PIE', 'self'])
    for i, dim in enumerate([40, 80, 200]):
        x_reduced = x_pca.reduce_dim(n_component = dim)
        x_reduced_test = x_pca.reduce_test_dim(test_x)
        preds = kNN(x_reduced, label_pca , x_reduced_test, args.k)
        acc1, acc2 = get_metrics(preds, test_y)
        print('Dimension:{}, ACC1:{}, ACC2:{}'.format(dim, acc1, acc2))
        results.loc[i,'Dimension'] = dim
        results.loc[i,'PIE'] = acc1
        results.loc[i,'self'] = acc2
    results.to_csv('./results/pca_classification.csv', index = False)

if args.model == 'LDA':
    # randomly choose 493 samples from PIE and 7 self samples
    idx = random.sample([i for i in range(len(train_x)-7)], 493)
    data_lda = np.concatenate([train_x[idx],train_x[-7:] ], 0)
    label_lda = np.concatenate([train_y[idx], [26 for i in range(7)]], 0)
    # apply lda
    x_lda = LDA(data_lda,label_lda, save_fig = args.save_fig)
    results = pd.DataFrame(columns=['Dimension','PIE', 'self'])
    for i, dim in enumerate([2, 3, 9]):
        x_reduced = x_lda.reduce_dim(n_component = dim)
        if dim != 9:
            x_lda.plot_data(x_reduced)
        x_reduced_test = x_lda.reduce_test_dim(test_x)
        preds = kNN(x_reduced, label_lda , x_reduced_test, args.k)
        acc1, acc2 = get_metrics(preds, test_y)
        print('Dimension:{}, ACC1:{}, ACC2:{}'.format(dim, acc1, acc2))
        results.loc[i,'Dimension'] = dim
        results.loc[i,'PIE'] = acc1
        results.loc[i,'self'] = acc2
    results.to_csv('./results/lda_classification.csv', index = False)
        
if args.model=='GMM':
    # use the raw face images
    x_gmm = train_x.reshape(-1, 32*32)
    
    # use the face vectors after PCA pre-processing
    x_pca = PCA(train_x, save_fig = args.save_fig)
    for i, dim in enumerate([200, 80]):
        x_reduced = x_pca.reduce_dim(n_component = dim)
        model = GMM(n_components = 3)
        

if args.model=='SVM':
    pass 
    
print('The results of method {} are saved in "./results"'.format(args.__module__))