import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import glob
import os

class PCA:
    def __init__(self, X, num_components = 2, save_fig = True):
        self.X = X.reshape(X.shape[0], -1)
        self.num_components = num_components
        self.save_fig = save_fig
        if save_fig:
            if not os.path.exists('./results'):
                os.mkdir('./results')
        
    def plot_raw(self, num_samples = 25):
        fig = plt.figure(figsize=(10,10), dpi = 200)
        for i in range(num_samples):
            ax = fig.add_subplot(5,5,i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(self.X[i].reshape(32,32),cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=0)
#         fig.suptitle('Original face image', fontsize=16)
        if self.save_fig:
            plt.savefig('./results/pca_origin.png')
            
        
    def reduce_dim(self):
        #Step-1 get the mean centering the data
        X_meaned = self.X - np.mean(self.X , axis = 0) 
        self.X_meaned = X_meaned
        #Step-2 Calculate the Covariance Matrix
        cov_mat = np.cov(X_meaned , rowvar = False)
        #Step-3 Compute the Eigenvalues and Eigenvectors
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
        #Step-4 Sort Eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        self.sorted_eigenvectors = sorted_eigenvectors
        #Step-5 Select a subset from the rearranged Eigenvalue matrix
        eigenvector_subset = sorted_eigenvectors[:,0:self.num_components] # [1024, num_components]
        self.eigenvector = eigenvector_subset
        #Step-6 Transform the data
        X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
        self.X_reduced = X_reduced
    
    def recon(self):
        tmp1 = np.dot(self.eigenvector, self.eigenvector.transpose())
        tmp2 = self.X - self.X_meaned
        x_recon = self.X_meaned + np.dot(tmp1, tmp2.transpose()).transpose()
        self.x_recon = x_recon.reshape(-1, 32, 32)

    def plot_pc(self, num_samples = 25):
        fig = plt.figure(figsize=(10,10), dpi = 200)
        for i in range(num_samples):
            ax = fig.add_subplot(5,5,i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(self.sorted_eigenvectors[:,i].reshape(32,32),cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=0)
#         fig.suptitle('Eigenfaces', fontsize=16)
        if self.save_fig:
            plt.savefig('./results/pca_eigenfaces.png')
            
    def plot_recon(self, num_samples = 25):
        fig = plt.figure(figsize=(10,10), dpi = 200)
        for i in range(num_samples):
            ax = fig.add_subplot(5,5,i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(self.x_recon[i],cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=0)
#         fig.suptitle('Reconstruction with PCs', fontsize=16)
        if self.save_fig:
            plt.savefig('./results/pca_recon.png')