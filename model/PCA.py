import numpy as np
import matplotlib.pyplot as plt
import os

class PCA:
    def __init__(self, X, save_fig = True):
        self.X = X.reshape(X.shape[0], -1)
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
        if self.save_fig:
            plt.savefig('./results/pca_origin.png')
            
    
    def reduce_dim(self, n_component=None):
        # get the mean and center the data
        X_meaned = self.X - np.mean(self.X , axis = 0) 
        self.X_meaned = X_meaned
        # Calculate the Covariance Matrix
        cov_mat = np.cov(X_meaned , rowvar = False)
        # Compute the Eigenvalues and Eigenvectors
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        self.sorted_eigenvectors = sorted_eigenvectors
        
        if n_component == None:
            for n_component in range(len(sorted_eigenvalue)):
                if sum(sorted_eigenvalue[:n_component])/sum(sorted_eigenvalue) > 0.95:
                    break
            print('Choose p={} based on 0.95 of variation to retain.'.format(n_component))
         
        eigenvector_subset = sorted_eigenvectors[:,0: n_component] # [1024, n_component]
        self.eigenvector = eigenvector_subset
        # Transform the data
        X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
        self.X_reduced = X_reduced
        return X_reduced
    
    def plot_data(self, X_reduced):
        if X_reduced.shape[-1] == 2:
            plt.figure(figsize=(10,10), dpi = 200)
            plt.scatter(X_reduced[:-7,0], X_reduced[:-7, 1], label= 'PIE photo')
            plt.scatter(X_reduced[-7:,0], X_reduced[-7:, 1], label= 'self photo')
            plt.legend()
            if self.save_fig:
                plt.savefig('./results/pca_2d.png')
        elif X_reduced.shape[-1] == 3:
            fig = plt.figure(figsize=(10,10), dpi = 200)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X_reduced[:-7,0], X_reduced[:-7, 1],X_reduced[:-7, 2], label= 'PIE photo')
            ax.scatter(X_reduced[-7:,0], X_reduced[-7:, 1],X_reduced[-7:, 2], label= 'self photo')
            plt.legend()
            if self.save_fig:
                plt.savefig('./results/pca_3d.png')
        
    
    
    def recon(self):
        '''
        reconstruct the image with PCs
        '''
        tmp1 = np.dot(self.eigenvector, self.eigenvector.transpose())
        tmp2 = self.X - self.X_meaned
        x_recon = self.X_meaned + np.dot(tmp1, tmp2.transpose()).transpose()
        self.x_recon = x_recon.reshape(-1, 32, 32)

    def plot_pc(self, num_samples = 3):
        fig = plt.figure(figsize=(10,10), dpi = 200)
        for i in range(num_samples):
            ax = fig.add_subplot(1,3,i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(self.sorted_eigenvectors[:,i].reshape(32,32),cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=0)
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
        if self.save_fig:
            plt.savefig('./results/pca_recon.png')