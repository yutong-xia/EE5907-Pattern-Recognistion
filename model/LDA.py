import numpy as np
import os
import matplotlib.pyplot as plt

class LDA:
    def __init__(self, X, y, save_fig = True):
        self.X = X.reshape(X.shape[0], -1)
        self.y = y
        self.n_samples = X.shape[0]
        self.unique_classes = np.unique(y)
        self.n_classes = len(self.unique_classes)
        self.eigenvector = None
        self.save_fig = save_fig
        if save_fig:
            if not os.path.exists('./results'):
                os.mkdir('./results')
        
    def reduce_dim(self, n_component):

        # total covariance
        scatter_t = np.cov(self.X.T)/ self.n_samples
        
        # within-class scatter
        scatter_w = 0
        for i in range(self.n_classes):
            class_items_idx = np.flatnonzero(self.y == self.unique_classes[i])
            scatter_w += np.cov(self.X[class_items_idx].T) * (len(class_items_idx) / self.n_samples)
        
        # between-class scatter
        scatter_b = scatter_t - scatter_w
        eigen_values , eigen_vectors = np.linalg.eigh(np.linalg.pinv(scatter_w).dot(scatter_b))

        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        eigenvector_subset = sorted_eigenvectors[:,0: n_component]
        self.eigenvector = eigenvector_subset
        # Transform the data
        X_meaned = self.X - np.mean(self.X , axis = 0) 
        X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
        
        return X_reduced

        
    def reduce_test_dim(self, test_x):
        test_x = test_x.reshape(-1, 32*32)
        X_meaned = test_x - np.mean(test_x , axis = 0) 
        X_reduced = np.dot(self.eigenvector.transpose() , X_meaned.transpose() ).transpose()
        return X_reduced
        
    def plot_data(self, X_reduced, figsize = (5,5)):
        if X_reduced.shape[-1] == 2:
            plt.figure(figsize=figsize, dpi = 200)
            plt.scatter(X_reduced[:-7,0], X_reduced[:-7, 1], label= 'PIE photo')
            plt.scatter(X_reduced[-7:,0], X_reduced[-7:, 1], label= 'self photo')
        elif X_reduced.shape[-1] == 3:
            fig = plt.figure(figsize=figsize, dpi = 200)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X_reduced[:-7,0], X_reduced[:-7, 1],X_reduced[:-7, 2], label= 'PIE photo')
            ax.scatter(X_reduced[-7:,0], X_reduced[-7:, 1],X_reduced[-7:, 2], label= 'self photo')
        plt.legend()
        if self.save_fig:
            plt.tight_layout()
            plt.savefig('./results/lda_{}d.png'.format(X_reduced.shape[-1]))
                
