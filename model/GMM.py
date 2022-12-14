
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components = 3, max_iter=10):
        self.n_components = n_components
        self.max_iter = max_iter

    def initialize(self, X):
        self.n, self.m = X.shape[0], X.shape[1]

        self.phi = np.full(shape=self.n_components, fill_value=1/self.n_components)
        self.weights = np.full(shape=X.shape, fill_value=1/self.n_components)
        
        # self.phi = None
        # self.weights = None
        
        random_row = np.random.randint(low=0, high=self.n, size=self.n_components)
        self.mu = [X[row_index] for row_index in random_row ]
        
        # self.mu = []
        # offset = 1 // (self.n_components-1)
        # for i in range(self.n_components):
        #     if i == 0:
        #         self.mu.append(np.mean(X, 0) - offset)
        #     elif i == 1:
        #         self.mu.append(np.mean(X, 0))
        #     elif i == 2:
        #         self.mu.append(np.mean(X, 0) + offset)
                
        # offset = np.max(X, 0) - np.min(X, 0) // (self.n_components-1)
        # for i in range(self.n_components):
        #     self.mu.append(np.min(X, 0) + offset * i)
            
        # self.sigma = [ np.cov(X.T) for _ in range(self.n_components) ]
        self.sigma = [ np.cov(X.T)for _ in range(self.n_components) ]
        
    def fit(self, X):
        self.initialize(X)
        
        for iter in range(self.max_iter):
            print('iter: {}'.format(iter))
            self.e_step(X)
            self.m_step(X)
    
    def e_step(self, X):
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)
    
    def m_step(self, X):
        for i in range(self.n_components):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)
            
    def predict_proba(self, X):
        likelihood = np.zeros( (self.n, self.n_components) )
        for i in range(self.n_components):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i], allow_singular=True)
            likelihood[:,i] = distribution.pdf(X)  # pdf : probability denisty function
            # print(distribution.pdf(X))
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
    
    def visual(self, X, preds, figsize = (10, 10)):
        unique_class = np.unique(preds)
        for i, c in enumerate(unique_class):
            class_items_idx = np.flatnonzero(preds == c)
            class_items = X[class_items_idx]
            n_cols = 10
            n_rows = class_items.shape[0] // 10
            fig = plt.figure(figsize=figsize, dpi = 200)
            for j in range(class_items.shape[0]):
                ax = fig.add_subplot(n_rows, n_cols ,j+1)
                ax.set_xticks([])
                ax.set_yticks([])
                img = class_items[j]
                plt.imshow(img, cmap='gray')
            plt.tight_layout()
            plt.savefig('./results/gmm_{}.png'.format(i))
        
        