
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components = 3, max_iter=10):
        self.n_components = n_components
        self.max_iter = max_iter

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.n_components, fill_value=1/self.n_components)
        self.weights = np.full( shape=self.shape, fill_value=1/self.n_components)
        
        random_row = np.random.randint(low=0, high=self.n, size=self.n_components)
        self.mu = [X[row_index,:] for row_index in random_row ]
        self.sigma = [ np.cov(X.T) for _ in range(self.n_components) ]
        
    def fit(self, X):
        self.initialize(X)
        
        for _ in range(self.max_iter):
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
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
    
    def visual(self, figsize = (10, 10)):
        plt.figure(figsize=figsize, dpi = 200)
        plt.tight_layout()
        plt.savefig('./results/gmm.png')
        
        