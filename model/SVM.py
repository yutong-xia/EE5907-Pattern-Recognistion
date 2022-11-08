from libsvm.svmutil import *


class SVM:
    def __init__(self, c):
        self.c = c
        self.m = None
        
    def trans_svm_data(self, X):
        '''
        X (n_sample, dim)
        '''
        X_svms = []
        for i in range(X.shape[0]):
            sample = X[i]
            X_svm = {}
            for dim in range(X.shape[1]):
                X_svm[dim+1] = sample[dim]
            X_svms.append(X_svm)
        return X_svms
    
    def fit(self, X, y):
        X = self.trans_svm_data(X)
        self.m = svm_train(y, X, '-c {}'.format(self.c))
    
    def predict(self, test_x, test_y):
        test_x = self.trans_svm_data(test_x)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, self.m)   

        return  p_label, p_acc, p_val
