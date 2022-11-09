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
            # X_svm = {}
            # for dim in range(X.shape[1]):
            #     X_svm[dim+1] = sample[dim]
            # X_svms.append(X_svm)
            X_svms.append(sample)
        return X_svms
    
    def fit(self, X, y):
        X = self.trans_svm_data(X)
        # self.m = svm_train(y, X, '-c {}'.format(self.c))
        
        prob  = svm_problem(y, X)
        param = svm_parameter('-t 0 -c {} -b 1'.format(self.c))
        # param = svm_parameter('-c {}'.format(self.c))
        self.m = svm_train(prob, param)
    
    def predict(self, test_x, test_y):
        test_x = self.trans_svm_data(test_x)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, self.m)   

        return  p_label, p_acc, p_val
