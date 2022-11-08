import numpy as np
from scipy.stats import mode
import sklearn.metrics as metrics


def get_distance(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

# def get_distance(instance1,instance2):
#     distance = 0
#     for i in range(instance1.shape[0]):
#         distance += (instance1[i] - instance2[i])**2

#     return math.sqrt(distance)

 

def kNN(train_x, train_y , test_x, k):
    preds = []
     
    #Loop through the Datapoints to be classified
    for test_idx in range(test_x.shape[0]): 
        
        item = test_x[test_idx]

        point_dist = []
         
        #Loop through each training Data
        for j in range(train_x.shape[0]): 
            distances = get_distance(train_x[j], item) 
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
        dist = np.argsort(point_dist)[:k] # idx
        #Labels of the K datapoints from above
        labels = train_y[dist]
        #Majority voting
        lab = mode(labels) 
        lab = lab.mode[0]
        preds.append(lab)
 
    return preds

def get_metrics(preds, labels):
    acc1 = metrics.accuracy_score(preds[:-3], labels[:-3]) ## test samples from PIE
    acc2 = metrics.accuracy_score(preds[-3:], labels[-3:]) ## test samples from self photo
    return acc1 * 100, acc2 * 100
    
