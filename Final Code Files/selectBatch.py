#Prelim
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

def get_batch(X_training, X_pool, X_uncertainty, metric):
    
    #Extract the number of labeled and unlabeled instances
    n_labeled, *rest = X_training.shape
    n_unlabeled, *rest = X_pool.shape

    #Determine alpha parameter. Note: 
    #because X_training and X_pool change alpha will change every iteration
    alpha = n_unlabeled / (n_unlabeled + n_labeled)

    #Compute pairwise distance and similarity scores from every unlabeled point
    #to every point in X_training. 
    _, distance_scores = pairwise_distances_argmin_min(X_pool.reshape(n_unlabeled, -1), X_training.reshape(n_labeled, -1), metric=metric)

    similarity_scores = 1 / (1 + distance_scores)

    #Compute our final scores, which are a balance between how dissimilar a 
    #given point is with the points in X_uncertainty and how uncertain we are 
    #about its value.
    scores = alpha * (1 - similarity_scores) + (1 - alpha) * X_uncertainty

    #Isolate and return our best instance for labeling as the one with the largest score.
    best_instance_index_in_unlabeled = np.argmax(scores)
    n_pool, *rest = X_pool.shape
    unlabeled_indices = [i for i in range(n_pool)]
    best_instance_index = unlabeled_indices[best_instance_index_in_unlabeled]

    return best_instance_index