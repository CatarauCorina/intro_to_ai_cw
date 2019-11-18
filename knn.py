import sys
sys.path.append('/Users/ckxz/Documents/GitHub/intro_to_ai_cw/verification_siamese_conv_net/config')
import data_preproc as dp
import numpy as np


def nearest_neighbour(pairs, targets): #recheck concept 
#image similarity ~ 1/vec proximity
    L2_distance = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distance[i] = np.sum(np.absolute(pairs[0][i] - pairs[1][i]))
    if np.argmin(L2_distance) == np.argmax(targets):
        return 1
    else:
        return 0

 def nn_accuracy(N_ways,n_trials):
        """Returns accuracy of NN approach """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))
    n_right = 0
    
    for i in range(n_trials):
        pairs,targets = make_oneshot_task(N_ways,"val")
        correct = nearest_neighbour_correct(pairs,targets)
        n_right += correct
    return 100.0 * n_right / n_trials

def main():
    pairs, targets = dp.SiameseDatasetCreator.create_pair_siamese() 
    neighbours = nearest_neighbours(pairs, targets)
    evaluate = nn_accuracy(N_ways, n_trials)
