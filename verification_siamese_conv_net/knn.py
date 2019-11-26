import sys
import data_preproc as dp
import numpy as np
import pandas as pd

def nearest_neighbour(pairs, targets):  # recheck concept
    # image similarity ~ 1/vec proximity
    L2_distance = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distance[i] = np.sum(np.absolute(pairs[0][i] - pairs[1][i]))
    if np.argmin(L2_distance) == np.argmax(targets):
        return 1
    else:
        return 0


def make_one_shot_task(n_way=5):
    data_creator = dp.SiameseDatasetCreator(split_size=1)
    ds = data_creator.celeb_loader.dataset
    sample_classes = pd.Series(ds.targets).unique()
    results = []
    count = 0
    for class_identity in sample_classes:
        print(f'{count} iteration , true class: {class_identity}')
        pairs_class, target_class = data_creator.create_pair_siamese(class_identity, 'train')
        pair_true = pairs_class[((len(pairs_class) // 2)-1):][:n_way]
        pair_true_target = target_class[((len(pairs_class) // 2)-1):][:n_way]
        comparison_result = nearest_neighbour(pair_true, pair_true_target)
        results.append(comparison_result)
        count = count + 1
    return results, count


def nn_accuracy(n_ways, n_trials):
    """Returns accuracy of NN approach """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials, n_ways))
    sum_corrects = 0
    nr_pairs = 0

    for i in range(n_trials):
        results, nr = make_one_shot_task(n_ways)
        sum_corrects = sum_corrects + int(np.array(results).sum())
        nr_pairs = nr_pairs + nr

    return 100.0 * sum_corrects / nr_pairs


def main():
    evaluate_4_way = nn_accuracy(4, 5)
    print(f'4-way one shot accuracy{evaluate_4_way}')
    evaluate_6_way = nn_accuracy(6, 3)
    print(f'6-way one shot accuracy{evaluate_6_way}')
    return


if __name__ == '__main__':
    main()
