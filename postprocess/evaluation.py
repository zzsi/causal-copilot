def evaluate_ground_truth(graph, truth):
    '''
    :param graph: the result of causal discovery.
    :param truth: ground truth, represented by adjacent matrix - Matrix[i,j] indicates j->i
    :return: Structural Hamming Distance, precision, recall, F1 score.
    '''

    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Structural Hamming Distance (SHD)
    diff = np.abs(graph - truth)
    # Count how many edges are different
    shd = np.sum(diff)

    # Precision, Recall and F1-score
    precision = precision_score(truth, graph, average=None)
    recall = recall_score(truth, graph, average=None)
    f1 = f1_score(truth, graph, average=None)

    return shd, precision, recall, f1


