import os
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


def load_data(dataset, view, method, k, show_details=True):
    folder = './input/' + dataset + '/'
    label = np.load('{}label.npy'.format(folder), allow_pickle=True)
    fea = np.load('{}{}_fea.npy'.format(folder, view), allow_pickle=True)

    graph_path = '{}{}_{}_{}.npz'.format(folder, view, method, k)
    if not os.path.exists(graph_path):
        _, adj = get_adj(count=fea, k=k)
        num = len(label)
        counter = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[0]):
                if adj[i][j] == 0 or i==j:
                    pass
                else:
                    if label[i] != label[j]:
                        counter += 1
        print('error rate: {}'.format(counter / (num * k)))
        sp.save_npz(graph_path, sp.csr_matrix(adj))

    adj = sp.load_npz(graph_path).toarray()

    if show_details:
        print("---details of graph dataset---")
        print("------------------------------")
        print("dataset name         :", dataset + '_' + view)
        print("feature shape        :", fea.shape)
        print("label shape          :", label.shape)
        print("adj shape            :", adj.shape)
        print("category num         :", max(label)-min(label)+1)
        print("category distribution:")
        for i in range(max(label)):
            print("label", i, end=":")
            print(len(label[np.where(label == i+1)]))
        print("------------------------------")

    return fea, label, adj


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def get_adj(count, k=15, mode="connectivity"):
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n
