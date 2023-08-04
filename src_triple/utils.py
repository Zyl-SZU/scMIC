import torch
import random
import numpy as np
from sklearn import metrics
from munkres import Munkres
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_mutual_info_score
import opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def numpy_to_torch(a, sparse=False):
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


# the reconstruction function
def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
    loss_ae = F.mse_loss(X_hat, X)
    loss_w = F.mse_loss(Z_hat, torch.spmm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm.to_dense())
    loss_igae = loss_w + opt.args.alpha_value * loss_a
    loss_rec = loss_ae + loss_igae
    return loss_rec


def target_distribution(Q):
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


# clustering guidance
def distribution_loss(Q, P):
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')

    return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m

    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cross_correlation(Z_v1, Z_v2):

    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def correlation_reduction_loss(S):

    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def drr_loss(cons):

    S_N1 = cross_correlation(cons[0], cons[1])
    S_N2 = cross_correlation(cons[0], cons[2])
    S_N3 = cross_correlation(cons[1], cons[2])
    L_N = correlation_reduction_loss(S_N1)+correlation_reduction_loss(S_N2)+correlation_reduction_loss(S_N3)

    S_F1 = cross_correlation(cons[3], cons[4])
    S_F2 = cross_correlation(cons[3], cons[5])
    S_F3 = cross_correlation(cons[4], cons[5])

    L_F = correlation_reduction_loss(S_F1)+correlation_reduction_loss(S_F2)+correlation_reduction_loss(S_F3)

    loss_drr = opt.args.lambda1 * L_N + opt.args.lambda2 * L_F

    return loss_drr


def clustering(Z, y):
    model = KMeans(n_clusters=opt.args.n_clusters, n_init=10)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())

    ari, nmi, ami, acc = eva(y, cluster_id, show_details=True)

    return ari, nmi, ami, acc, model.cluster_centers_


def assignment(Q, y):
    y_pred = torch.argmax(Q, dim=1).data.cpu().numpy()
    ari, nmi, ami, acc = eva(y, y_pred, show_details=False)
    return ari, nmi, ami, acc, y_pred


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0

    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)

    return acc


def eva(y_true, y_pred, show_details=True):

    acc = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    
    if show_details:
        print("\n","ARI: {:.4f},".format(ari), "NMI: {:.4f},".format(nmi), "AMI: {:.4f}".format(ami), "ACC: {:.4f},".format(acc))
        
    return ari, nmi, ami, acc
