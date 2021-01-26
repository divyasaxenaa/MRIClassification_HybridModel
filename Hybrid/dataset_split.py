import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
import load_data_sets as load_data_sets

IMG_SIZE = 112


def load_data(k=8, noise_level=0.0):
    A = _img_grid_graph(k)
    A = _flip_random_edges(A, noise_level).astype(np.float32)
    (X_train, y_train), (X_test, y_test),  (X_val, y_val) = load_data_sets.load_data()
    X_train, X_test, X_val = X_train / 255.0, X_test / 255.0, X_val / 255.0
    X_train = X_train.reshape(-1, IMG_SIZE ** 2)
    X_test = X_test.reshape(-1, IMG_SIZE ** 2)
    X_val = X_val.reshape(-1, IMG_SIZE ** 2)
    return X_train, y_train, X_val, y_val, X_test, y_test, A


def _grid_coordinates(side):
    M = side ** 2
    x = np.linspace(0, 1, side, dtype=np.float32)
    y = np.linspace(0, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), np.float32)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def _get_adj_from_data(X, k, **kwargs):
    A = kneighbors_graph(X, k, **kwargs).toarray()
    A = sp.csr_matrix(np.maximum(A, A.T))

    return A


def _img_grid_graph(k):
    X = _grid_coordinates(IMG_SIZE)
    A = _get_adj_from_data(
        X, k, mode='connectivity', metric='euclidean', include_self=False
    )
    return A


def _flip_random_edges(A, percent):
    if not A.shape[0] == A.shape[1]:
        raise ValueError('A must be a square matrix.')
    dtype = A.dtype
    A = sp.lil_matrix(A).astype(np.bool)
    n_elem = A.shape[0] ** 2
    n_elem_to_flip = round(percent * n_elem)
    unique_idx = np.random.choice(n_elem, replace=False, size=n_elem_to_flip)
    row_idx = unique_idx // A.shape[0]
    col_idx = unique_idx % A.shape[0]
    idxs = np.stack((row_idx, col_idx)).T
    for i in idxs:
        i = tuple(i)
        A[i] = np.logical_not(A[i])
    A = A.tocsr().astype(dtype)
    A.eliminate_zeros()
    return A
