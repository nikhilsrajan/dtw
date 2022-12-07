import numpy as np
import numba


@numba.njit()
def _njit_compute_accumulated_cost_matrix(a:np.ndarray, b:np.ndarray, ord:float=2)->np.ndarray:
    len_a = a.shape[0]
    len_b = b.shape[0]
    cost_matrix = np.full((len_a+1, len_b+1), np.inf)
    cost_matrix[0][0] = 0
    for r in range(1,len_a+1):
        for c in range(1,len_b+1):
            cost = np.power(np.sum(np.power(np.abs(a[r-1,:] - b[c-1,:]), ord)), 1/ord)
            cost_matrix[r,c] = cost + min([cost_matrix[r-1,c],
                                           cost_matrix[r,c-1],
                                           cost_matrix[r-1,c-1]])
    return cost_matrix[1:,1:]


def trace_warp_indexes(cost_matrix:np.ndarray)->list:
    R, C = cost_matrix.shape
    cost_matrix_aug = np.full((R+1,C+1), np.inf)
    cost_matrix_aug[1:,1:] = cost_matrix
    r, c = R-1, C-1
    warp_indexes = [(r, c)]
    while r > 0 or c > 0:
        r, c = [(r-1, c-1), 
                (r-1, c  ), 
                (r,   c-1)][np.argmin([cost_matrix_aug[r,   c  ],
                                       cost_matrix_aug[r,   c+1],
                                       cost_matrix_aug[r+1, c  ]])]
        warp_indexes = [(r, c)] + warp_indexes
    return warp_indexes


def compute_accumulated_cost_matrix(a:np.ndarray, b:np.ndarray, ord:float=2)->np.ndarray:
    if len(a.shape) != len(b.shape):
        raise ValueError("Dimensions of a and b don't match.")
    if len(a.shape) <= 2:
        if len(a.shape) == 1:
            axis = (1, 2)
        elif len(a.shape) == 2:
            axis = 2
        a = np.expand_dims(a, axis=axis)
        b = np.expand_dims(b, axis=axis)
    return _njit_compute_accumulated_cost_matrix(a=a, b=b, ord=ord)


def distance(a:np.ndarray, b:np.ndarray, ord:float=2, return_warp_indexes:bool=False)->float:
    cost_matrix = compute_accumulated_cost_matrix(a=a, b=b, ord=ord)
    if return_warp_indexes:
        return cost_matrix[-1,-1], trace_warp_indexes(cost_matrix)
    return cost_matrix[-1,-1]
