"""Calculate a pseudo-analytic transition matrix for a potential."""

import numpy as np
from msmtoys import plotting
from matplotlib import pyplot as pp
import scipy.sparse

EPSILON = 1.0e-10

def _state_id(x, y, shape):
    return shape[1] * y + x

def _rowcol(state_id, shape):
    x = state_id % shape[1]
    y = (state_id - x) // shape[1]
    return x, y

def _xy(state_id, grid):
    row, col = _rowcol(state_id, grid.shape)
    x = grid[0, row, 0]
    y = grid[1, 0, col]
    return x, y

def _neighbors():
    neighbors = np.zeros((0, 2), dtype='int')
    for row_i in xrange(-1, 2):
        for col_i in xrange(-1, 2):
            if not (row_i == 0 and col_i == 0):
                neighbors = np.append(neighbors, [[row_i, col_i]], axis=0)

    return neighbors


def calculate_transition_matrix(potential, resolution, beta, bounds=None):
    grid, grid_width = plotting._get_grid(potential, bounds, resolution)

    n_states = grid.shape[1] * grid.shape[2]

    # The transition matrix will be constructed as a sparse COO matrix
    t_matrix_rows = np.zeros((0,))
    t_matrix_cols = np.zeros((0,))
    t_matrix_data = np.zeros((0,))

    neighbors = _neighbors()
    potential = potential.potential(grid[0], grid[1])

    for row_i in xrange(grid.shape[1]):
        for col_i in xrange(grid.shape[2]):
            # Loop through each starting point
            pot = potential[row_i, col_i]
            from_state = _state_id(row_i, col_i, grid.shape)
            normalization = 0.0
            # Only do nearest-neighbor
            for neighs in neighbors:
                nrow_i = row_i + neighs[0]
                ncol_i = col_i + neighs[1]
                if nrow_i < 0 or nrow_i >= grid.shape[1] or ncol_i < 0 or ncol_i >= grid.shape[2]:
                    # Transition probability to states outside our state space
                    # is zero. This is not in the transition matrix
                    pass
                else:
                    to_state = _state_id(nrow_i, ncol_i, grid.shape)
                    delta_pot = potential[nrow_i, ncol_i] - pot
                    t_prob = np.exp(-beta * delta_pot)

                    # Store info for our sparse matrix
                    t_matrix_rows = np.append(t_matrix_rows, from_state)
                    t_matrix_cols = np.append(t_matrix_cols, to_state)
                    t_matrix_data = np.append(t_matrix_data, t_prob)

                    normalization += t_prob



    t_matrix = scipy.sparse.coo_matrix((t_matrix_data, (t_matrix_rows, t_matrix_cols)), shape=(n_states, n_states)).tocsr()

    # Normalize
    for trow_i in xrange(n_states):
        rfrom = t_matrix.indptr[trow_i]
        rto = t_matrix.indptr[trow_i + 1]
        normalization = np.sum(t_matrix.data[rfrom:rto])
        if normalization < EPSILON:
            print("No transitions %d %d" % (row_i, col_i))
        else:
            t_matrix.data[rfrom:rto] = t_matrix.data[rfrom:rto] / normalization

    return t_matrix, grid

def propogate_t_matrix(pi, t_matrix, n_steps):
    normalize = np.sum(pi)
    if np.abs(normalize - 1.0) > EPSILON:
        print("Warning: initial state is not normalized")
        print("Performing normalization")
        v = pi / normalize
    else:
        v = pi

    for _ in xrange(n_steps):
        v = np.dot(v, t_matrix)

    return v

def _sample(weights, indices=None):
    cum_weights = np.cumsum(weights)
    result = np.sum(cum_weights < np.random.rand())
    if indices is not None:
        return indices[result]
    else:
        return result


def get_traj(t_matrix, length, grid, stride=1, initial_id=None):
    if initial_id is None:
        initial_id = np.random.randint(t_matrix.shape[0])

    state_id = initial_id
    xy = np.zeros((0, 2))

    if isinstance(t_matrix, scipy.sparse.csr_matrix):
        stride_trigger = stride
        for _ in xrange(length):
            rfrom = t_matrix.indptr[state_id]
            rto = t_matrix.indptr[state_id + 1]
            state_id = _sample(t_matrix.data[rfrom:rto], t_matrix.indices[rfrom:rto])

            if stride_trigger == stride:
                xy = np.append(xy, [_xy(state_id, grid)], axis=0)
                stride_trigger = 0

            stride_trigger += 1

    return xy

def plot_vector(vec, grid, scale=1.0):
    """Plot the centroids of a particular clustering scheme.

    If marker_sizes is given, it will use this array as the sizes for the
    various centroids. This is useful for visualizing eigenvectors.
    """

    for i in xrange(len(vec)):
        marker_size = np.abs(vec[i])
        if vec[i] < 0:
            color_string = 'ro'
        else:
            color_string = 'wo'

        x, y = _xy(i, grid)
        pp.plot(x, y, color_string, markersize=12 * scale * marker_size, zorder=10)

