"""Calculate a pseudo-analytic transition matrix for a potential."""

#TODO: This module is deprecated. Do not use. If you see this code used
#TODO: somewhere, try to replace it.


from msmtoys import plotting
from scipy.sparse import csr, coo
import numpy as np
import mdtraj

EPSILON = 1.0e-10


def _state_id(x, y, shape):
    """Take a row, column description into a state id."""
    return shape[1] * y + x


def _rowcol(state_id, shape):
    """Translate a state id into a row, column."""
    x = state_id % shape[1]
    y = (state_id - x) // shape[1]
    return x, y


def _xy(state_id, grid):
    """Translate a state id into x, y coordinates."""
    row, col = _rowcol(state_id, grid.shape)
    x = grid[0, row, 0]
    y = grid[1, 0, col]
    return x, y


def _neighbors():
    """Get an array of x,y offsets to index nearest neighbors."""
    neighbors = np.zeros((0, 2), dtype='int')
    for row_i in xrange(-1, 2):
        for col_i in xrange(-1, 2):
            if not (row_i == 0 and col_i == 0):
                neighbors = np.append(neighbors, [[row_i, col_i]], axis=0)

    return neighbors


def calculate_transition_matrix(potential, resolution, beta, bounds=None):
    """Calculate a transition matrix from a potential."""
    if bounds is None:
        bounds = potential.bounds
    grid = plotting.get_grid(bounds, resolution)

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
                if (nrow_i < 0 or nrow_i >= grid.shape[1]
                    or ncol_i < 0 or ncol_i >= grid.shape[2]):
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

    t_matrix = coo.coo_matrix((t_matrix_data, (t_matrix_rows, t_matrix_cols)),
                              shape=(n_states, n_states)).tocsr()

    # Normalize
    for trow_i in xrange(n_states):
        rfrom = t_matrix.indptr[trow_i]
        rto = t_matrix.indptr[trow_i + 1]
        normalization = np.sum(t_matrix.data[rfrom:rto])
        if normalization < EPSILON:
            print("No transitions from %d" % (trow_i))
        else:
            t_matrix.data[rfrom:rto] = t_matrix.data[rfrom:rto] / normalization

    return t_matrix, grid


def propogate_t_matrix(pi, t_matrix, n_steps):
    """Apply a transition matrix n_steps times.

    pi - initial state vector.
    """
    normalize = np.sum(pi)
    if np.abs(normalize - 1.0) > EPSILON:
        print("Warning: initial state is not normalized")
        print("Performing normalization")
        vec = pi / normalize
    else:
        vec = pi

    for _ in xrange(n_steps):
        vec = np.dot(vec, t_matrix)

    return vec


def _sample(weights, indices=None):
    """Get the next state from weights

    indices - if using a sparse matrix, pass the indices.
    """
    cum_weights = np.cumsum(weights)
    result = np.sum(cum_weights < np.random.rand())
    if indices is not None:
        return indices[result]
    else:
        return result


def get_traj(t_matrix, length, grid, stride=1, initial_id=None):
    """Sample a trajectory from the transition matrix."""
    if initial_id is None:
        initial_id = np.random.randint(t_matrix.shape[0])

    state_id = initial_id
    xy = np.zeros((length / stride, 2))

    if isinstance(t_matrix, csr.csr_matrix):
        stride_trigger = stride
        for i in xrange(length):
            rfrom = t_matrix.indptr[state_id]
            rto = t_matrix.indptr[state_id + 1]
            state_id = _sample(t_matrix.data[rfrom:rto],
                               t_matrix.indices[rfrom:rto])

            if stride_trigger == stride:
                # xy = np.append(xy, [_xy(state_id, grid)], axis=0)
                xy[i / stride] = _xy(state_id, grid)
                stride_trigger = 0

            stride_trigger += 1
    return xy


def get_trajlist(t_matrix, grid, num_trajs, traj_len, stride, random_seed=None):
    """Get a list of trajectories."""

    if random_seed is not None:
        np.random.seed(random_seed)
        initial_ids = np.random.random_integers(t_matrix.shape[0],
                                                size=num_trajs)
    else:
        initial_ids = [None for i in xrange(num_trajs)]

    traj_list = list()
    for i in xrange(num_trajs):
        traj = get_traj(t_matrix, length=traj_len, grid=grid, stride=stride,
                        initial_id=initial_ids[i])
        traj_list.append(traj)

    return traj_list



