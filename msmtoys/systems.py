__author__ = 'harrigan'

from msmtoys import systems_baseclasses as bc, plotting, muller
from msmbuilder import MSMLib as msml
import numpy as np
import scipy.sparse

EPSILON = 1e-6


class FourStateTmat(bc.TransitionSystem):
    """Set up a system with n=4 states

    """

    def __init__(self):
        super(FourStateTmat, self).__init__()

        counts = [
            [100, 30, 1, 1],
            [30, 100, 1, 1],
            [3, 3, 100, 30],
            [3, 3, 30, 100]
        ]
        counts = np.array(counts)
        counts = scipy.sparse.csr_matrix(counts, dtype=np.int)
        rev_counts, tmat, populations, mapping = msml.build_msm(
            counts, symmetrize='MLE', ergodic_trimming=True)

        self.n_states = tmat.shape[0]
        self.tmat = tmat
        self.counts = counts
        self.rev_counts = rev_counts
        self.step_func = self.step_sparse


class EightStateTmat(FourStateTmat):
    """Use outerproduct to double the number of states

    :param link_prob_f: the off diagonal conectybits
    :param link_prob_b:
    """

    def __init__(self, link_prob_f, link_prob_b):
        super(EightStateTmat, self).__init__()

        n = self.n_states * 2

        # Do outer product
        connecty_mat = np.array([
            [1.0, link_prob_b],
            [link_prob_f, 1.0]
        ])
        double_counts = np.multiply.outer(connecty_mat, self.counts.todense())
        # Turn it into a 2d matrix
        double_counts = np.swapaxes(double_counts, 1, 2)
        double_counts = np.reshape(double_counts, (n, n))

        double_counts = scipy.sparse.csr_matrix(double_counts, dtype=np.int)
        rev_counts, tmat, populations, mapping = msml.build_msm(
            double_counts, symmetrize='MLE', ergodic_trimming=True)

        # Record that we now have twice as many states
        self.n_states = n
        self.tmat = tmat
        self.counts = double_counts
        self.rev_counts = rev_counts


class TmatFromPotential(bc.TransitionSystem):
    def __init__(self):
        super(bc.TransitionSystem, self).__init__()
        self.grid = None


    def calculate_transition_matrix(self, potential, resolution, beta,
                                    bounds=None):
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

        t_matrix = scipy.sparse.coo_matrix(
            (t_matrix_data, (t_matrix_rows, t_matrix_cols)),
            shape=(n_states, n_states)).tocsr()

        # Normalize
        for trow_i in xrange(n_states):
            rfrom = t_matrix.indptr[trow_i]
            rto = t_matrix.indptr[trow_i + 1]
            normalization = np.sum(t_matrix.data[rfrom:rto])
            if normalization < EPSILON:
                print("No transitions from %d" % (trow_i))
            else:
                t_matrix.data[rfrom:rto] = t_matrix.data[
                                           rfrom:rto] / normalization

        self.grid = grid
        self.tmat = t_matrix


class MullerTmat(TmatFromPotential):
    def __init__(self, resolution, beta):
        super(TmatFromPotential, self).__init__()
        self.calculate_transition_matrix(muller.MullerForce, resolution, beta)


def _neighbors():
    """Get an array of x,y offsets to index nearest neighbors."""
    neighbors = np.zeros((0, 2), dtype='int')
    for row_i in xrange(-1, 2):
        for col_i in xrange(-1, 2):
            if not (row_i == 0 and col_i == 0):
                neighbors = np.append(neighbors, [[row_i, col_i]], axis=0)

    return neighbors


def _state_id(x, y, shape):
    """Take a row, column description into a state id."""
    return shape[1] * y + x
