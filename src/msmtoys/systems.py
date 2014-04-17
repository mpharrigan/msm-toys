__author__ = 'harrigan'

from msmtoys import systems_baseclasses as bc
from msmbuilder import MSMLib as msml
import numpy as np
import scipy.sparse


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

