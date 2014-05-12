__author__ = 'harrigan'

import numpy as np
from scipy import sparse, linalg
import logging
import sys
from sklearn import manifold

log = logging.getLogger()
log.addHandler(logging.StreamHandler(stream=sys.stdout))
log.setLevel(logging.INFO)


class ToySystem(object):
    pass


class PotentialSystem(ToySystem):
    """A system defined by its potential energy surface."""
    pass


class TransitionSystem(ToySystem):
    """Any system that directly samples based on transition probabilities."""

    def __init__(self):
        self.tmat = None
        self.step_func = None

    @property
    def step(self):
        """Call the correct step function

        Note: this is only supposed to infer the function
        on first call, but it seems to do it for every step in
        sample. Calling sample a second time removes the issue.
        Perhaps it is optimizing away the branch [check if step_func is none]
        """

        if self.step_func is None:
            if self.tmat is None:
                raise ValueError('No transition matrix!')
            else:
                if sparse.issparse(self.tmat):
                    log.info("Using sparse sampling")
                    self.step_func = self.step_sparse
                else:
                    log.info("Using dense sampling")
                    self.step_func = self.step_dense

        return self.step_func

    def sample(self, init_state, n_steps):
        """
        Sample a trajectory in state space

        :param init_state: State to start from
        :param n_steps: number of steps to take
        :return:
        """

        states = np.zeros(n_steps, dtype=int)
        states[0] = init_state
        for i in range(1, n_steps):
            states[i] = self.step(states[i - 1])
        return states

    def step_sparse(self, state_i):
        """Perform one step in a csr transition matrix

        :param state_i: The state to start from
        """
        t_matrix = self.tmat

        csr_slicer = slice(t_matrix.indptr[state_i],
                           t_matrix.indptr[state_i + 1])
        probs = t_matrix.data[csr_slicer]
        colinds = t_matrix.indices[csr_slicer]

        # Find our new state and translate to actual indices
        prob_i = np.sum(np.cumsum(probs) < np.random.rand())
        state_i = colinds[prob_i]

        return state_i

    def step_dense(self, state_i):
        """Perform one step in a dense transition matrix

        :param state_i: The state to start from
        :return: a new state index
        """

        t_matrix = self.tmat

        probs = t_matrix[state_i, :]
        log.debug(np.cumsum(probs))
        state_i = np.sum(np.cumsum(probs) < np.random.rand())

        return state_i

    def get_centers(self, ndim=2):
        """Use MDS to get n-dim rep.
        :param ndim: number of dimension
        :return:
        """

        mds = manifold.MDS(n_components=ndim, dissimilarity='precomputed')
        # TODO: make rev_counts in init
        dtmat = self.rev_counts.toarray()

        # Microcount to define distance
        dtmat = dtmat + 0.01

        # Make symmetric
        dtmat = 0.5 * (dtmat + np.transpose(dtmat))

        # Turn from similarity to dissimilarity
        dtmat = 1.0 / dtmat

        # Do it to it
        pos = mds.fit(dtmat).embedding_
        return pos

    def format_mat(self, matname):
        mat = self.__getattribute__(matname)
        if sparse.issparse(mat):
            mat = mat.toarray()
        for row in mat:
            yield ' '.join(["{: =10g}".format(r) for r in row])

    def print_mat(self, matname):
        for line in self.format_mat(matname):
            print line



