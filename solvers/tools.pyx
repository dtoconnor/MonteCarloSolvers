# encoding: utf-8
# cython: profile=False
# filename: tools.pyx
'''

File: tools.py
Author: Hadayat Seddiqi
Date: 10.07.14
Description: A few helpful functions for doing simulated
             and quantum annealing simulations.

'''

import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sps


def bits2spins(vec):
    """ Convert a bitvector @vec to a spinvector. """
    return [ -1 if k == 1 else 1 for k in vec ]

def spins2bits(vec):
    """ Convert a spinvector @vec to a bitvector. """
    return [ 0 if k == 1 else 1 for k in vec ]

cpdef GenerateNeighbors(int nspins, 
                        J,  # scipy.sparse matrix
                        int maxnb, 
                        str savepath=None):
    """
    Precompute a list that include neighboring indices to each spin
    and the corresponding coupling value. Specifically, build:

    neighbors = [
           [ [ ni_0, J[0, ni_0] ], 
             [ ni_1, J[0, ni_1] ], 
               ... ],

           [ [ ni_0, J[1, ni_0] ], 
             [ ni_1, J[1, ni_1] ], 
               ... ],

            ...

           [ [ ni_0, J[nspins-1, ni_0]], 
             [ ni_1, J[nspins-1, ni_1]],                   
               ... ]
     ]

    For graphs that are not completely "regular", there will be
    some rows in the neighbor matrix for each spin that will show
    [0,0]. This is required to keep the neighbors data structure
    an N-dimensional array, but in the energy calculations will have
    no contribution as the coupling strength is essentially zero.
    On the other hand, this is why @maxnb must be set to as high a
    number as necessary, but no more (otherwise it will incur some
    computational cost).

    Args:
        @nspins (np.array, float): number of spins in the 2D lattice
        @J (sp.sparse.matrix, float): Ising coupling matrix
        @maxnb (int): the maximum number of neighbors for any spin
                  (if self-connections representing local field
                  terms are present along the diagonal of @J, 
                  this counts as a "neighbor" as well)

    Returns:
        np.ndarray, float:  the above specified "neighbors" list 
                            as a 3D numpy array
    """
    # predefining vars
    cdef int ispin = 0
    cdef int ipair = 0
    # the neighbors data structure
    cdef np.float_t[:, :, :]  nbs = np.zeros((nspins, maxnb, 2))
    # dictionary of keys type makes this easy
    J = J.todok()
    # Iterate over all spins
    for ispin in range(nspins):
        ipair = 0
        # Find the pairs including this spin
        for pair in J.keys():
            if pair[0] == ispin:
                nbs[ispin, ipair, 0] = pair[1]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
            elif pair[1] == ispin:
                nbs[ispin, ipair, 0] = pair[0]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
    J = J.tocsr()  # DOK is really slow for multiplication
    if savepath is not None:
        np.save(savepath, nbs)
    return nbs

@cython.embedsignature(True)
def ClassicalIsingEnergy(spins, J):
    """
    Calculate energy for Ising graph @J in configuration @spins.
    Generally not needed for the annealing process but useful to
    have around at the end of simulations.

    Args:
        @spins (np.array, float): configuration of spins (values +/-1)
        @J (np.ndarray, float): coupling matrix where off-diagonals
                                store coupling values and diagonal
                                stores local field biases

    Returns:
        float: the energy of configuration @spins in an Ising
               system specified by @J
    """
    J = np.asarray(J.todense())
    d = np.diag(np.diag(J))
    np.fill_diagonal(J, 0.0)
    return np.dot(spins, np.dot(J, spins)) + np.sum(np.dot(d,spins))

