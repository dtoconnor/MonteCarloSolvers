# encoding: utf-8
# cython: profile=False
# filename: sa.pyx

import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange, parallel
from libc.math cimport exp as cexp
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX
# from libc.stdio cimport printf as cprintf


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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal(np.ndarray[np.float64_t, ndim=1] sched,
             int mcsteps,
             np.ndarray[np.int_t, ndim=1] svec,
             np.float64_t[:, :, :] nbs,
             int maxnb,
             rng):
    """
    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is 
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec (np.array, float): contains the starting configuration
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int nspins = svec.shape[0]
    cdef int schedsize = sched.size
    cdef int itemp = 0
    cdef double temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = rng.permutation(range(nspins))
    # Loop over temperatures
    for itemp in xrange(schedsize):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over spins
            for sidx in sidx_shuff:
                # loop through the given spin's neighbors
                for si in xrange(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[sidx,si,0])
                    # get the coupling value to that neighbor
                    jval = nbs[sidx,si,1]
                    # self-connections are not quadratic
                    if spinidx == sidx:
                        ediff += -2.0*float(svec[sidx])*jval
                    # calculate the energy diff of flipping this spin
                    else:
                        ediff += -2.0*float(svec[sidx])*(jval*float(svec[spinidx]))
                # Metropolis accept or reject
                if ediff <= 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(-1.0 * ediff/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)
