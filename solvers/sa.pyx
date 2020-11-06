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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal(np.ndarray[np.float64_t, ndim=1] sched,
             int mcsteps,
             np.ndarray[np.int_t, ndim=1] svec,
             np.float64_t[:, :, :] nbs):
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
    cdef int maxnb = nbs[0].shape[0]
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
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    # Loop over temperatures
    with nogil:
        for itemp in xrange(schedsize):
            # Get temperature
            temp = sched[itemp]
            # Do some number of Monte Carlo steps
            for step in xrange(mcsteps):
                # Fisher-Yates shuffling algorithm
                # cannot use numpy.random.permutation due to nogil
                for i in xrange(nspins):
                    ispins[i] = i
                for i in xrange(nspins, 0, -1):
                    j = crand() % i
                    t = ispins[i-1]
                    ispins[i-1] = ispins[j]
                    ispins[j] = t
                # Loop over spins
                for ispin in xrange(nspins):
                    sidx = ispins[ispin]
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal_parallel(np.float_t[:] sched,
                      int mcsteps,
                      np.int_t[:] svec,
                      np.float_t[:, :, :] nbs,
                      int nthreads):
    """
    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.
    This version uses straightforward OpenMP threading to parallelize
    over inner spin-update loop.
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
        @nthreads (int): number of threads to execute in parallel
    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int schedsize = sched.size
    cdef int itemp = 0
    cdef double temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int ispin = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef np.ndarray[np.float_t, ndim=1] ediffs = np.zeros(nspins)
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    with nogil, parallel(num_threads=nthreads):
        # Loop over temperatures
        for itemp in xrange(schedsize):
            # Get temperature
            temp = sched[itemp]
            # Do some number of Monte Carlo steps
            for step in xrange(mcsteps):
                # Fisher-Yates shuffling algorithm
                # cannot use numpy.random.permutation due to nogil
                for i in xrange(nspins):
                    ispins[i] = i
                for i in xrange(nspins, 0, -1):
                    j = crand() % i
                    t = ispins[i-1]
                    ispins[i-1] = ispins[j]
                    ispins[j] = t
                # Loop over spins
                for ispin in prange(nspins, schedule='static'):
                    sidx = ispins[ispin]
                    ediffs[sidx] = 0.0  # reset
                    # loop through the neighbors
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediffs[sidx] += -2.0*svec[sidx]*jval
                        else:
                            ediffs[sidx] += -2.0*svec[sidx]*(jval*svec[spinidx])
                    # Accept or reject
                    if ediffs[sidx] <= 0.0:  # avoid overflow
                        svec[sidx] *= -1
                    elif cexp(-1.0*ediffs[sidx]/temp) > crand()/float(RAND_MAX):
                        svec[sidx] *= -1
