# encoding: utf-8
# cython: profile=False
# filename: svmc.pyx

import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange, parallel
from libc.math cimport exp as cexp
from libc.math cimport sin as csin
from libc.math cimport cos as ccos
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef SpinVectorMonteCarlo(np.ndarray[np.float64_t, ndim=1] A_sched,
                           np.ndarray[np.float64_t, ndim=1] B_sched,
                           int mcsteps,
                           float temp,
                           np.ndarray[np.float64_t, ndim=1] svec,
                           np.float64_t[:, :, :] nbs):
    """
    Execute spin vector monte carlo according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    Args:
        @A_sched (np.array, float): an array of transverse field values that specify
                                   the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field values that specify
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
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double a_coeff = 0.0
    cdef double b_coeff = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double theta_prop = 0.0
    cdef double zmagdiff = 0.0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double pi = np.pi
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int ispin = 0
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    # loop through schedule
    with nogil:
        for ifield in xrange(schedsize):
            a_coeff = A_sched[ifield]
            b_coeff = B_sched[ifield]
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
                    # propose new theta
                    theta_prop = pi * crand()/float(RAND_MAX)
                    zmagdiff = ccos(theta_prop) - ccos(svec[sidx])
                    # loop through the given spin's neighbors and add z components
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx,si,0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx,si,1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += b_coeff*jval*zmagdiff
                        # calculate the energy diff of flipping this spin
                        else:
                            ediff += b_coeff*jval*zmagdiff*ccos(svec[spinidx])
                    # add x component
                    ediff += a_coeff * (csin(svec[sidx]) - csin(theta_prop))
                    # Metropolis accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        svec[sidx] = theta_prop
                    elif cexp(-1.0 * ediff/temp) > crand()/float(RAND_MAX):
                        svec[sidx] = theta_prop
                    # Reset energy diff value
                    ediff = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef SpinVectorMonteCarlo_parallel(np.ndarray[np.float64_t, ndim=1] A_sched,
                           np.ndarray[np.float64_t, ndim=1] B_sched,
                           int mcsteps,
                           float temp,
                           np.ndarray[np.float_t, ndim=2] svec,
                           np.float64_t[:, :, :] nbs,
                           int nthreads):
    """
    Execute spin vector monte carlo according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    Args:
        @A_sched (np.array, float): an array of transverse field values that specify
                                   the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field values that specify
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
    cdef int nspins = svec.shape[1]
    cdef int nruns = svec.shape[0]
    cdef int run = 0
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double a_coeff = 0.0
    cdef double b_coeff = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double theta_prop = 0.0
    cdef double zmagdiff = 0.0
    cdef double jval = 0.0
    cdef double pi = np.pi
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nruns, nspins))
    cdef np.ndarray[np.int_t, ndim=2] ispins = np.repeat([np.arange(nspins),], nruns, axis=0)
    cdef int ispin = 0
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    with nogil, parallel(num_threads=nthreads):
        for ifield in xrange(schedsize):
            a_coeff = float(A_sched[ifield])
            b_coeff = float(B_sched[ifield])
            # Do some number of Monte Carlo steps
            for step in xrange(mcsteps):
                for run in prange(nruns, schedule='dynamic'):
                    # Fisher-Yates shuffling algorithm
                    for i in xrange(nspins):
                        ispins[run, i] = i
                    for i in xrange(nspins, 0, -1):
                        j = crand() % i
                        t = ispins[run, i-1]
                        ispins[run, i-1] = ispins[run, j]
                        ispins[run, j] = t
                    # Loop over spins
                    for ispin in xrange(nspins):
                        sidx = ispins[run, ispin]
                        ediffs[run, sidx] = 0.0  # reset
                        # propose new theta
                        theta_prop = pi * crand()/float(RAND_MAX)
                        zmagdiff = ccos(theta_prop) - ccos(svec[run, sidx])
                        # loop through the given spin's neighbors and add z components
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[sidx,si,0])
                            # get the coupling value to that neighbor
                            jval = nbs[sidx,si,1]
                            # self-connections are not quadratic
                            if spinidx == sidx:
                                ediffs[run, sidx] += b_coeff*jval*zmagdiff
                            # calculate the energy diff of flipping this spin
                            else:
                                ediffs[run, sidx] += b_coeff*jval*zmagdiff*ccos(svec[run, spinidx])
                        # add x component
                        ediffs[run, sidx] += a_coeff*(csin(svec[run, sidx]) - csin(theta_prop))
                        # Metropolis accept or reject
                        if ediffs[run, sidx] <= 0.0:  # avoid overflow
                            svec[run, sidx] = theta_prop
                        elif cexp(-1.0*ediffs[run, sidx]/temp) > crand()/float(RAND_MAX):
                            svec[run, sidx] = theta_prop


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef SpinVectorMonteCarloTF(np.ndarray[np.float64_t, ndim=1] A_sched,
                           np.ndarray[np.float64_t, ndim=1] B_sched,
                           int mcsteps,
                           float temp,
                           np.ndarray[np.float_t, ndim=1] svec,
                           np.float64_t[:, :, :] nbs):
    """
    Execute spin vector monte carlo according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    Args:
        @A_sched (np.array, float): an array of transverse field values that specify
                                   the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field values that specify
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
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double a_coeff = 0.0
    cdef double b_coeff = 0.0
    cdef double ab_ratio = 1.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double zmagdiff = 0.0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double pi = np.pi
    cdef double theta_prop = 0.0
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int ispin = 0
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    # loop through schedule
    with nogil:
        for ifield in xrange(schedsize):
            a_coeff = A_sched[ifield]
            b_coeff = B_sched[ifield]
            # Do some number of Monte Carlo steps
            for step in xrange(mcsteps):
                # Fisher-Yates shuffling algorithm
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
                    # propose new theta
                    ab_ratio = a_coeff/b_coeff
                    if ab_ratio > 1:
                        theta_prop = (2.0 * pi * crand()/float(RAND_MAX)) - pi
                    else:
                        theta_prop = ab_ratio * ((2.0 * pi * crand()/float(RAND_MAX)) - pi)
                    theta_prop = theta_prop + svec[sidx]
                    if theta_prop < 0:
                        theta_prop = 0.0
                    elif theta_prop > pi:
                        theta_prop = pi
                    zmagdiff = ccos(theta_prop) - ccos(svec[sidx])
                    # loop through the given spin's neighbors and add z components
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx,si,0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx,si,1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += b_coeff*jval*zmagdiff
                        # calculate the energy diff of flipping this spin
                        else:
                            ediff += b_coeff*jval*zmagdiff*ccos(svec[spinidx])
                    # add x component
                    ediff += a_coeff * (csin(svec[sidx]) - csin(theta_prop))
                    # Metropolis accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        svec[sidx] = theta_prop
                    elif cexp(-1.0 * ediff/temp) > crand()/float(RAND_MAX):
                        svec[sidx] = theta_prop
                    # Reset energy diff value
                    ediff = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef SpinVectorMonteCarloTF_parallel(np.ndarray[np.float64_t, ndim=1] A_sched,
                           np.ndarray[np.float64_t, ndim=1] B_sched,
                           int mcsteps,
                           float temp,
                           np.ndarray[np.float_t, ndim=2] svec,
                           np.float64_t[:, :, :] nbs,
                           int nthreads):
    """
    Execute spin vector monte carlo according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    Args:
        @A_sched (np.array, float): an array of transverse field values that specify
                                   the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field values that specify
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
    cdef int nspins = svec.shape[1]
    cdef int nruns = svec.shape[0]
    cdef int run = 0
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double a_coeff = 0.0
    cdef double b_coeff = 0.0
    cdef double ab_ratio = 1.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double theta_prop = 0.0
    cdef double zmagdiff = 0.0
    cdef double jval = 0.0
    cdef double pi = np.pi
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nruns, nspins))
    cdef np.ndarray[np.int_t, ndim=2] ispins = np.repeat([np.arange(nspins),], nruns, axis=0)
    cdef int ispin = 0
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    with nogil, parallel(num_threads=nthreads):
        for ifield in xrange(schedsize):
            a_coeff = float(A_sched[ifield])
            b_coeff = float(B_sched[ifield])
            # Do some number of Monte Carlo steps
            for step in xrange(mcsteps):
                for run in prange(nruns, schedule='dynamic'):
                    # Fisher-Yates shuffling algorithm
                    for i in xrange(nspins):
                        ispins[run, i] = i
                    for i in xrange(nspins, 0, -1):
                        j = crand() % i
                        t = ispins[run, i-1]
                        ispins[run, i-1] = ispins[run, j]
                        ispins[run, j] = t
                    # Loop over spins
                    for ispin in xrange(nspins):
                        sidx = ispins[run, ispin]
                        ediffs[run, sidx] = 0.0  # reset
                        # propose new theta
                        ab_ratio = a_coeff/b_coeff
                        if ab_ratio > 1:
                            theta_prop = (2.0 * pi * crand()/float(RAND_MAX)) - pi
                        else:
                            theta_prop = ab_ratio * ((2.0 * pi * crand()/float(RAND_MAX)) - pi)
                        theta_prop = theta_prop + svec[run, sidx]
                        if theta_prop < 0:
                            theta_prop = 0.0
                        elif theta_prop > pi:
                            theta_prop = pi
                        zmagdiff = ccos(theta_prop) - ccos(svec[run, sidx])
                        # loop through the given spin's neighbors and add z components
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[sidx,si,0])
                            # get the coupling value to that neighbor
                            jval = nbs[sidx,si,1]
                            # self-connections are not quadratic
                            if spinidx == sidx:
                                ediffs[run, sidx] += b_coeff*jval*zmagdiff
                            # calculate the energy diff of flipping this spin
                            else:
                                ediffs[run, sidx] += b_coeff*jval*zmagdiff*ccos(svec[run, spinidx])
                        # add x component
                        ediffs[run, sidx] += a_coeff*(csin(svec[run, sidx]) - csin(theta_prop))
                        # Metropolis accept or reject
                        if ediffs[run, sidx] <= 0.0:  # avoid overflow
                            svec[run, sidx] = theta_prop
                        elif cexp(-1.0*ediffs[run, sidx]/temp) > crand()/float(RAND_MAX):
                            svec[run, sidx] = theta_prop