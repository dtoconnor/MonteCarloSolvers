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
    cdef np.ndarray[np.float64_t, ndim=4] randuni = np.random.uniform(size=(schedsize, mcsteps, nspins, 2))
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
                    theta_prop = pi * randuni[ifield, step, ispin, 0]
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
                    elif cexp(-1.0 * ediff/temp) > randuni[ifield, step, ispin, 1]:
                        svec[sidx] = theta_prop
                    # Reset energy diff value
                    ediff = 0.0

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
    cdef np.ndarray[np.float64_t, ndim=4] randuni = np.random.uniform(size=(schedsize, mcsteps, nspins, 2))
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
                        theta_prop = (2.0 * pi * randuni[ifield, step, ispin, 0]) - pi
                    else:
                        theta_prop = ab_ratio * ((2.0 * pi * randuni[ifield, step, ispin, 0]) - pi)
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
                    elif cexp(-1.0 * ediff/temp) > randuni[ifield, step, ispin, 1]:
                        svec[sidx] = theta_prop
                    # Reset energy diff value
                    ediff = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef NoisySVMC(np.ndarray[np.float64_t, ndim=1] A_sched,
                           np.ndarray[np.float64_t, ndim=1] B_sched,
                           int mcsteps,
                           float temp,
                           np.ndarray[np.float_t, ndim=1] svec,
                           np.float64_t[:, :, :, :] nbs):
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
        @nbs (np.ndarray, float): 4D array whose 1st dimension indexes
                                  a new nbs array in 'time' with either,
                                  noise or CT added, 2st dimension indexes
                                  each spin, 3nd dimension indexes
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
    cdef int maxnb = nbs.shape[2]
    cdef int nspins = svec.shape[0]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double a_coeff = 0.0
    cdef double b_coeff = 0.0
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
    cdef np.ndarray[np.float64_t, ndim=4] randuni = np.random.uniform(size=(schedsize, mcsteps, nspins, 2))
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
                    theta_prop = pi * randuni[ifield, step, ispin, 0]
                    zmagdiff = ccos(theta_prop) - ccos(svec[sidx])
                    # loop through the given spin's neighbors and add z components
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ifield,sidx,si,0])
                        # get the coupling value to that neighbor
                        jval = nbs[ifield,sidx,si,1]
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
                    elif cexp(-1.0 * ediff/temp) > randuni[ifield, step, ispin, 1]:
                        svec[sidx] = theta_prop
                    # Reset energy diff value
                    ediff = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef NoisySVMCTF(np.ndarray[np.float64_t, ndim=1] A_sched,
                           np.ndarray[np.float64_t, ndim=1] B_sched,
                           int mcsteps,
                           float temp,
                           np.ndarray[np.float_t, ndim=1] svec,
                           np.float64_t[:, :, :, :] nbs):
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
        @nbs (np.ndarray, float): 4D array whose 1st dimension indexes
                                  a new nbs array in 'time' with either,
                                  noise or CT added, 2st dimension indexes
                                  each spin, 3nd dimension indexes
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
    cdef int maxnb = nbs.shape[2]
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
    cdef np.ndarray[np.float64_t, ndim=4] randuni = np.random.uniform(size=(schedsize, mcsteps, nspins, 2))
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
                        theta_prop = (2.0 * pi * randuni[ifield, step, ispin, 0]) - pi
                    else:
                        theta_prop = ab_ratio * ((2.0 * pi * randuni[ifield, step, ispin, 0]) - pi)
                    theta_prop = theta_prop + svec[sidx]
                    if theta_prop < 0:
                        theta_prop = 0.0
                    elif theta_prop > pi:
                        theta_prop = pi
                    zmagdiff = ccos(theta_prop) - ccos(svec[sidx])
                    # loop through the given spin's neighbors and add z components
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ifield,sidx,si,0])
                        # get the coupling value to that neighbor
                        jval = nbs[ifield,sidx,si,1]
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
                    elif cexp(-1.0 * ediff/temp) > randuni[ifield, step, ispin, 1]:
                        svec[sidx] = theta_prop
                    # Reset energy diff value
                    ediff = 0.0
