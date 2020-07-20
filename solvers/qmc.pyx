# encoding: utf-8
# cython: profile=False
# filename: qmc.pyx

cimport cython
import numpy as np
cimport numpy as np
cimport openmp
from cython.parallel import prange, parallel
from libc.math cimport exp as cexp
from libc.math cimport tanh as ctanh
from libc.math cimport sinh as csinh
from libc.math cimport sqrt as csqrt
from libc.math cimport log as clog
from libc.math cimport sin as csin
from libc.math cimport abs as cabs
from libc.math cimport pow as cpow
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnneal(np.float_t[:] A_sched,
                 np.float_t[:] B_sched,
                 int mcsteps,
                 float temp,
                 np.int_t[:, :] confs,
                 np.float_t[:, :, :] nbs,
                 int nthreads):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single spin flips only.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @nthreads (int): number of parallel threads to use

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef int sidx = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nspins, slices))
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    with nogil, parallel(num_threads=nthreads):
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = -2.0*B_sched[ifield]
            for step in xrange(mcsteps):
                # Loop over Trotter slices
                for islice in xrange(slices):
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
                    for sidx in prange(nspins, schedule='static'):
                        ispin = ispins[sidx]
                        ediffs[ispin, islice] = 0.0
                        # loop through the neighbors
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[ispin, si, 0])
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            # self-connections are not quadratic
                            if spinidx == ispin:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*jval
                            else:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*(
                                    jval*float(confs[spinidx, islice])
                                )
                        # periodic boundaries
                        if islice == 0:
                            tleft = slices-1
                            tright = 1
                        elif islice == slices-1:
                            tleft = slices-2
                            tright = 0
                        else:
                            tleft = islice-1
                            tright = islice+1
                        # now calculate between neighboring slices
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tleft]))
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tright]))
                        # Accept or reject
                        if ediffs[ispin, islice] <= 0.0:  # avoid overflow
                            confs[ispin, islice] *= -1
                        elif cexp(-1.0 * ediffs[ispin, islice]/teff) > crand()/float(RAND_MAX):
                            confs[ispin, islice] *= -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef DissipativeQuantumAnneal(np.float_t[:] A_sched,
                             np.float_t[:] B_sched,
                             int mcsteps,
                             float temp,
                             np.float_t[:] lookuptable,
                             np.int_t[:, :] confs,
                             np.float_t[:, :, :] nbs,
                             int nthreads):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single spin flips only.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i ) +
        alpha \sum_i^N (\sum_k^P (\sum_k'^P s^k_i s^k'_i * (pi / (P*sin(pi * |k - k'|/P))^2)))

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @temp (float): ambient temperature
        @lookuptable (np.ndarray, float): table of system bath coupling strengths for different distances
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @nthreads (int): number of parallel threads to use

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int k = 0
    cdef int bslice = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef int sidx = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef double pi = np.pi
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nspins, slices))
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    with nogil, parallel(num_threads=nthreads):
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = -2.0*B_sched[ifield]
            for step in xrange(mcsteps):
                # Loop over Trotter slices
                for islice in xrange(slices):
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
                    for sidx in prange(nspins, schedule='static'):
                        ispin = ispins[sidx]
                        ediffs[ispin, islice] = 0.0
                        # loop through the neighbors
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[ispin, si, 0])
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            # self-connections are not quadratic
                            if spinidx == ispin:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*jval
                            else:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*(
                                    jval*float(confs[spinidx, islice])
                                )
                        # periodic boundaries
                        if islice == 0:
                            tleft = slices-1
                            tright = 1
                        elif islice == slices-1:
                            tleft = slices-2
                            tright = 0
                        else:
                            tleft = islice-1
                            tright = islice+1
                        # now calculate between neighboring slices
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tleft]))
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tright]))
                        # system bath coupling
                        for k in xrange(1, slices):
                            bslice = (islice+k)%slices
                            ediffs[ispin, islice] += 2.0*teff*float(confs[ispin, islice]*confs[
                                ispin, bslice])*lookuptable[k-1]
                        # Accept or reject
                        if ediffs[ispin, islice] <= 0.0:  # avoid overflow
                            confs[ispin, islice] *= -1
                        elif cexp(-1.0 * ediffs[ispin, islice]/teff) > crand()/float(RAND_MAX):
                            confs[ispin, islice] *= -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealGlobal(np.float_t[:] A_sched,
                             np.float_t[:] B_sched,
                             int mcsteps,
                             float temp,
                             np.int_t[:, :] confs,
                             np.float_t[:, :, :] nbs,
                             int nthreads):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes local and global updates.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @nthreads (int): number of parallel threads to use

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int islice2 = 0
    cdef int ispin = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef int sidx = 0
    cdef int sidx2 = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nspins, slices))
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    with nogil, parallel(num_threads=nthreads):
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = -2.0*B_sched[ifield]
            for step in xrange(mcsteps):
                # Loop over Trotter slices
                for islice in xrange(slices):
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
                    for sidx in prange(nspins, schedule='static'):
                        ispin = ispins[sidx]
                        ediffs[ispin, islice] = 0.0
                        # loop through the neighbors
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[ispin, si, 0])
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            # self-connections are not quadratic
                            if spinidx == ispin:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*jval
                            else:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*(
                                    jval*float(confs[spinidx, islice])
                                )
                        # periodic boundaries
                        if islice == 0:
                            tleft = slices-1
                            tright = 1
                        elif islice == slices-1:
                            tleft = slices-2
                            tright = 0
                        else:
                            tleft = islice-1
                            tright = islice+1
                        # now calculate between neighboring slices
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tleft]))
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tright]))
                        # Accept or reject
                        if ediffs[ispin, islice] <= 0.0:  # avoid overflow
                            confs[ispin, islice] *= -1
                        elif cexp(-1.0 * ediffs[ispin, islice]/teff) > crand()/float(RAND_MAX):
                            confs[ispin, islice] *= -1
                # global updates
                # Fisher-Yates shuffling algorithm
                # cannot use numpy.random.permutation due to nogil
                for i in xrange(nspins):
                    ispins[i] = i
                for i in xrange(nspins, 0, -1):
                    j = crand() % i
                    t = ispins[i-1]
                    ispins[i-1] = ispins[j]
                    ispins[j] = t
                for sidx2 in prange(nspins, schedule='static'):
                    ispin = ispins[sidx2]
                    ediffs[ispin, 0] = 0.0
                    for islice2 in xrange(slices):
                        # loop through the neighbors
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[ispin, si, 0])
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            # self-connections are not quadratic
                            if spinidx == ispin:
                                ediffs[ispin, 0] += b_coeff*float(confs[ispin, islice2])*jval
                            else:
                                ediffs[ispin, 0] += b_coeff*float(confs[ispin, islice2])*(
                                    jval*float(confs[spinidx, islice2])
                             )
                    # Accept or reject
                    if ediffs[ispin, 0] <= 0.0:  # avoid overflow
                        for islice2 in xrange(slices):
                            confs[ispin, islice2] *= -1
                    elif cexp(-1.0 * ediffs[ispin, 0]/teff) > crand()/float(RAND_MAX):
                        for islice2 in xrange(slices):
                            confs[ispin, islice2] *= -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef DissipativeQuantumAnnealGlobal(np.float_t[:] A_sched,
                             np.float_t[:] B_sched,
                             int mcsteps,
                             float temp,
                             np.float_t[:] lookuptable,
                             np.int_t[:, :] confs,
                             np.float_t[:, :, :] nbs,
                             int nthreads):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes local and global updates.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i ) +
        alpha \sum_i^N (\sum_k^P (\sum_k'^P s^k_i s^k'_i * (pi / (P*sin(pi * |k - k'|/P))^2)))

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @temp (float): ambient temperature
        @lookuptable (np.ndarray, float): table of system bath coupling strengths for different distances
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @nthreads (int): number of parallel threads to use

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int islice2 = 0
    cdef int ispin = 0
    cdef int k = 0
    cdef int bslice = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef int sidx = 0
    cdef int sidx2 = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef double pi = np.pi
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nspins, slices))
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0

    with nogil, parallel(num_threads=nthreads):
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = -2.0*B_sched[ifield]
            for step in xrange(mcsteps):
                # Loop over Trotter slices
                for islice in xrange(slices):
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
                    for sidx in prange(nspins, schedule='static'):
                        ispin = ispins[sidx]
                        ediffs[ispin, islice] = 0.0
                        # loop through the neighbors
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[ispin, si, 0])
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            # self-connections are not quadratic
                            if spinidx == ispin:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*jval
                            else:
                                ediffs[ispin, islice] += b_coeff*float(confs[ispin, islice])*(
                                    jval*float(confs[spinidx, islice])
                                )
                        # periodic boundaries
                        if islice == 0:
                            tleft = slices-1
                            tright = 1
                        elif islice == slices-1:
                            tleft = slices-2
                            tright = 0
                        else:
                            tleft = islice-1
                            tright = islice+1
                        # now calculate between neighboring slices
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tleft]))
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tright]))
                        # system bath coupling
                        for k in xrange(1, slices):
                            bslice = (islice+k)%slices
                            ediffs[ispin, islice] += 2.0*teff*float(confs[ispin, islice]*confs[
                                ispin, bslice])*lookuptable[k-1]
                        # Accept or reject
                        if ediffs[ispin, islice] <= 0.0:  # avoid overflow
                            confs[ispin, islice] *= -1
                        elif cexp(-1.0 * ediffs[ispin, islice]/teff) > crand()/float(RAND_MAX):
                            confs[ispin, islice] *= -1
                # global updates
                # Fisher-Yates shuffling algorithm
                # cannot use numpy.random.permutation due to nogil
                for i in xrange(nspins):
                    ispins[i] = i
                for i in xrange(nspins, 0, -1):
                    j = crand() % i
                    t = ispins[i-1]
                    ispins[i-1] = ispins[j]
                    ispins[j] = t
                for sidx2 in prange(nspins, schedule='static'):
                    ispin = ispins[sidx2]
                    ediffs[ispin, 0] = 0.0
                    for islice2 in xrange(slices):
                        # loop through the neighbors
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[ispin, si, 0])
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            # self-connections are not quadratic
                            if spinidx == ispin:
                                ediffs[ispin, 0] += b_coeff*float(confs[ispin, islice2])*jval
                            else:
                                ediffs[ispin, 0] += b_coeff*float(confs[ispin, islice2])*(
                                    jval*float(confs[spinidx, islice2])
                             )
                    # Accept or reject
                    if ediffs[ispin, 0] <= 0.0:  # avoid overflow
                        for islice2 in xrange(slices):
                            confs[ispin, islice2] *= -1
                    elif cexp(-1.0 * ediffs[ispin, 0]/teff) > crand()/float(RAND_MAX):
                        for islice2 in xrange(slices):
                            confs[ispin, islice2] *= -1


#######################################################################################################################
# Function under test
#######################################################################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealWCL(np.float_t[:] A_sched,
                       np.float_t[:] B_sched,
                       int mcsteps,
                       float temp,
                       np.int_t[:, :] confs,
                       np.float_t[:, :, :] nbs):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes Wolff cluster updates.
    Probability of forming cluster bonds depends on local energy between the two spins only.
    This version removes the need for a register as we automatically flips spins as soon as they are added
    to the cluster making spins the wrong sign for future attempts to add it to the stack.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int si = 0
    cdef int si2 = 0
    cdef int spinidx = 0
    cdef int spinidx2 = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef int i = 0
    cdef int j = 0
    cdef np.ndarray[np.int_t, ndim=2] cluster = -np.ones((nspins*slices, 2), dtype=np.intc)
    cdef int k = 0
    cdef int stack = 1
    cdef int stackidx = 1
    cdef int cluster_count = 0
    cdef double r = 1.0

    with nogil:
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = B_sched[ifield]
            for step in xrange(mcsteps):
                # single Wolff cluster update
                # pick random start point that aligns with local field
                ispin = crand() % nspins
                islice = crand() % slices
                # initialize
                cluster[0, 0] = ispin
                cluster[0, 1] = islice
                k = confs[ispin, islice]
                # confs[ispin, islice] *= -1
                stack = 1
                stackidx = 1
                cluster_count = 0
                r = 1.0
                while True:
                    ispin = cluster[cluster_count, 0]
                    islice = cluster[cluster_count, 1]
                    # add spacial neighbours to stack
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ispin, si, 0])
                        # is same spin
                        if confs[spinidx, islice] == k:
                            ediff = 0.0
                            ediff += 2.0*b_coeff*nbs[ispin, si, 1]
                            # add bias energy
                            for si2 in xrange(maxnb):
                                 spinidx2 = int(nbs[spinidx, si2, 0])
                                 if spinidx == spinidx2:
                                    ediff += -2.0*b_coeff*nbs[spinidx, si2, 1]*k
                            if ediff < 0:
                                p = 1 - cexp(ediff/teff)
                                if r*p > crand()/float(RAND_MAX):
                                    # add to cluster
                                    r *= p
                                    cluster[stackidx, 0] = spinidx
                                    cluster[stackidx, 1] = islice
                                    confs[spinidx, islice] *= -1
                                    stack += 1
                                    stackidx += 1
                    # add trotter neighbours to stack
                    if islice == 0:
                        tleft = slices - 1
                        tright = 1
                    elif islice == slices - 1:
                        tleft = slices - 2
                        tright = 0
                    else:
                        tleft = islice - 1
                        tright = islice + 1
                    if confs[ispin, tleft] == k:
                        ediff = 0.0
                        ediff += -2.0 * jperp
                        # add bias energy
                        for si2 in xrange(maxnb):
                            spinidx2 = int(nbs[ispin, si2, 0])
                            if ispin == spinidx2:
                                ediff += -2.0*b_coeff*nbs[ispin, si2, 1]*k
                        if ediff < 0:
                            p = 1 - cexp(ediff/teff)
                            if r*p > crand()/float(RAND_MAX):
                                # add to cluster
                                r *= p
                                cluster[stackidx, 0] = ispin
                                cluster[stackidx, 1] = tleft
                                confs[ispin, tleft] *= -1
                                stack += 1
                                stackidx += 1
                    if confs[ispin, tright] == k:
                        ediff = 0.0
                        ediff += -2.0 * jperp
                        # add bias energy
                        for si2 in xrange(maxnb):
                            spinidx2 = int(nbs[ispin, si2, 0])
                            if ispin == spinidx2:
                                ediff += -2.0*b_coeff*nbs[ispin, si2, 1]*k
                        if ediff < 0:
                            p = 1 - cexp(ediff/teff)
                            if r*p > crand()/float(RAND_MAX):
                                # add to cluster
                                r *= p
                                cluster[stackidx, 0] = ispin
                                cluster[stackidx, 1] = tright
                                confs[ispin, tright] *= -1
                                stack += 1
                                stackidx += 1
                    cluster_count += 1
                    stack += -1
                    if stack == 0:
                        break


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef DissaptiveQuantumAnnealWCL(np.float_t[:] A_sched,
                       np.float_t[:] B_sched,
                       int mcsteps,
                       float temp,
                       np.float_t[:] lookuptable,
                       np.int_t[:, :] confs,
                       np.float_t[:, :, :] nbs):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes Wolff cluster updates.
    Probability of forming cluster bonds depends on local energy between the two spins only.
    This version removes the need for a register as we automatically flips spins as soon as they are added
    to the cluster making spins the wrong sign for future attempts to add it to the stack.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @temp (float): ambient temperature
        @lookuptable (np.ndarray, float): contains information regarding system bath coupling
                                          as a function of distance
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int si = 0
    cdef int si2 = 0
    cdef int spinidx = 0
    cdef int spinidx2 = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef int i = 0
    cdef int j = 0
    cdef np.ndarray[np.int_t, ndim=2] cluster = -np.ones((nspins*slices, 2), dtype=np.intc)
    cdef int k = 0
    cdef int b = 0
    cdef int bslice = 0
    cdef int stack = 1
    cdef int stackidx = 1
    cdef int cluster_count = 0
    cdef double r = 1.0

    with nogil:
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = B_sched[ifield]
            for step in xrange(mcsteps):
                # single Wolff cluster update
                # pick random start point that aligns with local field
                ispin = crand() % nspins
                islice = crand() % slices
                j = islice * nspins + ispin
                for i in xrange(1, nspins*slices):
                    ediff = 0.0
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ispin, si, 0])
                        jval = nbs[ispin, si, 1]
                        if spinidx == ispin:
                            ediff += -2.0 * jval * b_coeff * confs[ispin, islice]
                    if ediff <= 0:
                        break
                    elif cexp(-1.0*ediff/teff) > crand()/float(RAND_MAX):
                        break
                    else:
                        ispin = (j+i) % nspins
                        islice = ((j+i)/nspins) % slices
                # initialize
                cluster[0, 0] = ispin
                cluster[0, 1] = islice
                k = confs[ispin, islice]
                confs[ispin, islice] *= -1
                stack = 1
                stackidx = 1
                cluster_count = 0
                r = 1.0
                while True:
                    ispin = cluster[cluster_count, 0]
                    islice = cluster[cluster_count, 1]
                    # add bath neighbours
                    for b in xrange(1, slices):
                        bslice = (islice+b)%slices
                        if confs[ispin, bslice] == k:
                            ediff = 0.0
                            ediff += -2.0*teff*lookuptable[b-1]
                            # add bias energy
                            for si2 in xrange(maxnb):
                                 spinidx2 = int(nbs[ispin, si2, 0])
                                 if ispin == spinidx2:
                                    ediff += -2.0*b_coeff*nbs[ispin, si2, 1]*k
                            if ediff < 0:
                                p = 1 - cexp(ediff/teff)
                                if r*p > crand()/float(RAND_MAX):
                                    # add to cluster
                                    r *= p
                                    cluster[stackidx, 0] = ispin
                                    cluster[stackidx, 1] = bslice
                                    confs[ispin, bslice] *= -1
                                    stack += 1
                                    stackidx += 1
                    # add spacial neighbours to stack
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ispin, si, 0])
                        # is same spin
                        if confs[spinidx, islice] == k:
                            ediff = 0.0
                            ediff += 2.0*b_coeff*nbs[ispin, si, 1]
                            # add bias energy
                            for si2 in xrange(maxnb):
                                 spinidx2 = int(nbs[spinidx, si2, 0])
                                 if spinidx == spinidx2:
                                    ediff += -2.0*b_coeff*nbs[spinidx, si2, 1]*k
                            if ediff < 0:
                                p = 1 - cexp(ediff/teff)
                                if r*p > crand()/float(RAND_MAX):
                                    # add to cluster
                                    r *= p
                                    cluster[stackidx, 0] = spinidx
                                    cluster[stackidx, 1] = islice
                                    confs[spinidx, islice] *= -1
                                    stack += 1
                                    stackidx += 1
                    # add trotter neighbours to stack
                    if islice == 0:
                        tleft = slices - 1
                        tright = 1
                    elif islice == slices - 1:
                        tleft = slices - 2
                        tright = 0
                    else:
                        tleft = islice - 1
                        tright = islice + 1
                    if confs[ispin, tleft] == k:
                        ediff = 0.0
                        ediff += -2.0 * jperp
                        # add bias energy
                        for si2 in xrange(maxnb):
                            spinidx2 = int(nbs[ispin, si2, 0])
                            if ispin == spinidx2:
                                ediff += -2.0*b_coeff*nbs[ispin, si2, 1]*k
                        if ediff < 0:
                            p = 1 - cexp(ediff/teff)
                            if r*p > crand()/float(RAND_MAX):
                                # add to cluster
                                r *= p
                                cluster[stackidx, 0] = ispin
                                cluster[stackidx, 1] = tleft
                                confs[ispin, tleft] *= -1
                                stack += 1
                                stackidx += 1
                    if confs[ispin, tright] == k:
                        ediff = 0.0
                        ediff += -2.0 * jperp
                        # add bias energy
                        for si2 in xrange(maxnb):
                            spinidx2 = int(nbs[ispin, si2, 0])
                            if ispin == spinidx2:
                                ediff += -2.0*b_coeff*nbs[ispin, si2, 1]*k
                        if ediff < 0:
                            p = 1 - cexp(ediff/teff)
                            if r*p > crand()/float(RAND_MAX):
                                # add to cluster
                                r *= p
                                cluster[stackidx, 0] = ispin
                                cluster[stackidx, 1] = tright
                                confs[ispin, tright] *= -1
                                stack += 1
                                stackidx += 1
                    cluster_count += 1
                    stack += -1
                    if stack == 0:
                        break


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealWC(np.float_t[:] A_sched,
                       np.float_t[:] B_sched,
                       int mcsteps,
                       float temp,
                       np.int_t[:, :] confs,
                       np.float_t[:, :, :] nbs):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes Wolff cluster updates.
    Probability of forming cluster bonds depends on local energy change.
    This version removes the need for a register as we automatically flips spins as soon as they are added
    to the cluster making spins the wrong sign for future attempts to add it to the stack.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int si = 0
    cdef int si2 = 0
    cdef int spinidx = 0
    cdef int spinidx2 = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef int tleft2 = 0
    cdef int tright2 = 0
    cdef double teff = temp * float(slices)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0
    cdef np.ndarray[np.int_t, ndim=2] cluster = -np.ones((nspins*slices, 2), dtype=np.intc)
    cdef int k = 0
    cdef int stack = 1
    cdef int stackidx = 1
    cdef int cluster_count = 0
    cdef double r = 1.0
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.zeros(nspins, dtype=np.intc)

    with nogil:
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = B_sched[ifield]
            for step in xrange(mcsteps):
                 # Fisher-Yates shuffling algorithm
                # cannot use numpy.random.permutation due to nogil
                #for i in xrange(nspins):
                #    ispins[i] = i
                #for i in xrange(nspins, 0, -1):
                #    j = crand() % i
                #    t = ispins[i-1]
                #    ispins[i-1] = ispins[j]
                #    ispins[j] = t
                #for i in xrange(nspins):
                #    ispin = ispins[i]
                #    for islice in xrange(slices):
                # single Wolff cluster update
                # pick random start point
                ispin = crand() % nspins
                islice = crand() % slices
                cluster[0, 0] = ispin
                cluster[0, 1] = islice
                k = confs[ispin, islice]
                stack = 1
                stackidx = 1
                cluster_count = 0
                r = 1.0
                while True:
                    ispin = cluster[cluster_count, 0]
                    islice = cluster[cluster_count, 1]
                    # add trotter neighbours to stack
                    if islice == 0:
                        tleft = slices - 1
                        tright = 1
                    elif islice == slices - 1:
                        tleft = slices - 2
                        tright = 0
                    else:
                        tleft = islice - 1
                        tright = islice + 1
                    if confs[ispin, tleft] == k:
                        ediff = 0.0
                        # add energy change
                        for si2 in xrange(maxnb):
                            spinidx2 = int(nbs[ispin, si2, 0])
                            jval = nbs[spinidx, si2, 1]
                            if spinidx == spinidx2:
                                ediff += -2.0*b_coeff*jval*k
                            else:
                                ediff += -2.0*b_coeff*jval*k*confs[spinidx2, tleft]
                        if tleft == 0:
                            tleft2 = slices - 1
                            tright2 = 1
                        elif tleft == slices - 1:
                            tleft2 = slices - 2
                            tright2 = 0
                        else:
                            tleft2 = tleft - 1
                            tright2 = tleft + 1
                        ediff += 2.0*jperp*k*confs[ispin, tleft2]
                        ediff += 2.0*jperp*k*confs[ispin, tright2]
                        if ediff < 0:
                            p = 1 - cexp(ediff/teff)
                            if r * p > crand()/float(RAND_MAX):
                                # add to cluster
                                #r *= p
                                cluster[stackidx, 0] = ispin
                                cluster[stackidx, 1] = tleft
                                confs[ispin, tleft] *= -1
                                stack += 1
                                stackidx += 1
                    if confs[ispin, tright] == k:
                        ediff = 0.0
                        # add energy change
                        for si2 in xrange(maxnb):
                            spinidx2 = int(nbs[ispin, si2, 0])
                            jval = nbs[spinidx, si2, 1]
                            if spinidx == spinidx2:
                                ediff += -2.0*b_coeff*jval*k
                            else:
                                ediff += -2.0*b_coeff*jval*k*confs[spinidx2, tright]
                        if tright == 0:
                            tleft2 = slices - 1
                            tright2 = 1
                        elif tright == slices - 1:
                            tleft2 = slices - 2
                            tright2 = 0
                        else:
                            tleft2 = tright - 1
                            tright2 = tright + 1
                        ediff += 2.0*jperp*k*confs[ispin, tleft2]
                        ediff += 2.0*jperp*k*confs[ispin, tright2]
                        if ediff < 0:
                            p = 1 - cexp(ediff/teff)
                            if r*p > crand()/float(RAND_MAX):
                                # add to cluster
                                #r *= p
                                cluster[stackidx, 0] = ispin
                                cluster[stackidx, 1] = tright
                                confs[ispin, tright] *= -1
                                stack += 1
                                stackidx += 1
                    # add spacial neighbours to stack
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ispin, si, 0])
                        # is same spin
                        if confs[spinidx, islice] == k:
                            ediff = 0.0
                            # add energy change
                            for si2 in xrange(maxnb):
                                spinidx2 = int(nbs[spinidx, si2, 0])
                                jval = nbs[spinidx, si2, 1]
                                if spinidx == spinidx2:
                                    ediff += -2.0*b_coeff*jval*k
                                else:
                                    ediff += -2.0*b_coeff*jval*k*confs[spinidx2, islice]
                            if islice == 0:
                                tleft2 = slices - 1
                                tright2 = 1
                            elif islice == slices - 1:
                                tleft2 = slices - 2
                                tright2 = 0
                            else:
                                tleft2 = islice - 1
                                tright2 = islice + 1
                            ediff += 2.0*jperp*k*confs[spinidx, tleft2]
                            ediff += 2.0*jperp*k*confs[spinidx, tright2]
                            if ediff < 0:
                                p = 1 - cexp(ediff/teff)
                                if r*p > crand()/float(RAND_MAX):
                                    # add to cluster
                                    #r *= p
                                    cluster[stackidx, 0] = spinidx
                                    cluster[stackidx, 1] = islice
                                    confs[spinidx, islice] *= -1
                                    stack += 1
                                    stackidx += 1
                    cluster_count += 1
                    stack += -1
                    if stack == 0:
                        break


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef DissipativeQuantumAnnealWC2(np.float_t[:] A_sched,
                             np.float_t[:] B_sched,
                             int mcsteps,
                             float temp,
                             np.float_t[:] lookuptable,
                             np.int_t[:, :] confs,
                             np.float_t[:, :, :] nbs,
                             int nthreads):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes local and global updates.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i ) +
        alpha \sum_i^N (\sum_k^P (\sum_k'^P s^k_i s^k'_i * (pi / (P*sin(pi * |k - k'|/P))^2)))

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @nspins (int): number of spins in the spacial dimension
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @nthreads (int): number of parallel threads to use

    Returns:
        None: spins are flipped in-place within @confs
    """

    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int islice2 = 0
    cdef int ispin = 0
    cdef int si = 0
    cdef int si2 = 0
    cdef int spinidx = 0
    cdef int spinidx2 = 0
    cdef int sidx = 0
    cdef int sidx2 = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef double pi = np.pi
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nspins, slices))
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0
    cdef int b = 0
    cdef int b2 = 0
    cdef int bslice = 0
    cdef int cslice = 0
    cdef np.ndarray[np.int_t, ndim=2] cluster = -np.ones((slices, 2), dtype=np.intc)
    cdef int k = 0
    cdef int stack = 1
    cdef int stackidx = 1
    cdef int cluster_count = 0
    cdef double r = 1.0
    cdef int tleft2 = 0
    cdef int tright2 = 0
    cdef double e_total = 0.0

    with nogil:
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = B_sched[ifield]
            for step in xrange(mcsteps):
                # Loop over Trotter slices
                for islice in xrange(slices):
                    # Fisher-Yates shuffling algorithm
                    for i in xrange(nspins):
                        ispins[i] = i
                    for i in xrange(nspins, 0, -1):
                        j = crand() % i
                        t = ispins[i-1]
                        ispins[i-1] = ispins[j]
                        ispins[j] = t
                    # Loop over spins
                    for sidx in xrange(nspins):
                        ispin = ispins[sidx]
                        ediffs[ispin, islice] = 0.0
                        # loop through the neighbors
                        for si in xrange(maxnb):
                            # get the neighbor spin index
                            spinidx = int(nbs[ispin, si, 0])
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            # self-connections are not quadratic
                            if spinidx == ispin:
                                ediffs[ispin, islice] += -2.0*b_coeff*float(confs[ispin, islice])*jval
                            else:
                                ediffs[ispin, islice] += -2.0*b_coeff*float(confs[ispin, islice])*(
                                    jval*float(confs[spinidx, islice])
                                )
                        # periodic boundaries
                        if islice == 0:
                            tleft = slices-1
                            tright = 1
                        elif islice == slices-1:
                            tleft = slices-2
                            tright = 0
                        else:
                            tleft = islice-1
                            tright = islice+1
                        # now calculate between neighboring slices
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tleft]))
                        ediffs[ispin, islice] += 2.0*float(confs[ispin, islice])*(jperp*float(confs[ispin, tright]))
                        # system bath coupling
                        # system bath coupling
                        for b2 in xrange(1, slices):
                            cslice = (bslice+b2)%slices
                            ediffs[ispin, islice] += 2.0*teff*float(confs[ispin, islice]*confs[
                                ispin, cslice])*lookuptable[b2-1]
                        # Accept or reject
                        if ediffs[ispin, islice] <= 0.0:  # avoid overflow
                            confs[ispin, islice] *= -1
                        elif cexp(-1.0 * ediffs[ispin, islice]/teff) > crand()/float(RAND_MAX):
                            confs[ispin, islice] *= -1
                # Fisher-Yates shuffling algorithm
                for i in xrange(nspins):
                    ispins[i] = i
                for i in xrange(nspins, 0, -1):
                    j = crand() % i
                    t = ispins[i-1]
                    ispins[i-1] = ispins[j]
                    ispins[j] = t
                # start wolff updates
                for sidx2 in xrange(nspins):
                    ispin = ispins[sidx2]
                    islice = crand() % slices
                    cluster[0, 0] = ispin
                    cluster[0, 1] = islice
                    k = confs[ispin, islice]
                    stack = 1
                    stackidx = 1
                    cluster_count = 0
                    r = 1.0
                    e_total = 0.0
                    while True:
                        ispin = cluster[cluster_count, 0]
                        islice = cluster[cluster_count, 1]
                        # add bath neighbours
                        for b in xrange(1, slices):
                            bslice = (islice+b)%slices
                            if confs[ispin, bslice] == k:
                                p = 1 - cexp(-2.0*lookuptable[b-1])
                                if r*p > crand()/float(RAND_MAX):
                                    ediff = 0.0
                                    # add bias energy
                                    for si2 in xrange(maxnb):
                                        spinidx2 = int(nbs[ispin, si2, 0])
                                        if ispin == spinidx2:
                                            ediff += -2.0*b_coeff*nbs[ispin, si2, 1]*k
                                        else:
                                            ediff += -2.0*b_coeff*jval*k*confs[spinidx2, bslice]
                                    if bslice == 0:
                                        tleft2 = slices - 1
                                        tright2 = 1
                                    elif bslice == slices - 1:
                                        tleft2 = slices - 2
                                        tright2 = 0
                                    else:
                                        tleft2 = bslice - 1
                                        tright2 = bslice + 1
                                    ediff += 2.0*jperp*k*confs[ispin, tleft2]
                                    ediff += 2.0*jperp*k*confs[ispin, tright2]
                                    # system bath coupling
                                    for b2 in xrange(1, slices):
                                        cslice = (bslice+b2)%slices
                                        ediff += 2.0*teff*float(k*confs[
                                            ispin, cslice])*lookuptable[b2-1]
                                    # add to cluster
                                    r *= p
                                    e_total += ediff
                                    cluster[stackidx, 0] = ispin
                                    cluster[stackidx, 1] = bslice
                                    confs[ispin, bslice] *= -1
                                    stack += 1
                                    stackidx += 1
                        cluster_count += 1
                        stack += -1
                        if stack == 0:
                            break
                    # accept or reject cluster update
                    if e_total > 0:
                        if cexp(-1.0*e_total/teff) > crand()/float(RAND_MAX):
                            for stackidx in xrange(1, cluster_count):
                                confs[cluster[stackidx, 0], cluster[stackidx, 1]] *= -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef DissipativeQuantumAnnealWC3(np.float_t[:] A_sched,
                             np.float_t[:] B_sched,
                             int mcsteps,
                             float temp,
                             np.float_t[:] lookuptable,
                             np.int_t[:, :] confs,
                             np.float_t[:, :, :] nbs,
                             int nthreads):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes local and global updates.
    The Hamiltonian is:

    H = \sum_k^P(\sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i ) +
        alpha \sum_i^N (\sum_k^P (\sum_k'^P s^k_i s^k'_i * (pi / (P*sin(pi * |k - k'|/P))^2)))

    where J_perp = -PT/2 log(tanh(G/PT)).
    The quantum annealing is controlled by the strength of the transverse
    field. This is given as an array of field values in @A_sched. @confs
    stores the spin configurations for each replica which are updated
    sequentially.

    Args:
        @A_sched (np.array, float): an array of transverse field coefficients that specify
                                  the annealing schedule
        @B_sched (np.array, float): an array of longitudinal field coefficients that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @nspins (int): number of spins in the spacial dimension
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @nthreads (int): number of parallel threads to use

    Returns:
        None: spins are flipped in-place within @confs
    """

    cdef int maxnb = nbs[0].shape[0]
    cdef int nspins = confs.shape[0]
    cdef int slices = confs.shape[1]
    cdef int schedsize = A_sched.size
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int islice2 = 0
    cdef int ispin = 0
    cdef int si = 0
    cdef int si2 = 0
    cdef int spinidx = 0
    cdef int spinidx2 = 0
    cdef int sidx = 0
    cdef int sidx2 = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef double b_coeff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double teff = temp * float(slices)
    cdef double pi = np.pi
    cdef np.ndarray[np.float_t, ndim=2] ediffs = np.zeros((nspins, slices))
    cdef np.ndarray[np.int_t, ndim=1] ispins = np.arange(nspins)
    cdef int t = 0
    cdef int i = 0
    cdef int j = 0
    cdef int b = 0
    cdef int b2 = 0
    cdef int bslice = 0
    cdef int cslice = 0
    cdef np.ndarray[np.int_t, ndim=2] cluster = -np.ones((slices, 2), dtype=np.intc)
    cdef int k = 0
    cdef int stack = 1
    cdef int stackidx = 1
    cdef int cluster_count = 0
    cdef double r = 1.0
    cdef int tleft2 = 0
    cdef int tright2 = 0
    cdef double e_total = 0.0

    with nogil:
        for ifield in xrange(schedsize):
            # Calculate new coefficient for 1D Ising J
            jperp = -0.5*teff*clog(ctanh(A_sched[ifield]/teff))
            b_coeff = B_sched[ifield]
            for step in xrange(mcsteps):
                # Fisher-Yates shuffling algorithm
                for i in xrange(nspins):
                    ispins[i] = i
                for i in xrange(nspins, 0, -1):
                    j = crand() % i
                    t = ispins[i-1]
                    ispins[i-1] = ispins[j]
                    ispins[j] = t
                # start wolff updates
                for islice in xrange(slices):
                    for sidx2 in xrange(nspins):
                        e_total = 0.0
                        ispin = ispins[sidx2]
                        # islice = crand() % slices
                        cluster[0, 0] = ispin
                        cluster[0, 1] = islice
                        k = confs[ispin, islice]
                        confs[ispin, islice] *= -1
                        stack = 1
                        stackidx = 1
                        cluster_count = 0
                        r = 1.0
                        while True:
                            ispin = cluster[cluster_count, 0]
                            islice = cluster[cluster_count, 1]
                            # add energy
                            # loop through the neighbors
                            for si in xrange(maxnb):
                                # get the neighbor spin index
                                spinidx = int(nbs[ispin, si, 0])
                                # get the coupling value to that neighbor
                                jval = nbs[ispin, si, 1]
                                # self-connections are not quadratic
                                if spinidx == ispin:
                                    e_total += -2.0*b_coeff*float(k)*jval
                                else:
                                    e_total += -2.0*b_coeff*float(k)*(
                                        jval*float(confs[spinidx, islice])
                                    )
                            # periodic boundaries
                            if islice == 0:
                                tleft = slices-1
                                tright = 1
                            elif islice == slices-1:
                                tleft = slices-2
                                tright = 0
                            else:
                                tleft = islice-1
                                tright = islice+1
                            # now calculate between neighboring slices
                            e_total += 2.0*float(k)*(jperp*float(confs[ispin, tleft]))
                            e_total += 2.0*float(k)*(jperp*float(confs[ispin, tright]))
                            # system bath coupling
                            for b in xrange(1, slices):
                                bslice = (islice+b)%slices
                                # e_total += 2.0*teff*float(k*confs[ispin, bslice])*lookuptable[b-1]
                                if confs[ispin, bslice] == k:
                                    p = 1 - cexp(-2.0*lookuptable[b-1])
                                    if r*p > crand()/float(RAND_MAX):
                                        # add to cluster
                                        r *= p
                                        cluster[stackidx, 0] = ispin
                                        cluster[stackidx, 1] = bslice
                                        confs[ispin, bslice] *= -1
                                        stack += 1
                                        stackidx += 1
                            cluster_count += 1
                            stack += -1
                            if stack == 0:
                                break
                        if e_total > 0.0:
                            if 1 - cexp(-1.0*e_total/teff) > crand()/float(RAND_MAX):
                                # undo cluster move as increased system energy
                                for stackidx in xrange(cluster_count):
                                    confs[cluster[stackidx, 0], cluster[stackidx, 1]] *= -1

