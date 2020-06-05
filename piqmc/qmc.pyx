# encoding: utf-8
# cython: profile=False
# filename: qmc.pyx

cimport cython
import numpy as np
cimport numpy as np
# cimport openmp
from cython.parallel import prange
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
# from libc.stdio cimport printf as cprintf


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnneal(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
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
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @alpha (float): system bath coupling
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	    # Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(A_sched[ifield]/(slices*temp)))
        for step in range(mcsteps):
            # Loop over Trotter slices local updates
            for islice in range(slices):
                # Loop over spins
                for sidx in sidx_shuff:
                    # loop through the neighbors
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
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
                    ediff += 2.0*float(confs[sidx, islice])*(jperp*float(confs[sidx, tleft]))
                    ediff += 2.0*float(confs[sidx, islice])*(jperp*float(confs[sidx, tright]))
                    # Accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                    # reset energy diff
                    ediff = 0.0
                sidx_shuff = rng.permutation(sidx_shuff)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealExplicitBath(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    float alpha,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
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
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @alpha (float): system bath coupling
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int bslice = 1
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef int k = 0
    cdef double ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double pi = np.pi
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(A_sched[ifield]/(slices*temp)))
        # cn = cpow(csqrt(0.5 * csinh(2 * A_sched[ifield]/(slices*temp))), nspins)
        for step in range(mcsteps):
            # Loop over Trotter slices local updates
            for islice in range(slices):
                # Loop over spins
                for sidx in sidx_shuff:
                    # loop through the neighbors
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
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
                    ediff += 2.0*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tleft]))
                    ediff += 2.0*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tright]))
                   # system bath coupling
                    for k in range(1, slices):
                        bslice = (islice + k) % slices
                        ediff += 2.0*alpha*temp*slices*float(confs[sidx, islice]*confs[sidx, bslice]) * cpow(
                            pi / (csin(pi * float(cabs(bslice - islice)) / float(slices)) * float(slices)), 2
                        )
                    if ediff <= 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                    # reset energy diff
                    ediff = 0.0
                sidx_shuff = rng.permutation(sidx_shuff)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealGlobal(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single and global spin flips.
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef int k = 0
    cdef double ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(A_sched[ifield]/(slices*temp)))
        # cn = cpow(csqrt(0.5 * csinh(2 * A_sched[ifield]/(slices*temp))), nspins)
        for step in range(mcsteps):
            # Loop over Trotter slices local updates
            for islice in range(slices):
                # Loop over spins
                for sidx in sidx_shuff:
                    # loop through the neighbors
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
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
                    ediff += 2.0*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tleft]))
                    ediff += 2.0*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tright]))
                    # Accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                    # reset energy diff
                    ediff = 0.0
                sidx_shuff = rng.permutation(sidx_shuff)
            # global moves along trotter direction
            for sidx in sidx_shuff:
                for islice in range(slices):
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
                                jval*confs[spinidx, islice]
                            )
                # Accept or reject
                if ediff <= 0.0:  # avoid overflow
                    for islice in range(slices):
                        confs[sidx, islice] *= -1
                elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                    for islice in range(slices):
                        confs[sidx, islice] *= -1
                # reset energy diff
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealExplicitBathGlobal(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    float alpha,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single and global spin flips.
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
        @alpha (float): system bath coupling
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int bslice = 1
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef int k = 0
    cdef double ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef double pi = np.pi
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(A_sched[ifield]/(slices*temp)))
        # cn = cpow(csqrt(0.5 * csinh(2 * A_sched[ifield]/(slices*temp))), nspins)
        for step in range(mcsteps):
            # Loop over Trotter slices local updates
            for islice in range(slices):
                # Loop over spins
                for sidx in sidx_shuff:
                    # loop through the neighbors
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
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
                    ediff += 2.0*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tleft]))
                    ediff += 2.0*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tright]))
                    # system bath coupling
                    for k in range(1, slices):
                        bslice = (islice + k) % slices
                        ediff += 2.0*alpha*temp*slices*float(confs[sidx, islice] * confs[sidx, bslice]) * cpow(
                            pi / (csin(pi * float(cabs(bslice - islice)) / float(slices)) * float(slices)), 2
                        )
                    # Accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                    # reset energy diff
                    ediff = 0.0
                sidx_shuff = rng.permutation(sidx_shuff)
           # global moves along trotter direction
            for sidx in sidx_shuff:
                for islice in range(slices):
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
                                jval*confs[spinidx, islice]
                            )
                # Accept or reject
                if ediff <= 0.0:  # avoid overflow
                    for islice in range(slices):
                        confs[sidx, islice] *= -1
                elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                    for islice in range(slices):
                        confs[sidx, islice] *= -1
                # reset energy diff
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealSWCluster(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single and Swendsen-Wang cluster spin flips.
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
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @alpha (float): system bath coupling
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int bslice = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef double ebond = 0.0
    cdef double ediff = 0.0
    cdef int nsites = nspins * slices
    cdef int spin_nb = 0
    cdef int ic = 0
    cdef int isite = 0
    cdef int isite_nb = 0
    cdef int cspin = 0
    cdef int cslice = 0
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=3] cluster_bonds = -np.ones((nsites, nsites, 2), dtype=np.intc)
    cdef np.ndarray[np.int_t, ndim=1] cluster_sizes = np.zeros(nsites, dtype=np.intc)
    cdef cluster_count = 0
    cdef double pi = np.pi
    cdef np.ndarray[np.int_t, ndim=1] ispins = rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(A_sched[ifield]/(slices*temp)))
        # cn = cpow(csqrt(0.5 * csinh(2 * A_sched[ifield]/(slices*temp))), nspins)
        for step in range(mcsteps):
            # Loop over Trotter slices local updates
            # build clusters
            for islice in range(slices):
                for ispin in rng.permutation(ispins):
                    # try place spin in an existing cluster
                    for ic in rng.permutation(range(cluster_count)):
                        # check if cluster is of same spin
                        if confs[ispin, islice] == confs[cluster_bonds[ic, 0, 0], cluster_bonds[ic, 0, 1]]:
                            # try to bond to a site in the cluster
                            for isite in rng.permutation(range(cluster_sizes[ic])):
                                ebond = 0.0
                                cspin = cluster_bonds[ic, isite, 0]
                                cslice = cluster_bonds[ic, isite, 1]
                                # check if along the same trotter slice
                                if cspin == ispin:
                                    if cabs(cslice - islice) == 1:
                                        if 1.0 - cexp(-2.0*jperp / (temp * slices)) > crand() / float(RAND_MAX):
                                            cluster_bonds[ic, cluster_sizes[ic], 0] = ispin
                                            cluster_bonds[ic, cluster_sizes[ic], 1] = islice
                                            placed = True
                                            cluster_sizes[ic] += 1
                                            break
                                # check if adjacent in space dim
                                elif cslice == islice:
                                    for spin_nb in range(maxnb):
                                        if nbs[ispin, spin_nb, 0] == cspin:
                                            ebond = B_sched[ifield]*nbs[ispin, spin_nb, 1]
                                            break
                                    if ebond > 0.0:
                                        if 1 - cexp(-2.0 * ebond / (temp*slices)) > crand() / float(RAND_MAX):
                                            cluster_bonds[ic, cluster_sizes[ic], 0] = ispin
                                            cluster_bonds[ic, cluster_sizes[ic], 1] = islice
                                            placed = True
                                            cluster_sizes[ic] += 1
                                            break
                            # break this loop as well if placed in a cluster
                            if placed:
                                break
                    # start a new cluster
                    if not placed:
                        cluster_bonds[cluster_count, 0, 0] = ispin
                        cluster_bonds[cluster_count, 0, 1] = islice
                        cluster_sizes[cluster_count] += 1
                        cluster_count += 1
                        ediff = 0.0
                    else:
                        # reset if placed
                        placed = False
            # Flip clusters with prob 1/2
            for ic in range(cluster_count):
                if 0.5 > crand()/float(RAND_MAX):
                    for isite in range(cluster_sizes[ic]):
                        confs[cluster_bonds[ic, isite, 0], cluster_bonds[ic, isite, 1]] *= -1
            # reset clusters
            cluster_count = 0
            cluster_sizes[:] = 0
            # single spin flips
            for islice in range(slices):
                for ispin in rng.permutation(ispins):
                    # calculate energy of flipping the spin
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ispin, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[ispin, si, 1]
                        # self-connections are not quadratic
                        if spinidx == ispin:
                            ediff += -2.0 * B_sched[ifield] * float(confs[ispin, islice]) * jval
                        # only account for bonds not being flipped with the cluster
                        else:
                            ediff += -2.0 * B_sched[ifield] * float(confs[ispin, islice]) * (
                                jval * float(confs[spinidx, islice]))
                    # periodic boundaries
                    if islice == 0:
                        tleft = slices - 1
                        tright = 1
                    elif islice == slices - 1:
                        tleft = slices - 2
                        tright = 0
                    else:
                        tleft = islice - 1
                        tright = islice + 1
                    # now calculate between neighboring slices
                    ediff += 2.0 * float(confs[ispin, islice]) * (jperp * float(confs[ispin, tleft]))
                    ediff += 2.0 * float(confs[ispin, islice]) * (jperp * float(confs[ispin, tright]))
                    # Accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        confs[ispin, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[ispin, islice] *= -1
                    # reset energy diff
                    ediff = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealExplicitBathSWCluster(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    float alpha,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single and Swendsen-Wang cluster spin flips.
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
        @alpha (float): system bath coupling
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int bslice = 0
    cdef int k = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef double ebond = 0.0
    cdef double ediff = 0.0
    cdef int nsites = nspins * slices
    cdef int spin_nb = 0
    cdef int ic = 0
    cdef int isite = 0
    cdef int isite_nb = 0
    cdef int cspin = 0
    cdef int cslice = 0
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=3] cluster_bonds = -np.ones((nsites, nsites, 2), dtype=np.intc)
    cdef np.ndarray[np.int_t, ndim=1] cluster_sizes = np.zeros(nsites, dtype=np.intc)
    cdef cluster_count = 0
    cdef double pi = np.pi
    cdef np.ndarray[np.int_t, ndim=1] ispins = rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(A_sched[ifield]/(slices*temp)))
        # cn = cpow(csqrt(0.5 * csinh(2 * A_sched[ifield]/(slices*temp))), nspins)
        for step in range(mcsteps):
            # Loop over Trotter slices local updates
            # build cluster
            for islice in range(slices):
                for ispin in rng.permutation(ispins):
                    # try place spin in an existing cluster
                    for ic in rng.permutation(range(cluster_count)):
                        # check if cluster is of same spin
                        if confs[ispin, islice] == confs[cluster_bonds[ic, 0, 0], cluster_bonds[ic, 0, 1]]:
                            # try to bond to a site in the cluster
                            for isite in rng.permutation(range(cluster_sizes[ic])):
                                ebond = 0.0
                                cspin = cluster_bonds[ic, isite, 0]
                                cslice = cluster_bonds[ic, isite, 1]
                                # check if along the same trotter slice
                                if cspin == ispin:
                                    if cabs(cslice - islice) == 1:
                                        ebond += jperp / (temp * slices)
                                    ebond += alpha * cpow(pi / (csin(pi * float(cabs(cslice - islice)) / float(slices))
                                        * float(slices)), 2)

                                    if 1.0 - cexp(-2.0*ebond) > crand() / float(RAND_MAX):
                                        cluster_bonds[ic, cluster_sizes[ic], 0] = ispin
                                        cluster_bonds[ic, cluster_sizes[ic], 1] = islice
                                        placed = True
                                        cluster_sizes[ic] += 1
                                        break
                                # check if adjacent in space dim
                                elif cslice == islice:
                                    for spin_nb in range(maxnb):
                                        if nbs[ispin, spin_nb, 0] == cspin:
                                            ebond = B_sched[ifield]*nbs[ispin, spin_nb, 1]
                                            break
                                    if ebond > 0.0:
                                        if 1 - cexp(-2.0 * ebond / (temp*slices)) > crand() / float(RAND_MAX):
                                            cluster_bonds[ic, cluster_sizes[ic], 0] = ispin
                                            cluster_bonds[ic, cluster_sizes[ic], 1] = islice
                                            placed = True
                                            cluster_sizes[ic] += 1
                                            break
                            # break this loop as well if placed in a cluster
                            if placed:
                                break
                    # start a new cluster
                    if not placed:
                        cluster_bonds[cluster_count, 0, 0] = ispin
                        cluster_bonds[cluster_count, 0, 1] = islice
                        cluster_sizes[cluster_count] += 1
                        cluster_count += 1
                        ebond = 0.0
                    else:
                        # reset if placed
                        ebond = 0.0
                        placed = False
            # Flip clusters with prob 1/2
            for ic in range(cluster_count):
                if 0.5 > crand()/float(RAND_MAX):
                    for isite in range(cluster_sizes[ic]):
                        confs[cluster_bonds[ic, isite, 0], cluster_bonds[ic, isite, 1]] *= -1
            # reset clusters
            cluster_count = 0
            cluster_sizes[:] = 0
            # single spin flips
            for islice in range(slices):
                for ispin in rng.permutation(ispins):
                    # calculate energy of flipping the spin
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[ispin, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[ispin, si, 1]
                        # self-connections are not quadratic
                        if spinidx == ispin:
                            ediff += -2.0 * B_sched[ifield] * float(confs[ispin, islice]) * jval
                        # only account for bonds not being flipped with the cluster
                        else:
                            ediff += -2.0 * B_sched[ifield] * float(confs[ispin, islice]) * (
                                jval * float(confs[spinidx, islice]))
                    # periodic boundaries
                    if islice == 0:
                        tleft = slices - 1
                        tright = 1
                    elif islice == slices - 1:
                        tleft = slices - 2
                        tright = 0
                    else:
                        tleft = islice - 1
                        tright = islice + 1
                    # now calculate between neighboring slices
                    ediff += 2.0 * float(confs[ispin, islice]) * (jperp * float(confs[ispin, tleft]))
                    ediff += 2.0 * float(confs[ispin, islice]) * (jperp * float(confs[ispin, tright]))
                     # system bath coupling
                    for k in range(1, slices):
                        bslice = (islice + k) % slices
                        ediff += 2.0*alpha*temp*slices*float(confs[ispin, islice] * confs[ispin, bslice]) * cpow(
                            pi / (csin(pi * float(cabs(bslice - islice)) / float(slices)) * float(slices)), 2
                        )
                    # Accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        confs[ispin, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[ispin, islice] *= -1
                    # reset energy diff
                    ediff = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealWolffCluster(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single and Wolff cluster spin flips.
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
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @alpha (float): system bath coupling
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int bslice = 0
    cdef int si = 0
    cdef int sidx = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef int nsites = nspins * slices
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=2] cluster= -np.ones((nsites, 2), dtype=np.intc)
    cdef np.ndarray[np.int_t, ndim=1] coord = np.zeros(2, dtype=np.intc)
    cdef int stack = 0
    cdef int stackidx = 0
    cdef int cluster_count = 0
    cdef int coordidx = 0
    cdef np.ndarray[np.int_t, ndim=1] register = np.zeros(nsites, dtype=np.intc)
    cdef double pi = np.pi
    cdef np.ndarray[np.int_t, ndim=1] ispins = rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*clog(ctanh(A_sched[ifield]/(slices*temp)))
        # cn = cpow(csqrt(0.5 * csinh(2 * A_sched[ifield]/(slices*temp))), nspins)
        for step in range(mcsteps):
            # single Wolff cluster update before local updates
            # pick random start point
            cluster[0, 0] = rng.permutation(ispins)[0]
            cluster[0, 1] = rng.permutation(range(slices))[0]
            register[0] = cluster[0, 1] * nspins + cluster[0, 0]
            stack = 1
            stackidx = 1
            cluster_count = 0
            while stack:
                ispin = cluster[cluster_count, 0]
                islice = cluster[cluster_count, 1]
                # add neighbours to stack
                for si in range(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[ispin, si, 0])
                    coordidx = islice * nspins + spinidx
                    # check if already in the cluster
                    if not np.isin(coordidx, register):
                        # check if spins aligned
                        if confs[ispin, islice] == confs[spinidx, islice]:
                            # get the coupling value to that neighbor
                            jval = nbs[ispin, si, 1]
                            if jval < 0:
                                if 1 - cexp(2.0 * B_sched[ifield] * jval / temp) > crand()/float(RAND_MAX):
                                    # add to cluster
                                    cluster[stackidx, 0] = spinidx
                                    cluster[stackidx, 1] = islice
                                    register[stackidx] = coordidx
                                    stack += 1
                                    stackidx += 1
                # periodic boundaries
                if islice == 0:
                    tleft = slices - 1
                    tright = 1
                elif islice == slices - 1:
                    tleft = slices - 2
                    tright = 0
                else:
                    tleft = islice - 1
                    tright = islice + 1
                # now calculate between neighboring slices
                coordidx = tleft * nspins + ispin
                if not np.isin(coordidx, register):
                    if confs[ispin, tleft] == confs[ispin, islice]:
                        if 1 - cexp(-2.0 * jperp) > crand()/float(RAND_MAX):
                            cluster[stackidx, 0] = ispin
                            cluster[stackidx, 1] = tleft
                            register[stackidx] = coordidx
                            stack += 1
                            stackidx += 1
                coordidx = tright * nspins + ispin
                if not np.isin(coordidx, register):
                    if confs[ispin, tright] == confs[ispin, islice]:
                        if 1 - cexp(-2.0 * jperp) > crand()/float(RAND_MAX):
                            cluster[stackidx, 0] = ispin
                            cluster[stackidx, 1] = tright
                            register[stackidx] = coordidx
                            stack += 1
                            stackidx += 1
                cluster_count += 1
                stack -= 1
            # Flip cluster
            for coord in cluster[:cluster_count, :]:
                confs[coord[0], coord[1]] *= -1
            # reset clusters
            cluster_count = 0

            # single spin flip updates
            for islice in range(slices):
                # Loop over spins
                for sidx in ispins:
                    # loop through the neighbors
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
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
                    ediff += 2.0*slices*temp*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tleft]))
                    ediff += 2.0*slices*temp*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tright]))
                    # Accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                    # reset energy diff
                    ediff = 0.0
                sidx_shuff = rng.permutation(ispins)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnnealExplicitBathWolffCluster(np.float_t[:] A_sched,
                    np.float_t[:] B_sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    float alpha,
                    int nspins,
                    np.int_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    This implementation includes single and Wolff cluster spin flips.
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
        @alpha (float): system bath coupling
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
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @confs
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef double jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int ispin = 0
    cdef int bslice = 0
    cdef int k = 0
    cdef int si = 0
    cdef int sidx = 0
    cdef int spinidx = 0
    cdef double jval = 0.0
    cdef double ediff = 0.0
    cdef int nsites = nspins * slices
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=2] cluster= -np.ones((nsites, 2), dtype=np.intc)
    cdef np.ndarray[np.int_t, ndim=1] coord = np.zeros(2, dtype=np.intc)
    cdef int stack = 0
    cdef int stackidx = 0
    cdef int cluster_count = 0
    cdef double pi = np.pi
    cdef int coordidx = 0
    cdef np.ndarray[np.int_t, ndim=1] register = np.zeros(nsites, dtype=np.intc)
    cdef np.ndarray[np.int_t, ndim=1] ispins = rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in range(A_sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*clog(ctanh(A_sched[ifield]/(slices*temp)))
        # cn = cpow(csqrt(0.5 * csinh(2 * A_sched[ifield]/(slices*temp))), nspins)
        for step in range(mcsteps):
            # single Wolff cluster update
            # pick random start point
            cluster[0, 0] = rng.permutation(ispins)[0]
            cluster[0, 1] = rng.permutation(range(slices))[0]
            register[0] = cluster[0, 1] * nspins + cluster[0, 0]
            stack = 1
            stackidx = 1
            cluster_count = 0
            while stack:
                ispin = cluster[cluster_count, 0]
                islice = cluster[cluster_count, 1]
                # add neighbours to stack
                for si in range(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[ispin, si, 0])
                    coordidx = islice * nspins + spinidx
                    if not np.isin(coordidx, register):
                        # is same spin
                        if spinidx != ispin:
                            if confs[ispin, islice] == confs[spinidx, islice]:
                                # get the coupling value to that neighbor
                                jval = nbs[ispin, si, 1]
                                if jval < 0:
                                    if 1 - cexp(2.0*B_sched[ifield]*jval/(temp*slices)) > crand()/float(RAND_MAX):
                                        cluster[stackidx, 0] = spinidx
                                        cluster[stackidx, 1] = islice
                                        register[stackidx] = coordidx
                                        stack += 1
                                        stackidx += 1
                # periodic boundaries
                if islice == 0:
                    tleft = slices - 1
                    tright = 1
                elif islice == slices - 1:
                    tleft = slices - 2
                    tright = 0
                else:
                    tleft = islice - 1
                    tright = islice + 1
               # now calculate between neighboring slices
                coordidx = tleft * nspins + ispin
                if not np.isin(coordidx, register):
                    if confs[ispin, tleft] == confs[ispin, islice]:
                        if 1 - cexp(-2.0 * jperp) > crand()/float(RAND_MAX):
                            cluster[stackidx, 0] = ispin
                            cluster[stackidx, 1] = tleft
                            register[stackidx] = coordidx
                            stack += 1
                            stackidx += 1
                coordidx = tright * nspins + ispin
                if not np.isin(coordidx, register):
                    if confs[ispin, tright] == confs[ispin, islice]:
                        if 1 - cexp(-2.0 * jperp) > crand()/float(RAND_MAX):
                            cluster[stackidx, 0] = ispin
                            cluster[stackidx, 1] = tright
                            register[stackidx] = coordidx
                            stack += 1
                            stackidx += 1
                # system bath coupling
                for bslice in range(1, slices):
                    coordidx = bslice * nspins + ispin
                    if not np.isin(coordidx, register):
                        if confs[ispin, bslice] == confs[ispin, islice]:
                            ediff = alpha * cpow(pi/(slices*csin(pi * float(cabs(bslice - islice)) / float(slices))), 2)
                            if 1 - cexp(-2.0 * ediff) > crand()/float(RAND_MAX):
                                cluster[stackidx, 0] = ispin
                                cluster[stackidx, 1] = bslice
                                register[stackidx] = coordidx
                                stack += 1
                                stackidx += 1
                            ediff = 0.0
                cluster_count += 1
                stack -= 1
            # Flip clusters
            for coord in cluster[:cluster_count, :]:
                confs[coord[0], coord[1]] *= -1
            # reset clusters
            cluster_count = 0

            # single spin flip updates
            for islice in range(slices):
                # Loop over spins
                for sidx in ispins:
                    # loop through the neighbors
                    for si in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*jval
                        else:
                            ediff += -2.0*B_sched[ifield]*float(confs[sidx, islice])*(
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
                    ediff += 2.0*slices*temp*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tleft]))
                    ediff += 2.0*slices*temp*float(confs[sidx, islice])*(
                        jperp*float(confs[sidx, tright]))
                   # system bath coupling
                    for k in range(1, slices):
                        bslice = (islice + k) % slices
                        ediff += 2.0*alpha*temp*slices*float(confs[sidx, islice] * confs[sidx, bslice]) * cpow(
                            pi / (csin(pi * float(cabs(bslice - islice)) / float(slices)) * float(slices)), 2
                        )
                    # Accept or reject
                    if ediff <= 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(-1.0 * ediff/(temp*slices)) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                    # reset energy diff
                    ediff = 0.0
                sidx_shuff = rng.permutation(ispins)