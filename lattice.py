# lattice.py

import numpy as np
from numba import njit


@njit
def build_neighbour_lists(Lx, Ly):
    N = Lx * Ly
    nn = np.empty((N, 4), dtype=np.int32)
    nnn = np.empty((N, 4), dtype=np.int32)

    for iy in range(Ly):
        for ix in range(Lx):
            i = (ix % Lx) + Lx * (iy % Ly)

            nn[i, 0] = ((ix + 1) % Lx) + Lx * (iy % Ly)
            nn[i, 1] = ((ix - 1) % Lx) + Lx * (iy % Ly)
            nn[i, 2] = (ix % Lx) + Lx * ((iy + 1) % Ly)
            nn[i, 3] = (ix % Lx) + Lx * ((iy - 1) % Ly)

            nnn[i, 0] = ((ix + 1) % Lx) + Lx * ((iy + 1) % Ly)
            nnn[i, 1] = ((ix + 1) % Lx) + Lx * ((iy - 1) % Ly)
            nnn[i, 2] = ((ix - 1) % Lx) + Lx * ((iy + 1) % Ly)
            nnn[i, 3] = ((ix - 1) % Lx) + Lx * ((iy - 1) % Ly)

    return nn, nnn