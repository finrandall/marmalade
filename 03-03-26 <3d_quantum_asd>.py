import time
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

# ---------- Constants ----------
    
μ = 9.274e-24  
λ = 0.0001
γ = 1.7609e11 
ħ = 1.054571817e-34
kB = 1.380649e-23
eV = 1.602176634e-19

# ---------- Temperature ----------

T = 80
kT = kB * T

# ---------- Hamiltonian Constants ----------

J = eV * 1e-2
K = 0

# ---------- Lattice size/indexing ----------

L = 128
Lz = 1
N = L * L * Lz

# ---------- Timestep and Endtime ----------

dt = 1e-15
stride = 1

burn_in_time = 1e-11
prod_time = 3e-11

# ---------- Analytic spectrum? ----------

def ω_analytic(x):
    
    return 4 * J * (1 - np.cos(x)) / ħ 

# ---------- Quantum Noise ----------

Γ = (λ * kT) / ((1.0 + λ*λ) * γ * μ)

def Φ(ω, x_max=708):
    x = ħ * np.abs(ω) / kT
    phi = np.empty_like(x, dtype=np.float32)

    phi[x == 0.0] = 1.0

    mid = (x > 0.0) & (x < x_max)
    phi[mid] = x[mid] / np.expm1(x[mid])

    phi[x >= x_max] = 0.0
    
    return phi

def generate_noise(iterations, N, dt, Γ, blockN=1024):

    ξ_mm = np.empty((iterations, N, 3), dtype=np.float32)
    
    ν = np.fft.rfftfreq(iterations, d=dt)
    ω = 2 * np.pi * ν
    phi = Φ(ω)

    shape = (np.sqrt((2.0 * Γ / dt)).astype(np.float32) * np.sqrt(phi).astype(np.float32))[:, None, None]

    for i0 in range(0, N, blockN):
        i1 = min(i0 + blockN, N)
        b = i1 - i0

        ξ = np.random.normal(0.0, 1.0, size=(iterations, b, 3)).astype(np.float32)

        Ξ = np.fft.rfft(ξ, axis=0)

        Ξ *= shape

        ξ_col = np.fft.irfft(Ξ, n=iterations, axis=0).astype(np.float32)

        ξ_mm[:, i0:i1, :] = ξ_col

    return ξ_mm

# ---------- Lattice Initialization ----------

rng = np.random.default_rng()

def init_spins(N, mode="aligned"):    
    if mode == "random":
        S = rng.normal(size=(N, 3)).astype(np.float64)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        return S
    
    elif mode == "aligned":
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (N, 1))

    else:
        raise ValueError("mode must be 'random' or 'aligned'")

# ---------- Nearest neighbour function ----------

def build_neighbour_list(L, Lz):
    N = L * L * Lz
    nl = np.empty((N, 6), dtype=np.int32)

    def idx(ix, iy, iz):
        return (ix % L) + L * ((iy % L) + L * iz)

    for iz in range(Lz):
        for iy in range(L):
            for ix in range(L):
                i = idx(ix, iy, iz)
                nl[i, 0] = idx(ix + 1, iy, iz)
                nl[i, 1] = idx(ix - 1, iy, iz)
                nl[i, 2] = idx(ix, iy + 1, iz)
                nl[i, 3] = idx(ix, iy - 1, iz)
                nl[i, 4] = idx(ix, iy, iz + 1) if (iz + 1 < Lz) else -1
                nl[i, 5] = idx(ix, iy, iz - 1) if (iz - 1 >= 0) else -1
    return nl

# ---------- Effective Field ----------
        
@njit
def H_eff(S, neighbour_list, H):
    N_local = S.shape[0]

    for i in range(N_local):
        hx = 0.0; hy = 0.0; hz = 0.0
        for j in range(neighbour_list.shape[1]):
            n = neighbour_list[i, j]
            if n >= 0:
                sj = S[n]
                hx += sj[0]; hy += sj[1]; hz += sj[2]
        H[i, 0] = J / μ * hx
        H[i, 1] = J / μ * hy
        H[i, 2] = J / μ * hz + 2.0 * K / μ * S[i, 2]

# ---------- LLG Equation ---------- 
 
@njit(parallel=True)
def LLG_rhs(S, H, out):
    pref = -γ / (1.0 + λ*λ)

    N_local = S.shape[0]
    for i in prange(N_local):
        sx = S[i, 0]; sy = S[i, 1]; sz = S[i, 2]
        hx = H[i, 0]; hy = H[i, 1]; hz = H[i, 2]

        ax = sy*hz - sz*hy
        ay = sz*hx - sx*hz
        az = sx*hy - sy*hx

        bx = sy*az - sz*ay
        by = sz*ax - sx*az
        bz = sx*ay - sy*ax

        out[i, 0] = pref * (ax + λ*bx)
        out[i, 1] = pref * (ay + λ*by)
        out[i, 2] = pref * (az + λ*bz)

# ---------- Heun Step ---------- 

@njit(parallel=True)
def LLG_heun_step(S, neighbour_list, Γ, r, H_ex, H_ex_tilde, dS1, dS2, S_tilde, S_new, H_tot, H_tot_tilde):
    H_eff(S, neighbour_list, H_ex)

    for i in prange(S.shape[0]):
        H_tot[i,0] = H_ex[i,0] + Γ * r[i,0]
        H_tot[i,1] = H_ex[i,1] + Γ * r[i,1]
        H_tot[i,2] = H_ex[i,2] + Γ * r[i,2]

    LLG_rhs(S, H_tot, dS1)

    for i in prange(S.shape[0]):
        S_tilde[i,0] = S[i,0] + dt*dS1[i,0]
        S_tilde[i,1] = S[i,1] + dt*dS1[i,1]
        S_tilde[i,2] = S[i,2] + dt*dS1[i,2]

    for i in prange(S.shape[0]):
        x,y,z = S_tilde[i,0], S_tilde[i,1], S_tilde[i,2]
        inv = 1.0 / np.sqrt(x*x + y*y + z*z)
        S_tilde[i,0] = x*inv; S_tilde[i,1] = y*inv; S_tilde[i,2] = z*inv

    H_eff(S_tilde, neighbour_list, H_ex_tilde)

    for i in prange(S.shape[0]):
        H_tot_tilde[i,0] = H_ex_tilde[i,0] + Γ * r[i,0]
        H_tot_tilde[i,1] = H_ex_tilde[i,1] + Γ * r[i,1]
        H_tot_tilde[i,2] = H_ex_tilde[i,2] + Γ * r[i,2]

    LLG_rhs(S_tilde, H_tot_tilde, dS2)

    for i in prange(S.shape[0]):
        S_new[i,0] = S[i,0] + 0.5*dt*(dS1[i,0] + dS2[i,0])
        S_new[i,1] = S[i,1] + 0.5*dt*(dS1[i,1] + dS2[i,1])
        S_new[i,2] = S[i,2] + 0.5*dt*(dS1[i,2] + dS2[i,2])

    for i in prange(S.shape[0]):
        x,y,z = S_new[i,0], S_new[i,1], S_new[i,2]
        inv = 1.0 / np.sqrt(x*x + y*y + z*z)
        S_new[i,0] = x*inv; S_new[i,1] = y*inv; S_new[i,2] = z*inv

# ---------- Evolution ----------

burn_in_steps = int(burn_in_time / dt)
iterations = int((burn_in_time + prod_time) / dt)

rem = iterations - burn_in_steps

nsamp = 0 if rem <= 0 else (rem + stride - 1) // stride

dτ = stride * dt

@njit
def evolve(S, neighbour_list, ξ_series):
    Sx_samp = np.zeros((nsamp, N), dtype=np.float32)
    Sy_samp = np.zeros((nsamp, N), dtype=np.float32)
    Sz_samp = np.zeros((nsamp, N), dtype=np.float32)

    H_ex = np.empty((N, 3), dtype=np.float64)
    H_ex_tilde = np.empty((N, 3), dtype=np.float64)
    dS1 = np.empty((N, 3), dtype=np.float64)
    dS2 = np.empty((N, 3), dtype=np.float64)
    S_tilde = np.empty((N, 3), dtype=np.float64)
    S_new = np.empty((N, 3), dtype=np.float64)
    H_tot = np.empty((N, 3), dtype=np.float64)
    H_tot_tilde = np.empty((N, 3), dtype=np.float64)

    count = 0
    for step in range(iterations):
        LLG_heun_step(S, neighbour_list, Γ, ξ_series[step],
                      H_ex, H_ex_tilde, dS1, dS2, S_tilde, S_new, H_tot, H_tot_tilde)

        for i in range(N):
            S[i, 0] = S_new[i, 0]
            S[i, 1] = S_new[i, 1]
            S[i, 2] = S_new[i, 2]

        if step >= burn_in_steps and ((step - burn_in_steps) % stride == 0):
            if count < nsamp:
                for i in range(N):
                    Sx_samp[count, i] = S[i, 0]
                    Sy_samp[count, i] = S[i, 1]
                    Sz_samp[count, i] = S[i, 2]
                count += 1

    return Sx_samp, Sy_samp, Sz_samp

# ---------- Simulation ----------

start_time = time.time()

def main():
    
    S = init_spins(N, mode="aligned")
    
    neighbour_list = build_neighbour_list(L, Lz)
    
    ξ_series = generate_noise(iterations, N, dt, Γ, blockN=256)
    
    Sx, Sy, Sz = evolve(S, neighbour_list, ξ_series)
    
    φ = Sx + 1j * Sy
    
    fs = 1.0 / dτ          
    nperseg = nsamp // 12
    noverlap = nperseg // 2
    
    φ_zyx = φ.reshape(nsamp, Lz, L, L)
    φ_k = np.fft.fftn(φ_zyx, axes=(1, 2, 3), norm="ortho")
    φ_k0 = φ_k - φ_k.mean(axis=0, keepdims=True)
    
    a = 1.0
    
    kx = 2 * np.pi * np.fft.fftfreq(L, d=a)
    kx_s = np.fft.fftshift(kx)
    
    f, Pxx = welch(φ_k0, fs=fs, axis=0, window="hann", nperseg = nperseg, noverlap = noverlap, detrend=False, return_onesided=False, scaling="spectrum")
    
    ω = 2 * np.pi * f
    ω_s = np.fft.fftshift(ω)
    
    S_kω_s = np.fft.fftshift(Pxx, axes=(0,))
    
    S_kω_ss = np.fft.fftshift(S_kω_s, axes=(1, 2, 3))
    
    S_ω_kx = S_kω_ss[:, Lz//2, L//2, :]
    
    analytic = ω_analytic(kx)
    
    ω_max = 1.1 * max(analytic)
    ω_min = -0.1 * max(analytic)
    
    mask = (ω_s > ω_min) & (ω_s < ω_max)
    
    S_cut = S_ω_kx[mask, :]
    ω_cut = ω_s[mask]
    
    eps = 1e-30
    I = np.log(S_cut + eps)
    
    # ---------- Plots ----------
    
    plt.figure(figsize=(7,5))
    plt.imshow(I, origin="lower", aspect="auto", cmap='PuBu_r', extent=[kx_s[0], kx_s[-1], ω_cut[0], ω_cut[-1]])
    plt.plot(kx_s, ω_analytic(kx_s), color="red", linestyle="--", linewidth=0.6)
    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$\omega$")
    plt.colorbar(label=r"$\ln[S(k_x,\omega)]$")
    plt.tight_layout()
    plt.margins(0)
    plt.show()
    
main()

# ---------- Timer ----------

end_time = time.time()

time_array = (burn_in_steps + np.arange(nsamp) * stride) * dt

elapsed = float(end_time) - float(start_time)

h = int(elapsed // 3600)
m = int((elapsed % 3600) // 60)
s = elapsed % 60

print(f"Elapsed: {h:02d}:{m:02d}:{s:05.2f}")
