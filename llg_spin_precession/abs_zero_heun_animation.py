import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

dt = 1e-15
end_time = 2e-10

γ = 1.7609e11
λ = 0.1

B = np.array([0.0, 0.0, 1.0], dtype=np.float64)
S = np.array([1.0, 0.0, 0.2], dtype=np.float64)
S /= np.linalg.norm(S)


def LLG_rhs(S, B, γ, λ):
    pref = -γ / (1.0 + λ * λ)

    S_cross_B = np.cross(S, B)
    S_cross_S_cross_B = np.cross(S, S_cross_B)

    return pref * (S_cross_B + λ * S_cross_S_cross_B)


def heun_step(S, B, dt, γ, λ):
    dS1 = LLG_rhs(S, B, γ, λ)

    S_tilde = S + dt * dS1
    S_tilde /= np.linalg.norm(S_tilde)

    dS2 = LLG_rhs(S_tilde, B, γ, λ)

    S_new = S + 0.5 * dt * (dS1 + dS2)
    S_new /= np.linalg.norm(S_new)

    return S_new


iterations = int(end_time / dt)
t_series = np.empty(iterations, dtype=np.float64)
S_series = np.empty((iterations, 3), dtype=np.float64)
norm_series = np.empty(iterations, dtype=np.float64)

for n in range(iterations):
    t_series[n] = n * dt
    S_series[n] = S
    norm_series[n] = np.linalg.norm(S)

    S = heun_step(S, B, dt, γ, λ)


def make_spin_animation(t_series, S_series, frame_stride=1000, fps=30):
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel(r"$S_x$")
    ax.set_ylabel(r"$S_y$")
    ax.set_zlabel(r"$S_z$")

    ax.plot(S_series[:, 0], S_series[:, 1], S_series[:, 2], linewidth=0.8, alpha=0.4)
    arrow = [ax.quiver(0, 0, 0, S_series[0, 0], S_series[0, 1], S_series[0, 2], length=1.0)]
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    frames = np.arange(0, len(t_series), frame_stride)

    def update(frame):
        arrow[0].remove()

        Sx = S_series[frame, 0]
        Sy = S_series[frame, 1]
        Sz = S_series[frame, 2]

        arrow[0] = ax.quiver(0, 0, 0, Sx, Sy, Sz, length=1.0)
        time_text.set_text(rf"$t = {t_series[frame]:.2e}\,\mathrm{{s}}$")

        return arrow[0], time_text

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save("single_spin_llg.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)

    return anim


anim = make_spin_animation(t_series, S_series, frame_stride=1000, fps=40)