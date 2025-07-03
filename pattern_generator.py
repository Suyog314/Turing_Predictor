
import numpy as np

def run_simulation(Du, Dv, F, k, dt=1.0, dx=1.0, steps=10000, size=128, num_seeds=None):
    """
    Simulates the Gray-Scott reaction-diffusion system with multiple initial origin points.

    Parameters:
        Du, Dv, F, k: Turing parameters
        dt: Time step
        dx: Grid spacing
        steps: Number of simulation steps
        size: Grid size (size x size)
        num_seeds: Number of random origin points. If None, chosen randomly in [2, 5].

    Returns:
        V: Final concentration of chemical V (for pattern image)
    """
    U = np.ones((size, size))
    V = np.zeros((size, size))

    r = size // 12  # radius of each seed patch
    if num_seeds is None:
        num_seeds = np.random.randint(2, 6)  # inclusive low, exclusive high

    for _ in range(num_seeds):
        cx = np.random.randint(r, size - r)
        cy = np.random.randint(r, size - r)
        U[cx - r:cx + r, cy - r:cy + r] = 0.50
        V[cx - r:cx + r, cy - r:cy + r] = 0.25

    # Add small noise
    U += 0.05 * np.random.rand(size, size)
    V += 0.05 * np.random.rand(size, size)

    def laplacian(Z):
        return (
            -4 * Z
            + np.roll(Z, 1, axis=0)
            + np.roll(Z, -1, axis=0)
            + np.roll(Z, 1, axis=1)
            + np.roll(Z, -1, axis=1)
        ) / (dx ** 2)

    for _ in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)
        uvv = U * V * V
        U += dt * (Du * Lu - uvv + F * (1 - U))
        V += dt * (Dv * Lv + uvv - (F + k) * V)
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

    return V

