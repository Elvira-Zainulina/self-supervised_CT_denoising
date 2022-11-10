import numpy as np


def make_low_dose(hdif, fd_prj, mas, sig, a=0.005332484102435619, b=-2.6879278304206555e-05):
    """
        Simulates a low-dose projection from the full-dose projection.

        Args:
            hdif: incident photon distribution of the full-dose projection;
            fd_prj: image of the full-dose projection (np.array of shape HxW);
            mas: the low-dose tube current to simulate;
            sig: electronic noise variance to simulate;
            a: slope coefficient characterizing connection between incident flux levels
            of the different tube currents;
            b: bias coefficient characterizing connection between incident flux levels
            of the different tube currents.
        """
    k = mas * a + b
    ldif0 = k * hdif[:, None] * np.exp(-fd_prj)
    ldif = ldif0.copy()
    ldif[ldif0 <= 0] += 1.0
    ldif = np.random.poisson(ldif).astype(np.float64)
    ldif[ldif0 <= 0] -= 1.0
    ldif += np.sqrt(sig) * np.random.normal(size=ldif.shape)
    ldif[ldif <= 0] = 1e-6
    li = -np.log(ldif / (hdif[:, None] * k))
    return li
