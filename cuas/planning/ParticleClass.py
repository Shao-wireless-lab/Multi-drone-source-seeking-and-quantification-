# ParticleClass.py

import numpy as np

class Particle:
    def __init__(self, source_pos, params):

        """
        params: (1,4) param
        """

        self.q_ = params[0]           # emission rate (Q)
        self.xs_ = np.asarray(params[1:3], dtype=float)          # source location [x,y]
        self.layer_ = params[3]       # plume layer (usually 0)


    def setMOABSConc(self, LayerConc):
        self.LayerConc_ = LayerConc

    # 10 s is more stable: x0 = 12.82, a2 = 0.1814, b2 = 0.6968, a3 = 0.0160, b3 = 1.7890,
    # 15 s is less stable: x0=10.04m,a2=0.1796,b2=0.7829,a3=0.0589,b3= 1.4113
    # 10000s corrected data:  x0=2.13, a2=0.3845, b2=0.5218, a3=0.4611, b3=0.8678
    def getModelOutput(self, xm, wind, xs_override=None, x0=2.13, use_modified_sigma=True):

        """
        xm: (3,)
        wind: (2,)
        xs_override:
            None -> use self.xs_
            (N,2) -> vectorized source positions
        """
        single_mode = False
        if xs_override is None:
            xs = self.xs_.reshape(1, 2)
            single_mode = True
        else:
            xs = xs_override

        U = np.linalg.norm(wind) + 1e-8

        theta = np.arctan2(wind[1], wind[0])

        dx = xm[0] - xs[:, 0]
        dy = xm[1] - xs[:, 1]

        x_rel = dx * np.cos(theta) + dy * np.sin(theta)

        y_rel = -dx * np.sin(theta) + dy * np.cos(theta)

        if use_modified_sigma:
            x_eff = x_rel + x0  # Shifted effective downwind distance; use this everywhere below, no longer x_rel
            g = np.zeros_like(x_rel)
            mask = x_eff > 1e-3  # Use the shifted distance to judge validity
            x_eff_valid = x_eff[mask]  # Key fix: filter x_eff with the mask, not x_rel
            y_valid = y_rel[mask]
            # 10000s corrected data:
            sigma_y = np.maximum(0.3845 * np.power(x_eff_valid, 0.5218), 1e-6)
            sigma_z = np.maximum(0.4611 * np.power(x_eff_valid, 0.8678), 1e-6)

        else:
            # Derek Original Setting
            x0 = 0.0  # The original version does not apply a shift
            x_eff = x_rel + x0  # Shifted effective downwind distance; use this everywhere below, no longer x_rel
            g = np.zeros_like(x_rel)
            mask = x_eff > 1e-3  # Use the shifted distance to judge validity
            x_eff_valid = x_eff[mask]  # Key fix: filter x_eff with the mask, not x_rel
            y_valid = y_rel[mask]

            sigma_y = np.maximum(0.1547 * np.power(x_eff_valid, 0.8932), 1e-6)
            sigma_z = np.maximum(0.1098 * np.power(x_eff_valid, 1.064), 1e-6)

        z = xm[2]
        H = 0
        Q = 1.0

        g_valid = (
                Q
                / (2 * np.pi * U * sigma_y * sigma_z)
                * np.exp(-0.5 * (y_valid / sigma_y) ** 2)
                * (np.exp(-0.5 * ((z - H) / sigma_z) ** 2)
                   + np.exp(-0.5 * ((z + H) / sigma_z) ** 2))
        )

        g[mask] = g_valid

        if single_mode:
            return g[0]
        return g


    # Not used to build the SRM; multiplies by Q to obtain the true simulated concentration
    def getTrueOutput(self, xm, u):
        return float(self.q_ * self.getModelOutput(xm, u))