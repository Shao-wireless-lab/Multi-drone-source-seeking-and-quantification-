#SrmEstimator.py
import numpy as np

#logger
import logging
logger = logging.getLogger(__name__)

class SrmEstimator:
    def __init__(self, grid_cells, true_particle, lambda_reg=1e-5, **grid_d):
        self.grid_cells_ = grid_cells #() Shouldn't this be the 25*25 kind?
        self.source_positions_ = np.array([cell.particles_[0].xs_ for cell in self.grid_cells_])
        self.n_ = len(self.grid_cells_)

        self.Q_ = np.zeros((self.n_, 1))
        self.P_ = (1/lambda_reg) * np.eye(self.n_)  # This initialization affects the later normalization of logdet(P)

        self.cached_srm_ = {}

        self.dx_ = grid_d['dx']
        self.dy_ = grid_d['dy']
        self.dz_ = grid_d['dz']

        self.true_particle_ = true_particle

        # self.ch4_bk_ = 1.98  #background ch4 concentration
        self.ch4_bk_ = 0  # background ch4 concentration
        # These are the quantities used for the ppm <-> molecules unit conversion; no conversion is needed for now, so they are commented out
        # self.M_ = 16.04      # g/mol,   grams per mole  methane
        # self.Pr_ = 101325     # Pa,       standard conditions
        # self.Temp_ = 273        # K,        standard conditions
        # self.Rg_ = 8.314      # J/mol*K   universal gas constant

        self.layer_conc_ = None
        self.layer_ws_ = None

    def setLayerData(self, layer_conc, layer_ws):
        self.layer_conc_ = layer_conc   # No deepcopy, memory usage would be too large
        self.layer_ws_ = layer_ws

    # Lawnmower (back-and-forth) pattern, reads all measurements (written for testing only)
    def getMeasurement(self, xm, wind):
        c = self.true_particle_.getModelOutput(xm, wind)
        c += 1e-8 * np.random.randn()
        self.measurements_ = c

    # Build the H matrix using the GP forward model
    def getSRMrow(self, xm, wind):
        """
        SRM vectorized version
        """
        # U = np.linalg.norm(wind) + 1e-8
        g = self.grid_cells_[0].particles_[0].getModelOutput(xm, wind, xs_override=self.source_positions_)
        # g = g / U
        return g.reshape(1, -1) # Convert to a np array row


    # Used to solve for Q; L2 is equation (18)
    # @ is matrix multiplication in linear algebra, * is element-wise multiplication
    def rlsUpdate(self, g, c):
        """
        Corresponds to the MATLAB L2 RLS:
        ri = 1 + gPg'
        ki = P g' / ri
        Q = Q + ki * (c - gQ)
        P = P - ki ki' * ri
        """
        g = g.reshape(1, -1)

        ri = 1 + g @ self.P_ @ g.T  # Denominator of the second term of eq 18
        ki = self.P_ @g.T / ri  # Left part of the second term of eq 18 as a whole
        ei = c - g @ self.Q_    # Residual, the right part of the second term of eq 18

        # The estimated Q has no physical dimension; it is a normalized source weight / source intensity coefficient.
        # The dimension is in fact already contained in g, consistent with the unit (ppm) of the final MOABS-generated data
        self.Q_ = self.Q_ + ki * ei # Overall update of eq 1
        self.P_ = self.P_ - ki @ ki.T * ri  # eq 17; multiplying by the transpose introduces an extra denominator, so multiply by ri to cancel it

    def step(self, x, c, wind):
        g = self.getSRMrow(x, wind)   # Update the SRM row
        self.rlsUpdate(g, c)               # Solve for Q
        return self.getState()

    def step_by_cache(self, x, c, wind):
        g = self.getSRMrow(x, wind)   # Update the SRM row
        self.rlsUpdate(g, c)               # Solve for Q
        return self.getState()

    def getState(self):
        """
        Summary for RL use
        """
        return {
            "Q_mean": np.mean(self.Q_),
            "Q_max": np.max(self.Q_),
            "uncertainty_trace": np.log(np.trace(self.P_) / self.n_ + 1e-12),   # The division by n is needed here because this is a large 25*25-dimensional matrix; it is not needed for W, which is only 2*2, and P ∝ W^(-1)
            "uncertainty_logdet": np.linalg.slogdet(self.P_)[1] / self.n_,  # For scale, turn it into the average log-eigenvalue
        }


    def getModelOutputLookup(self, xm, t):
        i = int(round(xm[0] / self.dx_))
        j = int(round(xm[1] / self.dy_))
        k = int(round(xm[2] / self.dz_)) -1 # Layer 0 is the 1 m layer, hence the -1; this differs from MATLAB

        i = max(0, min(99, i))
        j = max(0, min(99, j))
        k = max(0, min(4, k))

        # logger.info(f'in Lookup: i={i}, j={j}')
        c = self.layer_conc_[k, t, i, j]    # What is read here should already be ppm, no further conversion needed
        c -= self.ch4_bk_
        # c = (c - self.ch4_bk_) * (self.M_ * self.Pr_ / (self.Rg_ * self.Temp_ * 1e6)) #
        # logger.info(f'in Lookup, xm=[{xm[0]:.4f}, {xm[1]:.4f}], c={c:e}')

        ws = self.layer_ws_[k, t, i, j, :]
        output = np.concatenate((ws, np.array([c])), axis=0)    # The first three dimensions are ws, the fourth is conc
        return output

    def getMeasurementLookUp(self, xm, t):
        output = self.getModelOutputLookup(xm, t)
        # output[3] += 1e-8 * np.random.randn()   # Note: the measurement noise (eta) was already added during MOABS data generation, so do not add it again here; commented out
        self.measurements_ = output
        return output

    def cal_idx(self, xm, wind):
        i = int(round(xm[0] / self.dx_))
        j = int(round(xm[1] / self.dy_))
        k = int(round(xm[2] / self.dz_))

        uw = round(wind[0], 3)
        vw = round(wind[1], 3)

        return (i, j, k, uw, vw)


    #Important: Correct the concept of the SRM. The SRM should still be built with the forward model
    def cacheAllSrm(self, xm_all, wind):
        time_steps = min(len(xm_all), 6000)
        for i in range(time_steps):
            if i%1000 == 0:
                logger.info(f"cache SRM for step {i}")
            key = self.cal_idx(xm_all[i], wind[i, int(xm_all[i][0]/self.dx_-1),
                                int(xm_all[i][1]/self.dy_-1), 0:2])

            if key in self.cached_srm_:
                continue

            # After the inner loop ends, compute the SRM response over the whole cell. Do not divide by U here; divide outside
            g = self.getSRMrow(xm_all[i], np.array((key[3], key[4])))  # Use the GPM forward model to obtain the measurement
            self.cached_srm_[key] = np.array(g).reshape(1, -1)

    def getCachedSrm(self, xm, wind):
        key = self.cal_idx(xm, wind)
        if key in self.cached_srm_:
            return self.cached_srm_[key]   # Look up the SRM row in the table
        else:
            # Not cached before, so this is a new position+wind; recompute the SRM row
            u = wind
            U = np.linalg.norm(wind) + 1e-8  # Take the norm of the wind

            # After the inner loop ends, compute the SRM response over the whole cell. Do not divide by U here; divide outside
            g = self.getSRMrow(xm, u)  # Use the GPM forward model to obtain the measurement
            g = np.array(g).reshape(1, -1)
            self.cached_srm_[key] = g
            return g

    def step_use_cache(self, xm, c, wind):
        g = self.getCachedSrm(xm, wind)
        self.rlsUpdate(g, c)  # Solve for Q
        return self.getState()


    #For sanity check
    def getMeasurementAnalytic(self, xm, wind, Q_true, noise=0.0):
        # Generate the concentration using the ground-truth particle's forward model + the true Q
        # Note: getModelOutput returns the unitary (Q=1) response; multiplying by Q_true gives the true concentration
        c = Q_true * self.true_particle_.getModelOutput(xm, wind)
        c += noise * np.random.randn()
        output = np.concatenate((wind, np.array([c])), axis=0)  # [u, v, c], aligned with the lookup format
        return output