import copy

import numpy as np
from SrmEstimator import SrmEstimator

class EmgrEstimator():
    def __init__(self, srm_estimator, dt=1.0, d=2.0, ap=1.0, bp=1e-13, ew=1e-16):
        self.srm_ = srm_estimator
        self.dt_ = dt
        self.d_ = d

        self.W_gamma_ = 0.99   #forgetting factor of W, otherwise, W will always increase.
                                #1/(1-gamma)=approximated window size，when 0.995 it's 200 steps' window,0.98=50steps，0.95 approximately equals to 20 steps

        self.ap_ = ap
        self.bp_ = bp
        self.ew_ = ew

        self.W_ = np.zeros((2, 2))

        self.score_history_ = []
        self.det_history_ = []
        self.min_svd_history_ = []

    #phi is essentially the gradient of current concentration field, here already use Y^=HQ^
    def compute_gradient(self, xm, wind):
        d = self.d_
        Q = self.srm_.Q_
        U = np.linalg.norm(wind) + 1e-12

        #x_direction
        g_x_plus = self.srm_.getCachedSrm(xm + np.array([d, 0, 0]), wind)
        g_x_minus = self.srm_.getCachedSrm(xm - np.array([d, 0, 0]), wind)

        #y_direction
        g_y_plus = self.srm_.getCachedSrm(xm + np.array([0, d, 0]), wind)
        g_y_minus = self.srm_.getCachedSrm(xm - np.array([0, d, 0]), wind)

        phi_x = ((g_x_plus - g_x_minus) @ Q)
        phi_y = ((g_y_plus - g_y_minus) @ Q)

        phi = np.array([phi_x.item(), phi_y.item()])

        return phi

    def calculate_W(self, phi, update_W=False):
        phi = phi.reshape(2, 1)
        W = copy.deepcopy(self.W_)
        # W += (phi @ phi.T) * self.dt_ / (4 * self.d_ ** 2)
        W = (self.W_gamma_ * W + (phi @ phi.T) * self.dt_ / (4 * self.d_ ** 2))

        if update_W:
            self.W_ = W
        return W

    def update_W(self, W):
        self.W_ = W

    def compute_score(self, W, record_met=False):
        # W = self.W_

        ew = self.ew_

        detW = np.linalg.det(W + ew * np.eye(2))

        min_svd = np.min(np.linalg.svd(W + ew * np.eye(2), compute_uv=False))

        score = (-self.ap_ * np.log(detW) + self.bp_ / min_svd)

        # record some metric
        if record_met:
            self.score_history_.append(score)
            self.det_history_.append(detW)
            self.min_svd_history_.append(min_svd)

        return score

    def step(self, xm, wind, update_W=False, record_met=False):
        phi = self.compute_gradient(xm, wind)

        W = self.calculate_W(phi, update_W)

        score = self.compute_score(W, record_met)

        return {
            "phi": phi,
            "W": W,
            "score": score,
            "detW":  np.linalg.det(W + self.ew_ * np.eye(2))
        }


