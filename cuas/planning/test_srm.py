import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

from ParticleClass import Particle
from GridCellClass import GridCell
from SrmEstimator import SrmEstimator
from EmgrEstimator import EmgrEstimator

#logger
import logging
logger = logging.getLogger(__name__)

class TestClass():
    def __init__(self):
        #1. Create the true source particle
        self.true_soure_loc_ = np.array([80,60])
        self.true_particle_ = Particle(source_pos = self.true_soure_loc_,
                                 params=[1.0, self.true_soure_loc_[0], self.true_soure_loc_[1], 0]
                                 )
        self.grid_cells_ = []

        self.dx_ = 2
        self.dy_ = 2
        self.dz_ = 1
        self.test_z_ = 5    #Test height

        self.xm_ = []   #Preset trajectory

        self.measurements_ = []

        self.wind_ = np.array([1.0, 0.0])


        # self.ch4_bk_ = 1.98  #background ch4 concentration
        self.ch4_bk_ = 0  #TODO: set the background noise to 0 for now
        self.M_ = 16.04      # g/mol,   grams per mole  methane
        self.Pr_ = 101325     # Pa,       standard conditions
        self.Temp_ = 298        # K,        standard conditions
        self.Rg_ = 8.314      # J/mol*K   universal gas constant

        self.LayerConc_ = None

        #OSP part
        self.warmup_steps_ = 1500
        #First-order position dynamics
        self.actions_x_ = np.array([
                        [2, 0, 0],
                        [-2, 0, 0],
                        [0, 2, 0],
                        [0, -2, 0]])
        #Second-order acceleration dynamics
        self.actions_v_ = np.array([
                        [1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0]])

        self.dt_ = 1.0

        self.traj_type_ = None

        #anchor point related
        self.anchor_loc_ = None
        self.anchor_conc_ = 0.0
        # self.conc_threshold_ = 0.02 #Threshold for the AP update; the newly calibrated value is 9.2695e-06
        self.conc_threshold_ = 8.4807e-08  # Threshold for the AP update; the newly calibrated value is 9.2695e-06
        self.beta_max_deg_ = 75.0
        self.AP_updated_ = False
        # Add a state in __init__ for smoothing the wind speed
        self.wind_ema_ = None
        self.wind_ema_alpha_ = 0.05  # Smoothing coefficient; the smaller it is, the longer the averaging window
        self.conc_ema_ = None
        self.conc_ema_alpha_ = 1   # Smoothing coefficient; can be tuned separately from the wind alpha. It was 0.1; setting it to 1 means no smoothing and uses the instantaneous measured value

        # score weights
        self.lambda_emgr_ = 0.01 #0.01
        self.lambda_anchor_ = 0.05  #1.0
        self.lambda_visit_ = 0.1    #changed from 0.01 to 0.1
        self.lambda_boundary_ = 2

        #500,300,214,1498(W60,3s),1441(W1200,60s),1471(W600,S20),
        self.Nt = 1000


    def setTrajectoryType(self, type):
        self.traj_type_ = type

    def createGridCells(self):
        #2. establish candidate grid
        xmin  = np.array([0, 0])
        xmax = np.array([200, 200])
        nGridCell = np.array([25, 25])
        dxGrid = (xmax - xmin) / nGridCell
        xGridCell1 = np.arange(dxGrid[0]/2 + xmin[0], xmax[0], dxGrid[0])
        xGridCell2 = np.arange(dxGrid[1] / 2 + xmin[1], xmax[1], dxGrid[1])

        XGridCell1, XGridCell2 = np.meshgrid(xGridCell1, xGridCell2)

        #shape = (625,2)
        XGridCell = np.column_stack([XGridCell1.ravel(), XGridCell2.ravel()])

        #3. establish GridCell
        for i in range(XGridCell.shape[0]):
            cell_center = XGridCell[i]

            cell = GridCell(x=cell_center, dx = dxGrid[0], dy = dxGrid[1], level=0)

            particle = Particle(source_pos = cell_center, params=[1.0, cell_center[0], cell_center[1], 0])

            cell.addParticle(particle)

            self.grid_cells_.append(cell)

        # Debug
        logger.info(f"Number of grid cells: {len(self.grid_cells_)}")
        logger.info(f"First grid center: {self.grid_cells_[0].x_}")
        logger.info(f"Last grid center: {self.grid_cells_[-1].x_}")

    def generate3DLawnmmowerTrajectory(self):
        Nt = self.Nt
        x_min = 0
        x_max = 200
        y_min = 0
        y_max = 200

        # dx/dy are no longer hard-coded but derived backwards from the current point budget Nt: the goal is for the whole zigzag
        # path to roughly cover the entire x_max*y_max area, rather than a dense grid like the original dx=4, dy=2
        # designed for 6000 points -- if the budget is only 100 points and the dense grid is still used, the path
        # would not even get out of the first column (x=0) by the end, everything after would be background noise, and no signal could be learned.
        # Simplifying assumption dx=dy=d: set (number of columns covered) x (points per column) ≈ Nt and solve for d.
        n_side = max(2, int(np.sqrt(Nt)))  # Roughly how many "stations" each side needs
        dx = max(1, (x_max - x_min) / (n_side - 1))
        dy = max(1, (y_max - y_min) / (n_side - 1))
        # logger.info(f"[generate3DLawnmmowerTrajectory] Nt={Nt}, derived dx={dx:.2f}, dy={dy:.2f}")
        print(f"[generate3DLawnmmowerTrajectory] Nt={Nt}, derived dx={dx:.2f}, dy={dy:.2f}")

        # z_range = np.array([2, 3, 4, 5]) # multiple height layers
        z_range = np.array([5])  # single height layer
        # Init
        flag = 1
        x = x_min
        i = 0
        z = z_range[i]
        xm = []
        k = 0
        # Main loop
        while k <= Nt:
            # Case 1: bottom -> top
            if flag == 1:
                y_arr = np.arange(y_min, y_max + dy, dy)
                for y in y_arr:
                    xm.append([x, y, z])
                flag = 2
            # Case 2: move right at top
            elif flag == 2:
                x += dx
                if x > x_max:
                    x = x_min
                    flag = 1
                    # switch height layer
                    i += 1
                    if i >= len(z_range):
                        i = 0
                    z = z_range[i]
                else:
                    xm.append([x, y_max, z])
                    flag = 3

            # Case 3: top -> bottom
            elif flag == 3:
                y_arr = np.arange(y_max, y_min - dy, -dy)
                for y in y_arr:
                    xm.append([x, y, z])
                flag = 4
            # Case 4: move right at bottom
            elif flag == 4:
                x += dx
                if x > x_max:
                    x = x_min
                    flag = 1
                    # switch height layer
                    i += 1
                    if i >= len(z_range):
                        i = 0
                    z = z_range[i]
                else:
                    xm.append([x, y_min, z])
                    flag = 1
            k = len(xm)  # update the trajectory length
        xm = np.array(xm)
        logger.info(f"[generate_uav_trajectory_vertical] size(xm) = {xm.shape}")
        print(f"[generate_uav_trajectory_vertical] size(xm) = {xm.shape}, xm: {xm}")
        self.xm_ = xm

    def generateEmgrOspWarmupTraj(self, Nt=6000):
        # max range
        # x_min = 0
        # x_max = 200
        # y_min = 0
        # y_max = 200

        x_min = 20  # Was 50, changed to 20, kept in sync with multi-agent
        x_max = 160  # Was 120, changed to 160
        y_min = 10  # Was 20, changed to 10
        y_max = 150  # Was 80, changed to 150

        dx = 2
        dy = 2
        # z_range = np.array([2, 3, 4, 5]) # multiple height layers
        z_range = np.array([5])  # single height layer
        # Init
        flag = 1
        i = 0
        z = z_range[i]
        xm = []
        k = 0
        while k <= Nt:
            # Case 1: bottom -> top
            if flag == 1:
                x = x_min
                y_arr = np.arange(y_min, y_max + dy, dy)
                for y in y_arr:
                    xm.append([x, y, z])
                flag = 2
            # Case 2: move right at top
            elif flag == 2:
                y = y_max
                x_arr = np.arange(x_min, x_max + dx, dx)
                for x in x_arr:
                    xm.append([x, y, z])
                flag = 3
            # Case 3: top -> bottom
            elif flag == 3:
                x = x_max
                y_arr = np.arange(y_max, y_min - dy, -dy)
                for y in y_arr:
                    xm.append([x, y, z])
                flag = 4
            # Case 4: move right at bottom
            elif flag == 4:
                y = y_min
                x_arr = np.arange(x_max, x_min - dx, dx)
                for x in x_arr:
                    xm.append([x, y, z])
                # flag = 1
                # Warmup is finished after just one loop, then EMGR OSP starts
                break
        xm = xm * 40
        xm = xm[0:3000]
        k = len(xm)  # update the trajectory length
        xm = np.array(xm)
        logger.info(f"[generateEmgrOspWarmupTraj] size(xm) = {xm.shape}")
        self.xm_ = xm

    def generateRandomTrajectory(self, Nt=6000):
        xm = [np.array([30.0, 30.0, 5.0])]

        for _ in range(Nt - 1):
            action = self.actions_x_[
                np.random.randint(len(self.actions_x_))
            ]

            xp = xm[-1] + action

            xp[0] = np.clip(xp[0], 0, 198)
            xp[1] = np.clip(xp[1], 0, 198)

            xm.append(xp.copy())

        self.xm_ = xm

        return np.array(xm)

    def generateFluxTrajectory(self, srm_estimator, Nt=6000):

        xm = np.array([30.0, 30.0, 5.0])

        traj = []

        for t in range(Nt):

            traj.append(xm.copy())

            best_c = -np.inf
            best_x = xm.copy()

            for action in self.actions_x_:

                xp = xm + action

                xp[0] = np.clip(xp[0], 0, 198)
                xp[1] = np.clip(xp[1], 0, 198)

                c = srm_estimator.getModelOutputLookup(xp, t)[3]

                if c > best_c:
                    best_c = c
                    best_x = xp

            xm = best_x.copy()

        self.xm_ = np.array(traj)

        return np.array(traj)

    # #New version, saved as v7.3
    def loadLayerData(self):
        import h5py
        Nt = self.Nt
        Nx = 100
        Ny = 100
        Nz = 5
        Nws = 3

        layer_conc = np.zeros((Nz, Nt, Nx, Ny), dtype=np.float32)
        layer_ws = np.zeros((Nz, Nt, Nx, Ny, Nws), dtype=np.float32)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../.."))

        for height in range(1, 6):
            if height == 5:
                # Note: this batch of data was saved in MATLAB -v7.3 (HDF5) format (because a single variable
                # exceeds 2GB and the default save() format cannot hold it); scipy.io.loadmat cannot read
                # v7.3 files (it raises NotImplementedError directly), so h5py must be used instead.
                # In addition, under the v7.3 format MATLAB stores the dimension order of 2D arrays reversed (this is
                # a known property of MATLAB HDF5 storage), so after reading it must be transposed back,
                # to match the original (Nt*Nx, Ny) structure of "one time step per 100 rows".
                filename = f"Plume-C-Data-Height-{height}.mat"
                filepath = os.path.join(project_root, "Data",
                                        "G=1_80_60_modified_Q_true_V7_eta_SNR_10_10000s_wind_and_quant_check/Fixed_Window/Time_avg_W200", filename)
                logger.info(f"Loading Conc matfile (v7.3/HDF5): {filepath}")
                with h5py.File(filepath, 'r') as f:
                    C = f['C'][:]  # h5py reads it out as (Ny_orig, Nt*Nx_orig), with the dimensions reversed
                    C = C.T  # transpose back to (Nt*Nx, Ny)
                logger.info(f"original shape after transpose: {C.shape}")
                C = np.reshape(C, (Nt, Nx, Ny))
                logger.info(f"after reshape: {C.shape}")
                layer_conc[height - 1] = C

                filename = f"Plume-Wind-Data-Height-{height}.mat"
                filepath = os.path.join(project_root, "Data",
                                        "G=1_80_60_modified_Q_true_V7_eta_SNR_10_10000s_wind_and_quant_check/Fixed_Window/Time_avg_W200", filename)
                logger.info(f"Loading Wind matfile (v7.3/HDF5): {filepath}")
                with h5py.File(filepath, 'r') as f:
                    ws = f['ws'][:]
                    ws = ws.T
                logger.info(f"original shape after transpose: {ws.shape}")
                ws = np.reshape(ws, (Nt, Nx, Ny, Nws))
                logger.info(f"after reshape: {ws.shape}")
                layer_ws[height - 1] = ws
        logger.info(f"Final LayerConc shape: {layer_conc.shape}, final layer_ws shape: {layer_ws.shape}")
        return layer_conc, layer_ws

    #TODO: the angle calculation here feels wrong; need to check how it was originally computed
    def updateAnchor(self, xm, measured_c, wind):
        self.AP_updated_ = False
        # Check whether the concentration exceeds the set threshold; any detect must have exceeded the threshold, since that is exactly how the detect flag is filtered
        if measured_c < self.conc_threshold_:
            logger.info(f"conc not high enough for updating AP! measured_c: {measured_c}")
            return

        if self.conc_ema_ is None:
            self.conc_ema_ = measured_c
        else:
            self.conc_ema_ = self.conc_ema_alpha_ * measured_c + (1 - self.conc_ema_alpha_) * self.conc_ema_
            logger.info(f"conc_ema_ updated: conc_ema: {self.conc_ema_}")

        if self.anchor_loc_ is None:
            self.anchor_loc_ = xm.copy()
            self.anchor_conc_ = self.conc_ema_
            logger.info(f"anchorpoint initialized, AP_conc:{self.anchor_conc_}, AP_loc: {self.anchor_loc_}")
            return  #Explicit return; do not go on to compute "the angle with itself"


        delta = xm - self.anchor_loc_  # △p
        delta_mag = np.linalg.norm(delta)  # magnitude of △p

        # Update the EMA with wind, instead of using the instantaneous wind directly
        if self.wind_ema_ is None:
            self.wind_ema_ = wind.copy()
        else:
            self.wind_ema_ = self.wind_ema_alpha_ * wind + (1 - self.wind_ema_alpha_) * self.wind_ema_
        Vbar_t = self.wind_ema_   # Use the average flow-field direction, not the instantaneous wind  # Average wind speed, used to compute the upwind vector; this is a single agent so it can be taken directly, but the interface is kept
        wind_mag = np.linalg.norm(-Vbar_t)  # wind speed magnitude

        if delta_mag < 1e-8 or wind_mag < 1e-8: #floats should not be checked against exactly 0 (use a tolerance)
            return

        # Compute β
        cos_angle = np.dot(delta, -Vbar_t) / (delta_mag * wind_mag)
        angle = math.acos(np.clip(cos_angle, -1, 1)) * 180 / math.pi

        # Strong-signal override: if the new reading is much stronger than the current anchor (not just slightly stronger,
        # but with a clear advantage), it is most likely a signal genuinely close to the plume, so it is allowed to ignore
        # the angle constraint and replace the anchor directly
        STRONG_OVERRIDE_FACTOR = 3.0
        factored_c = self.anchor_conc_ * STRONG_OVERRIDE_FACTOR
        is_strong_override = self.conc_ema_ > factored_c

        # Only consider if within angle threshold and best so far
        if (angle <= self.beta_max_deg_ or is_strong_override
                # and measured_c > self.anchor_conc_
        ):
            self.anchor_loc_ = xm.copy()
            self.anchor_conc_ = self.conc_ema_
            self.AP_updated_ = True
            logger.info(f"anchorpoint updated, AP_loc: {xm}, AP_c: {measured_c}, measured_Vbar_t: {Vbar_t}, angle: {angle}, "
                        f"is_strong_override: {is_strong_override}, factored_c: {factored_c}")
        else:
            logger.info(f"Not any best agent, AP cannot update!! measured_loc: {xm}, measured_c: {measured_c}, conc_ema: {self.conc_ema_},  "
                        f"measured_Vbar_t: {Vbar_t}, angle: {angle}, is_strong_override: {is_strong_override}, factored_c: {factored_c}")


    def calcScore(self, xp, emgr_score, visit_map):
        # 1. Anchor reward
        anchor_score = 0.0
        if self.anchor_loc_ is not None:
            dist_anchor = np.linalg.norm(xp[:2] - self.anchor_loc_)
            anchor_score = self.lambda_anchor_ * dist_anchor
            logger.info(f"self.anchor_loc_: {self.anchor_loc_}, dist_anchor: {dist_anchor}, anchor_score: {anchor_score}")
        else:
            logger.warning(f"self.anchor_loc_ is None!")
        # 2. boundary_penalty
        margin = 10
        boundary_penalty = (
                1 / (xp[0] + 1)
                + 1 / (200 - xp[0] + 1)
                + 1 / (xp[1] + 1)
                + 1 / (200 - xp[1] + 1))
        boundary_score = self.lambda_boundary_ * boundary_penalty
        # 3. Exploration penalty, avoid circling
        gx = int(xp[0] // self.dx_)
        gy = int(xp[1] // self.dy_)
        visit_score = (self.lambda_visit_ * visit_map[gx, gy])
        # 4. Total score
        weighted_emgr_score = self.lambda_emgr_ * emgr_score
        total_score = weighted_emgr_score + anchor_score + boundary_score + visit_score
        logger.info(f"total_score: {total_score}, weighted_emgr_score: {weighted_emgr_score}, anchor_score: {anchor_score},"
                    f" boundary_score: {boundary_score}, visit_score: {visit_score}")
        return total_score

    def testMeasure(self, k, estimator):
        xm = np.array((100,60,5))
        measured_c = estimator.getMeasurementLookUp(xm, k)
        logger.info(f"in testMeasure, xm={xm}, measured_c: {measured_c[3]}")

def setup_logger(log_file="test_srm.log"):

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    # Avoid adding handlers repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # -------------------------------------------------
    # File handler
    # -------------------------------------------------

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # -------------------------------------------------
    # Terminal handler
    # -------------------------------------------------

    stream_handler = logging.StreamHandler()

    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    logger.info("Logger initialized.")

if __name__ == "__main__":
    #logger setup
    setup_logger("test_srm.log")

    # test_init
    test_class = TestClass()
    test_class.setTrajectoryType("LawnMower")   # SRM_EMGR_OSP / LawnMower
    test_class.createGridCells()

    layer_conc, layer_ws = test_class.loadLayerData()
    srm_estimator = SrmEstimator(test_class.grid_cells_, test_class.true_particle_, dx = test_class.dx_, dy = test_class.dy_, dz = test_class.dz_)  #How should it be initialized?
    srm_estimator.setLayerData(layer_conc, layer_ws)
    del layer_conc  #save memory
    # estimated_wind = np.array([1.0, 0])    #Add a bit of bias to wind; GT is [1.0, 0]
    emgr_estimator = EmgrEstimator(srm_estimator)

    match test_class.traj_type_:
        case "SRM_EMGR_OSP":
            test_class.generateEmgrOspWarmupTraj()
        case "LawnMower":
            test_class.generate3DLawnmmowerTrajectory()
        case "Random":
            test_class.generateRandomTrajectory()
        case "Flux":
            test_class.generateFluxTrajectory(srm_estimator)

    #visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(
        np.zeros((25,25)),
        origin='lower',
        cmap='turbo',
        extent=[0, 200, 0, 200],
        interpolation='bilinear')
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Estimated Source Strength", fontsize=16)
    # True source position
    ax.scatter(
        test_class.true_soure_loc_[0],
        test_class.true_soure_loc_[1],
        c='green',
        marker='x',
        s=300)
    #Estimated source position
    estimated_source_plot = ax.scatter(0, 0, c='magenta', marker='o', s=100)
    # UAV current position
    uav_plot = ax.scatter( 0, 0, c='white', marker='o', s=120, edgecolors='black', label='UAV')
    # UAV trajectory line
    uav_traj_line, = ax.plot([], [], color='white', linewidth=2, alpha=0.7)
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel("x [m]", fontsize=18)
    ax.set_ylabel("y [m]", fontsize=18)

    # The ground-truth wind should not actually be used here; the actually measured wind should be used, otherwise this is data leakage
    srm_estimator.cacheAllSrm(test_class.xm_, layer_ws[test_class.test_z_-1,:])
    logger.info(f"cacheAllSrm complete!")
    time_steps = 6000

    # Initial
    xm_initial = np.array([30.0,30.0,5.0], dtype=np.float64)
    osp_trajectory = []
    #Second-order dynamics, velocity initialization
    vm = np.array([0.0, 0.0, 0.0])
    v_max = 4.0

    #For evaluation
    Q_error_history = []
    Q_rel_error_history = []
    emgr_score_history = []
    tmp_emgr_score = None
    logdetW_history = []
    min_svd_history = []
    conc_sum = 0
    conc_mean = 0
    eval_data = []
    best_W = None
    #Used to test the effectiveness of the EMGR traj
    c_traj_pred_his = []
    c_traj_true_his = []
    c_traj_rel_error_his = []
    c_traj_all_rrmse_his = []
    c_traj_window_rrmse_his = []



    osp_trajectory = []
    visit_map = np.zeros((int(200/test_class.dx_), int(200/test_class.dy_)))
    xmin = 0
    xmax = 200
    conc_history = []

    start_time = 3000
    xm = np.zeros((1,2), dtype=np.float64)
    for k in range(start_time, time_steps):
        if test_class.traj_type_ == "LawnMower" or test_class.traj_type_ == "Random" or test_class.traj_type_ == "Flux":
            xm = test_class.xm_[k]
            measured = srm_estimator.getMeasurementLookUp(xm, k)
            g = srm_estimator.getCachedSrm(xm, measured[0:2])
            srm_estimator.rlsUpdate(g, measured[3])    #update Q estimation
            emgr_info = emgr_estimator.step(xm, measured[0:2], update_W=True, record_met=True) #This includes the SRM computation
            best_W = emgr_info["W"]
            logger.info(f"Fixed traj: step {k+1}/{time_steps}, score: {emgr_info['score']}")
        elif test_class.traj_type_ == "SRM_EMGR_OSP":
            tmp_emgr_score = None
            tmp_k = k-start_time
            if tmp_k == 0:
                test_class.generateEmgrOspWarmupTraj()
            if tmp_k < test_class.xm_.shape[0] and test_class.AP_updated_ == False:
                xm = test_class.xm_[tmp_k]
                measured = srm_estimator.getMeasurementLookUp(xm, k)
                # conc_history.append(measured[3]) ###############################
                # test_class.testMeasure(k, srm_estimator)    #################################
                measured_c_anchor = max(measured[3], 0.0)
                test_class.updateAnchor(xm[0:2], measured_c_anchor, measured[0:2])
                if test_class.AP_updated_:  #After the AP update, EMGR OSP is about to start, so clear the previous W in advance
                    emgr_estimator.update_W(np.zeros((2, 2)))
                g = srm_estimator.getCachedSrm(xm, measured[0:2])
                srm_estimator.rlsUpdate(g, measured[3])    #update Q estimation
                emgr_info = emgr_estimator.step(xm, measured[0:2], update_W=True, record_met=True) #This includes the SRM computation
                best_W = emgr_info["W"]
                logger.info(f"Fixed traj: step {k+1}/{time_steps}, score: {emgr_info['score']}")
            else:
                # 1. Get measurement
                measured = srm_estimator.getMeasurementLookUp(xm, k)
                # conc_history.append(measured[3]) ###############################
                # test_class.testMeasure(k, srm_estimator)    ###############################
                measured_c_anchor = max(measured[3], 0.0)
                test_class.updateAnchor(xm[0:2], measured_c_anchor, measured[0:2])
                # 2. RLS update
                g = srm_estimator.getCachedSrm(xm, measured[0:2])
                srm_estimator.rlsUpdate(g, measured[3])    #update Q estimation
                # 3. OSP
                best_score = np.inf
                best_action = np.array([0, 0, 0])
                best_xp = None
                best_vp = None
                # Current SRM estimated source
                idx = np.argmax(srm_estimator.Q_)
                best_cell = test_class.grid_cells_[idx]
                source_est = best_cell.x_

                for action in test_class.actions_v_:
                    # velocity update
                    vp = vm + action * test_class.dt_
                    speed = np.linalg.norm(vp)
                    if speed > v_max:
                        vp = vp / speed * v_max
                    # position update
                    xp_raw = xm.astype(np.float64).copy()
                    xp_raw += vp * test_class.dt_
                    xp = np.clip(xp_raw, xmin, xmax)
                    if np.allclose(xp, xm):
                        logger.warning(f"two trajs are allclose, continue!")
                        continue
                    # xp[0] = np.clip(xp[0], 0, 198)
                    # xp[1] = np.clip(xp[1], 0, 198)
                    # EMGR information score
                    emgr_info = emgr_estimator.step(xp, measured[0:2], update_W=False, record_met=False)  # This includes the SRM computation
                    emgr_score = emgr_info["score"]
                    logger.info(f'phi: {emgr_info["phi"]}, norm(phi): {np.linalg.norm(emgr_info["phi"])}, detw: {emgr_info["detW"]}')
                    if not np.isfinite(emgr_score):
                        logger.warning(f"emgr_score is infinite!")
                        continue
                    score_temp = test_class.calcScore(xp, emgr_score, visit_map)
                    # 5. Choose best action
                    if score_temp < best_score:
                        best_score = score_temp
                        # tmp_emgr_score = score_temp #This is the weighted score, containing emgr_score and the other heuristic scores
                        tmp_emgr_score = emgr_score #Only emgr_score, used for the clean correlation study
                        best_action = action
                        best_W = emgr_info["W"]
                        best_xp = xp.copy()
                        best_vp = vp.copy()
                # 4. move UAV
                vm = best_vp.copy()
                xm = best_xp.copy()
                #update visit map
                gx = int(xm[0] // 2)
                gy = int(xm[1] // 2)
                visit_map[gx, gy] += 1
                emgr_estimator.compute_score(best_W, record_met=True)
                emgr_estimator.update_W(best_W)
                logger.info(f"OSP traj: step {k+1}/{time_steps}, best_score: {best_score}, best_action: {best_action}, best_xp: {best_xp}")
                # logger.info(f"conc history: min={np.min(conc_history)}, "f"max={np.max(conc_history)}, "f"mean={np.mean(conc_history)}")
        else:
            logger.error(f"Trajectory type error!")

        #find location of highest Q
        Q_map = srm_estimator.Q_.reshape(25, 25)  # python is row-major
        idx = np.argmax(srm_estimator.Q_)
        best_cell = test_class.grid_cells_[idx]
        Q_pred = srm_estimator.Q_[idx].item() #calculate estimated Q
        best_loc = best_cell.x_
        estimated_source_plot.set_offsets([best_loc[0], best_loc[1]])   #Update the display of the estimated source location
        # update UAV current position
        uav_plot.set_offsets([xm[0], xm[1]])
        osp_trajectory.append(xm.copy())
        traj_np = np.array(osp_trajectory)
        tail_length = 200
        if len(traj_np) > tail_length:
            traj_tail = traj_np[-tail_length:]
        else:
            traj_tail = traj_np
        uav_traj_line.set_data(traj_tail[:, 0], traj_tail[:, 1])


        im.set_data(Q_map)
        im.set_clim(vmin=0, vmax=np.max(Q_map))
        ax.set_title(f"Step = {k}, source_loc_est: [{best_loc[0]}, {best_loc[1]}], Q_est: {Q_pred*1e-3: .6f} kg/s", fontsize=18)
        plt.draw()
        plt.pause(0.01)


        #Evaluation
        assert best_W is not None
        #1.0 Q Error
        # Q_true = test_class.true_particle_.q_
        # Q_error = abs(Q_pred - test_class.true_particle_.q_)
        # Q_rel_error = (abs(Q_pred - Q_true)/ Q_true)
        # Q_error_history.append(Q_error)
        # Q_rel_error_history.append(Q_rel_error)

        #1.1 Traj Conc RMSE
        Q_hat = srm_estimator.Q_    #Q: 625*1, g: 1*625
        c_pred = (g @ Q_hat).item()
        c_true = measured[3]
        if c_true > test_class.conc_threshold_:
            pred_error = abs(c_pred - c_true)
            c_traj_pred_his.append(c_pred)
            c_traj_true_his.append(c_true)
            # c_traj_all_RMSE = np.sqrt(np.mean((np.array(c_traj_pred_his) - np.array(c_traj_true_his))**2))
            rel_error = (c_pred - c_true)**2 / (c_true**2 + 1e-8)
            c_traj_rel_error_his.append(rel_error)
            c_traj_all_rrmse = np.sqrt(np.mean(np.array(c_traj_rel_error_his)))
            c_traj_all_rrmse_his.append(c_traj_all_rrmse)
            rmse_window_size = 100
            c_traj_window_rrmse = np.sqrt(np.mean(np.array(c_traj_rel_error_his)[-rmse_window_size:]))
            c_traj_window_rrmse_his.append(c_traj_window_rrmse)
        else:
            c_traj_all_rrmse_his.append(-1)
            c_traj_window_rrmse_his.append(-1)

        #2. Source localization error
        source_error = np.linalg.norm(best_loc - test_class.true_soure_loc_)
        #3. Information metrics: log det(W) measures the overall information volume
        eps = 1e-8
        emgr_score_history.append(tmp_emgr_score)
        sign, logdetW = np.linalg.slogdet(best_W + eps * np.eye(2))
        logdetW_history.append(logdetW)
        #4. min_svd measures worst-direction observability, i.e. the hardest-to-observe direction; larger is better
        min_svd = np.min(np.linalg.svd(best_W + eps * np.eye(2), compute_uv=False))
        min_svd_history.append(min_svd)
        #5. cumulative sensed concentration
        conc_sum += measured[3]
        conc_mean = conc_sum / (k + 1)

        #csv for visualization
        eval_data.append({
            "step": k,
            # "Q_error": Q_error,
            # "Q_rel_error": Q_rel_error,
            "source_error": source_error,
            "logdetW": logdetW,
            "min_svd": min_svd,
            "conc_sum": conc_sum,
            "conc_mean": conc_mean,
        })

    # 6. Correlaation coiefficient: corr < 0 -> higher information → lower estimation error is what we want!
    look_H = 100
    emgr_arr = np.array(emgr_score_history, dtype=float)
    rrmse_arr = np.array(c_traj_all_rrmse_his, dtype=float)
    window_arr = np.array(c_traj_window_rrmse_his, dtype=float)
    valid_index = (np.array(np.array(c_traj_all_rrmse_his)) != -1) & ( ~np.isnan(emgr_arr) )
    improvement = window_arr[valid_index][:-look_H] - window_arr[valid_index][look_H:]

    print(f"valid_index num: {np.sum(valid_index)}")   #Check how many are valid, to decide how large an H can be used
    corr_all = np.corrcoef(emgr_arr[valid_index], rrmse_arr[valid_index])[0, 1]
    corr_window = np.corrcoef(emgr_arr[valid_index], window_arr[valid_index])[0, 1]
    corr_window_improv = np.corrcoef(emgr_arr[valid_index][:-look_H], improvement)[0, 1]
    print(f"corr_all: {corr_all}, corr_window: {corr_window}, corr_window_improv: {corr_window_improv}")

    plt.ioff()

    #figure 1: timestep vs Q_error  want to see EMGR decrease faster
    plt.figure(figsize=(8, 5))
    plt.plot(c_traj_all_rrmse_his)
    plt.xlabel("Timestep")
    plt.ylabel("c_traj_all_RMSE")
    plt.grid(True)
    plt.title("c_traj_all_RMSE")

    plt.figure(figsize=(8, 5))
    plt.plot(c_traj_window_rrmse_his)
    plt.xlabel("Timestep")
    plt.ylabel("c_traj_window_RMSE")
    plt.grid(True)
    plt.title("c_traj_window_RMSE")

    #figure 2: timestep vs logdet(W)
    plt.figure(figsize=(8, 5))
    plt.plot(emgr_score_history)
    plt.xlabel("Timestep")
    plt.ylabel("log det(W)")
    plt.grid(True)
    plt.title("Information Gain(log det(W))")

    #figure 3: timestep vs min_svd(W)
    plt.figure(figsize=(8, 5))
    plt.plot(min_svd_history)
    plt.xlabel("Timestep")
    plt.ylabel("Min Singular Value")
    plt.grid(True)
    plt.title("Worst-direction Observability(Min Singular Value)")

    #figure 4: Q_error vs logdet(W), correlation study  want to see a lower-right trend
    plt.figure(figsize=(7, 7))
    plt.scatter(emgr_score_history, c_traj_all_rrmse_his, s=10)
    plt.xlabel("emgr_score_history")
    plt.ylabel("c_traj_all_rmse_his")
    plt.grid(True)
    plt.title(f"Correlation = {corr_all:.3f}")

    plt.figure(figsize=(7, 7))
    plt.scatter(emgr_score_history, c_traj_window_rrmse_his, s=10)
    plt.xlabel("log det(W)")
    plt.ylabel("c_traj_window_rmse_his")
    plt.grid(True)
    plt.title(f"Correlation = {corr_window:.3f}")

    plt.figure(figsize=(7, 7))
    plt.scatter(emgr_arr[valid_index][:-look_H], improvement, s=10)
    plt.xlabel("emgr_arr[valid_index][:-look_H]")
    plt.ylabel("window_RRMSE_improvement")
    plt.grid(True)
    plt.title(f"Correlation = {corr_window_improv:.3f}")

    plt.figure()
    plt.plot(emgr_arr[valid_index])
    plt.title("EMGR")

    plt.figure()
    plt.plot(improvement)
    plt.title("Improvement")

    #Trajectory visualization
    if len(osp_trajectory) > 0:
        traj = np.array(osp_trajectory)
        plt.figure(figsize=(8, 8))
        plt.plot(traj[:, 0], traj[:, 1])
        plt.scatter(
            test_class.true_soure_loc_[0],
            test_class.true_soure_loc_[1],
            c='white',
            marker='x',
            s=200
        )
        plt.xlim([0, 200])
        plt.ylim([0, 200])
        plt.grid(True)
        plt.title("OSP Trajectory")


    df = pd.DataFrame(eval_data)
    df.to_csv("evaluation_metrics.csv", index=False)


    plt.show()