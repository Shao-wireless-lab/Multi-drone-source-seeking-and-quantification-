# planning_pipeline_lib.py: containing our proposed hybrid planning algo and other util functions.
# Tips: Correct data path is needed in test_srm.loadLayerData().

import numpy as np

from SrmEstimator import SrmEstimator
from EmgrEstimator import EmgrEstimator
from test_srm import TestClass
from QMetrics import q_metrics


Q_TRUE = 5.225e-5
LAMBDA_HYBRID = 1e-5
EVAL_EVERY = 2
SMOOTH_WINDOW_STEPS = 20
N_SEEDS = 20
LAMBDA_LAWNMOWER = 1e-10   # Optimal value from the lawnmower's own sweep
LAMBDA_SINGLE = 1e-7
CONC_THRESHOLD = 5e-09  # Threshold for the large-area search
ANCHOR_ARGMAX_RADIUS_M = 15.0  # Small radius used by anchor-restricted-argmax
N_AGENTS = 3
SCAN_RADIUS = 15.0         # Radius (m) of the scan around the anchor in the quantify phase; rough initial value, tunable
ANGULAR_STEP = 0.05        # Angular advance per step (rad); rough initial value, tunable
MANUAL_PHASE_LAG = 50     # Already scanned 30/50/80; 50 is the best of the three (inverted-U relationship, see earlier sensitivity analysis), fixed to this value
SWITCH_STABILITY_WINDOW = 30   # Length (in steps) of the sliding window for judging whether the anchor is stable
SWITCH_STABILITY_THRESH = 5.0  # Maximum anchor movement (m) within the window; below this it is judged stable
MAX_SPATIAL_STEPS = 400        # Even if the anchor is never stable, phase A runs at most this many steps before a forced switch
PHASE_LAG = 50                 # Number of steps by which followers trail the leader in phase B


# ============================================================
# 1) Lawnmower: build the measurement stream (reuse a single disk read) + RLS sweep with history
# ============================================================

def build_lawnmower_stream(seed=0):
    """
    Read the real MOABS data once + generate one lawnmower trajectory, and
    precompute the (g_k, c_k) sequence. The returned layer_conc/layer_ws are
    reused later by the OSP side, so the files do not need to be read again.
    """
    np.random.seed(seed)

    test = TestClass()
    test.createGridCells()
    layer_conc, layer_ws = test.loadLayerData()

    test.generate3DLawnmmowerTrajectory()
    max_t = layer_conc.shape[1]
    xm_all = test.xm_[:max_t]

    dummy_srm = SrmEstimator(test.grid_cells_, test.true_particle_,
                             lambda_reg=1.0, dx=test.dx_, dy=test.dy_, dz=test.dz_)
    dummy_srm.setLayerData(layer_conc, layer_ws)

    g_list, c_list = [], []
    for k, xm in enumerate(xm_all):
        measured = dummy_srm.getMeasurementLookUp(xm, k)
        g = dummy_srm.getSRMrow(xm, measured[0:2])
        g_list.append(g.copy())
        c_list.append(measured[3])
        if k % 1000 == 0:
            print(f"  [lawnmower] precompute step {k}/{len(xm_all)}")

    return test, xm_all, g_list, np.array(c_list), layer_conc, layer_ws

# ============================================================
# 3) Error convergence curve comparison
# ============================================================

def smooth_series(val_arr, smooth_window_steps, eval_every):
    """
    Apply a moving average to hist["xxx_err"] to remove the short-period
    oscillations caused by the OSP trajectory circling in a small area /
    small anchor adjustments, keeping only the real convergence trend.

    Note: hist["step"] stores the "raw step" index (recorded once every
    eval_every steps). smooth_window_steps is given in "raw steps" (directly
    corresponding to the window duration used when generating the data, e.g.
    with a 10 s window, 20 steps = 200 s of real time). The function converts
    it to a "number of recorded points" internally before the rolling mean,
    so the caller does not need to care what eval_every is.

    min_periods=1 guarantees that the first few points also have an output
    (they do not become NaN and get truncated because the window is too
    short). The cost is that the first few points of the curve are smoothed
    less than the later ones -- this is acceptable, since the beginning has
    not converged yet anyway and is not expected to be flat.
    """
    import pandas as pd
    win_pts = max(1, int(round(smooth_window_steps / eval_every)))
    s = pd.Series(val_arr)
    return s.rolling(window=win_pts, min_periods=1, center=True).mean().to_numpy()

##################################################
def run_lawnmower_with_history(lambda_reg, test, g_list, c_arr, radius=1, eval_every=10):
    """Run RLS once (fixed lambda), recording the error and a Q_ snapshot every eval_every steps."""
    srm = SrmEstimator(test.grid_cells_, test.true_particle_,
                       lambda_reg=lambda_reg, dx=test.dx_, dy=test.dy_, dz=test.dz_)

    hist = {"step": [], "argmax_err": [], "nbhd_err": [], "global_err": [], "Q_snapshot": []}
    n = len(g_list)
    for k in range(n):
        srm.rlsUpdate(g_list[k], c_arr[k])
        if k % eval_every == 0 or k == n - 1:
            m = q_metrics(srm.Q_, Q_TRUE, radius=radius)
            hist["step"].append(k)
            hist["argmax_err"].append(m["argmax"]["rel_error"])
            hist["nbhd_err"].append(m["neighborhood"]["rel_error"])
            hist["global_err"].append(m["global"]["rel_error"])
            hist["Q_snapshot"].append(srm.Q_.flatten().copy())

    return hist


def compute_robust_metrics(hist, key="global_err", eval_every=EVAL_EVERY,
                           smooth_window_steps=SMOOTH_WINDOW_STEPS,
                           threshold=0.20, tail_fraction=0.2):
    """
    For a single curve (hist[key]), compute three metrics that are more
    robust than "the last point". All of them are based on the smoothed
    curve (not the raw curve), because the raw curve often has single-point
    noise spikes (e.g. one step happens to sweep through an incidentally low
    error point, or happens to hit a spurious peak causing a momentary
    degradation), and judging directly from raw values is easily misled by
    such coincidences.

    Returns:
      time_to_threshold: the step count at which the curve first drops below
                         threshold (None if the curve never drops below
                         threshold over its whole length, meaning "failed to
                         converge to this accuracy within the given length").
      tail_avg: the mean of the smoothed values over the last tail_fraction
                of the recorded points, representing "steady-state
                accuracy" -- much more noise-resistant than looking at just
                the last point.
      min_smoothed: the lowest value the smoothed curve ever reaches -- note
                   this is only a theoretical bound for reference and is not
                   actually achievable (unless you have a termination
                   condition that does not depend on the ground truth
                   Q_true and happens to stop exactly at this point), so do
                   not use this number alone to claim "actual accuracy".
    """
    step = np.array(hist["step"])
    val = np.array(hist[key])
    smooth = smooth_series(val, smooth_window_steps, eval_every)

    below = np.where(smooth <= threshold)[0]
    time_to_threshold = int(step[below[0]]) if len(below) > 0 else None

    n_tail = max(1, int(round(len(smooth) * tail_fraction)))
    tail_avg = float(np.mean(smooth[-n_tail:]))

    min_smoothed = float(np.min(smooth))
    min_smoothed_step = int(step[np.argmin(smooth)])

    return {
        "time_to_threshold": time_to_threshold,
        "tail_avg": tail_avg,
        "min_smoothed": min_smoothed,
        "min_smoothed_step": min_smoothed_step,
    }

def calc_score_components_local(test, xp, emgr_score, visit_map):
    """Identical logic to calc_score_local, except the four components are
    returned separately, to diagnose which term numerically dominates
    total_score (rather than only looking at the final combined number)."""
    anchor_score = 0.0
    if test.anchor_loc_ is not None:
        dist_anchor = np.linalg.norm(xp[:2] - test.anchor_loc_)
        anchor_score = test.lambda_anchor_ * dist_anchor

    boundary_penalty = (
        1 / (xp[0] + 1) + 1 / (200 - xp[0] + 1)
        + 1 / (xp[1] + 1) + 1 / (200 - xp[1] + 1))
    boundary_score = test.lambda_boundary_ * boundary_penalty

    gx = int(np.clip(xp[0] // test.dx_, 0, visit_map.shape[0] - 1))
    gy = int(np.clip(xp[1] // test.dy_, 0, visit_map.shape[1] - 1))
    visit_score = test.lambda_visit_ * visit_map[gx, gy]

    weighted_emgr_score = test.lambda_emgr_ * emgr_score
    total_score = weighted_emgr_score + anchor_score + boundary_score + visit_score
    # print(f"total_score: {total_score}, weighted_emgr_score: {weighted_emgr_score}, anchor_score: {anchor_score},"
    #                 f" boundary_score: {boundary_score}, visit_score: {visit_score}")
    return {
        "emgr": weighted_emgr_score,
        "anchor": anchor_score,
        "boundary": boundary_score,
        "visit": visit_score,
        "total": total_score,
    }

def calc_score_local(test, xp, emgr_score, visit_map):
    return calc_score_components_local(test, xp, emgr_score, visit_map)["total"]

# ------------------------------------------------------------------
# Version of single OSP with time_offset support (the original has none,
# so one is added here; the logic is identical to the original
# run_emgr_osp_with_history, only the offset is added)
# ------------------------------------------------------------------

def single_run_emgr_osp_with_offset(lambda_reg, layer_conc, layer_ws, total_steps,
                                    conc_threshold=CONC_THRESHOLD, time_offset=0,
                                    eval_every=EVAL_EVERY, verbose=False):
    max_t = min(total_steps, layer_conc.shape[1])

    def data_k(k):
        return (k + time_offset) % max_t

    test = TestClass()
    test.createGridCells()
    test.conc_threshold_ = conc_threshold
    test.anchor_loc_ = None
    test.anchor_conc_ = 0.0
    test.AP_updated_ = False

    srm = SrmEstimator(test.grid_cells_, test.true_particle_,
                       lambda_reg=lambda_reg, dx=test.dx_, dy=test.dy_, dz=test.dz_)
    srm.setLayerData(layer_conc, layer_ws)
    emgr = EmgrEstimator(srm)
    visit_map = np.zeros((int(200 / test.dx_), int(200 / test.dy_)))

    hist = {"step": [], "argmax_err": [], "nbhd_err": [], "global_err": [],
           "Q_snapshot": [], "anchor_loc": []}

    def record(k):
        m = q_metrics(srm.Q_, Q_TRUE, radius=1)
        hist["step"].append(k)
        hist["argmax_err"].append(m["argmax"]["rel_error"])
        hist["nbhd_err"].append(m["neighborhood"]["rel_error"])
        hist["global_err"].append(m["global"]["rel_error"])
        hist["Q_snapshot"].append(srm.Q_.flatten().copy())
        hist["anchor_loc"].append(
            tuple(float(v) for v in test.anchor_loc_) if test.anchor_loc_ is not None else None)

    test.generateEmgrOspWarmupTraj()
    xm_warmup = test.xm_[:max_t]
    xm = None
    last_k = -1
    for k, xm_k in enumerate(xm_warmup):
        xm = xm_k
        measured = srm.getMeasurementLookUp(xm, data_k(k))
        wind = measured[0:2]
        test.updateAnchor(xm[0:2], max(measured[3], 0.0), wind)
        if test.AP_updated_:
            emgr.update_W(np.zeros((2, 2)))
        g = srm.getCachedSrm(xm, wind)
        srm.rlsUpdate(g, measured[3])
        emgr.step(xm, wind, update_W=True, record_met=True)
        record(k)
        last_k = k
        if test.AP_updated_:
            break

    vm = np.array([0.0, 0.0, 0.0])
    v_max = 4.0
    xmin, xmax = 0, 200
    if xm is None:
        xm = np.array([30.0, 30.0, 5.0])

    for k in range(last_k + 1, max_t):
        measured = srm.getMeasurementLookUp(xm, data_k(k))
        wind = measured[0:2]
        test.updateAnchor(xm[0:2], max(measured[3], 0.0), wind)
        g = srm.getCachedSrm(xm, wind)
        srm.rlsUpdate(g, measured[3])

        best_score = np.inf
        best_xp, best_vp, best_W = None, None, None
        for action in test.actions_v_:
            vp = vm + action * test.dt_
            speed = np.linalg.norm(vp)
            if speed > v_max:
                vp = vp / speed * v_max
            xp = np.clip(xm.astype(np.float64) + vp * test.dt_, xmin, xmax)
            if np.allclose(xp, xm):
                continue
            emgr_info = emgr.step(xp, wind, update_W=False, record_met=False)
            emgr_score = emgr_info["score"]
            if not np.isfinite(emgr_score):
                continue
            score_temp = calc_score_local(test, xp, emgr_score, visit_map)
            if score_temp < best_score:
                best_score = score_temp
                best_xp, best_vp, best_W = xp.copy(), vp.copy(), emgr_info["W"]

        if best_xp is None:
            best_xp, best_vp, best_W = xm.copy(), vm.copy(), np.zeros((2, 2))

        vm, xm = best_vp, best_xp
        gx = int(np.clip(xm[0] // test.dx_, 0, visit_map.shape[0] - 1))
        gy = int(np.clip(xm[1] // test.dy_, 0, visit_map.shape[1] - 1))
        visit_map[gx, gy] += 1
        emgr.compute_score(best_W, record_met=True)
        emgr.update_W(best_W)

        if k % eval_every == 0 or k == max_t - 1:
            record(k)

    return hist, test

# ------------------------------------------------------------------
# New: anchor-restricted-argmax metric (identical logic to the version
# validated in single_vs_temporal_decisive.py, moved here for reuse) --
# take the cell with the highest Q value within radius_m of the anchor as
# Q_hat, instead of summing over all 625 cells like the global metric; the
# circle center follows the anchor frame by frame (not just its final
# position), and frames where the anchor has not been triggered return NaN.
# ------------------------------------------------------------------

def compute_anchor_restricted_argmax_error_curve(hist, test, radius_m=ANCHOR_ARGMAX_RADIUS_M):
    if "anchor_loc" not in hist:
        raise ValueError("hist has no 'anchor_loc' field; cannot compute the anchor-restricted argmax metric")

    cell_locs = np.array([c.x_ for c in test.grid_cells_], dtype=float)
    eps = 1e-12
    errs = []
    for q_flat, anchor_loc in zip(hist["Q_snapshot"], hist["anchor_loc"]):
        if anchor_loc is None:
            errs.append(np.nan)
            continue
        anchor_arr = np.asarray(anchor_loc, dtype=float)
        dists = np.linalg.norm(cell_locs - anchor_arr, axis=1)
        mask = dists <= radius_m
        if mask.sum() == 0:
            errs.append(np.nan)
            continue
        q_hat = float(q_flat[mask].max())
        rel_err = abs(q_hat - Q_TRUE) / (abs(Q_TRUE) + eps)
        errs.append(rel_err)
    return np.array(errs)


def compute_robust_metrics_from_array(step_arr, val_arr, eval_every=EVAL_EVERY,
                                      smooth_window_steps=SMOOTH_WINDOW_STEPS,
                                      threshold=0.20, tail_fraction=0.2):
    """Same logic as compute_robust_metrics, but takes (step, val) arrays directly
    instead of a hist dict, so that an already-computed anchor-argmax curve can
    also reuse the same set of robust statistics."""
    hist_wrap = {"step": step_arr, "_val": val_arr}
    return compute_robust_metrics(hist_wrap, key="_val", eval_every=eval_every,
                                  smooth_window_steps=smooth_window_steps,
                                  threshold=threshold, tail_fraction=tail_fraction)


# ============================================================
# 1) Multi-agent warmup: each of the N agents takes a y-direction search band and scans back and forth in parallel
# ============================================================

def generate_square_perimeter_traj(x_min=50, x_max=120, y_min=20, y_max=80,
                                   dx=2, dy=2, z=5, clockwise=True, n_steps=1000):
    """
    Same perimeter logic as generateEmgrOspWarmupTraj in test_srm.py, only
    with a parameterized direction (clockwise/counter-clockwise), so that in
    the multi-agent setting different agents can circle in opposite
    directions, reflecting "path diversity" rather than simply copying the
    single-agent warmup.
    """
    xm = []
    if clockwise:
        # bottom->top (left side), left->right (top side), top->bottom (right side), right->left (bottom side)
        for y in np.arange(y_min, y_max + dy, dy):
            xm.append([x_min, y, z])
        for x in np.arange(x_min, x_max + dx, dx):
            xm.append([x, y_max, z])
        for y in np.arange(y_max, y_min - dy, -dy):
            xm.append([x_max, y, z])
        for x in np.arange(x_max, x_min - dx, -dx):
            xm.append([x, y_min, z])
    else:
        # counter-clockwise: bottom->top (right side), right->left (top side), top->bottom (left side), left->right (bottom side)
        for y in np.arange(y_min, y_max + dy, dy):
            xm.append([x_max, y, z])
        for x in np.arange(x_max, x_min - dx, -dx):
            xm.append([x, y_max, z])
        for y in np.arange(y_max, y_min - dy, -dy):
            xm.append([x_min, y, z])
        for x in np.arange(x_min, x_max + dx, dx):
            xm.append([x, y_min, z])

    xm = np.array(xm)
    # If shorter than n_steps, repeat the full loop to fill; if longer, truncate
    reps = int(np.ceil(n_steps / len(xm)))
    xm = np.tile(xm, (reps, 1))[:n_steps]
    return xm


def generate_circle_traj(center_x, center_y, radius, z=5, angular_step=0.1, n_steps=1000):
    """
    Circular warmup trajectory, used as the search path of the third agent;
    it contrasts geometrically with the two squares (clockwise/counter-
    clockwise) and increases exploration diversity.
    """
    angles = np.arange(n_steps) * angular_step
    xs = center_x + radius * np.cos(angles)
    ys = center_y + radius * np.sin(angles)
    zs = np.full(n_steps, z)
    return np.stack([xs, ys, zs], axis=1)


def generate_multi_agent_warmup_traj(n_agents=3, x_min=20, x_max=160,
                                     y_min=10, y_max=150, n_steps=1000):
    """
    The three agents each follow a different shape of path, reflecting the
    exploration diversity of multi-agent (rather than simply splitting the
    same region into three bands that are each scanned separately, as before):
      agent 0: square, clockwise
      agent 1: square, counter-clockwise
      agent 2: circle, radius equal to half of the search region
    If n_agents != 3, these three patterns are cycled (agent 3 reuses the
    clockwise square, and so on).

    The default search range is enlarged from the original x:50-120 (span 70)
    / y:20-80 (span 60) to x:20-160 (span 140) / y:10-150 (span 140), roughly
    half of the whole 200x200 domain -- the aim is to make the advantage of
    multi-agent "covering a larger area in parallel" show up more clearly;
    with the earlier small range the warmup itself ended very quickly, hiding
    the value of parallel exploration.
    Note: the search range on the single-agent side
    (generateEmgrOspWarmupTraj in test_srm.py) must be enlarged to the same
    boundaries, otherwise the comparison is unfair (the measured
    "multi-agent is faster" might just be because the single-agent search
    range did not grow with it).
    """
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    radius = min(x_max - x_min, y_max - y_min) / 2

    pattern_fns = [
        lambda: generate_square_perimeter_traj(x_min, x_max, y_min, y_max,
                                               clockwise=True, n_steps=n_steps),
        lambda: generate_square_perimeter_traj(x_min, x_max, y_min, y_max,
                                               clockwise=False, n_steps=n_steps),
        lambda: generate_circle_traj(center_x, center_y, radius, n_steps=n_steps),
    ]
    trajs = [pattern_fns[i % len(pattern_fns)]() for i in range(n_agents)]
    return trajs



def multi_agent_warmup_step(test, srm, xm_list, k):
    """
    At each step: first compute each of the N agents' measurements, pick the
    one with the highest concentration among those satisfying the
    conc_threshold_ condition, and call test.updateAnchor() once to update
    the shared anchor. Returns measured_list for the caller to feed into RLS.
    """
    measured_list = []
    candidates = []
    for xm in xm_list:
        measured = srm.getMeasurementLookUp(xm, k)
        measured_c = max(measured[3], 0.0)
        measured_list.append(measured)
        if measured_c >= test.conc_threshold_:
            candidates.append((measured_c, xm, measured[0:2]))

    if candidates:
        best_c, best_xm, best_wind = max(candidates, key=lambda item: item[0])
        test.updateAnchor(best_xm[0:2], best_c, best_wind)

    return measured_list


# ============================================================
# 2) Multi-agent quantify: once the anchor is stable, the N agents scan around the anchor in separate sectors
# ============================================================

def run_multi_agent_osp_with_history(lambda_reg, test, layer_conc, layer_ws,
                                     n_agents=N_AGENTS, total_steps=750,
                                     scan_radius=SCAN_RADIUS, angular_step=ANGULAR_STEP,
                                     eval_every=EVAL_EVERY, verbose=True,
                                     time_offset=0):
    """
    time_offset: the key parameter for implementing "multiple seeds" -- a
    different time_offset makes the whole simulation start reading from a
    different time point of the real MOABS data (taken modulo the total data
    length, used cyclically), so the UAV encounters different turbulence
    realizations of the same real dataset during warmup/quantify. This is a
    real, physically meaningful source of variation, not noise made up out
    of thin air (currently nowhere in the code actually calls np.random to
    generate random numbers, so merely changing the seed number would not
    change any result; the variation has to come from this time offset).
    The default time_offset=0 is exactly the same as before and does not
    affect the results you have already obtained.
    """
    max_t = min(total_steps, layer_conc.shape[1])

    def data_k(k):
        # Everywhere data needs to be looked up, use this function to convert
        # the absolute step into the data index after "offset + wrap-around",
        # instead of using k directly
        return (k + time_offset) % max_t

    test.anchor_loc_ = None
    test.anchor_conc_ = 0.0
    test.AP_updated_ = False

    srm = SrmEstimator(test.grid_cells_, test.true_particle_,
                       lambda_reg=lambda_reg, dx=test.dx_, dy=test.dy_, dz=test.dz_)
    srm.setLayerData(layer_conc, layer_ws)

    hist = {"step": [], "argmax_err": [], "nbhd_err": [], "global_err": [],
           "Q_snapshot": [], "anchor_loc": []}
    trajs = [[] for _ in range(n_agents)]

    def record(k):
        m = q_metrics(srm.Q_, Q_TRUE, radius=1)
        hist["step"].append(k)
        hist["argmax_err"].append(m["argmax"]["rel_error"])
        hist["nbhd_err"].append(m["neighborhood"]["rel_error"])
        hist["global_err"].append(m["global"]["rel_error"])
        hist["Q_snapshot"].append(srm.Q_.flatten().copy())
        hist["anchor_loc"].append(
            tuple(float(v) for v in test.anchor_loc_) if test.anchor_loc_ is not None else None)

    # ---- warmup: N agents search in parallel until the anchor is triggered ----  currently a boundary of 40
    warmup_trajs = generate_multi_agent_warmup_traj(n_agents=n_agents,
                                                     n_steps=min(1000, total_steps))
    max_warmup_len = max(len(t) for t in warmup_trajs)
    last_k = -1
    for k in range(max_warmup_len):
        xm_list = [warmup_trajs[i][min(k, len(warmup_trajs[i]) - 1)] for i in range(n_agents)]
        measured_list = multi_agent_warmup_step(test, srm, xm_list, data_k(k))
        for xm, measured in zip(xm_list, measured_list):
            g = srm.getCachedSrm(xm, measured[0:2])
            srm.rlsUpdate(g, measured[3])
        for i, xm in enumerate(xm_list):
            trajs[i].append(np.array(xm, dtype=float))
        record(k)
        last_k = k
        if test.AP_updated_ and test.anchor_loc_ is not None:
            if verbose:
                print(f"  [multi-agent] anchor triggered at warmup step {k}, "
                     f"anchor_loc={test.anchor_loc_}")
            break

    if test.anchor_loc_ is None:
        if verbose:
            print("  Warning: anchor was still not triggered after warmup ended; the quantify phase will degrade to staying in place")
        anchor = np.array([100.0, 50.0])  # fallback, should not be reached in theory
    else:
        anchor = test.anchor_loc_

    # ---- quantify: each agent independently runs EMGR-based action planning
    # (same scoring logic as single-agent OSP: EMGR information gain + anchor
    # attraction + boundary + revisit penalty), with an additional
    # separation penalty -- a candidate is penalized if it is too close to
    # the other agents' current positions. This way each agent still decides
    # independently under the objective of "maximize information", but
    # naturally avoids positions already being covered by the others, rather
    # than following a hard-coded fixed geometric shape.
    #
    # The anchor keeps being updated at every step during the quantify phase
    # (consistent with the free OSP phase of single-agent), instead of being
    # locked to the first value found during warmup as before.
    LAMBDA_SEPARATION = 0.15   # Weight of the separation penalty; larger means the agents stay farther apart
    MIN_SAFE_DIST = 8.0        # Penalty starts below this distance (m)

    def calc_multi_agent_score(test, xp, emgr_score, visit_map, other_positions):
        anchor_score = 0.0
        if test.anchor_loc_ is not None:
            dist_anchor = np.linalg.norm(xp[:2] - test.anchor_loc_)
            anchor_score = test.lambda_anchor_ * dist_anchor

        boundary_penalty = (1/(xp[0]+1) + 1/(200-xp[0]+1)
                            + 1/(xp[1]+1) + 1/(200-xp[1]+1))
        boundary_score = test.lambda_boundary_ * boundary_penalty

        gx = int(np.clip(xp[0] // test.dx_, 0, visit_map.shape[0] - 1))
        gy = int(np.clip(xp[1] // test.dy_, 0, visit_map.shape[1] - 1))
        visit_score = test.lambda_visit_ * visit_map[gx, gy]

        weighted_emgr_score = test.lambda_emgr_ * emgr_score

        sep_penalty = 0.0
        for op in other_positions:
            d = np.linalg.norm(xp[:2] - op[:2])
            if d < MIN_SAFE_DIST:
                sep_penalty += (MIN_SAFE_DIST - d)
        separation_score = LAMBDA_SEPARATION * sep_penalty

        return (weighted_emgr_score + anchor_score + boundary_score
               + visit_score + separation_score)

    # Each agent maintains its own EmgrEstimator (each tracking the
    # information matrix W of its own trajectory), all built on the same
    # shared srm -- so each agent's information gain is evaluated based on
    # the path it has walked itself, but the quantify target (Q_) is the
    # same estimate accumulated jointly by all agents.
    emgrs = [EmgrEstimator(srm) for _ in range(n_agents)]
    vms = [np.array([0.0, 0.0, 0.0]) for _ in range(n_agents)]
    xm_list = [np.array(warmup_trajs[i][min(last_k, len(warmup_trajs[i])-1)], dtype=float)
              for i in range(n_agents)]
    v_max = 4.0
    xmin, xmax = 0, 200
    visit_map = np.zeros((int(200/test.dx_), int(200/test.dy_)))

    max_t = min(total_steps, layer_conc.shape[1])
    for k in range(last_k + 1, max_t):
        # First each agent measures, updates the anchor, and updates Q via RLS (consistent with the free OSP phase of single-agent)
        measured_list = []
        for i, xm in enumerate(xm_list):
            measured = srm.getMeasurementLookUp(xm, data_k(k))
            measured_list.append(measured)
            wind = measured[0:2]
            test.updateAnchor(xm[0:2], max(measured[3], 0.0), wind)
            g = srm.getCachedSrm(xm, wind)
            srm.rlsUpdate(g, measured[3])
            emgrs[i].step(xm, wind, update_W=True, record_met=True)

        # Then plan the next step for each agent in turn (processed
        # sequentially so that later agents can see the new positions just
        # chosen by earlier agents in this step, naturally producing mutual avoidance)
        current_positions = list(xm_list)  # Will be updated one by one in the loop below
        for i in range(n_agents):
            xm = xm_list[i]
            wind = measured_list[i][0:2]
            vm = vms[i]
            other_positions = [current_positions[j] for j in range(n_agents) if j != i]

            best_score = np.inf
            best_xp, best_vp, best_W = None, None, None
            for action in test.actions_v_:
                vp = vm + action * test.dt_
                speed = np.linalg.norm(vp)
                if speed > v_max:
                    vp = vp / speed * v_max
                xp = np.clip(xm.astype(np.float64) + vp * test.dt_, xmin, xmax)
                if np.allclose(xp, xm):
                    continue
                emgr_info = emgrs[i].step(xp, wind, update_W=False, record_met=False)
                emgr_score = emgr_info["score"]
                if not np.isfinite(emgr_score):
                    continue
                score_temp = calc_multi_agent_score(test, xp, emgr_score, visit_map, other_positions)
                if score_temp < best_score:
                    best_score = score_temp
                    best_xp, best_vp, best_W = xp.copy(), vp.copy(), emgr_info["W"]

            if best_xp is None:
                best_xp, best_vp, best_W = xm.copy(), vm.copy(), np.zeros((2, 2))

            vms[i] = best_vp
            xm_list[i] = best_xp
            current_positions[i] = best_xp  # Update so that subsequent agents can see it
            gx = int(np.clip(best_xp[0] // test.dx_, 0, visit_map.shape[0]-1))
            gy = int(np.clip(best_xp[1] // test.dy_, 0, visit_map.shape[1]-1))
            visit_map[gx, gy] += 1
            emgrs[i].compute_score(best_W, record_met=True)
            emgrs[i].update_W(best_W)
            trajs[i].append(best_xp.copy())

        if k % eval_every == 0 or k == max_t - 1:
            record(k)
        if verbose and k % 200 == 0:
            print(f"  [multi-agent] quantify step {k}/{max_t}, "
                 f"anchor={test.anchor_loc_}")

    return hist, [np.array(t) for t in trajs]




# ============================================================
# 2) EMGR OSP (with AP): same logic as run_emgr_osp in the earlier
#    lambda_sweep_real_moabs.py, only with history recording added
#    (error + Q snapshot + full trajectory + anchor position history)
# ============================================================

def reset_osp_state(test):
    test.anchor_loc_ = None
    test.anchor_conc_ = 0.0
    test.AP_updated_ = False



# ============================================================
# 4) Temporal diversity scheme: N agents follow the same "leader"
#    trajectory staggered in time, instead of planning independently in
#    space and avoiding each other as before.
# ============================================================
def run_emgr_osp_with_history(lambda_reg, test, layer_conc, layer_ws, total_steps=6000,
                              radius=1, eval_every=10, verbose=True):
    """
    hist now additionally contains "anchor_loc": a snapshot of
    test.anchor_loc_ at each recorded point (None or an (x,y) tuple), used in
    animate_ap_debug() to draw how the anchor position changes over time.
    """
    reset_osp_state(test)

    srm = SrmEstimator(test.grid_cells_, test.true_particle_,
                       lambda_reg=lambda_reg, dx=test.dx_, dy=test.dy_, dz=test.dz_)
    srm.setLayerData(layer_conc, layer_ws)
    emgr = EmgrEstimator(srm)

    visit_map = np.zeros((int(200 / test.dx_), int(200 / test.dy_)))
    max_t = min(total_steps, layer_conc.shape[1])

    hist = {"step": [], "argmax_err": [], "nbhd_err": [], "global_err": [],
           "Q_snapshot": [], "anchor_loc": []}
    traj = []
    g_list_osp = []  # New
    c_list_osp = []  # New


    def record(k):
        m = q_metrics(srm.Q_, Q_TRUE, radius=radius)
        hist["step"].append(k)
        hist["argmax_err"].append(m["argmax"]["rel_error"])
        hist["nbhd_err"].append(m["neighborhood"]["rel_error"])
        hist["global_err"].append(m["global"]["rel_error"])
        hist["Q_snapshot"].append(srm.Q_.flatten().copy())
        hist["anchor_loc"].append(
            tuple(float(v) for v in test.anchor_loc_) if test.anchor_loc_ is not None else None)

    # ---- warmup: fixed perimeter trajectory until the anchor point update is triggered ----
    test.generateEmgrOspWarmupTraj()
    xm_warmup = test.xm_[:max_t]

    xm = None
    last_k = -1
    for k, xm_k in enumerate(xm_warmup):
        xm = xm_k
        measured = srm.getMeasurementLookUp(xm, k)
        wind = measured[0:2]
        measured_c_anchor = max(measured[3], 0.0)
        test.updateAnchor(xm[0:2], measured_c_anchor, wind)

        if test.AP_updated_:
            emgr.update_W(np.zeros((2, 2)))

        g = srm.getCachedSrm(xm, wind)
        srm.rlsUpdate(g, measured[3])
        g_list_osp.append(g.copy())  # New
        c_list_osp.append(measured[3])  # New
        emgr.step(xm, wind, update_W=True, record_met=True)

        traj.append(xm.copy())
        record(k)  # Few warmup steps, so record every step to see the details around the AP trigger
        last_k = k

        if test.AP_updated_:
            if verbose:
                print(f"  Anchor point update completed at warmup step {k}, switching to free OSP")
            break

    if not test.AP_updated_ and verbose:
        print(f"  Warning: anchor point update was still not triggered after warmup ended ({last_k + 1} steps in total)")

    # ---- free OSP ----
    vm = np.array([0.0, 0.0, 0.0])
    v_max = 4.0
    xmin, xmax = 0, 200
    if xm is None:
        xm = np.array([30.0, 30.0, 5.0])

    for k in range(last_k + 1, max_t):
        measured = srm.getMeasurementLookUp(xm, k)
        wind = measured[0:2]
        measured_c_anchor = max(measured[3], 0.0)
        test.updateAnchor(xm[0:2], measured_c_anchor, wind)

        g = srm.getCachedSrm(xm, wind)
        g_list_osp.append(g.copy())  # New
        c_list_osp.append(measured[3])  # New
        srm.rlsUpdate(g, measured[3])

        best_score = np.inf
        best_xp, best_vp, best_action_W = None, None, None

        for action in test.actions_v_:
            vp = vm + action * test.dt_
            speed = np.linalg.norm(vp)
            if speed > v_max:
                vp = vp / speed * v_max
            xp = np.clip(xm.astype(np.float64) + vp * test.dt_, xmin, xmax)
            if np.allclose(xp, xm):
                continue

            emgr_info = emgr.step(xp, wind, update_W=False, record_met=False)
            emgr_score = emgr_info["score"]
            if not np.isfinite(emgr_score):
                continue

            score_temp = calc_score_local(test, xp, emgr_score, visit_map)
            if score_temp < best_score:
                best_score = score_temp
                best_xp, best_vp, best_action_W = xp.copy(), vp.copy(), emgr_info["W"]

        if best_xp is None:
            best_xp, best_vp, best_action_W = xm.copy(), vm.copy(), np.zeros((2, 2))

        vm, xm = best_vp, best_xp
        gx = int(np.clip(xm[0] // test.dx_, 0, visit_map.shape[0] - 1))
        gy = int(np.clip(xm[1] // test.dy_, 0, visit_map.shape[1] - 1))
        visit_map[gx, gy] += 1
        emgr.compute_score(best_action_W, record_met=True)
        emgr.update_W(best_action_W)

        traj.append(xm.copy())
        if k % eval_every == 0 or k == max_t - 1:
            record(k)
        if verbose and k % 500 == 0:
            print(f"  OSP step {k}/{max_t}")

    return hist, np.array(traj), g_list_osp, np.array(c_list_osp)


def run_multi_agent_sequential_follow(lambda_reg, layer_conc, layer_ws,
                                      n_agents=N_AGENTS, phase_lag=50,
                                      total_steps=750, eval_every=EVAL_EVERY,
                                      conc_threshold=CONC_THRESHOLD,
                                      time_offset=0, verbose=True):
    """
    Test the hypothesis that "temporal diversity (ensemble averaging) improves
    the GP-MOABS mismatch more than spatial diversity does".

    Design: first use the single-agent EMGR OSP planning logic to generate a
    "leader trajectory" (run the full warmup+quantify once, borrowing only
    its position sequence, not reusing its Q_ estimate). Then all N agents
    walk along this same trajectory, but at absolute time k the position of
    agent i is leader_traj[k - i*phase_lag] (lagging i*phase_lag steps; until
    that time point is reached it stays at the starting point). Each agent
    queries data using its own "current" true absolute time k (not the k the
    leader had at that moment), so the same spatial position is sampled
    repeatedly at different real turbulence moments -- this is where the
    ensemble averaging really happens, and no extra averaging logic needs to
    be written; RLS itself naturally fuses these independent observations
    across time into the shared Q_ estimate.

    Compared with the earlier spatial-separation version
    (run_multi_agent_osp_with_history):
      - No separation_score (agents are no longer prevented from approaching/overlapping each other)
      - No visit_score penalizing revisits (revisiting is exactly the effect this scheme wants)
      - Each agent no longer makes its own independent EMGR decisions; it just passively follows the same trajectory
    """
    leader_test = TestClass()
    leader_test.createGridCells()
    leader_test.conc_threshold_ = conc_threshold
    if verbose:
        print(f"  First generating a leader trajectory (borrowing the single-agent EMGR OSP planning logic)...")
    _, leader_traj, _, _ = run_emgr_osp_with_history(
        lambda_reg, leader_test, layer_conc, layer_ws,
        total_steps=total_steps, eval_every=eval_every, verbose=False)

    T = len(leader_traj)
    max_t = min(total_steps, layer_conc.shape[1])

    def data_k(k):
        return (k + time_offset) % max_t

    test = TestClass()
    test.createGridCells()
    test.conc_threshold_ = conc_threshold
    srm = SrmEstimator(test.grid_cells_, test.true_particle_,
                       lambda_reg=lambda_reg, dx=test.dx_, dy=test.dy_, dz=test.dz_)
    srm.setLayerData(layer_conc, layer_ws)

    hist = {"step": [], "argmax_err": [], "nbhd_err": [], "global_err": [],
           "Q_snapshot": [], "anchor_loc": []}
    trajs = [[] for _ in range(n_agents)]

    def record(k):
        m = q_metrics(srm.Q_, Q_TRUE, radius=1)
        hist["step"].append(k)
        hist["argmax_err"].append(m["argmax"]["rel_error"])
        hist["nbhd_err"].append(m["neighborhood"]["rel_error"])
        hist["global_err"].append(m["global"]["rel_error"])
        hist["Q_snapshot"].append(srm.Q_.flatten().copy())
        hist["anchor_loc"].append(
            tuple(float(v) for v in test.anchor_loc_) if test.anchor_loc_ is not None else None)

    for k in range(T):
        for i in range(n_agents):
            idx = max(0, k - i * phase_lag)
            xm = leader_traj[idx]
            measured = srm.getMeasurementLookUp(xm, data_k(k))
            wind = measured[0:2]
            test.updateAnchor(xm[0:2], max(measured[3], 0.0), wind)
            g = srm.getCachedSrm(xm, wind)
            srm.rlsUpdate(g, measured[3])
            trajs[i].append(np.array(xm, dtype=float))

        if k % eval_every == 0 or k == T - 1:
            record(k)
        if verbose and k % 200 == 0:
            print(f"  [sequential-follow] step {k}/{T}, anchor={test.anchor_loc_}")

    return hist, [np.array(t) for t in trajs], test


def run_multi_agent_hybrid(lambda_reg, layer_conc, layer_ws, n_agents=N_AGENTS,
                           total_steps=1000, conc_threshold=CONC_THRESHOLD,
                           switch_stability_window=SWITCH_STABILITY_WINDOW,
                           switch_stability_thresh=SWITCH_STABILITY_THRESH,
                           max_spatial_steps=MAX_SPATIAL_STEPS,
                           phase_lag=PHASE_LAG, time_offset=0, verbose=True):
    """
    Returns: hist (same structure as the other schemes, with an extra "phase" field marking phase A/B),
    trajs (one full trajectory per agent), test, switch_step (the absolute step at which the switch actually happened)
    """
    max_t = min(total_steps, layer_conc.shape[1])

    def data_k(k):
        return (k + time_offset) % max_t

    test = TestClass()
    test.createGridCells()
    test.conc_threshold_ = conc_threshold
    test.anchor_loc_ = None
    test.anchor_conc_ = 0.0
    test.AP_updated_ = False

    srm = SrmEstimator(test.grid_cells_, test.true_particle_,
                       lambda_reg=lambda_reg, dx=test.dx_, dy=test.dy_, dz=test.dz_)
    srm.setLayerData(layer_conc, layer_ws)

    hist = {"step": [], "argmax_err": [], "nbhd_err": [], "global_err": [],
           "Q_snapshot": [], "anchor_loc": [], "phase": []}
    trajs = [[] for _ in range(n_agents)]

    def record(k, phase):
        m = q_metrics(srm.Q_, Q_TRUE, radius=1)
        hist["step"].append(k)
        hist["argmax_err"].append(m["argmax"]["rel_error"])
        hist["nbhd_err"].append(m["neighborhood"]["rel_error"])
        hist["global_err"].append(m["global"]["rel_error"])
        hist["Q_snapshot"].append(srm.Q_.flatten().copy())
        hist["anchor_loc"].append(
            tuple(float(v) for v in test.anchor_loc_) if test.anchor_loc_ is not None else None)
        hist["phase"].append(phase)

    # ================================================================
    # Phase A: spatial warmup + quantify, until the anchor is stable or max_spatial_steps is reached
    # ================================================================
    warmup_trajs = generate_multi_agent_warmup_traj(n_agents=n_agents,
                                                     n_steps=min(1000, total_steps))
    max_warmup_len = max(len(t) for t in warmup_trajs)
    last_k = -1
    for k in range(max_warmup_len):
        xm_list_w = [warmup_trajs[i][min(k, len(warmup_trajs[i]) - 1)] for i in range(n_agents)]
        measured_list = multi_agent_warmup_step(test, srm, xm_list_w, data_k(k))
        for xm, measured in zip(xm_list_w, measured_list):
            g = srm.getCachedSrm(xm, measured[0:2])
            srm.rlsUpdate(g, measured[3])
        for i, xm in enumerate(xm_list_w):
            trajs[i].append(np.array(xm, dtype=float))
        record(k, "A_warmup")
        last_k = k
        if test.AP_updated_ and test.anchor_loc_ is not None:
            if verbose:
                print(f"  [hybrid] anchor triggered at warmup step {k}, "
                     f"anchor_loc={test.anchor_loc_}")
            break

    LAMBDA_SEPARATION = 0.15
    MIN_SAFE_DIST = 8.0

    # Combination of the five spatial score terms, encouraging spatial coverage
    def calc_multi_agent_score(test, xp, emgr_score, visit_map, other_positions):
        anchor_score = 0.0
        if test.anchor_loc_ is not None:
            dist_anchor = np.linalg.norm(xp[:2] - test.anchor_loc_)
            anchor_score = test.lambda_anchor_ * dist_anchor
        boundary_penalty = (1/(xp[0]+1) + 1/(200-xp[0]+1)
                            + 1/(xp[1]+1) + 1/(200-xp[1]+1))
        boundary_score = test.lambda_boundary_ * boundary_penalty
        gx = int(np.clip(xp[0] // test.dx_, 0, visit_map.shape[0] - 1))
        gy = int(np.clip(xp[1] // test.dy_, 0, visit_map.shape[1] - 1))
        visit_score = test.lambda_visit_ * visit_map[gx, gy]
        weighted_emgr_score = test.lambda_emgr_ * emgr_score
        sep_penalty = 0.0
        for op in other_positions:
            d = np.linalg.norm(xp[:2] - op[:2])
            if d < MIN_SAFE_DIST:
                sep_penalty += (MIN_SAFE_DIST - d)
        separation_score = LAMBDA_SEPARATION * sep_penalty
        return weighted_emgr_score + anchor_score + boundary_score + visit_score + separation_score

    emgrs = [EmgrEstimator(srm) for _ in range(n_agents)]
    vms = [np.array([0.0, 0.0, 0.0]) for _ in range(n_agents)]
    xm_list = [np.array(warmup_trajs[i][min(last_k, len(warmup_trajs[i])-1)], dtype=float)
              for i in range(n_agents)]
    v_max = 4.0
    xmin, xmax = 0, 200
    visit_map = np.zeros((int(200/test.dx_), int(200/test.dy_)))

    anchor_history = []  # Record the anchor position at every step (not appended while untriggered), used for the stability check
    switch_step = None

    for k in range(last_k + 1, max_t):
        measured_list = []
        for i, xm in enumerate(xm_list):
            measured = srm.getMeasurementLookUp(xm, data_k(k))
            measured_list.append(measured)
            wind = measured[0:2]
            test.updateAnchor(xm[0:2], max(measured[3], 0.0), wind)
            g = srm.getCachedSrm(xm, wind)
            srm.rlsUpdate(g, measured[3])
            emgrs[i].step(xm, wind, update_W=True, record_met=True)

        current_positions = list(xm_list)
        for i in range(n_agents):
            xm = xm_list[i]
            wind = measured_list[i][0:2]
            vm = vms[i]
            other_positions = [current_positions[j] for j in range(n_agents) if j != i]
            best_score = np.inf
            best_xp, best_vp, best_W = None, None, None
            for action in test.actions_v_:
                vp = vm + action * test.dt_
                speed = np.linalg.norm(vp)
                if speed > v_max:
                    vp = vp / speed * v_max
                xp = np.clip(xm.astype(np.float64) + vp * test.dt_, xmin, xmax)
                if np.allclose(xp, xm):
                    continue
                emgr_info = emgrs[i].step(xp, wind, update_W=False, record_met=False)
                emgr_score = emgr_info["score"]
                if not np.isfinite(emgr_score):
                    continue
                score_temp = calc_multi_agent_score(test, xp, emgr_score, visit_map, other_positions)
                if score_temp < best_score:
                    best_score = score_temp
                    best_xp, best_vp, best_W = xp.copy(), vp.copy(), emgr_info["W"]
            if best_xp is None:
                best_xp, best_vp, best_W = xm.copy(), vm.copy(), np.zeros((2, 2))
            vms[i] = best_vp
            xm_list[i] = best_xp
            current_positions[i] = best_xp
            gx = int(np.clip(best_xp[0] // test.dx_, 0, visit_map.shape[0]-1))
            gy = int(np.clip(best_xp[1] // test.dy_, 0, visit_map.shape[1]-1))
            visit_map[gx, gy] += 1
            emgrs[i].compute_score(best_W, record_met=True)
            emgrs[i].update_W(best_W)
            trajs[i].append(best_xp.copy())

        if k % EVAL_EVERY == 0 or k == max_t - 1:
            record(k, "A_spatial")
        if verbose and k % 100 == 0:
            print(f"  [hybrid-A] step {k}/{max_t}, anchor={test.anchor_loc_}")

        # ---- Switching criterion: anchor stability (uses the anchor's own position, not argmax) ----
        if test.anchor_loc_ is not None:
            anchor_history.append(np.array(test.anchor_loc_, dtype=float))
        spatial_steps_done = k - last_k
        if len(anchor_history) >= switch_stability_window:
            recent = np.array(anchor_history[-switch_stability_window:])
            max_move = float(np.max(np.linalg.norm(recent - recent.mean(axis=0), axis=1)))
            if max_move < switch_stability_thresh:
                switch_step = k
                if verbose:
                    print(f"  [hybrid] max anchor displacement within the last {switch_stability_window} steps "
                         f"= {max_move:.2f}m < threshold {switch_stability_thresh}m, "
                         f"switching to phase B (temporal) at step {k}")
                break
        if spatial_steps_done >= max_spatial_steps:
            switch_step = k
            if verbose:
                print(f"  [hybrid] reached max_spatial_steps={max_spatial_steps}, "
                     f"forcing switch to phase B (temporal) at step {k}")
            break

    if switch_step is None:
        switch_step = max_t - 1
        if verbose:
            print(f"  [hybrid] Warning: the switching condition was never met; passively switching at the final data step "
                 f"{switch_step}")

    # ================================================================
    # Phase B: the leader (agent 0) continues single-agent OSP decisions, followers trail it staggered in time
    # ================================================================
    leader_idx = 0
    leader_xm = xm_list[leader_idx].copy()
    leader_vm = vms[leader_idx].copy()
    leader_emgr = emgrs[leader_idx]
    leader_visit_map = visit_map  # Keep using the visit_map accumulated in phase A, no reset

    follower_wait_positions = [xm_list[i].copy() for i in range(n_agents) if i != leader_idx]
    follower_indices = [i for i in range(n_agents) if i != leader_idx]

    leader_traj_since_switch = [leader_xm.copy()]

    for k in range(switch_step + 1, max_t):
        measured = srm.getMeasurementLookUp(leader_xm, data_k(k))
        wind = measured[0:2]
        test.updateAnchor(leader_xm[0:2], max(measured[3], 0.0), wind)
        g = srm.getCachedSrm(leader_xm, wind)
        srm.rlsUpdate(g, measured[3])

        best_score = np.inf
        best_xp, best_vp, best_W = None, None, None
        for action in test.actions_v_:
            vp = leader_vm + action * test.dt_
            speed = np.linalg.norm(vp)
            if speed > v_max:
                vp = vp / speed * v_max
            xp = np.clip(leader_xm.astype(np.float64) + vp * test.dt_, xmin, xmax)
            if np.allclose(xp, leader_xm):
                continue
            emgr_info = leader_emgr.step(xp, wind, update_W=False, record_met=False)
            emgr_score = emgr_info["score"]
            if not np.isfinite(emgr_score):
                continue
            score_temp = calc_score_local(test, xp, emgr_score, leader_visit_map)
            if score_temp < best_score:
                best_score = score_temp
                best_xp, best_vp, best_W = xp.copy(), vp.copy(), emgr_info["W"]
        if best_xp is None:
            best_xp, best_vp, best_W = leader_xm.copy(), leader_vm.copy(), np.zeros((2, 2))

        leader_vm, leader_xm = best_vp, best_xp
        gx = int(np.clip(leader_xm[0] // test.dx_, 0, leader_visit_map.shape[0]-1))
        gy = int(np.clip(leader_xm[1] // test.dy_, 0, leader_visit_map.shape[1]-1))
        leader_visit_map[gx, gy] += 1
        leader_emgr.compute_score(best_W, record_met=True)
        leader_emgr.update_W(best_W)
        trajs[leader_idx].append(leader_xm.copy())
        leader_traj_since_switch.append(leader_xm.copy())

        rel_k = k - switch_step
        for fi, agent_i in enumerate(follower_indices):
            lag = (fi + 1) * phase_lag
            idx = rel_k - lag
            if idx < 0:
                xm_f = follower_wait_positions[fi]
            else:
                idx = min(idx, len(leader_traj_since_switch) - 1)
                xm_f = leader_traj_since_switch[idx]
            measured_f = srm.getMeasurementLookUp(xm_f, data_k(k))
            wind_f = measured_f[0:2]
            test.updateAnchor(xm_f[0:2], max(measured_f[3], 0.0), wind_f)
            g_f = srm.getCachedSrm(xm_f, wind_f)
            srm.rlsUpdate(g_f, measured_f[3])
            trajs[agent_i].append(np.array(xm_f, dtype=float))

        if k % EVAL_EVERY == 0 or k == max_t - 1:
            record(k, "B_temporal")
        if verbose and k % 100 == 0:
            print(f"  [hybrid-B] step {k}/{max_t}, anchor={test.anchor_loc_}")

    return hist, [np.array(t) for t in trajs], test, switch_step



def run_five_way_multiseed(layer_conc, layer_ws, g_list, c_arr, n_seeds=N_SEEDS,
                           verbose=False):
    total_steps = layer_conc.shape[1]
    offsets = [int(i * total_steps / n_seeds) for i in range(n_seeds)]

    # method name -> {"global": [...], "anchor_argmax": [...]}
    results = {
        "lawnmower": {"global": []},
        "single": {"global": [], "anchor_argmax": []},
        "spatial": {"global": [], "anchor_argmax": []},
        "temporal": {"global": [], "anchor_argmax": []},
        "hybrid": {"global": [], "anchor_argmax": []},
    }

    # ---- Store the full error curve of every seed, used for the convergence-stability comparison plot ----
    # curves[method name]["global"/"anchor_argmax"] = [(step_arr, val_arr), ...]
    curves = {
        "lawnmower": {"global": []},
        "single": {"global": [], "anchor_argmax": []},
        "spatial": {"global": [], "anchor_argmax": []},
        "temporal": {"global": [], "anchor_argmax": []},
        "hybrid": {"global": [], "anchor_argmax": []},
    }

    # ---- lawnmower does not support time_offset (it uses the precomputed
    # fixed g_list/c_arr), so every seed gives exactly the same result; it is
    # moved outside the loop and computed only once to avoid repeated
    # computation -- this also fixes the earlier bug where "TestClass() was
    # used directly without calling createGridCells()", which left
    # grid_cells_ empty and the SRM matrix dimensions mismatched ----
    test_lawn_local = TestClass()
    test_lawn_local.createGridCells()
    hist_lawn = run_lawnmower_with_history(LAMBDA_LAWNMOWER, test_lawn_local, g_list, c_arr,
                                           eval_every=EVAL_EVERY)
    m_lawn = compute_robust_metrics(hist_lawn, key="global_err", eval_every=EVAL_EVERY,
                                    smooth_window_steps=SMOOTH_WINDOW_STEPS)
    print(f"\nlawnmower (computed once, does not vary with seed): global_err tail_avg={m_lawn['tail_avg']:.4f}")
    lawn_curve = (np.array(hist_lawn["step"]), np.array(hist_lawn["global_err"]))
    curves["lawnmower"]["global"].append(lawn_curve)

    print(f"\n{'='*60}")
    print(f"Five-method multi-seed comparison: n_seeds={n_seeds}")
    print(f"{'='*60}")

    for i, offset in enumerate(offsets):
        print(f"\n{'='*60}")
        print(f"seed {i} (time_offset={offset})")
        print(f"{'='*60}")

        results["lawnmower"]["global"].append(m_lawn["tail_avg"])
        print(f"  lawnmower: global_err tail_avg={m_lawn['tail_avg']:.4f} (reuse fixed value)")

        # ---- 2) single OSP ----
        hist_s, test_s = single_run_emgr_osp_with_offset(
            LAMBDA_SINGLE, layer_conc, layer_ws, total_steps,
            time_offset=offset, verbose=verbose)
        m_s_global = compute_robust_metrics(hist_s, key="global_err", eval_every=EVAL_EVERY,
                                            smooth_window_steps=SMOOTH_WINDOW_STEPS)
        curve_s = compute_anchor_restricted_argmax_error_curve(hist_s, test_s)
        m_s_aargmax = compute_robust_metrics_from_array(
            np.array(hist_s["step"]), curve_s, eval_every=EVAL_EVERY,
            smooth_window_steps=SMOOTH_WINDOW_STEPS)
        results["single"]["global"].append(m_s_global["tail_avg"])
        results["single"]["anchor_argmax"].append(m_s_aargmax["tail_avg"])
        curves["single"]["global"].append(
            (np.array(hist_s["step"]), np.array(hist_s["global_err"])))
        curves["single"]["anchor_argmax"].append(
            (np.array(hist_s["step"]), curve_s))
        print(f"  single   : global={m_s_global['tail_avg']:.4f}, "
             f"anchor_argmax={m_s_aargmax['tail_avg']:.4f}")

        # ---- 3) spatial ----
        test_sp = TestClass()
        test_sp.createGridCells()
        test_sp.conc_threshold_ = CONC_THRESHOLD
        hist_sp, _ = run_multi_agent_osp_with_history(
            LAMBDA_HYBRID, test_sp, layer_conc, layer_ws,
            n_agents=N_AGENTS, total_steps=total_steps,
            eval_every=EVAL_EVERY, verbose=verbose, time_offset=offset)
        m_sp_global = compute_robust_metrics(hist_sp, key="global_err", eval_every=EVAL_EVERY,
                                             smooth_window_steps=SMOOTH_WINDOW_STEPS)
        curve_sp = compute_anchor_restricted_argmax_error_curve(hist_sp, test_sp)
        m_sp_aargmax = compute_robust_metrics_from_array(
            np.array(hist_sp["step"]), curve_sp, eval_every=EVAL_EVERY,
            smooth_window_steps=SMOOTH_WINDOW_STEPS)
        results["spatial"]["global"].append(m_sp_global["tail_avg"])
        results["spatial"]["anchor_argmax"].append(m_sp_aargmax["tail_avg"])
        curves["spatial"]["global"].append(
            (np.array(hist_sp["step"]), np.array(hist_sp["global_err"])))
        curves["spatial"]["anchor_argmax"].append(
            (np.array(hist_sp["step"]), curve_sp))
        print(f"  spatial  : global={m_sp_global['tail_avg']:.4f}, "
             f"anchor_argmax={m_sp_aargmax['tail_avg']:.4f}")

        # ---- 4) temporal ----
        hist_t, _, test_t = run_multi_agent_sequential_follow(
            LAMBDA_HYBRID, layer_conc, layer_ws,
            n_agents=N_AGENTS, phase_lag=MANUAL_PHASE_LAG,
            total_steps=total_steps, eval_every=EVAL_EVERY,
            conc_threshold=CONC_THRESHOLD, time_offset=offset, verbose=verbose)
        m_t_global = compute_robust_metrics(hist_t, key="global_err", eval_every=EVAL_EVERY,
                                            smooth_window_steps=SMOOTH_WINDOW_STEPS)
        curve_t = compute_anchor_restricted_argmax_error_curve(hist_t, test_t)
        m_t_aargmax = compute_robust_metrics_from_array(
            np.array(hist_t["step"]), curve_t, eval_every=EVAL_EVERY,
            smooth_window_steps=SMOOTH_WINDOW_STEPS)
        results["temporal"]["global"].append(m_t_global["tail_avg"])
        results["temporal"]["anchor_argmax"].append(m_t_aargmax["tail_avg"])
        curves["temporal"]["global"].append(
            (np.array(hist_t["step"]), np.array(hist_t["global_err"])))
        curves["temporal"]["anchor_argmax"].append(
            (np.array(hist_t["step"]), curve_t))
        print(f"  temporal : global={m_t_global['tail_avg']:.4f}, "
             f"anchor_argmax={m_t_aargmax['tail_avg']:.4f}")

        # ---- 5) hybrid ----
        hist_h, _, test_h, switch_step = run_multi_agent_hybrid(
            LAMBDA_HYBRID, layer_conc, layer_ws, n_agents=N_AGENTS,
            total_steps=total_steps, time_offset=offset, verbose=verbose)
        m_h_global = compute_robust_metrics(hist_h, key="global_err", eval_every=EVAL_EVERY,
                                            smooth_window_steps=SMOOTH_WINDOW_STEPS)
        curve_h = compute_anchor_restricted_argmax_error_curve(hist_h, test_h)
        m_h_aargmax = compute_robust_metrics_from_array(
            np.array(hist_h["step"]), curve_h, eval_every=EVAL_EVERY,
            smooth_window_steps=SMOOTH_WINDOW_STEPS)
        results["hybrid"]["global"].append(m_h_global["tail_avg"])
        results["hybrid"]["anchor_argmax"].append(m_h_aargmax["tail_avg"])
        curves["hybrid"]["global"].append(
            (np.array(hist_h["step"]), np.array(hist_h["global_err"])))
        curves["hybrid"]["anchor_argmax"].append(
            (np.array(hist_h["step"]), curve_h))
        print(f"  hybrid   : global={m_h_global['tail_avg']:.4f}, "
             f"anchor_argmax={m_h_aargmax['tail_avg']:.4f}, switch_step={switch_step}")
    return results, curves



def compute_anchor_restricted_argmax_raw_curve(hist, test, radius_m=ANCHOR_ARGMAX_RADIUS_M):
    """
    Same search logic as compute_anchor_restricted_argmax_error_curve (take
    the Q-value peak within radius_m around the anchor), but returns the raw
    Q_hat value itself (not converted to an error) -- calibration needs "how
    large a number the model tends to read out", not "how far this number is
    from the ground truth".
    """
    if "anchor_loc" not in hist:
        raise ValueError("hist has no 'anchor_loc' field")

    cell_locs = np.array([c.x_ for c in test.grid_cells_], dtype=float)
    q_hats = []
    for q_flat, anchor_loc in zip(hist["Q_snapshot"], hist["anchor_loc"]):
        if anchor_loc is None:
            q_hats.append(np.nan)
            continue
        anchor_arr = np.asarray(anchor_loc, dtype=float)
        dists = np.linalg.norm(cell_locs - anchor_arr, axis=1)
        mask = dists <= radius_m
        if mask.sum() == 0:
            q_hats.append(np.nan)
            continue
        q_hats.append(float(q_flat[mask].max()))
    return np.array(q_hats)