"""
global time-avg MOABS GP joint fit code
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
HEIGHT = 5
NX, NY = 100, 100
XMIN, XMAX = 0, 200
YMIN, YMAX = 0, 200
SOURCE_XY = (80, 60)
Q_TRUE = 5.225e-5
FS = 20  # Hz, original sampling frequency

RAW_TOTAL_SECONDS = 10000   #data duration seconds
RAW_NT_EXPECTED = RAW_TOTAL_SECONDS * FS  # data volumn: 20hz * data duration seconds

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))
RAW_DATA_DIR = os.path.join(_project_root, "Data",
                            "G=1_80_60_modified_Q_true_V7_eta_SNR_10_10000s_wind_and_quant_check")  # <-- modify according to the data path

MEAS_Z = float(HEIGHT)
WIND = np.array([1.0, 0.0, 0.0])

DEREK_ORIGINAL = {"a2": 0.1547, "b2": 0.8932, "a3": 0.1098, "b3": 1.064}
INIT_PARAMS_BASE = {"x0": 13.0, "a2": 0.1547, "b2": 0.8932, "a3": 0.1098,
                    "yc0": 0.0, "yc_slope": 0.0}
# BOUNDS now include the search ranges for yc0 and yc_slope
# yc0: allows an overall shift of up to ±30 m (enough margin, but not so large that it covers the whole ±100 m half-width domain)
# yc_slope: theoretical expected value ~0.038 (= v_bias/U); a range of ±0.2 leaves
#   ample margin while keeping the optimizer from running to unreasonable extreme slopes
BOUNDS = {"x0": (0, 40), "a2": (0.01, 1.0), "b2": (0.3, 1.5),
         "a3": (0.01, 1.0), "b3": (0.3, 3.0),
         "yc0": (-30, 30), "yc_slope": (-0.2, 0.2)}
MULTI_START_B3 = [0.5, 1.0, 1.5, 2.0, 2.5]

WINDOW_LENGTHS_SEC = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                     130, 140, 150, 160, 170, 180, 200, 250, 300, 350, 400,
                     450, 500]
# WINDOW_LENGTHS_SEC = [5, 10, 15, 20, 25, 30]
N_SEEDS_PER_WINDOW = 20  # How many independent windows to draw per window length for the statistics (upper limit;
                        # for large windows the number actually used is reduced automatically, see the comment above)

CHUNK_NT = 2000


# ------------------------------------------------------------------
# Chunked reading utilities
# ------------------------------------------------------------------
def get_raw_dataset(height, data_dir=RAW_DATA_DIR):
    filepath = os.path.join(data_dir, f"Plume-C-Data-Height-{height}.mat")
    f = h5py.File(filepath, 'r')
    dset = f['C']
    print(f"  Raw h5py dataset shape (before transpose) = {dset.shape}")
    Nt_inferred = dset.shape[1] // NX if dset.ndim == 2 else dset.shape[0] // NX
    print(f"  Inferred Nt = {Nt_inferred} (expected {RAW_NT_EXPECTED}, "
         f"{'consistent' if Nt_inferred == RAW_NT_EXPECTED else 'inconsistent, please check'})")
    return f, dset, Nt_inferred


def read_timestep_range(dset, t_start, t_end, Nt_total):
    t_start = max(0, t_start)
    t_end = min(Nt_total, t_end)
    col_start, col_end = t_start * NX, t_end * NX
    chunk = dset[:, col_start:col_end]
    chunk = chunk.T
    n_t = t_end - t_start
    chunk = chunk.reshape(n_t, NX, NY)
    return chunk


def compute_global_average_field(dset, Nt_total, chunk_nt=CHUNK_NT):
    accum = np.zeros((NX, NY), dtype=np.float64)
    n_chunks = int(np.ceil(Nt_total / chunk_nt))
    for i in range(n_chunks):
        t0 = i * chunk_nt
        t1 = min(Nt_total, t0 + chunk_nt)
        chunk = read_timestep_range(dset, t0, t1, Nt_total)
        accum += chunk.sum(axis=0)
        if i % max(1, n_chunks // 10) == 0:
            print(f"    Global average: processed {t1}/{Nt_total} steps ({100*t1/Nt_total:.0f}%)")
    return accum / Nt_total


# ------------------------------------------------------------------
# GP forward model: add yc0 and yc_slope
# ------------------------------------------------------------------
def build_field_general(x0, a2, b2, a3, b3, yc0=0.0, yc_slope=0.0,
                        source_xy=SOURCE_XY, z=MEAS_Z, wind=WIND):
    """
    The crosswind Gaussian term used to use y_rel (the deviation from the
    source y coordinate) directly; it now uses y_rel - y_center, where
    y_center(x_eff) = yc0 + yc_slope * x_eff.
    With yc0=0, yc_slope=0 it reduces to the original model (convenient for
    comparison with the baseline).
    """
    xs = np.asarray(source_xy, dtype=float)
    U = np.linalg.norm(wind) + 1e-8
    theta = np.arctan2(wind[1], wind[0])

    xvals = np.linspace(XMIN, XMAX, NX, endpoint=False) + (XMAX - XMIN) / NX / 2
    yvals = np.linspace(YMIN, YMAX, NY, endpoint=False) + (YMAX - YMIN) / NY / 2
    X, Y = np.meshgrid(xvals, yvals, indexing="ij")

    dx = X - xs[0]
    dy = Y - xs[1]
    x_rel = dx * np.cos(theta) + dy * np.sin(theta)
    y_rel = -dx * np.sin(theta) + dy * np.cos(theta)

    x_eff = x_rel + x0
    valid = x_eff > 1e-3

    field = np.zeros((NX, NY))
    xe = x_eff[valid]
    ye_raw = y_rel[valid]

    # The crosswind center drifts linearly with x_eff
    y_center = yc0 + yc_slope * xe
    ye = ye_raw - y_center

    sigma_y = np.maximum(a2 * np.power(xe, b2), 1e-6)
    sigma_z = np.maximum(a3 * np.power(xe, b3), 1e-6)

    H = 0.0
    g = (
        1.0 / (2 * np.pi * U * sigma_y * sigma_z)
        * np.exp(-0.5 * (ye / sigma_y) ** 2)
        * (np.exp(-0.5 * ((z - H) / sigma_z) ** 2)
           + np.exp(-0.5 * ((z + H) / sigma_z) ** 2))
    )
    field[valid] = g
    return field, xvals, yvals


def fit_Q_given_field(empirical_field, unit_field):
    y = empirical_field.ravel()
    f = unit_field.ravel()
    denom = np.sum(f * f)
    if denom < 1e-300:
        return 0.0, -np.inf, np.inf
    Q_hat = np.sum(y * f) / denom
    pred = Q_hat * f
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    return Q_hat, r2, rmse


def joint_fit(empirical_field, init_params, bounds=BOUNDS,
             source_xy=SOURCE_XY, z=MEAS_Z, wind=WIND):
    # === Change: param_names now includes yc0, yc_slope, so 7 parameters are fitted jointly ===
    param_names = ["x0", "a2", "b2", "a3", "b3", "yc0", "yc_slope"]
    x_init = [init_params[k] for k in param_names]
    bnds = [bounds[k] for k in param_names]

    def neg_r2(params):
        x0, a2, b2, a3, b3, yc0, yc_slope = params
        field, _, _ = build_field_general(x0, a2, b2, a3, b3, yc0, yc_slope,
                                          source_xy, z, wind)
        _, r2, _ = fit_Q_given_field(empirical_field, field)
        if not np.isfinite(r2):
            return 1e6
        return -r2

    result = minimize(neg_r2, x_init, method="L-BFGS-B", bounds=bnds,
                      options={"maxiter": 300, "ftol": 1e-8})

    x0, a2, b2, a3, b3, yc0, yc_slope = result.x
    field, xvals, yvals = build_field_general(x0, a2, b2, a3, b3, yc0, yc_slope,
                                              source_xy, z, wind)
    Q_hat, r2, rmse = fit_Q_given_field(empirical_field, field)

    fitted = {"x0": x0, "a2": a2, "b2": b2, "a3": a3, "b3": b3,
             "yc0": yc0, "yc_slope": yc_slope,
             "Q_hat": Q_hat, "r2": r2, "rmse": rmse}
    return fitted, field, xvals, yvals


def multi_start_joint_fit(field, verbose_label=""):
    scan_results = []
    for b3_init in MULTI_START_B3:
        init_params = dict(INIT_PARAMS_BASE, b3=b3_init)
        fitted_scan, _, _, _ = joint_fit(field, init_params=init_params)
        scan_results.append((b3_init, fitted_scan))
    best_b3_init, fitted = max(scan_results, key=lambda item: item[1]["r2"])
    if verbose_label:
        print(f"  [{verbose_label}] Using the group with the highest R^2 (b3_init={best_b3_init}): "
             f"x0={fitted['x0']:.2f}, a2={fitted['a2']:.4f}, b2={fitted['b2']:.4f}, "
             f"a3={fitted['a3']:.4f}, b3={fitted['b3']:.4f}, "
             f"yc0={fitted['yc0']:.3f}, yc_slope={fitted['yc_slope']:.4f}, "
             f"R^2={fitted['r2']:.4f}")
    return fitted


def downwind_profile(field, xvals, yvals, y_center):
    j = np.argmin(np.abs(yvals - y_center))
    return xvals, field[:, j]


def crosswind_profile(field, xvals, yvals, x_probe):
    i = np.argmin(np.abs(xvals - x_probe))
    return yvals, field[i, :]


def plot_global_fit_downwind(global_avg_field, fitted, fixed_shape_field,
                             baseline_Q, baseline_field, xvals, yvals,
                             save_path="global_fit_downwind_yc.png"):
    xv, prof_emp = downwind_profile(global_avg_field, xvals, yvals, SOURCE_XY[1])
    _, prof_fit = downwind_profile(fitted["Q_hat"] * fixed_shape_field, xvals, yvals, SOURCE_XY[1])
    _, prof_base = downwind_profile(baseline_Q * baseline_field, xvals, yvals, SOURCE_XY[1])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xv, prof_emp, "o-", ms=3, label="Empirical (global 10000s average)", color="tab:blue")
    ax.plot(xv, prof_fit, "--", lw=2, color="black",
           label=f"Joint fit w/ yc (R^2={fitted['r2']:.3f})")
    ax.plot(xv, prof_base, ":", color="gray",
           label=f"Baseline (Derek coeffs, yc=0, Q refit)")
    ax.set_title(f"Global-average downwind profile @ y={SOURCE_XY[1]} (source y, not fitted center)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("concentration [kg/m^3]")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")


def plot_global_fit_crosswind(global_avg_field, fitted, fixed_shape_field,
                              xvals, yvals, save_path="global_fit_crosswind_yc.png"):
    probe_xs = [SOURCE_XY[0] + 10, SOURCE_XY[0] + 20, SOURCE_XY[0] + 40]
    fitted_field_full = fitted["Q_hat"] * fixed_shape_field

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, px in zip(axes, probe_xs):
        yv, prof_emp = crosswind_profile(global_avg_field, xvals, yvals, px)
        _, prof_fit = crosswind_profile(fitted_field_full, xvals, yvals, px)
        # Mark the fitted center position for this probe point, to make the drift directly visible
        x_eff_at_px = (px - SOURCE_XY[0]) + fitted["x0"]
        yc_at_px = fitted["yc0"] + fitted["yc_slope"] * x_eff_at_px
        ax.plot(yv, prof_emp, "o-", ms=3, label="Empirical (global average)", color="tab:blue")
        ax.plot(yv, prof_fit, "--", lw=2, color="black", label="Joint fit w/ yc")
        ax.axvline(SOURCE_XY[1], color="gray", linestyle=":", alpha=0.5, label="source y (old fixed center)")
        ax.axvline(SOURCE_XY[1] + yc_at_px, color="red", linestyle="-.", alpha=0.6,
                  label=f"fitted center (yc={yc_at_px:.2f}m)")
        ax.set_title(f"Crosswind @ x={px}m (global 10000s average)")
        ax.set_xlabel("y [m]"); ax.set_ylabel("concentration [kg/m^3]")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")


def plot_residual_heatmap(global_avg_field, fitted, fixed_shape_field, xvals, yvals,
                          save_path="global_fit_residual_heatmap_yc.png",
                          label_fontsize=20, tick_fontsize=18, cbar_fontsize=18,
                          panel_label_fontsize=24):
    """
    1) The layout is changed back to 1 row x 4 columns -- in the paper this
       figure will span both columns, so it is no longer limited by the
       single-column width; a 1x4 horizontal arrangement is more compact
       than 2x2 and the reading order is more natural (from left to right:
       empirical / fitted / residual / relative residual).
    2) Keep the plain-text panel labels (a)/(b)/(c)/(d) (ax.text, not
       titles); do not restore the original descriptive title of each
       subplot, and leave the details to the LaTeX figure caption.
    3) Font sizes are enlarged somewhat for the 1x4 layout (slightly larger
       than the defaults of the 2x2 version), because each subplot now gets
       more width than in 2x2, and the larger fonts remain clear.
    """
    fitted_field_full = fitted["Q_hat"] * fixed_shape_field
    residual = global_avg_field - fitted_field_full

    mask = global_avg_field > (0.01 * global_avg_field.max())
    rel_residual = np.full_like(global_avg_field, np.nan)
    rel_residual[mask] = residual[mask] / global_avg_field[mask]

    extent = [XMIN, XMAX, YMIN, YMAX]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6.5))
    ax0, ax1, ax2, ax3 = axes

    im0 = ax0.imshow(global_avg_field.T, extent=extent, origin="lower",
                     aspect="auto", cmap="viridis")
    ax0.set_xlabel("x [m] (downwind)", fontsize=label_fontsize)
    ax0.set_ylabel("y [m] (crosswind)", fontsize=label_fontsize)
    ax0.scatter([SOURCE_XY[0]], [SOURCE_XY[1]], c="red", marker="x", s=80)
    ax0.tick_params(axis="both", labelsize=tick_fontsize)
    cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046)
    cb0.ax.tick_params(labelsize=cbar_fontsize)
    ax0.text(0.04, 0.95, "(a)", transform=ax0.transAxes, fontsize=panel_label_fontsize,
            fontweight="bold", va="top", color="white")

    im1 = ax1.imshow(fitted_field_full.T, extent=extent, origin="lower",
                     aspect="auto", cmap="viridis")
    ax1.set_xlabel("x [m] (downwind)", fontsize=label_fontsize)
    ax1.set_ylabel("y [m] (crosswind)", fontsize=label_fontsize)
    ax1.scatter([SOURCE_XY[0]], [SOURCE_XY[1]], c="red", marker="x", s=80)
    ax1.tick_params(axis="both", labelsize=tick_fontsize)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cb1.ax.tick_params(labelsize=cbar_fontsize)
    ax1.text(0.04, 0.95, "(b)", transform=ax1.transAxes, fontsize=panel_label_fontsize,
            fontweight="bold", va="top", color="white")

    vmax_res = np.abs(residual).max()
    im2 = ax2.imshow(residual.T, extent=extent, origin="lower",
                     aspect="auto", cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res)
    ax2.set_xlabel("x [m] (downwind)", fontsize=label_fontsize)
    ax2.set_ylabel("y [m] (crosswind)", fontsize=label_fontsize)
    ax2.scatter([SOURCE_XY[0]], [SOURCE_XY[1]], c="black", marker="x", s=80)
    ax2.tick_params(axis="both", labelsize=tick_fontsize)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046)
    cb2.ax.tick_params(labelsize=cbar_fontsize)
    ax2.text(0.04, 0.95, "(c)", transform=ax2.transAxes, fontsize=panel_label_fontsize,
            fontweight="bold", va="top")

    im3 = ax3.imshow(rel_residual.T, extent=extent, origin="lower",
                     aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_xlabel("x [m] (downwind)", fontsize=label_fontsize)
    ax3.set_ylabel("y [m] (crosswind)", fontsize=label_fontsize)
    ax3.scatter([SOURCE_XY[0]], [SOURCE_XY[1]], c="black", marker="x", s=80)
    ax3.tick_params(axis="both", labelsize=tick_fontsize)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cb3.ax.tick_params(labelsize=cbar_fontsize)
    ax3.text(0.04, 0.95, "(d)", transform=ax3.transAxes, fontsize=panel_label_fontsize,
            fontweight="bold", va="top")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")

    # ---- marginals figure (unchanged, still saved separately, not covered by this layout request) ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    marginal_x = np.abs(residual).sum(axis=1)
    axes2[0].plot(xvals, marginal_x, color="tab:red")
    axes2[0].axvline(SOURCE_XY[0], color="gray", linestyle=":", alpha=0.6, label="source x")
    axes2[0].set_xlabel("x [m] (downwind distance)")
    axes2[0].set_ylabel("sum(|residual|) over y")
    axes2[0].set_title("Where along downwind distance does error concentrate?")
    axes2[0].legend(); axes2[0].grid(alpha=0.3)

    marginal_y = np.abs(residual).sum(axis=0)
    axes2[1].plot(yvals, marginal_y, color="tab:blue")
    axes2[1].axvline(SOURCE_XY[1], color="gray", linestyle=":", alpha=0.6, label="source y")
    axes2[1].set_xlabel("y [m] (crosswind position)")
    axes2[1].set_ylabel("sum(|residual|) over x")
    axes2[1].set_title("Is error symmetric across the plume centerline (source y)?")
    axes2[1].legend(); axes2[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("global_fit_residual_marginals_yc.png", dpi=150)
    print(f"Figure saved to global_fit_residual_marginals_yc.png")



def sweep_window_lengths(dset, Nt_total, fixed_shape_field, window_lengths_sec,
                         n_seeds=N_SEEDS_PER_WINDOW, fs=FS):
    results = {}
    for wl_sec in window_lengths_sec:
        wl_steps = int(round(wl_sec * fs))
        max_non_overlap = Nt_total // wl_steps

        if max_non_overlap <= n_seeds:
            starts = [i * wl_steps for i in range(max_non_overlap)]
            print(f"  Window length={wl_sec}s: the whole dataset can be cut into at most {max_non_overlap} non-overlapping windows "
                 f"(<=n_seeds={n_seeds}); using all {max_non_overlap} non-overlapping windows instead of spreading start points evenly")
        else:
            max_start = Nt_total - wl_steps
            starts = [int(i * max_start / n_seeds) for i in range(n_seeds)]

        if len(starts) == 0:
            print(f"  Warning: window length {wl_sec}s ({wl_steps} steps) exceeds the total data length, skipping")
            continue

        errs = []
        for t0 in starts:
            t1 = t0 + wl_steps
            window_data = read_timestep_range(dset, t0, t1, Nt_total)
            window_avg = window_data.mean(axis=0)
            Q_hat, r2, rmse = fit_Q_given_field(window_avg, fixed_shape_field)
            rel_err = abs(Q_hat - Q_TRUE) / Q_TRUE
            errs.append(rel_err)

        errs = np.array(errs)
        results[wl_sec] = errs
        print(f"  Window length={wl_sec:3d}s ({wl_steps} steps): "
             f"mean_rel_err={errs.mean():.4f}, std={errs.std():.4f}  "
             f"(n_windows={len(errs)})")
    return results


def fit_asymptotic_trend(wls, means):
    from scipy.optimize import curve_fit
    wls = np.array(wls, dtype=float)
    means = np.array(means, dtype=float)

    def model(w, mean_inf, k):
        return mean_inf + k / w

    try:
        popt, pcov = curve_fit(model, wls, means, p0=[means[-1], 1.0])
        mean_inf, k = popt
        perr = np.sqrt(np.diag(pcov))
        return mean_inf, perr[0], k, model
    except Exception as e:
        print(f"  Asymptotic fit failed: {e}")
        return None, None, None, None


def plot_window_length_sweep(results, save_path="window_length_sweep_yc.png",
                             label_fontsize=20, tick_fontsize=20,
                             legend_fontsize=20):
    wls = sorted(results.keys())
    means = [results[w].mean() for w in wls]
    stds = [results[w].std() for w in wls]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax1 = axes[0]
    ax1.errorbar(wls, means, yerr=stds, marker="o", capsize=3, ms=4,
                 color="tab:blue", ecolor="tab:blue", alpha=0.7, linewidth=1.5,
                 label="Empirical (mean $\\pm$ 1 std)")  # <-- newly added label

    mean_inf, mean_inf_err, k, model = fit_asymptotic_trend(wls, means)
    if mean_inf is not None:
        w_smooth = np.linspace(min(wls), max(wls), 200)
        ax1.plot(w_smooth, model(w_smooth, mean_inf, k), "--", color="tab:red",
                 alpha=0.8, label=f"fit: mean_inf={mean_inf:.4f}±{mean_inf_err:.4f}")
        ax1.axhline(mean_inf, color="tab:red", linestyle=":", alpha=0.4)
        ax1.legend(fontsize=legend_fontsize)  # No change needed here; legend() automatically collects all artists that have a label
        print(f"\nAsymptotic trend fit (mean(w) = mean_inf + k/w):")
        print(f"  mean_inf = {mean_inf:.4f} ± {mean_inf_err:.4f}")

    ax1.set_xlabel("time-average window length (s)", fontsize=label_fontsize)
    ax1.set_ylabel("relative quantification error", fontsize=label_fontsize)
    ax1.tick_params(axis="both", labelsize=tick_fontsize)
    # ---- Change: explicitly limit the x-axis range to the actual data range, leaving no blank space ----
    ax1.set_xlim(min(wls) - 10, max(wls) + 10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    data = [results[w] for w in wls]
    bp = ax2.boxplot(data, labels=[str(w) for w in wls], showmeans=True, widths=0.6)
    for i, w in enumerate(wls):
        x = np.full(len(results[w]), i + 1, dtype=float)
        ax2.scatter(x, results[w], color="tab:blue", alpha=0.5, s=12, zorder=5)
    ax2.set_xlabel("window length (s)", fontsize=label_fontsize)
    ax2.set_ylabel("relative quantification error", fontsize=label_fontsize)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=90, labelsize=tick_fontsize - 4)
    ax2.tick_params(axis="y", labelsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")


def plot_global_fit_downwind_crosswind(global_avg_field, fitted, fixed_shape_field,
                                       baseline_Q, baseline_field, xvals, yvals,
                                       crosswind_x=100,
                                       save_path="global_fit_downwind_crosswind.png",
                                       label_fontsize=20, tick_fontsize=20,
                                       legend_fontsize=20, panel_label_fontsize=22):
    """
    Merge the two original figures (plot_global_fit_downwind +
    plot_global_fit_crosswind) into a single figure with left and right subplots:
      (a) downwind profile
      (b) crosswind profile, using only the single probe point crosswind_x=100
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6.5))

    # ---- (a) downwind profile ----
    xv, prof_emp = downwind_profile(global_avg_field, xvals, yvals, SOURCE_XY[1])
    _, prof_fit = downwind_profile(fitted["Q_hat"] * fixed_shape_field, xvals, yvals, SOURCE_XY[1])
    _, prof_base = downwind_profile(baseline_Q * baseline_field, xvals, yvals, SOURCE_XY[1])

    ax_a.legend(fontsize=legend_fontsize, loc="upper right")
    ax_a.plot(xv, prof_emp, "o-", ms=3, label="Empirical", color="tab:blue")
    ax_a.plot(xv, prof_fit, "--", lw=2, color="black",  label="Joint fit")
    ax_a.plot(xv, prof_base, ":", color="gray", label="Baseline in [15]")
    ax_a.set_xlabel("x [m]", fontsize=label_fontsize)
    ax_a.set_ylabel("concentration [kg/m$^3$]", fontsize=label_fontsize)
    ax_a.tick_params(axis="both", labelsize=tick_fontsize)
    ax_a.grid(alpha=0.3)
    ax_a.text(0.02, 0.95, "(a)", transform=ax_a.transAxes,
              fontsize=panel_label_fontsize, fontweight="bold", va="top")

    # ---- (b) crosswind profile, plotting only the single probe point crosswind_x=100 ----
    yv, prof_emp_cw = crosswind_profile(global_avg_field, xvals, yvals, crosswind_x)
    fitted_field_full = fitted["Q_hat"] * fixed_shape_field
    _, prof_fit_cw = crosswind_profile(fitted_field_full, xvals, yvals, crosswind_x)
    x_eff_at_px = (crosswind_x - SOURCE_XY[0]) + fitted["x0"]
    yc_at_px = fitted["yc0"] + fitted["yc_slope"] * x_eff_at_px

    ax_b.legend(fontsize=legend_fontsize, loc="upper right")

    ax_b.plot(yv, prof_emp_cw, "o-", ms=3, label="Empirical", color="tab:blue")
    ax_b.plot(yv, prof_fit_cw, "--", lw=2, color="black", label="Joint fit")
    ax_b.axvline(SOURCE_XY[1], color="gray", linestyle=":", alpha=0.5)  # label removed so it does not appear in the legend
    ax_b.axvline(SOURCE_XY[1] + yc_at_px, color="red", linestyle="-.", alpha=0.6,
                 label="Fitted center")
    ax_b.set_xlabel("y [m]", fontsize=label_fontsize)
    ax_b.set_ylabel("concentration [kg/m$^3$]", fontsize=label_fontsize)
    ax_b.tick_params(axis="both", labelsize=tick_fontsize)
    ax_b.grid(alpha=0.3)
    ax_b.text(0.02, 0.95, "(b)", transform=ax_b.transAxes,
              fontsize=panel_label_fontsize, fontweight="bold", va="top")



    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    print(f"Config check: FS={FS}Hz, total duration={RAW_TOTAL_SECONDS}s, "
         f"expected raw step count Nt={RAW_NT_EXPECTED}")

    print(f"\nOpening the raw data file (not loading into memory, only opening a handle)...")
    f, dset, Nt = get_raw_dataset(HEIGHT)

    if Nt != RAW_NT_EXPECTED:
        print(f"\n!! Warning: inferred Nt={Nt} does not match the expected {RAW_NT_EXPECTED} !!\n")

    try:
        print(f"\n===== Step 1: Compute the global time-averaged field (all {Nt} steps) =====")
        global_avg_field = compute_global_average_field(dset, Nt)
        print(f"  Global average field statistics: max={global_avg_field.max():.4e}, "
             f"mean={global_avg_field.mean():.4e}")

        print(f"\n===== Step 2: Refit jointly with the global average field (including yc0, yc_slope, multi-start scan over b3) =====")
        scan_results = []
        for b3_init in MULTI_START_B3:
            init_params = dict(INIT_PARAMS_BASE, b3=b3_init)
            fitted_scan, _, _, _ = joint_fit(global_avg_field, init_params=init_params)
            scan_results.append((b3_init, fitted_scan))
            print(f"  b3_init={b3_init:.2f}: x0={fitted_scan['x0']:.2f}, "
                 f"a2={fitted_scan['a2']:.4f}, b2={fitted_scan['b2']:.4f}, "
                 f"a3={fitted_scan['a3']:.4f}, b3={fitted_scan['b3']:.4f}, "
                 f"yc0={fitted_scan['yc0']:.3f}, yc_slope={fitted_scan['yc_slope']:.4f}, "
                 f"R^2={fitted_scan['r2']:.4f}")

        best_b3_init, fitted = max(scan_results, key=lambda item: item[1]["r2"])
        print(f"\n  Using the group with the highest R^2 (starting point b3_init={best_b3_init}) as the final global fit result:")
        print(f"  x0={fitted['x0']:.2f}, a2={fitted['a2']:.4f}, b2={fitted['b2']:.4f}, "
             f"a3={fitted['a3']:.4f}, b3={fitted['b3']:.4f}")
        print(f"  yc0={fitted['yc0']:.3f}m, yc_slope={fitted['yc_slope']:.4f} "
             f"(theoretical expected value if the v_bias assumption holds: ~v_bias/U ≈ 0.05/1.32 ≈ 0.038)")
        print(f"  Q_hat (direct fit to the global average field)={fitted['Q_hat']:.4e}, "
             f"Q_true={Q_TRUE:.4e}, R^2={fitted['r2']:.4f}")

        fixed_shape_field, xvals, yvals = build_field_general(
            fitted["x0"], fitted["a2"], fitted["b2"], fitted["a3"], fitted["b3"],
            fitted["yc0"], fitted["yc_slope"])

        print(f"\n===== Step 2b: Plot the downwind/crosswind profiles of the global fit =====")
        baseline_field, _, _ = build_field_general(0.0, **DEREK_ORIGINAL, yc0=0.0, yc_slope=0.0)
        baseline_Q, baseline_r2, _ = fit_Q_given_field(global_avg_field, baseline_field)
        print(f"  baseline (Derek's original coefficients, yc=0, Q refitted): Q_hat={baseline_Q:.4e}, "
             f"R^2={baseline_r2:.4f}  (for comparison; the lower it is, the larger the improvement after adding yc)")
        #Plot GP joint fit from downwind and crosswind
        plot_global_fit_downwind_crosswind(global_avg_field, fitted, fixed_shape_field,
                                           baseline_Q, baseline_field, xvals, yvals,
                                           crosswind_x=100)

        print(f"\n===== Step 2c: Spatial residual diagnostics (after adding yc, check whether the 'red on top, blue on the bottom' pattern disappears) =====")
        plot_residual_heatmap(global_avg_field, fitted, fixed_shape_field, xvals, yvals)

        print(f"\n===== Step 3: Fix the shape parameters (including yc), sweep window lengths {WINDOW_LENGTHS_SEC} =====")
        sweep_results = sweep_window_lengths(dset, Nt, fixed_shape_field,
                                             WINDOW_LENGTHS_SEC)

        plot_window_length_sweep(sweep_results)
        plt.show()

    finally:
        f.close()

    print("\nDone.")