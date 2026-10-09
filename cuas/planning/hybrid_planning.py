#hybrid_planning.py: run this code to get multi-seeds comparison experiments between single, sapatial, temporal and hybrid strategies.

import numpy as np
import matplotlib.pyplot as plt

from planning_pipeline_lib import (SMOOTH_WINDOW_STEPS, EVAL_EVERY, N_SEEDS,
                                   build_lawnmower_stream, run_five_way_multiseed)


# Error convergence curve comparison
def smooth_series(val_arr, smooth_window_steps, eval_every):
    """
    Apply a moving average to hist["xxx_err"] to remove the short-period
    oscillations caused by the OSP trajectory circling in a small area /
    small anchor adjustments, keeping only the real convergence trend.
    """
    import pandas as pd
    win_pts = max(1, int(round(smooth_window_steps / eval_every)))
    s = pd.Series(val_arr)
    return s.rolling(window=win_pts, min_periods=1, center=True).mean().to_numpy()

def compute_mean_std_band(curve_list, common_steps,
                          smooth_window_steps=SMOOTH_WINDOW_STEPS,
                          eval_every=EVAL_EVERY):
    aligned = []
    for step_arr, val_arr in curve_list:
        step_arr = np.asarray(step_arr, dtype=float)
        val_arr = np.asarray(val_arr, dtype=float)

        smooth_val = smooth_series(val_arr, smooth_window_steps, eval_every)
        smooth_val = np.asarray(smooth_val, dtype=float)

        mask = ~np.isnan(smooth_val)
        if mask.sum() < 2:
            # This seed is almost entirely NaN (e.g. the anchor was never
            # triggered); skip it so it does not take part in the mean/std
            # statistics of this method
            continue

        s = step_arr[mask]
        v = smooth_val[mask]
        interp_val = np.interp(common_steps, s, v, left=v[0], right=v[-1])
        aligned.append(interp_val)

    if len(aligned) == 0:
        return np.full_like(common_steps, np.nan, dtype=float), \
            np.full_like(common_steps, np.nan, dtype=float)

    aligned = np.array(aligned)  # shape (n_valid_seeds, n_common_steps)
    mean_arr = np.nanmean(aligned, axis=0)
    std_arr = np.nanstd(aligned, axis=0)
    return mean_arr, std_arr


def compute_mean_curve_metrics(mean_arr, std_arr, common_steps,
                               threshold=0.40, tail_fraction=0.2):
    valid = ~np.isnan(mean_arr)
    below = np.where(valid & (mean_arr <= threshold))[0]
    time_to_threshold = int(common_steps[below[0]]) if len(below) > 0 else None

    valid_idx = np.where(valid)[0]
    n_tail = max(1, int(round(len(valid_idx) * tail_fraction)))
    tail_idx = valid_idx[-n_tail:]
    tail_mean = float(np.mean(mean_arr[tail_idx]))
    tail_std = float(np.mean(std_arr[tail_idx])) if std_arr is not None else None

    return {"time_to_threshold": time_to_threshold,
           "tail_mean": tail_mean, "tail_std": tail_std}


def print_paper_numbers_table(curves, compute_mean_std_band_fn,
                              names=("spatial", "temporal", "hybrid"),
                              threshold=0.40, tail_fraction=0.2):
    all_max_steps = []
    for name in names:
        for step_arr, _ in curves[name]["anchor_argmax"]:
            all_max_steps.append(np.max(step_arr))
    max_step = int(max(all_max_steps))
    sample_steps = np.asarray(curves[names[0]]["anchor_argmax"][0][0])
    eval_every_guess = int(np.median(np.diff(np.unique(sample_steps)))) if len(sample_steps) > 1 else 1
    common_steps = np.arange(0, max_step + 1, max(eval_every_guess, 1))

    print(f"\n{'='*60}")
    print(f"Paper numbers check table (threshold={threshold:.0%}, tail_fraction={tail_fraction:.0%})")
    print(f"{'='*60}")
    print(f"{'method':>10s} {'time_to_thresh':>16s} {'tail_mean':>12s} {'tail_std':>10s}")
    for name in names:
        mean_arr, std_arr = compute_mean_std_band_fn(curves[name]["anchor_argmax"],
                                                      common_steps)
        m = compute_mean_curve_metrics(mean_arr, std_arr, common_steps,
                                       threshold=threshold, tail_fraction=tail_fraction)
        ttt = f"{m['time_to_threshold']}" if m['time_to_threshold'] is not None else "not reached"
        print(f"{name:>10s} {ttt:>16s} {m['tail_mean']:>12.4f} {m['tail_std']:>10.4f}")

def plot_mean_std_grid(curves, compute_mean_std_band_fn,
                       save_path="four_way_mean_std_grid.png",
                       label_fontsize=30, tick_fontsize=30,
                       panel_label_fontsize=24, legend_fontsize=35):
    method_colors = {
        "single": "#e6194b",
        "spatial": "#3cb44b",
        "temporal": "#f58231",
        "hybrid": "#911eb4",
    }
    names = ["single", "spatial", "temporal", "hybrid"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    all_max_steps = []
    for name in names:
        for step_arr, _ in curves[name]["anchor_argmax"]:
            all_max_steps.append(np.max(step_arr))
    max_step = int(max(all_max_steps))
    sample_steps = np.asarray(curves[names[0]]["anchor_argmax"][0][0])
    eval_every_guess = int(np.median(np.diff(np.unique(sample_steps)))) if len(sample_steps) > 1 else 1
    common_steps = np.arange(0, max_step + 1, max(eval_every_guess, 1))

    # ---- 1x4 layout, to be placed spanning both columns (figure*) in the paper ----
    fig, axes = plt.subplots(1, 4, figsize=(28, 6.5), sharey=True)

    for ax, name, panel in zip(axes, names, panel_labels):
        mean_arr, std_arr = compute_mean_std_band_fn(curves[name]["anchor_argmax"],
                                                      common_steps)
        color = method_colors[name]
        ax.plot(common_steps, mean_arr, color=color, linewidth=2.5, label=name)
        ax.fill_between(common_steps, mean_arr - std_arr, mean_arr + std_arr,
                        color=color, alpha=0.22)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, panel, transform=ax.transAxes,
                fontsize=panel_label_fontsize, fontweight="bold", va="top")
        ax.legend(fontsize=legend_fontsize, loc="upper right")
        ax.set_xlabel("step", fontsize=label_fontsize)

    axes[0].set_ylabel("relative quantification error", fontsize=label_fontsize)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")


def plot_mean_std_comparison_three_way(curves, compute_mean_std_band_fn, EVAL_EVERY,
                                       save_path="three_way_mean_std_comparison.png",
                                       label_fontsize=30,
                                       legend_fontsize=35, tick_fontsize=30):
    method_colors = {
        "spatial": "#3cb44b",  # green
        "temporal": "#f58231",  # orange, kept well apart from red/purple
        "hybrid": "#911eb4",  # purple
    }
    method_labels = {
        "spatial": "spatial",
        "temporal": "temporal",
        "hybrid": "hybrid",
    }
    names = ["spatial", "temporal", "hybrid"]

    all_max_steps = []
    for name in names:
        for step_arr, _ in curves[name]["anchor_argmax"]:
            all_max_steps.append(np.max(step_arr))
    max_step = int(max(all_max_steps))
    common_steps = np.arange(0, max_step + 1, EVAL_EVERY)

    series_data = []
    for name in names:
        mean_arr, std_arr = compute_mean_std_band_fn(curves[name]["anchor_argmax"],
                                                     common_steps)
        band_area = np.nansum(std_arr)
        series_data.append((band_area, method_colors[name], method_labels[name],
                            mean_arr, std_arr))

    series_data.sort(key=lambda t: t[0], reverse=True)  # Draw the one with larger variance first (bottom layer)

    fig, ax = plt.subplots(figsize=(11, 7.5))

    for _, color, label, mean_arr, std_arr in series_data:
        ax.fill_between(common_steps, mean_arr - std_arr, mean_arr + std_arr,
                        color=color, alpha=0.15, zorder=1)
        ax.plot(common_steps, mean_arr - std_arr, color=color, alpha=0.55,
                linewidth=0.8, linestyle=":", zorder=2)
        ax.plot(common_steps, mean_arr + std_arr, color=color, alpha=0.55,
                linewidth=0.8, linestyle=":", zorder=2)

    for _, color, label, mean_arr, std_arr in series_data:
        ax.plot(common_steps, mean_arr, color=color, linewidth=2.4, label=label,
                zorder=3)

    ax.set_xlabel("step", fontsize=label_fontsize)
    ax.set_ylabel("relative quantification error", fontsize=label_fontsize)

    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    print("=== Data loading for all methods ===")
    test_lawn, xm_all, g_list, c_arr, layer_conc, layer_ws = build_lawnmower_stream(seed=0)

    print("\n=== multi seeds experiment of 5 methods ===\n")
    results, curves = run_five_way_multiseed(layer_conc, layer_ws, g_list, c_arr,
                                             n_seeds=N_SEEDS)
    # print results
    print_paper_numbers_table(curves, compute_mean_std_band)

    print("\n=== Plot comparison ===")

    plot_mean_std_grid(curves, compute_mean_std_band)

    plot_mean_std_comparison_three_way(curves, compute_mean_std_band, EVAL_EVERY)

    plt.show(block=True)