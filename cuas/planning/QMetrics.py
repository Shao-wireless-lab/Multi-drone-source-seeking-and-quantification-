# QMetrics.py
import numpy as np


def read_Q(Q_flat, n_grid=25, radius=1):
    """
    From the SRM's Q_ vector, read out the emission rate estimate in three ways.
    Returns a dict: {'argmax', 'neighborhood', 'global', 'argmax_idx', 'argmax_loc_rc'}

    - argmax:       Q value of the peak cell
    - neighborhood: sum over the (2r+1)x(2r+1) neighborhood around the argmax (more robust in noisy environments)
    - global:       sum over the whole field (most accurate in clean environments)
    """
    #    """Sum over the (2*radius+1)x(2*radius+1) neighborhood around the argmax cell"""
    Q_2d = Q_flat.reshape(n_grid, n_grid)   # Keep the reshape consistent with the visualization
    idx = int(np.argmax(Q_flat))            # Convert the flat index back to 2D
    row, col = divmod(idx, n_grid)

    r0, r1 = max(0, row - radius), min(n_grid, row + radius + 1)
    c0, c1 = max(0, col - radius), min(n_grid, col + radius + 1)

    return {
        "argmax":       float(Q_flat[idx]),
        "neighborhood": float(Q_2d[r0:r1, c0:c1].sum()),
        "global":       float(Q_flat.sum()),
        "argmax_idx":   idx,
        "argmax_loc_rc": (row, col),
    }


def q_metrics(Q_flat, Q_true, n_grid=25, radius=1, eps=1e-12):
    """
    Compute the full set of metrics for the Q estimation.
    For each of the three readout methods (argmax/neighborhood/global), give:
      - Q_hat:      the estimated value
      - ratio:      Q_hat / Q_true  (Derek's normalized rate, ideal = 1.0)
      - abs_error:  |Q_hat - Q_true|
      - rel_error:  |Q_hat - Q_true| / |Q_true|  (dimensionless, comparable across different Q_true)

    Returns a nested dict: {'argmax': {...}, 'neighborhood': {...}, 'global': {...}}
    """
    reads = read_Q(Q_flat, n_grid=n_grid, radius=radius)
    out = {}
    for key in ("argmax", "neighborhood", "global"):
        q_hat = reads[key]
        out[key] = {
            "Q_hat":     q_hat,
            "ratio":     q_hat / (Q_true + eps),
            "abs_error": abs(q_hat - Q_true),
            "rel_error": abs(q_hat - Q_true) / (abs(Q_true) + eps),
        }
    out["argmax_idx"] = reads["argmax_idx"]
    out["argmax_loc_rc"] = reads["argmax_loc_rc"]
    return out


def loc_error(argmax_idx, grid_cells, true_loc):
    """source localization error: Euclidean distance from the center of the argmax cell to the true source"""
    est_loc = np.asarray(grid_cells[argmax_idx].x_, dtype=float)
    true_loc = np.asarray(true_loc, dtype=float)
    return float(np.linalg.norm(est_loc[:2] - true_loc[:2])), est_loc


def print_q_report(Q_flat, Q_true, grid_cells, true_loc, n_grid=25, radius=1):
    """Print line by line, convenient for a visual check after each experiment"""
    m = q_metrics(Q_flat, Q_true, n_grid=n_grid, radius=radius)
    le, est_loc = loc_error(m["argmax_idx"], grid_cells, true_loc)
    print(f"--- Q estimation report (Q_true={Q_true}) ---")
    print(f"location error: {le:.2f} m  (argmax loc rc={m['argmax_loc_rc']}), est_loc: {est_loc}, true_loc: {true_loc}")
    for key in ("argmax", "neighborhood", "global"):
        d = m[key]
        print(f"  {key:12s}: Q_hat={d['Q_hat']:.4f}  "
              f"Q/Q*={d['ratio']:.4f}  "
              f"abs_err={d['abs_error']:.4f}  "
              f"rel_err={d['rel_error']:.2%}")
    return m

def flatten_for_log(m, lam, Q_true):
    row = {"lambda_reg": lam, "Q_true": Q_true}
    for key in ("argmax", "neighborhood", "global"):
        for metric in ("Q_hat", "ratio", "abs_error", "rel_error"):
            row[f"{key}_{metric}"] = m[key][metric]
    return row