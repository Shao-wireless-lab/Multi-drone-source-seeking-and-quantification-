# single_vs_hybrid_visualization.py

# Intention: Used to visualize and compare path planning process of both single and hybrid strategy.

# Hybrid strategy:
# Stage A (spatial): N agents follow the spatial logic (each makes its own EMGR decisions + separation
#   for mutual avoidance) and explore around the anchor until the anchor position is stable (judged by the anchor's own movement,
#   not argmax(Q_) -- since argmax has been confirmed to be contaminated by ill-conditioned solutions at low lambda,
#   whereas the anchor itself (with angle constraint + EMA smoothing) is unaffected, so using it as the switching criterion is safe).
# Stage B (temporal): at the switch point, one agent (agent 0) is promoted to "leader",
#   which keeps exploring freely with the single-agent OSP decision logic (no separation interference); the other
#   agents, starting from the switch point, each follow the trajectory the leader newly traces out afterwards, staggered (phase_lag),
#   and use the time offset to repeatedly sample the same spatial position at different turbulence instants, achieving ensemble averaging.

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

from planning_pipeline_lib import (
    build_lawnmower_stream, run_multi_agent_hybrid,
    N_AGENTS, CONC_THRESHOLD, LAMBDA_HYBRID)


def mp4_to_gif(mp4_path, gif_path, fps=10, width=640):
    """Convert mp4 to a compact GIF with a custom palette (much smaller than the pillow writer)."""
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path,
                    "-vf", vf, "-loop", "0", gif_path], check=True)


def animate_multi_agent(test, trajs, hist, layer_conc_height, title="Multi-agent OSP",
                        tail_length=200, interval=40, save_path=None, time_offset=0,
                        plume_max_alpha=0.85, plume_gamma=0.35, plume_threshold=None,
                        render_steps=None, formats=None, gif_fps=10, gif_width=640, speed=0.5):
    import matplotlib.animation as animation

    max_t = layer_conc_height.shape[0]
    if plume_threshold is None:
        plume_threshold = test.conc_threshold_

    def data_k(k):
        return (k + time_offset) % max_t

    fig, ax = plt.subplots(figsize=(9, 9))

    im = ax.imshow(np.zeros((25, 25)), origin='lower', cmap='turbo',
                   extent=[0, 200, 0, 200], interpolation='bilinear', zorder=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Estimated Source Strength")

    im_plume = ax.imshow(np.zeros((100, 100, 4)), origin='lower',
                         extent=[0, 200, 0, 200], interpolation='bilinear', zorder=2)

    ax.scatter(test.true_soure_loc_[0], test.true_soure_loc_[1],
              c='white', marker='x', s=220, linewidths=3, label='true source', zorder=6)

    n_agents = len(trajs)
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    agent_scatters = [ax.scatter([0], [0], c=[colors[i]], marker='o', s=90,
                                 edgecolors='black', label=f'agent {i}', zorder=6)
                      for i in range(n_agents)]
    tail_lines = [ax.plot([], [], color=colors[i], linewidth=1.5, alpha=0.6, zorder=5)[0]
                 for i in range(n_agents)]

    anchor_scatter = ax.scatter([], [], c='orange', marker='*', s=260,
                                edgecolors='black', label='anchor', zorder=6)

    ax.set_xlim([0, 200]); ax.set_ylim([0, 200])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc='upper right', fontsize=8)
    title_obj = ax.set_title(title)

    hist_steps_arr = np.array(hist["step"])
    Q_snaps = hist["Q_snapshot"]
    anchor_locs = hist["anchor_loc"]

    if render_steps is not None:
        # Target frame sequence: for each step to render, use searchsorted to find the nearest
        # (not later than it) recorded-point index in hist to take the snapshot, ensuring the frame count / step number per frame are consistent on both sides
        target_steps = np.array(render_steps)
        snap_idx_for_frame = np.searchsorted(hist_steps_arr, target_steps, side='right') - 1
        snap_idx_for_frame = np.clip(snap_idx_for_frame, 0, len(hist_steps_arr) - 1)
        frame_steps = target_steps
    else:
        frame_steps = hist_steps_arr
        snap_idx_for_frame = np.arange(len(hist_steps_arr))

    n_frames = len(frame_steps)

    def update(frame_idx):
        step = int(frame_steps[frame_idx])
        snap_i = int(snap_idx_for_frame[frame_idx])

        traj_step = min(step, min(len(t) for t in trajs) - 1)

        Q_map = Q_snaps[snap_i].reshape(25, 25)
        im.set_data(Q_map)
        im.set_clim(vmin=0, vmax=max(float(Q_map.max()), 1e-12))

        plume_frame_raw = layer_conc_height[data_k(step)]
        plume_frame = np.where(plume_frame_raw >= plume_threshold, plume_frame_raw, 0.0)
        plume_max = plume_frame.max()
        if plume_max > 0:
            norm_p = np.clip(plume_frame / plume_max, 0, 1) ** plume_gamma
        else:
            norm_p = np.zeros_like(plume_frame)
        rgba = np.zeros((*plume_frame.shape, 4))
        rgba[..., 0] = 0.15
        rgba[..., 1] = 0.95
        rgba[..., 2] = 0.15
        rgba[..., 3] = norm_p * plume_max_alpha
        im_plume.set_data(rgba.transpose(1, 0, 2))

        for i in range(n_agents):
            traj = trajs[i]
            s = min(traj_step, len(traj) - 1)
            traj_upto = traj[:s + 1]
            tail = traj_upto[-tail_length:] if len(traj_upto) > tail_length else traj_upto
            tail_lines[i].set_data(tail[:, 0], tail[:, 1])
            agent_scatters[i].set_offsets([[traj_upto[-1, 0], traj_upto[-1, 1]]])

        frame_anchor = anchor_locs[snap_i]
        if frame_anchor is not None:
            anchor_scatter.set_offsets([[frame_anchor[0], frame_anchor[1]]])
        else:
            anchor_scatter.set_offsets(np.empty((0, 2)))

        title_obj.set_text(f"{title}\nstep={step}")
        return im, im_plume, anchor_scatter, *agent_scatters, *tail_lines

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path is None:
        plt.show()
        return ani

    fps = max(1, (1000 // interval) * (2 / 3) * speed)
    base, ext = os.path.splitext(save_path)  # prefix + suffix
    formats = tuple(f.lower().lstrip(".") for f in (formats or (ext or ".mp4",)))
    saved = []

    mp4_path = f"{base}.mp4"
    if "mp4" in formats or "gif" in formats:  # render only once
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
        ani.save(mp4_path, writer=writer,
                 progress_callback=lambda i, n: print(f"  rendering frame {i}/{n}") if i % 10 == 0 else None)
        if "mp4" in formats:
            saved.append(mp4_path)

    if "gif" in formats:
        gif_path = f"{base}.gif"
        mp4_to_gif(mp4_path, gif_path, fps=min(gif_fps, fps), width=gif_width)
        saved.append(gif_path)
        if "mp4" not in formats:  # user only wanted the gif
            os.remove(mp4_path)

    print("Animation saved to: " + ", ".join(saved))
    return ani

################# video stitching
import subprocess
import shutil

def combine_videos_side_by_side(video_left, video_right, output_path,
                                target_height=1080, labels=None):
    """
    Stitch two already generated mp4 videos side by side into a new video, so that both
    methods' animations can be compared in one playback -- no need to re-run the simulation or re-render; this is purely post-processing of the two existing mp4 files
    (stitching).

    video_left/video_right: the respective mp4 paths, e.g. "single_debug.mp4" and
        "hybrid_debug.mp4".
    target_height: before stitching, scale both videos to the same height (in pixels), otherwise
        hstack errors out when the left and right heights differ -- the width does not matter, ffmpeg keeps
        each original aspect ratio and only adjusts the height.
    labels: optional, a (left_label, right_label) tuple, e.g.
        ("Single EMGR OSP", "Hybrid") -- if given, text labels are overlaid on the top-left corner of each video,
        making it easy to tell which method is on the left and right during playback (especially
        when the matplotlib titles of the two videos themselves are shrunk and hard to read).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg executable not found, please install it first: "
                          "conda install -c conda-forge ffmpeg")

    # Scale both videos to the same height first, keeping their respective aspect ratios
    scale_filter = f"scale=-2:{target_height}"

    if labels is not None:
        left_label, right_label = labels
        # drawtext overlays a text label at the top-left, white text with a black box (box=1) so it is legible on any background
        filter_complex = (
            f"[0:v]{scale_filter},drawtext=text='{left_label}':"
            f"x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[left];"
            f"[1:v]{scale_filter},drawtext=text='{right_label}':"
            f"x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[right];"
            f"[left][right]hstack=inputs=2[out]"
        )
    else:
        filter_complex = (
            f"[0:v]{scale_filter}[left];"
            f"[1:v]{scale_filter}[right];"
            f"[left][right]hstack=inputs=2[out]"
        )

    cmd = [
        "ffmpeg", "-y",              # -y: overwrite an existing output file directly, no interactive confirmation
        "-i", video_left,
        "-i", video_right,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-shortest",                 # when the two lengths are not exactly equal, truncate to the shorter one,
                                     # to avoid the finished stream freezing into a black screen with a long tail
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",       # consistent with the earlier mp4 export, ensuring it plays normally in Office
        output_path,
    ]

    print(f"Stitching videos: {video_left} + {video_right} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:")
        print(result.stderr[-2000:])  # only print the tail, which is usually the real cause of the error
        raise RuntimeError("Video stitching failed, please check the ffmpeg error message above")

    print(f"Stitching done, saved to {output_path}")


if __name__ == "__main__":
    print("=== Loading real MOABS data ===")
    _, _, _, _, layer_conc, layer_ws = build_lawnmower_stream(seed=0)
    total_steps = layer_conc.shape[1]

    #Run single OSP
    print(f"\n=== Running single (lambda={LAMBDA_HYBRID:.1e}, N={N_AGENTS}) ===")
    hist_osp, traj_osp, test_osp, switch_step = run_multi_agent_hybrid(
        LAMBDA_HYBRID, layer_conc, layer_ws,
        n_agents=1, total_steps=total_steps, verbose=True)
    test_osp.conc_threshold_ = CONC_THRESHOLD
    layer_conc_height = layer_conc[test_osp.test_z_ - 1]  # (Nt, Nx, Ny)

    #vedio settings
    TOTAL_STEPS = test_osp.Nt
    RENDER_STRIDE = 10  # Render one frame every 5 steps; the smaller the number, the smoother the video but the larger the file / the slower the rendering
    shared_render_steps = list(range(0, TOTAL_STEPS, RENDER_STRIDE))
    vedio_interval = 70

    animate_multi_agent(test_osp, traj_osp, hist_osp, layer_conc_height,
                        title="Single EMGR OSP", save_path="single_debug",
                        formats=("mp4",), interval=vedio_interval, render_steps=shared_render_steps, speed = 0.5)  # Use the same interval=40 as hybrid


    print(f"\n=== Running hybrid (lambda={LAMBDA_HYBRID:.1e}, N={N_AGENTS}) ===")
    hist, trajs, test, switch_step = run_multi_agent_hybrid(
        LAMBDA_HYBRID, layer_conc, layer_ws,
        n_agents=N_AGENTS, total_steps=total_steps, verbose=True)
    print(f"\nswitch_step = {switch_step}")

    animate_multi_agent(test, trajs, hist, layer_conc_height,
                        title=f"Hybrid (switch@{switch_step})", save_path="hybrid_debug.mp4",
                        formats=("mp4",), interval=vedio_interval, render_steps=shared_render_steps, speed = 0.5)
    print("Stitching start")
    combine_videos_side_by_side(
        "single_debug.mp4", "hybrid_debug.mp4",
        output_path="single_vs_hybrid_combined.mp4",
        labels=("Single EMGR OSP", "Hybrid")
    )

    # Side-by-side gif for the README: use a larger width because two frames are placed side by side
    mp4_to_gif("single_vs_hybrid_combined.mp4", "single_vs_hybrid_combined.gif",
                fps=10, width=1000)

    plt.show(block=True)