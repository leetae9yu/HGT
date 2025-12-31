import argparse
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.decomposition import PCA


def load_history(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No probe history found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_frames(history, n_components=2):
    coords_all = np.array([rec["coord"] for snap in history for rec in snap["records"]], dtype=float)
    pca = PCA(n_components=n_components)
    pca.fit(coords_all)

    frames = []
    for snap in history:
        coords = np.array([rec["coord"] for rec in snap["records"]], dtype=float)
        coords_2d = pca.transform(coords)
        masses = np.array([rec["mass"] for rec in snap["records"]], dtype=float)
        groups = [rec["group"] for rec in snap["records"]]
        labels = [f"{rec['word']}:{rec['char']}" for rec in snap["records"]]
        frames.append(
            {
                "step": snap["step"],
                "coords": coords_2d,
                "masses": masses,
                "groups": groups,
                "labels": labels,
            }
        )
    return frames


def animate_frames(frames, output_path, fps=3):
    if not frames:
        raise ValueError("No frames available for animation.")

    all_coords = np.vstack([f["coords"] for f in frames])
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    pad_x = (x_max - x_min) * 0.1 + 1e-6
    pad_y = (y_max - y_min) * 0.1 + 1e-6

    unique_groups = sorted({g for frame in frames for g in frame["groups"]})
    palette = plt.cm.get_cmap("tab10", max(len(unique_groups), 1))
    group_to_color = {g: palette(i) for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=[])
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def init():
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        return scat, title

    def update(frame):
        coords = frame["coords"]
        masses = frame["masses"]
        colors = [group_to_color[g] for g in frame["groups"]]
        sizes = 50.0 * (masses / (masses.max() + 1e-9) + 0.1)
        scat.set_offsets(coords)
        scat.set_sizes(sizes)
        scat.set_color(colors)
        title.set_text(f"Step {frame['step']}")
        return scat, title

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)

    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".gif", ""):
        writer = animation.PillowWriter(fps=fps)
        ani.save(output_path or "gravity_evolution.gif", writer=writer)
    else:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize latent gravity evolution.")
    parser.add_argument(
        "--input",
        default=os.path.join("checkpoints", "gravity_evolution.pkl"),
        help="Path to the probe history pickle file.",
    )
    parser.add_argument(
        "--output",
        default="gravity_evolution.gif",
        help="Output path for the animation (gif or mp4).",
    )
    parser.add_argument("--fps", type=int, default=3, help="Frames per second for the animation.")
    args = parser.parse_args()

    history = load_history(args.input)
    frames = prepare_frames(history)
    animate_frames(frames, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
