import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import animation
import torch.nn.functional as F

from models import *

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from trained flow")

    parser.add_argument("--checkpoint", type=str, default="output/TwoMoons.pth")
    parser.add_argument("--gif", type=str, default="output/TwoMoons.gif")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_config", type=str, default='{"model" : "MLP", "h" : 16}')

    return parser.parse_args()

def sample(model, n_samples, dim, n_steps, device, c, DDIM):
    model.eval()

    x = torch.randn(n_samples, *dim, device=device)
    time_steps = torch.linspace(0., 1., n_steps + 1, device=device)
    cond = None
    if c > 0:
        cond = torch.randint(0, max(c,2), (n_samples, 1)).to(device)

    xs = [x.detach().cpu()]

    with torch.no_grad():
        for i in range(n_steps):
            x = model.step(
                x_t=x,
                t_start=time_steps[i],
                t_end=time_steps[i + 1],
                cond=F.one_hot(cond.squeeze(), num_classes=c) if c > 1 else cond,
                DDIM=DDIM,
            )
            xs.append(x.detach().cpu())

    return xs, cond, time_steps.cpu()

def animate(xs, cond, dim, time_steps, kind=None, cmap="gray", vmin=0, vmax=1, save_gif=None, fps=30):

    if kind is None:
        if len(dim) == 1 and dim[0] == 2:
            kind = "2d"
        elif xs[0].dim() == 4:
            kind = "images"
        else:
            raise ValueError("Cannot infer kind from xs[0].shape")

    if kind == "images":
        N = xs[0].shape[0]
        ncols = int(N**0.5)
        nrows = (N + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = axes.flatten()

        artists = []

        for i, ax in enumerate(axes[:N]):
            im = ax.imshow(xs[0][i, 0], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")

            if cond is not None:
                ax.text(
                    0.5,
                    -0.1,
                    f"{cond[i].item()}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )

            artists.append(im)

        for ax in axes[N:]:
            ax.axis("off")

    elif kind == "2d":

        palette = [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        if cond is not None:
            colors = [palette[lbl % len(palette)] for lbl in cond.squeeze().tolist()]
        else:
            colors = "blue"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

        scat = ax.scatter(xs[0][:, 0], xs[0][:, 1], s=10, c=colors)
        artists = [scat]

    title = fig.suptitle(f"t = {time_steps[0]:.2f}")

    def draw_frame(frame):

        if kind == "images":
            imgs = xs[frame].view(-1, 1, *xs[0].shape[2:])
            for i, im in enumerate(artists):
                im.set_data(imgs[i, 0])

        elif kind == "2d":
            artists[0].set_offsets(xs[frame])

        title.set_text(f"t = {time_steps[frame]:.2f}")

    # ----------------
    # GIF export
    # ----------------

    if save_gif is not None:

        pause_frames = int(fps * 1.5)  # pause duration (seconds)

        forward = list(range(len(xs)))
        pause_end = [len(xs) - 1] * pause_frames
        backward = list(range(len(xs) - 2, -1, -1))
        pause_start = [0] * pause_frames

        frames = forward + pause_end + backward + pause_start

        def anim_update(frame):
            draw_frame(frame)
            return artists

        ani = animation.FuncAnimation(
            fig,
            anim_update,
            frames=frames,
            interval=1000 // fps,
            blit=False,
        )

        ani.save(save_gif, writer="pillow", fps=fps)
        print(f"GIF saved to {save_gif}")

    # ----------------
    # Interactive slider
    # ----------------
    else:
        plt.subplots_adjust(bottom=0.2)

        slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(slider_ax, "Step", valmin=0, valmax=len(xs) - 1, valinit=0, valstep=1)

        def update(val):
            frame = int(slider.val)
            draw_frame(frame)
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()

if __name__ == "__main__":
    args = parse_args()
    print(f"using {args.device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    dim = checkpoint["dim"]
    c = checkpoint["c"]
    model = load(args.model_config, dim, c).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    DDIM = checkpoint["DDIM"] if "DDIM" in checkpoint else False

    # Sample
    xs, cond, time_steps = sample(
        model,
        n_samples=args.n_samples,
        dim=dim,
        n_steps=args.n_steps,
        device=args.device,
        c=c,
        DDIM=DDIM,
    )

    animate(xs, cond, dim, time_steps, save_gif=args.gif)
