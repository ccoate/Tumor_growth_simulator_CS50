import argparse
import time
import matplotlib.pyplot as plt
from .config import SimConfig
from .sim import Simulator
from .viz import render_frame

def parse_args():   # Add command line arguments to change run parameters w/o changing base model
    ap = argparse.ArgumentParser(description="Tumor Growth Simulator (MVP)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-every", type=int, default=1)
    return ap.parse_args()

# ChatGPT for configuring with MatLab
def main():     # Run  the model
    args = parse_args()
    cfg = SimConfig(size=args.size, steps=args.steps, seed=args.seed, show_every=args.show_every)
    sim = Simulator(cfg)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plt.ion()
    plt.show(block=False)

    last_render = time.time()
    for t in range(cfg.steps):
        sim.step()
        if (t % cfg.show_every) == 0:
            ax.clear()
            render_frame(sim.cells, sim.nutrient, ax=ax)
            ax.set_title(f"Step {sim.step_count} — Tumor cells: {(sim.cells==1).sum():,}")
            fig.canvas.draw()
            fig.canvas.flush_events()

    # Keep window open at the end
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

