import numpy as np
import matplotlib.pyplot as plt
from .config import EMPTY, TUMOR

def render_frame(cells: np.ndarray, nutrient: np.ndarray, ax=None):
    """Combine nutrient (grayscale background) + cells (colored overlay) into one RGB image"""
    # Background: nutrient as grayscale
    bg = nutrient.copy()
    bg = (bg - bg.min()) / (bg.ptp() + 1e-8)
    rgb = np.stack([bg, bg, bg], axis=-1)  # shape (H, W, 3)

    # Overlay tumor cells in red
    tumor_mask = (cells == TUMOR)
    rgb[tumor_mask, 0] = 1.0  # R
    rgb[tumor_mask, 1] = 0.15 # G
    rgb[tumor_mask, 2] = 0.15 # B

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    ax.imshow(rgb, interpolation="nearest", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    return ax

