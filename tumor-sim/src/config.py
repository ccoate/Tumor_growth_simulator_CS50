from dataclasses import dataclass

@dataclass
class SimConfig:
    size: int = 256                     # Matlab grid width/height
    steps: int = 2000                   # Total number of steps
    seed: int = 42                      # Seed for reproducibility

    # Nutrient field parameters
    nutrient_init: float = 0.8          # Starting nutrient level everywhere
    nutrient_boundary: float = 1.0      # Nutrient at boundary (constant)
    nutrient_diffusion: float = 0.18    # Diffusion constant
    nutrient_decay: float = 0.0005      # Decay to avoid runaway

    # Tumor parameters
    tumor_divide_thresh: float = 0.5    # Nutrient level needed for tumor to attempt division
    tumor_divide_prob: float = 0.25     # Prob tumor will successfully divide, given sufficient nutrient
    tumor_consume: float = 0.02         # Max nutrient each tumor cell and consume per step
    tumor_starve_thresh: float = 0.08   # Level of nutrient below which tumor cell can die
    tumor_starve_die_prob: float = 0.10  # Prob of tumor cell death if below tumor_starve_thresh

    # Rendering
    show_every: int = 1                 # Show updated grid every step

# Cell types (int codes)
EMPTY, TUMOR = 0, 1     # Normal cells = 0, Tumor = 1
