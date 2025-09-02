import numpy as np
from dataclasses import dataclass
from .config import SimConfig, EMPTY, TUMOR

# Saves simulator's state at a given step/point in time
@dataclass
class State:
    cells: np.ndarray         # Grid of discrete cell types; int8: 0=EMPTY, 1=TUMOR
    nutrient: np.ndarray      # Continuous nutrient grid; float32 field
    step: int                 # Current step/point in time (tick)

class Simulator:
    """
    Tumor cells consume nutrient, divide if enough nutrient, die if starving.
    Nutrient is replenished at the boundary and diffuses inward.
    """

    def __init__(self, cfg: SimConfig):             # Imports values from config.py
        self.cfg = cfg                              # Draws parameters from config.py
        self.rng = np.random.default_rng(cfg.seed)  # Creates Numpy random number generator with a fixed seed

        # Grid (cells) and continuous field (nutrient)
        self.cells = np.zeros((cfg.size, cfg.size), dtype=np.int8)  # Builds cell grid, starts with all 0's (Healthy)
        self.nutrient = np.full((cfg.size, cfg.size), cfg.nutrient_init, dtype=np.float32)  # Whole grid starts with initial nutrient parameter (eg. 0.8)

        # Seed a small tumor in the center
        r = cfg.size // 2  # Finds center of grid (r)
        rad = np.maximum(3, cfg.size // 40)  # Small circle of radius (max) between 1/40 of grid (256 / 40) and 3
        yy, xx = np.ogrid[:cfg.size, :cfg.size]  # Creates coordinate grids (yy, xx). NumPy arrays are row-major, hence y, x
        tumor_mask = (xx - r) ** 2 + (yy - r) ** 2 <= rad ** 2  # Builds boolean mask that is equivalent to radius
        self.cells[tumor_mask] = TUMOR  # Converts cells within that mask to TUMOR (ie. 0 -> 1)

        self.step_count = 0  # Sets step count = 0 / initial step

        # Neighbor offsets (direction)
        self.neighbor_offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int8)  # Creates small 2D array of offset vectors (directions) relative to a given cell

    def spread_pressure(self, A: np.ndarray):
        """Compute Laplacian "spread pressure" for nutrients by checking concentrations of neighbors"""
        # If grid point has a higher nutrient value (Laplacian) than its neighbors, is negative; if lower, is positive
        # Negative values tend towards diffusion, positive will attract nutrients
        # Numpy array passed in, transformed array (with pressure considered) passed out
        # ChatGPT used to develop this function
        up = np.roll(A, -1, axis=0)  # Computes nutrient for neighbor below
        down = np.roll(A, 1, axis=0)  # Computes nutrient for neighbor above
        left = np.roll(A, -1, axis=1)  # Computes nutrient for neighbor to right
        right = np.roll(A, 1, axis=1)  # Computes nutrient for neighbor to left
        return(up + down + left + right - 4.0 * A)  # Returns Laplacian values for plane passed in

    def apply_boundary_inflow(self, A: np.ndarray, value: float):
        """Simulates nutrient inflow from boundaries"""
        # Replaces all boundary pixels (edges) with  a constant value (cfg.nutrient_boundary)
        A[0, :] = value     # All top row values (y=10) replaced with value
        A[-1, :] = value    # All bottom row values (y=246) replaced with value
        A[:, 0] = value     # All left-most col values (x=10) replaced with value
        A[:, -1] = value    # All right-most col values (x=246) replaced with value

    def diffuse_nutrient(self):
        """Simulates nutrient diffusion across plane"""
        cfg = self.cfg
        self.apply_boundary_inflow(self.nutrient, cfg.nutrient_boundary)

        spread = self.spread_pressure(self.nutrient)
        self.nutrient += cfg.nutrient_diffusion * spread    # Nutrient spots now equal to diffusion coefficient * neighbor values (pressure)
        self.nutrient -= cfg.nutrient_decay * self.nutrient  # Control for unlimited accumulation of nutrients

        np.clip(self.nutrient, 0.0, 1.5, out=self.nutrient)  # Overwrites nutrients below or above reasonable range

    def tumor_consume(self):
        """Simulate tumor consuming nutrients"""
        consume = np.zeros_like(self.nutrient, dtype=np.float32)  # Establishes new grid for consumption of nutrient
        consume[self.cells == TUMOR] = self.cfg.tumor_consume
        self.nutrient -= consume
        np.maximum(self.nutrient, 0.0, out=self.nutrient)  # Set boundary so nutrient does not become negative

    def tumor_divide_and_die(self):
        """Simulate tumor division and death"""
        # Use global variables locally
        cfg = self.cfg
        cells = self.cells
        nutrient = self.nutrient
        rng = self.rng

        H, W = cells.shape  # Height, width of grid
        tumor_mask = (cells == TUMOR)

        # Division
        can_divide = tumor_mask & (nutrient >= cfg.tumor_divide_thresh)  # if cell == TUMOR and nutrient >= thresh, it can divide

        # Division = each candidate tries to place a daughter in an empty neighbor
        # Randomize order to avoid directional bias
        ys, xs = np.where(can_divide)  # Locations of cells that can divide
        order = rng.permutation(len(ys))  # Returns randomized array of numbers (0, len(ys))
        ys, xs = ys[order], xs[order]  # Shuffled coordinates returned

        # ChatGPT for zip function
        # Loops through all cells in can_divide
        for y, x in zip(ys, xs):
            if rng.random() > cfg.tumor_divide_prob:    # Outputs a float from [0,1], continue if beats divide prob
                continue
            for i in range(4):
                dy, dx = self.neighbor_offsets[rng.integers(0, 4)]  # Pick one of 4 random neighbors
                ny, nx = y+dy, x+dx     # Add neighbor offsets to coordinates to pick division spot
                if 0 <= ny < H and 0 <= nx < W and cells[ny, nx] == EMPTY:  # Checks cells are not on edge, then checks whether neighbor is EMPTY (ie. not TUMOR)
                    cells[ny, nx] = TUMOR   # If that's all true, make neighbor TUMOR
                    break

        # Starvation
        starving = tumor_mask & (nutrient < cfg.tumor_starve_thresh)  # If point tumor and below thresh, starving
        die = (rng.random(starving.shape) < cfg.tumor_starve_die_prob) & starving
        cells[die] = EMPTY  # Sets cells where death occurred back to empty (same as healthy in this case)

    def step(self):
        """Advance simulation by one step; Run"""
        # 1) Field dynamics first
        self.diffuse_nutrient()

        # 2) Cell-field coupling (tumor consumes)
        self.tumor_consume()

        # 3) Cell rules (divide/die)
        self.tumor_divide_and_die()

        self.step_count += 1

    def snapshot(self) -> State:
        return State(cells=self.cells.copy(),
                     nutrient=self.nutrient.copy(),
                     step=self.step_count)
