# TUMOR INVASION SIMULATOR
#### Video Demo: <https://youtu.be/XF3OSmaczqI>
#### Description: 
This is a tumor growth simulation using Python and MATLAB. The simulation is projected onto a 2D grid (matplotlib), and is designed to model tumor invasion and motility given unlimited nutrients. Currently, I’m a Master’s student and researcher in Cancer Biology with a focus on Systems Biology, so this was my attempt to create a simple model tumor system in-silico. 

config.py: This file contains configuration parameters for the model. Notably, this include the size of the grid, the number of steps (timepoints), as well as parameters for nutrients and tumor cells. The arguments ‘seed’, ‘steps’, and ‘show_every’, the number of steps before an updated grid is rendered, are stored here as defaults but can be changed in the command line.

sim.py: This file contains rules by which the model runs and cells interact. 
__init__: Gives initial placements of nutrients and tumor cells on grid.
spread_pressure: Performs vectorized operations on the nutrient grid to return a Laplacian score.  This works by computing and then adding the nutrient scores of neighboring cells, and then comparing to initial cell nutrient level. If a grid point has a higher nutrient score than its neighbors (on average), it receives a negative Laplacian score, while if it has a lower nutrient score, it receives a positive score. These scores are then stored as a new grid.
apply_boundary_inflow: Assigns constant nutrient value to edges of screen in order to simulate constant inflow of nutrients from other parts of the body.
diffuse_nutrient: This function uses the spread_pressure function to simulate diffusion of nutrients. After generating a grid of Laplacian scores, it adds these scores to the existing nutrient grid in order to compensate for diffusion of nutrients (ie. areas high in nutrient will receive lower scores during the next step, and vice versa). This simulates nutrients flowing freely across the grid from areas of high concentration to low concentration. Additionally, this function renews nutrients at the boundaries via the apply_boundary_inflow function. Values are clipped when they are above or below a reasonable range, and decay is factored in.
tumor_consume: This function simulates consumption of nutrients via tumor cells. It works by establishing a new grid, and then subtracting the amount of nutrients specified by the configuration parameters (tumor_consume) from all cells which are identified currently as tumor.
divide_or_die: Based on a combination of nutrient levels in tumor cells and random chance, tumor cells will either divide, die, or remain stable in a given step. Cells are defined as ‘can_divide’ if they are tumor cells and have sufficient nutrients (nutrient >= tumor_divide_thresh). Locations of cells in can_divide are then recorded, and randomized to avoid directional bias before proceeding to division. Looping through randomized cells in can_divide, a random number generator decides whether division occurs, and a random empty neighbor is then chosen as the spot where division occurs. If this proceeds successfully, the chosen neighbor is converted to a tumor cell. Starvation also comes into effect here. If tumor cells have an insufficient level of nutrients, they are marked as starving. For starving cells, a random number is compared to the probability of death, and if lower, will result in these tumor cells being converted back into empty cells.
step: This function calls all previous functions in sim.py. This was ordered firstly by field dynamics, nutrient consumption, and then division.
snapshot: This function allows a single step to be saved or photographed later on. This is tool a utility function for research purposes.

viz.py: This file consists of one function called render_frame, and is used to visualize the simulation on MATLAB. In it, nutrient concentration is visualized on a scale from black to white (grayscale), and empty cells simply take on the color of nutrient in the cell. For specific cell types such as tumors, their locations are recorded and then passed in as objects called rgbs, which colors them as specified. It also contains instructions for creating the matplotlib window and creating axes.

run.py: This file compiles config.py, sim.py, and viz.py in order to run the simulation. It’s first function, ‘parse_args’, takes command line arguments to provide the option of changing small details within the simulation. The next function, ‘main’, uses parse_args, config.py, sim.py, and viz.py to run the program and render each step accordingly.


## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.run --size 256 --steps 800 --seed 42 --show-every 1

