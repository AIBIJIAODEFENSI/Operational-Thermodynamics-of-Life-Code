# Repository: Information-Energy-Criterion-Simulations
# This repository contains three Python scripts demonstrating:
#   1. error_catastrophe.py      - Quasispecies error threshold simulation
#   2. turing_mutation.py        - Turing pattern collapse due to mutation
#   3. lineage_phase_diagram.py  - Lineage persistence phase diagram (fixation probability)

## Required software environment

### Python version
- Python 3.6 or higher (the code uses f-strings and type hints)
- Recommended: Python 3.8 or later

### Required Python packages
- numpy (tested with version 1.19+)
- matplotlib (tested with version 3.3+)

## Installation instructions

It is recommended to use a virtual environment:

```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy matplotlib
```

## Running the scripts

```bash
python error_catastrophe.py
python turing_mutation.py
python lineage_phase_diagram.py
```

Each script will generate a PDF figure:
- error_catastrophe.pdf
- turing_mutation.pdf
- lineage_phase_diagram.pdf

## Additional notes
- The code uses explicit Euler integration with small time steps; no additional ODE solvers are required.
- For the lineage phase diagram, the fixation probability is computed via Kimura's diffusion approximation (vectorized).
- All scripts are self-contained and require no external data files.

## License
MIT