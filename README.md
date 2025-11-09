
---

# Environment Setup Guide

## Required Versions
- IsaacSim 4.5
- IsaacLab 2.2
- unitree_rl_lab

## Installation Steps
1. Clone the project repository:
   ```sh
   git clone https://github.com/cytochronne/State-Estimation-AMP-Lab.git
   ```
2. Follow the official IsaacLab documentation to install and configure IsaacSim and IsaacLab.
   - It is recommended to create a new conda environment for isolation.
   - Complete all dependency installations as described in the IsaacLab docs.
3. Set up unitree_rl_lab:
   - Enter the unitree_rl_lab directory and follow its README to install dependencies and configure the environment.
4. Install rsl_rl in editable (development) mode:
   ```sh
   cd IsaacLab
   ./isaaclab.sh -p -m pip install -e /path/to/rsl_rl
   ```
   - Replace `/path/to/rsl_rl` with the actual path to your rsl_rl source directory.

## Additional Notes
- Using a dedicated conda environment is strongly recommended to avoid dependency conflicts.
- If you encounter installation or runtime issues, consult the official documentation for each component and check the project's issues page.

