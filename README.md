# AsyDM-SST

**AsyDM-SST** is the code repository for the Asynchronous Diffusion Model based on Schrödinger Bridge for Sea Surface Temperature Reconstruction.

This project implements a diffusion-based model to reconstruct missing SST data caused by cloud cover. It incorporates cloud masks and anomaly detection for high-accuracy reconstruction and visualization.

## Environment Setup

All dependencies are listed in `environment.yml`. We recommend using Conda.

```bash
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate dsb

# 2. (Optional) Verify the environment
python --version
Dataset
Example datasets are provided in the data/ folder:

0.1_Cloud_mask_68_train_South_Sea_miss.h5 — Dataset
Cloud_mask_68.h5 — Cloud mask data

The data is stored in HDF5 (.h5) format and includes SST values along with corresponding cloud masks. It can be used directly for training and inference.
Usage
The main entry point of the program is main.py.
Bash# Run the main program
python main.py

The script will automatically load the datasets from the data/ folder.
Training/inference modes and hyperparameters can be adjusted directly in main.py.

Project Structure
textAsyDM-SST/
├── data/                    # Example datasets (.h5 files)
├── guided_diffusion/        # Core diffusion model modules
├── anomaly_mask.py          # Anomaly mask generation
├── deal_sst_util.py         # SST data preprocessing utilities
├── diffusion.py             # Diffusion process implementation
├── main.py                  # Main program entry (training/inference)
├── mask_obtain.py           # Mask acquisition script
├── model.py                 # Model definition
├── reconstru_visual.py      # Reconstruction visualization
├── utils.py                 # General utility functions
├── environment.yml          # Conda environment configuration
└── README.md
