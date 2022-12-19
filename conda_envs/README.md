### INSTALLATION TUTORIAL with CONDA (miniconda) in Windows

This tutorial shows how to install, run and update the tools in this repository. 

#### Installation

1. Install MINICONDA: https://docs.conda.io/en/latest/miniconda.html#windows-installers
2. Download the conda environment from this folder (or copy paste with the same file name).
3. Install the environment with conda:
    1. Open the Anaconda Prompt (search anaconda in the windows Start Menu)
    2. In the terminal, navigate to the folder that contains the environment file using the command `cd`.
    3. In the terminal, type `conda env create --file=tlag_env_windows.yml`


#### Running the tools

1. Activate the environment:
    1. Open the Anaconda Prompt
    2. Type `conda activate TLAG-analysis`.
2. Run the tools:
    1. In the terminal, type `visualizer` or `peak_fitting`.

#### Updating the repository

1. Activate the environment:
    1. Open the Anaconda Prompt
    2. Type `conda activate TLAG-analysis`.

2. In the terminal, type `pip install --upgrade -e git+https://github.com/Jordi-aguilar/TLAGToolkit@master#egg=tlagtk`.
