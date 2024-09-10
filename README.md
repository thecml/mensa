# MENSA: A Multi-Event Network for Survival Analysis under Informative Censoring

Preprint: TBA

Structure
--------
- /configs contains the configuration files for MENSA, the synthetic data generator and the Hierarchical model.
- /data contains the data files used for training the models.
- /notebooks contains Jupyter notebooks to plot the results in the paper.
- /results contains the model results once they have been executed.
- /scripts contains bash scripts to run the experiments.
- /src contains the source code.
- /src/data contains Python files to load and preprocess the raw data, once it has been put in /data.

Reproducing the results
--------
The code was tested in virtual environment with Python 3.9 and PyTorch 1.13.1.
By default, the experiments are performed 5 times with randomization seeds (0 1 2 3 4) for splitting the dataset.

1. First, install the required packages specified in the [Requirements.txt](https://github.com/thecml/mensa/blob/main/requirements.txt) file.
2. Install the src directory by runnning: pip install -e .
3. Refer to config.py to set appropriate paths. By default, the results are in results folder.
4. Download the data needed to train the models. Rotterdam is provided in the /data directory. The MIMIC-IV,
SEER and PRO-ACT datasets need to be downloaded from their respective website and preprocessed according to the
description in the Supplementary Material.
5. The experiments can be performed by running the bash scripts in /scripts.

License
--------
To view the license for this work, visit https://github.com/thecml/mensa/blob/main/LICENSE

Citation
--------
TBA
