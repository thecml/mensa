# MENSA: A Multi-Event Network for Survival Analysis under Informative Censoring

MENSA is a novel, deep-learning model for multi-event survival analysis. It works by jointly learning K event distributions as a convex combination of Weibull distributions, treating the censoring distribution as just another event. This allows predicting if an instance will likely experience the event of interest or be censored, e.g., drop out of the study or be event-free at its termination. The model support single-event, competing risks and multi-event scenarios. Please consider citing the paper if you find this work useful.

<b>Preprint: https://arxiv.org/abs/2409.06525</b>

<p align="left"><img src="https://github.com/thecml/mensa/blob/main/mensa.png" width="60%" height="60%">
  
*Figure 1. MENSA generates survival distributions for K different events based on patient covariates, X. Θ represents the shared MLP layer that processes these covariates and
supports the prediction of survival outcomes across all K events.*

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
