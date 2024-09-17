# MENSA: A Multi-Event Network for Survival Analysis under Informative Censoring

MENSA is a novel, deep-learning model for multi-event survival analysis. Given an instance, a multi-event survival model predicts the time until that instance experiences each of several different events. These events are not mutually exclusive and there are often statistical dependencies between them. MENSA works by jointly learning the K event distributions as a convex combination of Weibull distributions. This approach leverages mutual information between events that may be lost in models that assume independence. MENSA does also support single-event and competing risks scenarios by treating the censoring distribution as just another event. This allows predicting if a patient will likely experience the event of interest or be censored, e.g., drop out of the study or be event-free at its termination.

<b>Preprint: https://arxiv.org/abs/2409.06525</b>

<p align="left"><img src="https://github.com/thecml/mensa/blob/main/mensa.png">
  
*Figure 1. MENSA generates survival distributions for K different events based on patient covariates, X. Θ represents the shared MLP layer that processes these covariates and
supports the prediction of survival outcomes across all K events.*

How to use
--------
The code was tested in virtual environment with Python 3.9 and PyTorch 1.13.1.

1. First, install the required packages specified in the [Requirements.txt](https://github.com/thecml/mensa/blob/main/requirements.txt) file.
2. Install the src directory by runnning: pip install -e .
3. Refer to config.py to set appropriate paths. By default, the results are in results folder.
4. This [Demo Notebook](https://github.com/thecml/mensa/blob/main/notebooks/demo.ipynb) shows practical examples on how to train MENSA.
5. To reproduce the paper results, please first obtain the evaluated datasets. Rotterdam is provided in the /data directory. The MIMIC-IV, SEER and PRO-ACT datasets need to be downloaded from their respective website and preprocessed according to the description in the Supplementary Material. The experiments can then be performed by running the bash scripts in /scripts.

License
--------
To view the license for this work, visit https://github.com/thecml/mensa/blob/main/LICENSE

Citation
--------
Please consider citing the paper if you find this work useful.
 
```
@article{lillelund_mensa_2024,
  title={MENSA: A Multi-Event Network for Survival Analysis under Informative Censoring}, 
  author={Christian Marius Lillelund and Ali Hossein Gharari Foomani and Weijie Sun and Shi-ang Qi and Russell Greiner},
  journal={preprint, arXiv:2409.06525},
  year={2024},
}
```
