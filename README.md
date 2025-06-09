# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based Likelihood Estimation

Given an instance, a multi-event survival model predicts the time until that instance experiences each of several different events. These events are not mutually exclusive and there are often statistical dependencies between them. MENSA is a novel, deep-learning model for multi-event survival analysis. MENSA works by jointly learning the event distributions as a convex combination of Weibull distributions. This approach leverages mutual information between events that may be lost in models that assume independence. MENSA also supports single-event and competing-risks predictions.

Preprint: https://arxiv.org/abs/2409.06525

<p align="left"><img src="https://github.com/thecml/mensa/blob/main/mensa.png">
Figure 1. MENSA generates survival predictions for multiple events based on patient covariates.

How to reproduce the results
--------
The code was tested in a virtual environment with Python 3.9 and PyTorch 1.13.1.

1. First, install the required packages specified in the [Requirements.txt](https://github.com/thecml/mensa/blob/main/requirements.txt) file.
2. Install the src directory by runnning: pip install -e .
3. Refer to config.py to set appropriate paths. By default, the results are in the /results folder.
4. To reproduce the paper results, please first obtain the raw datasets and put them in the /data folder.
5. After obtaining the raw datasets, run the data preprocessing scripts in /src/data/.
6. The experiments can then be executed by running the bash scripts in /scripts.

Demo
--------
See this [Notebook](https://github.com/thecml/mensa/blob/main/notebooks/demo.ipynb) for practical examples.

License
--------
To view the license for this work, visit https://github.com/thecml/mensa/blob/main/LICENSE

Citation
--------
If you find this paper useful in your work, please consider citing it:
 
```
@article{lillelund_mensa_2025,
  title={MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based Likelihood Estimation}, 
  author={Christian Marius Lillelund and Ali Hossein Gharari Foomani and Weijie Sun and Shi-ang Qi and Russell Greiner},
  journal={preprint, arXiv:2409.06525},
  year={2025},
}
```
