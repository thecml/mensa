# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based Likelihood Estimation

Source code for "MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based Likelihood Estimation", 2025. Under review.

Preprint: https://arxiv.org/abs/2409.06525

How to reproduce the results
--------
The code was tested in a virtual environment with Python 3.9 and PyTorch 1.13.1.

1. First, install the required packages specified in the [Requirements.txt](https://github.com/thecml/mensa/blob/main/requirements.txt) file.
2. Install the src directory by runnning: pip install -e .
3. Refer to config.py to set appropriate paths. By default, the results are in the /results folder.
4. To reproduce the paper results, please first obtain the raw datasets and put them in the /data folder.
5. After obtaining the raw datasets, run the data preprocessing scripts in /src/data/.
6. The experiments can then be executed by running the bash scripts in /scripts.

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
