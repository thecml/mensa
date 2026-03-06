# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based Likelihood Estimation

**Accepted at ML4H 2025**  
[Preprint on arXiv](https://arxiv.org/abs/2409.06525)

Most existing time-to-event methods focus on either single-event or competing-risk settings, leaving multi-event scenarios relatively underexplored. In many real-world applications, the same patient may experience multiple, potentially semi-competing events. A common workaround is to train separate single-event models, but this approach fails to exploit shared dependencies and structure across events.

MENSA (Multi-Event Network for Survival Analysis) jointly models all events using shared neural representations and Weibull mixtures. It flexibly supports:
- Single-transition (competing-risks) and multi-transition (multi-state) survival formulations.
- An optional trajectory-based likelihood enforcing valid temporal ordering between events.
- Event-free (transient) states for modeling transitions from baseline to terminal or intermediate states.

Across four benchmark datasets, MENSA consistently improves predictive performance over many state-of-the-art baselines.

<p align="left">
  <img src="https://github.com/thecml/mensa/blob/main/mensa.png" width="700">
  <br>
  <em>Figure 1. MENSA jointly learns survival distributions across multiple events.</em>
</p>

Using MENSA
--------

The codebase supports both research replication and standalone model training.  
Below are concise examples to help you train and use MENSA on your own data.

### 1. Setup
```python
from mensa.models.mensa import MENSA

# Example: 50 features, 4 events, 3 Weibull components per event
model = MENSA(
    n_features=50,
    n_events=4,
    n_dists=3,
    layers=[32],
    dropout_rate=0.3,
    trajectories=[(1, 4), (2, 4), (3, 4)],  # optional
    use_transient=True,
    device='cuda'
)
```

### 2. Training
```python
train_dict = {
    'X': X_train,   # torch.Tensor [N, D]
    'T': T_train,   # torch.Tensor [N] or [N, K]
    'E': E_train    # torch.Tensor [N] or [N, K]
}

valid_dict = {
    'X': X_val,
    'T': T_val,
    'E': E_val
}

model.fit(
    train_dict,
    valid_dict,
    batch_size=32,
    n_epochs=100,
    patience=10,
    verbose=True
)
```

The training automatically:

- Adds an event-free transient state if enabled.
- Uses conditional Weibull log-likelihood (multi-event or single-event).
- Optionally adds trajectory-based regularization.
- Performs gradient clipping and early stopping.

### 3. Prediction

After training, you can predict survival or CDF curves for any event/state:

```python
import torch
import numpy as np

time_bins = torch.linspace(0, 1000, 200)  # time grid
risk = 2  # select which event/state to predict

# Predict survival probabilities
S = model.predict_survival(X_test, time_bins, risk=risk)

# Or the corresponding cumulative distribution (1 - S)
F = model.predict_cdf(X_test, time_bins, risk=risk)
```

Structure overview
--------
```
mensa/
├── models/
│   ├── mlp.py          # Base MLP architecture
│   ├── mensa.py        # MENSA wrapper (fit, predict, multi-event logic)
│   ├── losses.py       # Loss functions
│   ├── utility.py      # Utility
├── scripts/            # Example training scripts
├── notebooks/          # Notebooks
└── requirements.txt
```

Environment setup
--------
The code was tested with:

- Python 3.9
- PyTorch 1.13.1

```
# Install dependencies
pip install -r requirements.txt

# Install the source package
pip install -e .
```

Demo notebook
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
