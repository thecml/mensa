import torch
import torch.nn as nn

def create_representation(
    input_dim: int,
    layers: list,
    dropout_rate: float,
    activation: str,
    bias: bool = True
) -> nn.Sequential:
    """
    Build a feedforward representation network.

    Constructs an MLP with the specified layer sizes, activation function,
    batch normalization, and dropout for regularization.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    layers : list of int
        Sizes of hidden layers.
    dropout_rate : float
        Dropout probability applied after each hidden layer.
    activation : {'ReLU', 'ReLU6', 'SeLU', 'Tanh'}
        Activation function to apply between layers.
    bias : bool, optional (default=True)
        Whether to include bias terms in linear layers.

    Returns
    -------
    nn.Sequential
        A sequential container of fully connected layers with the
        specified configuration.
    """
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = input_dim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=bias))
        modules.append(nn.BatchNorm1d(hidden))
        modules.append(act)
        modules.append(nn.Dropout(p=dropout_rate))
        prevdim = hidden

    return nn.Sequential(*modules)

class MLP(torch.nn.Module):
    """
    Multi-layer perceptron for parameterizing Weibull mixture survival models.

    This network maps input features to Weibull mixture parameters
    (shape, scale, and mixture logits) for each event or state in the MENSA
    architecture. Each state has its own parameter-generating heads,
    while a shared representation network extracts common features.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    n_dists : int
        Number of Weibull mixture components per state.
    layers : list of int
        Sizes of hidden layers for the shared feature extractor (e.g., [32, 32]).
        If empty or None, no hidden layers are used.
    dropout_rate : float
        Dropout probability applied after each hidden layer.
    temp : float
        Temperature parameter for mixture softmax scaling (default = 1000).
    n_states : int
        Total number of modeled states (P), typically equal to K events (+1 if transient state is used).
    discount : float, optional (default=1.0)
        Reserved for potential temporal discounting; not currently used.

    Attributes
    ----------
    embedding : nn.Sequential
        Shared representation network for feature extraction.
    shapeg : nn.ModuleList
        Linear heads generating Weibull shape parameters for each state.
    scaleg : nn.ModuleList
        Linear heads generating Weibull scale parameters for each state.
    gate : nn.ModuleList
        Linear heads producing unnormalized mixture logits for each state.
    adapters : nn.ModuleList
        Lightweight adapters refining the shared representation per state.

    Notes
    -----
    - The model produces one set of (shape, scale, gate) parameters per state.
    - Both `shape` and `scale` are initialized to small negative values to
      promote stable early training.
    - Activations and batch normalization in the embedding network are handled
      by `create_representation()`.
    """
    def __init__(
        self,
        input_dim: int,
        n_dists: int,
        layers: list,
        dropout_rate: float,
        temp: float,
        n_states: int,
        discount: float = 1.0
    ):
        super(MLP, self).__init__()

        self.n_dists = n_dists
        self.temp = float(temp)
        self.discount = float(discount)
        self.n_states = n_states

        if layers is None:
            layers = []
        self.layers = layers

        lastdim = input_dim if len(layers) == 0 else layers[-1]

        self.act = nn.SELU()
        self.shape = nn.Parameter(-torch.ones(self.n_dists * n_states))
        self.scale = nn.Parameter(-torch.ones(self.n_dists * n_states))

        self.embedding = create_representation(input_dim, layers, dropout_rate, 'ReLU6')

        self.shapeg = nn.ModuleList([nn.Linear(lastdim, self.n_dists, bias=True) for _ in range(n_states)])
        self.scaleg = nn.ModuleList([nn.Linear(lastdim, self.n_dists, bias=True) for _ in range(n_states)])
        self.gate   = nn.ModuleList([nn.Linear(lastdim, self.n_dists, bias=False) for _ in range(n_states)])
        
        adapter_hidden = max(16, lastdim // 2)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lastdim, adapter_hidden, bias=True),
                nn.ReLU6(),
                nn.Linear(adapter_hidden, lastdim, bias=True),
            ) for _ in range(n_states)
        ])
        
    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor]]:
        """
        Forward pass through the MLP to produce Weibull mixture parameters
        for each modeled state.

        Parameters
        ----------
        x : torch.Tensor, shape (N, D)
            Input feature matrix.

        Returns
        -------
        outcomes : list of tuples
            A list of length `n_states`, where each element contains:
                (shape, scale, gate_logits)
            with each tensor of shape (N, n_dists):
                - shape : log(shape) parameters per mixture component
                - scale : log(scale) parameters per mixture component
                - gate_logits : unnormalized mixture logits (before log-softmax)

        Notes
        -----
        - The input is first passed through a shared embedding network, followed
        by per-state adapter modules to produce state-specific representations.
        - Each state has separate linear heads (`shapeg`, `scaleg`, `gate`)
        that predict Weibull mixture parameters.
        - Base parameters (`self.shape`, `self.scale`) act as learned offsets
        shared across all samples for stability.
        - Mixture logits are divided by `self.temp` to control softmax sharpness.

        Example
        -------
        >>> params = model(x_batch)
        >>> len(params)
        n_states
        >>> params[0][0].shape  # shape parameters for state 0
        (batch_size, n_dists)
        """
        outcomes = []
        n_samples = x.shape[0]

        xrep_shared = self.embedding(x)

        base_shape = self.shape.view(self.n_states, self.n_dists)
        base_scale = self.scale.view(self.n_states, self.n_dists)

        for i in range(self.n_states):
            xrep = xrep_shared
            
            xrep = xrep + self.adapters[i](xrep)

            shp_lin = self.shapeg[i](xrep)
            scl_lin = self.scaleg[i](xrep)

            shp_act = self.act(shp_lin)
            scl_act = self.act(scl_lin)

            shape = shp_act + base_shape[i].expand(n_samples, -1)
            scale = scl_act + base_scale[i].expand(n_samples, -1)

            gate_logits = self.gate[i](xrep) / self.temp

            outcomes.append((shape, scale, gate_logits))

        return outcomes