

class MultiNDE(nn.Module):
    def __init__(self, inputdim, n_events,
                 layers = [32, 32, 32],
                 layers_surv = [100, 100, 100],
                 dropout = 0.,
                 optimizer = "Adam"):
        super(MultiNDE, self).__init__()
        self.input_dim = inputdim
        self.dropout = dropout
        self.optimizer = optimizer
        self.n_events = n_events
        
        # Shared embedding
        self.embedding = create_representation(inputdim, layers, self.dropout)
        
        # Individual outputs  
        self.outcome = nn.ModuleList([create_representation_positive(1 + layers[-1],
                                                                     layers_surv + [1],
                                                                     dropout) for _ in range(n_events)])

    def forward(self, x, horizon, gradient=False):
        # Go through neural network
        x_embed = self.embedding(x) # Extract unconstrained NN
        time_outcome = horizon.clone().detach().requires_grad_(gradient) # Copy with independent gradient
        
        survival = [output_layer(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1))
                    for output_layer in self.outcome]

        # survival = self.outcome(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1)) # Compute survival % TODO: Why concat?
        survival = [surv.sigmoid() for surv in survival] # apply sigmoid func
        # Compute gradients
        if gradient:
            intensities = [grad(surv.sum(), time_outcome, create_graph = True)[0].unsqueeze(1) for surv in survival]
        else:
            intensities = None

        # return 1 - survival, intensity
        return [1 - surv for surv in survival], intensities

    def survival(self, x, horizon):  
        with torch.no_grad():
            horizon = horizon.expand(x.shape[0])
            temp = self.forward(x, horizon)[0]
        return temp