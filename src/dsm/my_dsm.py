import torch 
import torch.nn as nn

def create_representation(inputdim, layers, activation, bias=False):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=bias))
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)


class DeepSurvivalMachinesTorch(torch.nn.Module):
    """"
    k: number of distributions
    temp: 1000 default, temprature for softmax
    discount: not used yet 
    """
    def __init__(self, inputdim, k, layers, temp, risks, dropout_prob=0.25, discount=1.0):
        super(DeepSurvivalMachinesTorch, self).__init__()

        self.k = k
        self.temp = float(temp)
        self.discount = float(discount)
        
        self.risks = risks

        if layers is None: layers = []
        self.layers = layers

        if len(layers) == 0: lastdim = inputdim
        else: lastdim = layers[-1]

        self.act = nn.SELU()
        self.shape = nn.Parameter(-torch.ones(self.k *  risks))#(k * risk)
        self.scale = nn.Parameter(-torch.ones(self.k * risks))

        self.gate = nn.Linear(lastdim, self.k * self.risks, bias=False)
        self.scaleg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.shapeg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.embedding = create_representation(inputdim, layers, 'ReLU6')


    def forward(self, x):
        xrep = self.embedding(x)
        dim = x.shape[0]
        shape = self.act(self.shapeg(xrep))  + self.shape.expand(dim,-1)
        scale = self.act(self.scaleg(xrep)) + self.scale.expand(dim,-1)
        
        
        gate = self.gate(xrep) / self.temp
        outcomes = []
        for i in range(self.risks):
            outcomes.append((shape[:,i*self.k:(i+1)* self.k], scale[:,i*self.k:(i+1)* self.k], gate[:,i*self.k:(i+1)* self.k]))
        return outcomes
    
if __name__ == "__main__":
    x = torch.rand(10)
    x = x.reshape(-1,1).expand(-1, 5)
    print(x)
    #assert 0
    model = DeepSurvivalMachinesTorch(10, 5, [32,32], 1000, 3)
    x = torch.rand((2, 10))
    tmp = model(x)
    print(tmp[0])

