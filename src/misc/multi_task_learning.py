import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

def gen_data(N):
    X = np.random.randn(N, Q)
    w1 = 2.
    b1 = 8.
    sigma1 = 1e1  # ground truth
    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    w2 = 3
    b2 = 3.
    sigma2 = 1e0  # ground truth
    Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2

def criterion(y_pred, y_true, log_vars):
  loss = 0
  for i in range(len(y_pred)):
    precision = torch.exp(-log_vars[i])
    diff = (y_pred[i]-y_true[i])**2.
    loss += torch.sum(precision * diff + log_vars[i], -1)
  return torch.mean(loss)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output1_size, output2_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output1_size)
        self.fc3 = nn.Linear(hidden_size, output2_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out1 = self.fc2(out)
        out2 = self.fc3(out)
        return out1, out2

if __name__ == "__main__":
    N = 100
    nb_epoch = 2000
    batch_size = 20
    nb_features = 1024
    Q = 1
    nb_output = 2  # total number of output
    D1 = 1  # first output
    D2 = 1  # second output
    
    model = Net(Q, nb_features, D1, D2)
    
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    
    std_1 = torch.exp(log_var_a)**0.5
    std_2 = torch.exp(log_var_b)**0.5
    print([std_1.item(), std_2.item()])
    
    # get all parameters (model parameters + task dependent log variances)
    params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b])
    
    optimizer = optim.Adam(params)
    
    X, Y1, Y2 = gen_data(N)
    X = X.astype('float32')
    Y1 = Y1.astype('float32')
    Y2 = Y2.astype('float32')
    
    loss_history = np.zeros(nb_epoch)

    for i in range(1):

        epoch_loss = 0
        
        for j in range(N//batch_size):
            
            optimizer.zero_grad()
            
            inp = torch.from_numpy(X[(j*batch_size):((j+1)*batch_size)])
            target1 = torch.from_numpy(Y1[(j*batch_size):((j+1)*batch_size)])
            target2 = torch.from_numpy(Y2[(j*batch_size):((j+1)*batch_size)])
            
            out = model(inp)
            
            print(log_var_a)
            loss = criterion(out, [target1, target2], [log_var_a, log_var_b])
            print(log_var_a)
            
            epoch_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
    
        loss_history[i] = epoch_loss * batch_size / N    