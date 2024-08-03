from DGP_models import Weibull_linear
#from Copula import  Clayton
from utils import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from final_model import *
from Copula_final import Nested_Convex_Copula, Clayton_Triple, Frank_Triple


if __name__ =="__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    #DEVICE = 'cpu'
    nf = 10
    n_train = 10000
    n_val = 5000
    n_test = 5000
    x_dict = synthetic_x(n_train, n_val, n_test, nf, device='cpu')
    
    dgp1 = Weibull_linear(nf, alpha=17, gamma=3, device='cpu')
    dgp2 = Weibull_linear(nf, alpha=16, gamma=3, device='cpu')
    dgp3 = Weibull_linear(nf, alpha=17, gamma=4, device='cpu')


    dgp1.coeff = torch.rand((nf,), device='cpu')
    dgp2.coeff = torch.rand((nf,), device='cpu')
    dgp3.coeff = torch.rand((nf,), device='cpu')
    copula_dgp = 'frank'
    theta_dgp = 1.0
    eps = 1e-3
    train_dict, val_dict, test_dict = \
        generate_data(x_dict, dgp1, dgp2, dgp3, device = 'cpu', copula = copula_dgp, theta = theta_dgp)
    
    print(torch.unique(train_dict['E'], return_counts=True))
    copula = Frank_Triple(theta = theta_dgp, eps = 1e-3, dtype=torch.float32, device='cpu')
    
    dgp_loss_train = loss_triple(dgp1, dgp2, dgp3, train_dict, copula)
    dgp_loss_val = loss_triple(dgp1, dgp2, dgp3, val_dict, copula)
    dgp_loss_test = loss_triple(dgp1, dgp2, dgp3, test_dict, copula)
    print(dgp_loss_train, dgp_loss_val, dgp_loss_test)
    copula = Clayton_Triple(theta = theta_dgp, eps = 1e-3, dtype=torch.float32, device=DEVICE)
    copula = Nested_Convex_Copula(['fr', 'fr'], ['fr'], [1,1], [1], 1e-3, dtype=torch.float32, device=DEVICE)
    mensa = Mensa(nf, 3, copula = copula, device = DEVICE)
    paramnet, copula = mensa.fit(train_dict, val_dict, n_epochs=1000)
    print(copula.parameters())
    print(mensa.Evaluate_L1(dgp1, dgp2, dgp3, test_dict, 200))
    """k1, k2, k3, lam1, lam2, lam3 = paramnet(test_dict['X'])
    with torch.no_grad():
        s1, s2, t, t_max = survival(dgp1, k1, lam1, test_dict['X'])
        plt.plot(t[0], s1[0])
        plt.plot(t[0], s2[0])

        s1, s2, t, t_max = survival(dgp2, k2, lam2, test_dict['X'])
        plt.plot(t[0], s1[0])
        plt.plot(t[0], s2[0])
        
        s1, s2, t, t_max = survival(dgp3, k3, lam3, test_dict['X'])
        plt.plot(t[0], s1[0])
        plt.plot(t[0], s2[0])
        
        plt.show()"""