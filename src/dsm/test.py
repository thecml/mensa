#from dsm import DeepSurvivalMachinesTorch
import torch 
from utilities import *
from DGP import Weibull_linear
from my_dsm import DeepSurvivalMachinesTorch

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

    copula_dgp = None
    theta_dgp = 2

    train_dict, val_dict, test_dict = \
        generate_data(x_dict, dgp1, dgp2, dgp3, device = 'cpu', copula = copula_dgp, theta = theta_dgp)
    
    print(torch.unique(train_dict['E'], return_counts=True))

    dgp_loss_train = loss_triple(dgp1, dgp2, dgp3, train_dict, None)
    dgp_loss_val = loss_triple(dgp1, dgp2, dgp3, val_dict, None)
    dgp_loss_test = loss_triple(dgp1, dgp2, dgp3, test_dict, None)
    print(dgp_loss_train, dgp_loss_val, dgp_loss_test)
    
    

    X = train_dict['X']
    T = train_dict['T']
    E = train_dict['E']

    """dgp_pdf_train = safe_log(dgp2.PDF(T, X))
    dgp_surv_train = safe_log(dgp2.survival(T, X))
    dgp_loss_train = -1.0 * torch.mean(dgp_pdf_train * (E==1)*1.0 + dgp_surv_train * (1-(E==1)*1.0))

    dgp_pdf_val = safe_log(dgp2.PDF(val_dict['T'], val_dict['X']))
    dgp_surv_val = safe_log(dgp2.survival(val_dict['T'], val_dict['X']))
    dgp_loss_val = -1.0 * torch.mean(dgp_pdf_val * (val_dict['E']==1)*1.0 + dgp_surv_val * (1-(val_dict['E']==1)*1.0))"""

    
    

    

    model = DeepSurvivalMachinesTorch(10, 1, [32,32], 1000, 3)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.005)
    for itr in range(20000):
        optimizer.zero_grad()
        loss = conditional_weibull_loss(model, X, T, E)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = conditional_weibull_loss(model, val_dict['X'], val_dict['T'], val_dict['E'])
        if itr % 100 == 0:
            print(loss, val_loss, dgp_loss_train, dgp_loss_val)
    

