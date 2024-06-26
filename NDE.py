import torch
from l1_eval import Survival, surv_diff, SurvivalNDE, surv_diff_NDE
from test_script import LOG
from Copula import Clayton
import itertools
# from torch.distributions import Weibull,LogNormal

def linear(x):
    return x
class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        y = exp.log1p()
        return x.where(torch.isinf(exp),y.half() if x.type()=='torch.cuda.HalfTensor' else y )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (-x).exp().half() if x.type()=='torch.cuda.HalfTensor' else (-x).exp()
        return grad_output / (1 + y)

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class CustomSwish(torch.nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)

class multi_input_Sequential(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class multi_input_Sequential_res_net(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                output = module(inputs)
                if inputs.shape[1]==output.shape[1]:
                    inputs = inputs+output
                else:
                    inputs = output
        return inputs

log1plusexp = Log1PlusExp.apply
class nn_node(torch.nn.Module): #Add dropout layers, Do embedding layer as well!
    def __init__(self,d_in,d_out,cat_size_list,dropout=0.1,transformation=torch.tanh):
        super(nn_node, self).__init__()

        self.has_cat = len(cat_size_list)>0
        self.latent_col_list = []
        print('cat_size_list',cat_size_list)
        for i,el in enumerate(cat_size_list):
            col_size = el//2+2
            setattr(self,f'embedding_{i}',torch.nn.Embedding(el,col_size))
            self.latent_col_list.append(col_size)
        self.w = torch.nn.Linear(d_in+sum(self.latent_col_list),d_out)
        self.f = transformation
        self.dropout = torch.nn.Dropout(dropout)
        self.lnorm = torch.nn.LayerNorm(d_out)
    def forward(self,X,x_cat=[]):
        if not isinstance(x_cat,list):
            seq = torch.unbind(x_cat,1)
            cat_vals = [X]
            for i,f in enumerate(seq):
                o = getattr(self,f'embedding_{i}')(f)
                cat_vals.append(o)
            X = torch.cat(cat_vals,dim=1)
        return self.dropout(self.f(self.lnorm(self.w(X))))


class bounded_nn_layer(torch.nn.Module): #Add dropout layers
    def __init__(self, d_in, d_out, bounding_op, transformation=torch.tanh):
        super(bounded_nn_layer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(*(d_in,d_out))/d_in**0.5,requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out)/d_out**0.5,requires_grad=True)

    def forward(self,X):
        return self.f(X@self.bounding_op(self.W)+self.bias)

class bounded_nn_layer_last(torch.nn.Module):  # Add dropout layers
    def __init__(self, d_in, d_out, bounding_op, transformation=torch.tanh):
        super(bounded_nn_layer_last, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(*(d_in, d_out))/d_in**0.5, requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out)/d_out**0.5, requires_grad=True)

    def forward(self, X):
        return X @ self.bounding_op(self.W) + self.bias


class unbounded_nn_layer(torch.nn.Module): #Add dropout layers
    def __init__(self, d_in, d_out, bounding_op, transformation=torch.tanh,dropout=0.1):
        super(unbounded_nn_layer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(*(d_in,d_out))/d_in**0.5,requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out)/d_out**0.5,requires_grad=True)
        self.dropout = torch.nn.Dropout(p=dropout)
    def forward(self,X):
        return self.dropout(self.f(X@self.W+self.bias))

class unbounded_nn_layer_last(torch.nn.Module):  # Add dropout layers
    def __init__(self, d_in, d_out, bounding_op, transformation=torch.tanh):
        super(unbounded_nn_layer_last, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(*(d_in, d_out))//d_in**0.5, requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out)/d_out**0.5, requires_grad=True)

    def forward(self, X):
        return X @ self.W + self.bias

class mixed_layer(torch.nn.Module): #Add dropout layers
    def __init__(self, d_in, d_in_bounded, d_out, bounding_op, transformation=torch.tanh):
        super(mixed_layer, self).__init__()
        self.pos_weights = torch.nn.Parameter(torch.randn(*(d_in_bounded, d_out))/d_in_bounded**0.5, requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out)/d_out**0.5, requires_grad=True)
        self.w = torch.nn.Linear(d_in,d_out)

    def forward(self,X,x_bounded):
        return self.f(x_bounded @ self.bounding_op(self.pos_weights) + self.bias + self.w(X))

class mixed_layer_all(torch.nn.Module): #Add dropout layers
    def __init__(self, d_in, d_in_bounded,cat_size_list,d_out,bounding_op, transformation=torch.tanh,dropout=0.0):
        super(mixed_layer_all, self).__init__()
        self.pos_weights = torch.nn.Parameter(torch.randn(*(d_in_bounded, d_out))/d_in_bounded**0.5, requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out)/d_out**0.5, requires_grad=True)
        self.x_node = nn_node(d_in=d_in,d_out=d_out,cat_size_list=cat_size_list,dropout=dropout,transformation=linear)

    def forward(self,X,x_cat,x_bounded):
        return  self.f(x_bounded @ self.bounding_op(self.pos_weights) + self.bias + self.x_node(X,x_cat))

class mixed_layer_2(torch.nn.Module): #Add dropout layers
    def __init__(self, d_in, d_in_bounded, d_out, bounding_op, transformation=torch.tanh):
        super(mixed_layer_2, self).__init__()
        self.pos_weights = torch.nn.Parameter(torch.randn(*(d_in_bounded, d_out//2))/d_in_bounded**0.5, requires_grad=True)
        self.f = transformation
        self.bounding_op = bounding_op
        self.bias = torch.nn.Parameter(torch.randn(d_out//2)/d_out**0.5, requires_grad=True)
        self.w = torch.nn.Linear(d_in,d_out//2)

    def forward(self,X,x_bounded):
        bounded_part = x_bounded @ self.bounding_op(self.pos_weights) + self.bias
        regular_part = self.w(X)
        return self.f( torch.cat([bounded_part,regular_part],dim=1))


class survival_net_basic(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers_t,
                 layers,
                 dropout=0.9,
                 bounding_op=torch.relu,
                 transformation=torch.relu,
                 direct_dif = True,
                 objective = 'hazard',
                 eps=1e-6
                 ):
        super(survival_net_basic, self).__init__()
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,dropout)
        self.init_middle_net(dx_in=layers_x[-1], d_in_y=d_in_y, d_out=d_out, layers=layers,
                             transformation=transformation, bounding_op=bounding_op)
        self.eps = eps
        self.direct = direct_dif
        self.objective  = objective

        if self.objective in ['hazard','hazard_mean']:
            self.f = self.forward_hazard
            self.f_cum = self.forward_cum_hazard
        elif self.objective in ['S','S_mean']:
            self.f=self.forward_f
            self.f_cum=self.forward_S

    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,dropout):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation,dropout=dropout)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation,dropout=dropout))
        self.covariate_net = multi_input_Sequential_res_net(*module_list)

    def init_middle_net(self, dx_in, d_in_y, d_out, layers, transformation, bounding_op):
        
        module_list = [mixed_layer_2(d_in=dx_in, d_in_bounded=d_in_y, d_out=layers[0], bounding_op=bounding_op,
                                   transformation=transformation)]
        for l_i in range(1,len(layers)):
            module_list.append(bounded_nn_layer(d_in=layers[l_i - 1], d_out=layers[l_i], bounding_op=bounding_op,
                                                transformation=transformation))
        module_list.append(
            bounded_nn_layer_last(d_in=layers[-1], d_out=d_out, bounding_op=bounding_op, transformation=linear))
        self.middle_net = multi_input_Sequential_res_net(*module_list)

    def forward(self,x_cov,y,x_cat=[]):
        return self.f(x_cov,y,x_cat)

    def forward_cum(self,x_cov,y,mask,x_cat=[]):
        return self.f_cum(x_cov, y,mask,x_cat)

    def forward_S(self,x_cov,y,mask,x_cat=[]):#return log(S)
        x_cov = x_cov#[~mask,:]
        y = y#[~mask,:]
        if not isinstance(x_cat,list):
            x_cat=x_cat#[~mask,:]
        #Fix categorical business
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov,y))
        return -log1plusexp(h)

    def forward_f(self,x_cov,y,x_cat=[]): #Figure out how to zero out grad
        #y = torch.autograd.Variable(y,requires_grad=True)
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y))
        F = h.sigmoid()
        if self.direct=='full' or True:
            h_forward = self.middle_net((x_cov, y + self.eps))
            F_forward = h_forward.sigmoid()
            f = ((F_forward - F) / self.eps)
        elif self.direct=='semi':
            h_forward = self.middle_net((x_cov, y + self.eps))
            dh = (h_forward - h) /self.eps
            f =dh*F*(1-F)
        """else:
            f, = torch.autograd.grad(
                outputs=[F],
                inputs=[y],
                grad_outputs=torch.ones_like(F),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
                allow_unused=True
            )"""

        return (f+1e-6).log()

    def forward_cum_hazard(self, x_cov, y, mask,x_cat=[]):
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y))
        return log1plusexp(h)

    def forward_hazard(self, x_cov, y,x_cat=[]):
        y = torch.autograd.Variable(y,requires_grad=True)
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov,y))
        if self.direct=='full':
            h_forward = self.middle_net((x_cov, y + self.eps))
            hazard = (log1plusexp(h_forward) - log1plusexp(h)) / self.eps
        elif self.direct == 'semi':
            h_forward = self.middle_net((x_cov, y + self.eps))
            hazard = torch.sigmoid(h) * ((h_forward - h) / self.eps)
        else:
            H=log1plusexp(h)
            hazard, = torch.autograd.grad(
                outputs=[H],
                inputs=[y],
                grad_outputs=torch.ones_like(H),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
                allow_unused=True
            )
        return hazard

    def forward_S_eval(self,x_cov,y,x_cat=[]):
        if self.objective in ['hazard','hazard_mean']:
            S = torch.exp(-self.forward_cum_hazard(x_cov, y, [],x_cat))
            return S
        elif self.objective in ['S','S_mean']:
            x_cov = self.covariate_net((x_cov,x_cat))
            h = self.middle_net((x_cov, y))
            return 1-h.sigmoid_()

class survival_GWI(survival_net_basic):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers_t,
                 layers,
                 dropout=0.9,
                 bounding_op=torch.relu,
                 transformation=torch.tanh,
                 direct_dif = True,
                 objective = 'hazard',
                 eps=1e-6
                 ):
        super(survival_GWI, self).__init__(d_in_x,
                                           cat_size_list,
                                           d_in_y,
                                           d_out,
                                           layers_x,
                                           layers_t,
                                           layers,
                                           dropout,
                                           bounding_op,
                                           transformation,
                                           direct_dif,
                                           objective,
                                           eps)
    def forward_S(self,x_cov,y,mask,x_cat=[],L_reparam = None):
        x_cov = x_cov[~mask,:]
        y = y[~mask,:]
        if not isinstance(x_cat,list):
            x_cat=x_cat[~mask,:]
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov,y))
        if L_reparam is not None:
            h = h + L_reparam
        return -log1plusexp(h), h

    def forward_f(self, x_cov, y_with_grad, x_cat=[], L_reparam = None): #Figure out how to zero out grad
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y_with_grad))
        if L_reparam is not None:
            h = h + L_reparam
        F = h.sigmoid()
        if self.direct=='full':
            h_forward = self.middle_net((x_cov, y_with_grad + self.eps))
            F_forward = h_forward.sigmoid()
            f = ((F_forward - F) / self.eps)
        elif self.direct=='semi':
            h_forward = self.middle_net((x_cov, y_with_grad + self.eps))
            dh = (h_forward - h) /self.eps
            f =dh*F*(1-F)
        else:
            f, = torch.autograd.grad(
                outputs=[F],
                inputs=[y_with_grad],
                grad_outputs=torch.ones_like(F),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
                allow_unused=True
            )

        return (torch.relu(f)+1e-6).log(),h
    def forward_h(self,x_cov,y,x_cat=[]):
        x_cov = self.covariate_net((x_cov,x_cat))
        h = self.middle_net((x_cov, y))
        return h

class weibull_net(torch.nn.Module):

    def __init__(self, d_in_x,
                 cat_size_list,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers_t,
                 layers,
                 dropout=0.9,
                 bounding_op=torch.relu,
                 transformation=torch.tanh,
                 direct_dif = True,
                 objective = 'hazard',
                 eps=1e-6):
        super(weibull_net, self).__init__()
        self.eps = eps
        self.direct = direct_dif
        self.objective  = objective
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,dropout)

        if self.objective in ['hazard','hazard_mean']:
            self.f = self.forward_hazard
            self.f_cum = self.forward_cum_hazard
        elif self.objective in ['S','S_mean']:
            self.f=self.forward_f
            self.f_cum=self.forward_S

        self.a_net = self.init_middle_net(dx_in=layers_x[-1]+1, d_in_y=d_in_y, d_out=d_out, layers=layers,
                             transformation=transformation, bounding_op=bounding_op,dropout=dropout)
        self.b_net = self.init_middle_net(dx_in=layers_x[-1]+1, d_in_y=d_in_y, d_out=d_out, layers=layers,
                             transformation=transformation, bounding_op=bounding_op,dropout=dropout)

    def f_func(self,t,k,lamb):
        k = k.exp().clip(1,5)
        lamb = lamb.exp().clip(0.5,100)
        # k = k.clip(1e-6,10)
        # lamb = lamb.clip(1e-6,100)
        # f = (k/lamb)*(torch.pow((t/lamb+1e-3),(k-1))) * torch.exp(-torch.pow((t/lamb+1e-3),k)) + 1e-3
        f = k.log()-lamb.log()+(k-1)*((t+1e-6).log()-lamb.log()) - torch.pow((t/lamb+1e-6),k)
        return f

    def S_func(self,t,k,lamb):
        k = k.exp().clip(1,5)
        lamb = lamb.exp().clip(0.5,100)
        return torch.exp(-torch.pow(t/lamb+1e-6,k))

    def init_middle_net(self, dx_in, d_in_y, d_out, layers, transformation, bounding_op,dropout):
        module_list = [unbounded_nn_layer(d_in=dx_in, d_out=layers[0], bounding_op=bounding_op,
                                     transformation=transformation,dropout=dropout)]
        for l_i in range(1, len(layers)):
            module_list.append(unbounded_nn_layer(d_in=layers[l_i - 1], d_out=layers[l_i], bounding_op=bounding_op,
                                                transformation=transformation,dropout=dropout))
        module_list.append(
            unbounded_nn_layer_last(d_in=layers[-1], d_out=d_out, bounding_op=bounding_op, transformation=linear))
        return multi_input_Sequential(*module_list)
    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,dropout):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation,dropout=dropout)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation,dropout=dropout))
        self.covariate_net = multi_input_Sequential(*module_list)
    def forward_S(self,x_cov,y,mask,x_cat=[]):
        x_cov = x_cov[~mask,:]
        y = y[~mask,:]
        if not isinstance(x_cat,list):
            x_cat=x_cat[~mask,:]
        #Fix categorical business
        x_cov = self.covariate_net((x_cov,x_cat))
        cat_dat = torch.cat([x_cov,y],dim=1)
        a = self.a_net(cat_dat) #this is wrong...
        b = self.b_net(cat_dat)
        S = self.S_func(y,a,b)+1e-6
        return S.log()
    def forward(self,x_cov,y,x_cat=[]):
        return self.f(x_cov,y,x_cat)

    def forward_cum(self,x_cov,y,mask,x_cat=[]):
        return self.f_cum(x_cov, y,mask,x_cat)

    def forward_f(self,x_cov,y,x_cat=[]):
        y = torch.autograd.Variable(y,requires_grad=True)

        x_cov = self.covariate_net((x_cov,x_cat))
        cat_dat = torch.cat([x_cov,y],dim=1)
        a = self.a_net(cat_dat) #this is wrong...
        b = self.b_net(cat_dat)
        # f = self.f_func(y,a,b)
        F= 1-self.S_func(y,a,b)
        f, = torch.autograd.grad(
            outputs=[F],
            inputs=[y],
            grad_outputs=torch.ones_like(F),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
            allow_unused=True
        )

        return (f+1e-6).log()

    def forward_S_eval(self,x_cov,y,x_cat=[]):
        x_cov = self.covariate_net((x_cov, x_cat))
        cat_dat = torch.cat([x_cov,y],dim=1)
        a = self.a_net(cat_dat) #this is wrong...
        b = self.b_net(cat_dat)
        S = self.S_func(y,a,b)
        return S

class lognormal_net(weibull_net):
    def __init__(self, d_in_x,
                 cat_size_list,
                 d_in_y,
                 d_out,
                 layers_x,
                 layers_t,
                 layers,
                 dropout=0.9,
                 bounding_op=torch.relu,
                 transformation=torch.tanh,
                 direct_dif=True,
                 objective='hazard',
                 eps=1e-6):
        super(lognormal_net, self).__init__(
            d_in_x,
            cat_size_list,
            d_in_y,
            d_out,
            layers_x,
            layers_t,
            layers,
            dropout,
            bounding_op,
            transformation,
            direct_dif,
            objective,
            eps
        )
        self.const_1 = 2**0.5
        self.const_2 = 2**0.5 * 3.1415927410125732**0.5


    def f_func(self,t,mu,std):
        std = std.sigmoid()*100+1e-6
        # f = 1/(t*std*self.const_2) * torch.exp(-(torch.log(t)-mu)**2/(2*std**2))
        f = -((t+1e-6) * std  *self.const_2 ).log()  - (torch.log(t+1e-6)-mu)**2/(2*std**2)
        return f

    def S_func(self,t,mu,std):
        std = std.sigmoid()*100+1e-6
        S = 0.5-0.5*torch.erf((torch.log(t+1e-6)-mu)/(self.const_1*std))
        return S
def get_objective(objective):
    if objective == 'hazard':
        return log_objective_hazard
    if objective == 'hazard_mean':
        return log_objective_hazard_mean
    elif objective == 'S':
        return log_objective
    elif objective=='S_mean':
        return log_objective_mean

def log_objective(S,f):
    return -(f).sum()-S.sum()

def log_objective_mean(S,f):
    n = S.shape[0]+f.shape[0]
    return -(f.sum()+S.sum())/n


def log_objective_hazard(cum_hazard,hazard): #here cum_hazard should be a vector of
    # length n, and hazard only needs to be computed for all individuals with
    # delta = 1 I'm not sure how to implement that best?
    return -(  (hazard+1e-6).log().sum()-cum_hazard.sum() )

def log_objective_hazard_mean(cum_hazard,hazard):
    n = cum_hazard.shape[0]
    return -(  (hazard+1e-6).log().sum()-cum_hazard.sum() )/n




"""def generate_events(dgp1, dgp2, x, device,copula=None):
    if copula is None:
        uv = torch.rand((x.shape[0],2), device=device)
    else:
        uv = copula.rvs(x.shape[0])
    t1 = dgp1.rvs(x, uv[:,0])
    t2 = dgp2.rvs(x, uv[:,1])
    E = (t1 < t2).type(torch.float32)
    T = E * t1 + t2 *(1-E)
    return {'X':x,'E':E, 'T':T, 't1':t1, 't2':t2}"""


def synthetic_x(n_train, n_val, n_test, nf, device):
    x_train = torch.rand((n_train, nf), device=device)
    x_val = torch.rand((n_val, nf), device=device)
    x_test = torch.rand((n_test, nf), device=device)
    return {"x_train":x_train, "x_val":x_val, "x_test":x_test}

"""def generate_data(x_dict, dgp1, dgp2,device, copula=None):
    train_dict = generate_events(dgp1, dgp2, x_dict['x_train'],device, copula)
    val_dict = generate_events(dgp1, dgp2, x_dict['x_val'],device, copula)
    test_dict = generate_events(dgp1, dgp2, x_dict['x_test'],device, copula)
    return train_dict, val_dict, test_dict"""


def loss_triple(model1, model2, model3, X, T, E, copula=None):#estimates the joint loss
    s1 = model1.survival(T, X)
    s2 = model2.survival(T, X)
    s3 = model3.survival(T, X)
    f1 = model1.PDF(T, X)
    f2 = model2.PDF(T, X)
    f3 = model3.PDF(T, X)
    w = torch.mean(E)
    if copula is None:
        p1 = LOG(f1) + LOG(s2) + LOG(s3)
        p2 = LOG(f2) + LOG(s1) + LOG(s3)
        p3 = LOG(f3) + LOG(s1) + LOG(s2)
    else:
        
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1), s3.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
        p3 = LOG(f3) + LOG(copula.conditional_cdf("w", S))
        
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (E == 0)*1.0
    e2 = (E == 1)*1.0
    e3 = (E == 2)*1.0
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2) + torch.sum(p3*e3)
    loss = -loss/E.shape[0]
    return loss


def loss_tripleNDE(model, X, T, E, copula=None):#estimates the joint loss
    log_f = model.forward_f(X, T.reshape(-1,1))
    log_s = model.forward_S(X, T.reshape(-1,1), mask=0)
    f1 = log_f[:,0:1]
    s1 = log_s[:,0:1]
    f2 = log_f[:,1:2]
    s2 = log_s[:,1:2]
    f3 = log_f[:,2:3]
    s3 = log_s[:,2:3]
    w = torch.mean(E)
    if copula is None:
        p1 = f1 + s2 + s3
        p2 = s1 + f2 + s3
        p3 = s1 + s2 + f3
    else:
        S = torch.cat([torch.exp(s1).reshape(-1,1), torch.exp(s2).reshape(-1,1), torch.exp(s3).reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = f1 + LOG(copula.conditional_cdf("u", S)).reshape(-1,1)
        p2 = f2 + LOG(copula.conditional_cdf("v", S)).reshape(-1,1)
        p3 = f3 + LOG(copula.conditional_cdf("w", S)).reshape(-1,1)
    
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (E == 0)*1.0
    e2 = (E == 1)*1.0
    e3 = (E == 2)*1.0
    
    loss = torch.sum(p1 * e1.reshape(-1,1)) + torch.sum(p2*e2.reshape(-1,1)) + torch.sum(p3*e3.reshape(-1,1))
    loss = -loss/E.shape[0]
    
    return loss

if __name__ == "__main__":
    from test_script import Weibull_linear
    from test_script import generate_data
    import numpy as np
    torch.manual_seed(0)
    np.random.seed(0)
    DEVICE = 'cpu'
    nf = 1
    n_train = 10000
    n_val = 5000
    n_test = 5000

    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    dgp1 = Weibull_linear(nf, alpha=14, gamma=3, device=DEVICE)
    dgp2 = Weibull_linear(nf, alpha=14.5, gamma=3, device=DEVICE)
    dgp3 = Weibull_linear(nf, alpha=13, gamma=3, device=DEVICE)
    

    
    dgp1.coeff = torch.rand((nf,),device=DEVICE)
    dgp2.coeff = torch.rand((nf,), device=DEVICE)
    dgp3.coeff = torch.rand((nf,), device=DEVICE)
    theta = 3.0
    
    train_dict, val_dict, tst_dict = \
        generate_data(x_dict, dgp1, dgp2,dgp3,'device', copula='clayton', theta=theta)
    
    import matplotlib.pyplot as plt
    """plt.hist(train_dict['E'])
    plt.show()
    assert 0"""
    
    print(train_dict['E'].mean())
    sn1 = survival_net_basic(d_in_x=nf, 
                            cat_size_list=[],
                            d_in_y=1,
                            d_out=3,
                            layers_x=[32],
                            layers_t=[],
                            layers=[32,32],
                            transformation=torch.tanh,
                            dropout=0.2, eps=1e-3)
    
    

    optimizer = torch.optim.Adam(sn1.parameters(), lr=1e-3)
    f1 = torch.log(dgp1.PDF(train_dict['t1'], train_dict['X'])+1e-20)
    f2 = torch.log(dgp2.PDF(train_dict['t2'], train_dict['X'])+1e-20)
    f3 = torch.log(dgp3.PDF(train_dict['t3'], train_dict['X'])+1e-20)
    loss_train =-1 * (f1 + f2 + f3).mean()
    f1 = torch.log(dgp1.PDF(val_dict['t1'], val_dict['X'])+1e-20)
    f2 = torch.log(dgp2.PDF(val_dict['t2'], val_dict['X'])+1e-20)
    f3 = torch.log(dgp3.PDF(val_dict['t3'], val_dict['X'])+1e-20)
    loss_val =-1 * (f1 + f2 + f3).mean()
    print(loss_train, loss_val)
    
    """for itr in range(2000):
        sn1.train()
        optimizer.zero_grad()
        f1 = sn1.forward_f(train_dict['X'], train_dict['t1'].reshape(-1,1))[:,0]
        f2 = sn1.forward_f(train_dict['X'], train_dict['t2'].reshape(-1,1))[:,1]
        f3 = sn1.forward_f(train_dict['X'], train_dict['t3'].reshape(-1,1))[:,2]
        loss = -1.0 * (f1+f2+f3).mean() 
        loss.backward()
        optimizer.step()
        if itr %100 == 0:
            sn1.eval()
            with torch.no_grad():
                
                f1 = sn1.forward_f(val_dict['X'], val_dict['t1'].reshape(-1,1))[:,0]
                f2 = sn1.forward_f(val_dict['X'], val_dict['t2'].reshape(-1,1))[:,1]
                f3 = sn1.forward_f(val_dict['X'], val_dict['t3'].reshape(-1,1))[:,2]
                loss_val = -1.0 * (f1+f2+f3).mean() 
                print(loss_val, loss,surv_diff_NDE(dgp1, sn1, val_dict['X'], 0, 500),\
                       surv_diff_NDE(dgp2, sn1, val_dict['X'], 1, 500), surv_diff_NDE(dgp3, sn1, val_dict['X'], 2, 500))
    
    print(surv_diff_NDE(dgp1, sn1, val_dict['X'], 0, 500),\
                       surv_diff_NDE(dgp2, sn1, val_dict['X'], 1, 500), surv_diff_NDE(dgp3, sn1, val_dict['X'], 2, 500))
    torch.save(sn1.state_dict(), 'pre_trained.pt')
    sn1.load_state_dict(torch.load('pre_trained.pt'))
    print(surv_diff_NDE(dgp1, sn1, val_dict['X'], 0, 500),\
                       surv_diff_NDE(dgp2, sn1, val_dict['X'], 1, 500), surv_diff_NDE(dgp3, sn1, val_dict['X'], 2, 500))
    assert 0
    """
    sn1.load_state_dict(torch.load('pre_trained.pt'))
    print(surv_diff_NDE(dgp1, sn1, val_dict['X'], 0, 500),\
                       surv_diff_NDE(dgp2, sn1, val_dict['X'], 1, 500), surv_diff_NDE(dgp3, sn1, val_dict['X'], 2, 500))














    log_f_1 = torch.log(1e-20+dgp1.PDF(val_dict['T'], val_dict['X']))
    log_s_1 = torch.log(1e-20+dgp1.survival(val_dict['T'], val_dict['X']))
    log_f_2 = torch.log(1e-20+dgp2.PDF(val_dict['T'], val_dict['X']))
    log_s_2 = torch.log(1e-20+dgp2.survival(val_dict['T'], val_dict['X']))
    log_f_3 = torch.log(1e-20+dgp3.PDF(val_dict['T'], val_dict['X']))
    log_s_3 = torch.log(1e-20+dgp3.survival(val_dict['T'], val_dict['X']))
    
    p1 = log_f_1 + log_s_2 + log_s_3
    p2 = log_s_1 + log_f_2 + log_s_3
    p3 = log_s_1 + log_s_2 + log_f_3
    e1 = (val_dict['E'] == 0)*1.0
    e2 = (val_dict['E'] == 1)*1.0
    e3 = (val_dict['E'] == 2)*1.0
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2) + torch.sum(p3*e3)
    loss_dgp = -loss/val_dict['E'].shape[0]
    copula = Clayton(torch.tensor([theta]).type(torch.float32))
    print(loss_triple(dgp1, dgp2, dgp3, val_dict['X'], val_dict['T'], val_dict['E'], copula))
    
    """loss = []
    thetas = torch.linspace(0.01, 12, 100)
    copula = Clayton(torch.tensor([theta]).type(torch.float32))
    for theta in thetas:
        copula = Clayton(torch.tensor([theta]).type(torch.float32))
        loss.append(loss_triple(dgp1, dgp2, dgp3, val_dict, copula).detach().numpy())
    plt.plot(thetas,loss)
    print(min(loss))
    plt.show()
    assert 0"""
    loss_dgp  = loss_triple(dgp1, dgp2, dgp3, val_dict['X'], val_dict['T'], val_dict['E'], copula)
    #print(loss_triple(dgp1, dgp2, dgp3, train_dict['X'], train_dict['T'], train_dict['E'], copula))
    #assert 0
    #print(loss_triple(dgp1, dgp2, dgp3, val_dict, None))
    copula = Clayton(torch.tensor([theta]).type(torch.float32))
    copula.enable_grad()
   
    optimizer = torch.optim.Adam([  {"params": sn1.parameters(), "lr": 1e-4},
                        {"params": copula.parameters(), "lr": 0e-3},
                    ], weight_decay=0.000)
    copula.flag = True
    copula_grad = []
    copula_grad_m = []
    import matplotlib.pyplot as plt
    idx = torch.randperm(train_dict['E'].shape[0])
    for i in range(10000):
        if i==0:
            
            sn1.eval()
            loss_tr = loss_tripleNDE(sn1, train_dict['X'], train_dict['T'], train_dict['E'], copula)
            loss_val = loss_tripleNDE(sn1, val_dict['X'], val_dict['T'], val_dict['E'], copula)
            print(111111, copula.theta.detach().numpy(), loss_tr, loss_val, loss_dgp, surv_diff_NDE(dgp1, sn1, val_dict['X'], 0, 200),\
                    surv_diff_NDE(dgp2, sn1, val_dict['X'], 1, 200), surv_diff_NDE(dgp3, sn1, val_dict['X'], 2, 200))
        
        sn1.train()
        optimizer.zero_grad()
        """log_f = sn1.forward_f(train_dict['X'], train_dict['T'].reshape(-1,1))
        log_s = sn1.forward_S(train_dict['X'], train_dict['T'].reshape(-1,1), mask=0)
        

        log_f_1 = log_f[:,0:1]
        log_s_1 = log_s[:,0:1]
        log_f_2 = log_f[:,1:2]
        log_s_2 = log_s[:,1:2]
        log_f_3 = log_f[:,2:3]
        log_s_3 = log_s[:,2:3]

        p1 = log_f_1 + log_s_2 + log_s_3
        p2 = log_s_1 + log_f_2 + log_s_3
        p3 = log_s_1 + log_s_2 + log_f_3
        e1 = (train_dict['E'] == 0)*1.0
        e2 = (train_dict['E'] == 1)*1.0
        e3 = (train_dict['E'] == 2)*1.0
        loss = torch.sum(p1 * e1.reshape(-1,1)) + torch.sum(p2*e2.reshape(-1,1)) + torch.sum(p3*e3.reshape(-1,1))
        loss = -loss/train_dict['E'].shape[0]
        print(loss)"""
        X = train_dict['X']
        T = train_dict['T']
        E = train_dict['E']
        
        for b in range(1+int(E.shape[0]/4096)):
            e = E[b*128:min((b+1)*128, E.shape[0])]
            t = T[b*128:min((b+1)*128, E.shape[0])]
            x = X[b*128:min((b+1)*128, E.shape[0])]
            
            loss = loss_tripleNDE(sn1, x, t, e, copula)
        
            loss.backward()
        
            copula_grad.append(copula.theta.grad.clone().detach().numpy())
            for p in copula.parameters():
                p.grad = p.grad *1.0
                p.grad.clamp_(torch.tensor([-1.0]), torch.tensor([1.0]))
            copula_grad_m.append(copula.theta.grad.clone().detach().numpy())
            
            optimizer.step()
        
            for p in copula.parameters():
                if p <= 0.01:
                    with torch.no_grad():
                        p[:] = torch.clamp(p,0.01, 100)
        """if i%1000==0:
            print('change theta')
            min_loss = 10000
            min_theta = 1.0
            loss_theta = []
            for theta in torch.linspace(0.01, 8, 100):
                copula.theta = theta
                loss = loss_tripleNDE(sn1, train_dict['X'], train_dict['T'], train_dict['E'], copula)
                loss_theta.append(loss.detach().clone())
                if loss < min_loss:
                    min_loss = loss.detach().clone()
                    
                    min_theta = theta
            
            copula.theta = min_theta"""
        if i%50 == 0:
            
            plt.plot(copula_grad)
            plt.savefig('copula.png')
            plt.cla()
            plt.plot(copula_grad_m)
            plt.savefig('copula_m.png')
            plt.cla()
        
        
        if i%100==0:
            #with torch.no_grad():
            if True:
                sn1.eval()
                """log_f = sn1.forward_f(val_dict['X'], val_dict['T'].reshape(-1,1))
                log_s = sn1.forward_S(val_dict['X'], val_dict['T'].reshape(-1,1), mask=0)

                log_f_1 = log_f[:,0:1]
                log_s_1 = log_s[:,0:1]
                log_f_2 = log_f[:,1:2]
                log_s_2 = log_s[:,1:2]
                log_f_3 = log_f[:,2:3]
                log_s_3 = log_s[:,2:3]

                p1 = log_f_1 + log_s_2 + log_s_3
                p2 = log_s_1 + log_f_2 + log_s_3
                p3 = log_s_1 + log_s_2 + log_f_3

                e1 = (val_dict['E'] == 0)*1.0
                e2 = (val_dict['E'] == 1)*1.0
                e3 = (val_dict['E'] == 2)*1.0
                loss_val = torch.sum(p1 * e1.reshape(-1,1)) + torch.sum(p2*e2.reshape(-1,1)) + torch.sum(p3*e3.reshape(-1,1))
                loss_val = -loss_val/val_dict['E'].shape[0]"""
                loss_val = loss_tripleNDE(sn1, val_dict['X'], val_dict['T'], val_dict['E'], copula)
                #scheduler.step(loss_val)
                
                print(copula.theta.detach().numpy(), loss.detach().numpy(), loss_val.detach().numpy(), loss_dgp, surv_diff_NDE(dgp1, sn1, val_dict['X'], 0, 200),\
                       surv_diff_NDE(dgp2, sn1, val_dict['X'], 1, 200), surv_diff_NDE(dgp3, sn1, val_dict['X'], 2, 200))
               
    
      
        

    """sn = survival_net_basic(d_in_x=10,
                 cat_size_list=[],
                 d_in_y=1,
                 d_out=1,
                 layers_x=[32,32,32],
                 layers_t=[],
                 layers=[32,32,32,32]
                 ,dropout=0)
    x = torch.randn((5,10))
    t = torch.ones((5,1))
    print(log1plusexp(t*0))
    assert 0
    print(sn.forward_S(x, t, torch.ones_like(t).squeeze()==0, []))
    print(sn.forward_f(x, t, []))"""






