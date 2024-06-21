import torch
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
                 transformation=torch.tanh,
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




def generate_events(dgp1, dgp2, x, device,copula=None):
    if copula is None:
        uv = torch.rand((x.shape[0],2), device=device)
    else:
        uv = copula.rvs(x.shape[0])
    t1 = dgp1.rvs(x, uv[:,0])
    t2 = dgp2.rvs(x, uv[:,1])
    E = (t1 < t2).type(torch.float32)
    T = E * t1 + t2 *(1-E)
    return {'X':x,'E':E, 'T':T, 't1':t1, 't2':t2}


def synthetic_x(n_train, n_val, n_test, nf, device):
    x_train = torch.rand((n_train, nf), device=device)
    x_val = torch.rand((n_val, nf), device=device)
    x_test = torch.rand((n_test, nf), device=device)
    return {"x_train":x_train, "x_val":x_val, "x_test":x_test}

def generate_data(x_dict, dgp1, dgp2,device, copula=None):
    train_dict = generate_events(dgp1, dgp2, x_dict['x_train'],device, copula)
    val_dict = generate_events(dgp1, dgp2, x_dict['x_val'],device, copula)
    test_dict = generate_events(dgp1, dgp2, x_dict['x_test'],device, copula)
    return train_dict, val_dict, test_dict


if __name__ == "__main__":
    from test_script import Weibull_linear
    torch.manual_seed(0)
    DEVICE = 'cpu'
    nf = 3
    n_train = 10000
    n_val = 5000
    n_test = 5000

    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    dgp1 = Weibull_linear(nf, alpha=17, gamma=3, device=DEVICE)
    dgp2 = Weibull_linear(nf, alpha=16, gamma=3, device=DEVICE)
    

    
    dgp1.coeff = torch.rand((nf,),device=DEVICE)
    dgp2.coeff = torch.rand((nf,), device=DEVICE)
    
    train_dict, val_dict, tst_dict = \
        generate_data(x_dict, dgp1, dgp2,DEVICE, copula=None)
    
    sn1 = survival_net_basic(d_in_x=nf, 
                            cat_size_list=[],
                            d_in_y=1,
                            d_out=1,
                            layers_x=[32,32,32],
                            layers_t=[],
                            layers=[32,32,32,32],
                            dropout=0.4, eps=1e-3)
    
    

    optimizer = torch.optim.Adam(sn1.parameters(), lr=1e-3)
    log_f = torch.log(1e-10+dgp1.PDF(val_dict['T'], val_dict['X']))
    log_s = torch.log(1e-10+dgp1.survival(val_dict['T'], val_dict['X']))
    print(-(log_f * val_dict['E'] + log_s * (1-val_dict['E'])).mean())
    for i in range(5000):
        optimizer.zero_grad()
        log_f = sn1.forward_f(train_dict['X'], train_dict['T'].reshape(-1,1))
        log_s = sn1.forward_S(train_dict['X'], train_dict['T'].reshape(-1,1), mask=0)
        loss = -(log_f * train_dict['E'].reshape(-1,1) + log_s * (1-train_dict['E'].reshape(-1,1))).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            with torch.no_grad():
                log_f = sn1.forward_f(val_dict['X'], val_dict['T'].reshape(-1,1))
                log_s = sn1.forward_S(val_dict['X'], val_dict['T'].reshape(-1,1), mask=0)
                loss_val = -(log_f * val_dict['E'].reshape(-1,1) + log_s * (1-val_dict['E'].reshape(-1,1))).mean()
                print(loss, loss_val)
        

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






