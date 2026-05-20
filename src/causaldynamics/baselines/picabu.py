# from climatem.climatem.model.utils import ALM

import torch 
import torch.distributions as distr
import torch.nn as nn

from collections import OrderedDict

# import numpy as np
# import pandas as pd
# from torch.autograd import Variable

# from tqdm import trange 

class PICABU:
    """
    PICABU baseline wrapper that accepts in-memory numpy arrays.

    Reference:
        Hickman, S. et al. Causal Climate Emulation with Bayesian Filtering (PICABU)”
        https://github.com/RolnickLab/climatem
    """

    def __init__(
        self,
        d_x: int, #Input dimension
        seed: int = 1111,
        cuda: bool = False,
        optimizer: str = "rmsprop",
        reg_coeff: float = 0.2,
        epochs: int = 5_000,
        num_layers: int = 2, # number of layers for non linear dynamics
        num_hidden: int = 4,
        tau: int = 1, #Causal dynamics support 1 time lag for now
        nonlinear_dynamics: bool = True,
        lr = 0.001,
    ):
        self.cuda = cuda
        self.seed = seed
        self.reg_coeff = reg_coeff
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr = lr

        self.d_x = d_x
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = tau
        self.nonlinear_dynamics = nonlinear_dynamics

    def get_regularisation(self, model) -> float:
        adj = model.get_adj()
        return self.reg_coeff * torch.norm(adj, p=1)


    def train_step(self, X_train, Y_train, model, optimizer):
        
        model.train()

        # Different optimizer... + sparsity constraint ---
        optimizer.zero_grad()
        nll = model(X_train, Y_train)

        # Can't use ALMs + constraint because would need train/valid split
        loss = nll + self.get_regularisation(model)
        # ADD CRPS + spatial spectra
        # Add scheduler spectra?

        self.optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()

        return model.get_adj(), loss.item()

    
    def run(self, X):

        torch.manual_seed(self.seed)

        B, T, D = X.shape

        X_source = X
        X_target = X

        # X_source = torch.from_numpy(X[:-self.tau, :])
        # X_target = torch.from_numpy(X[self.tau:, :])

        model = CDSD(
            d_x = self.d_x, #Input dimension
            num_layers = self.num_layers, # number of layers for non linear dynamics
            num_hidden = self.num_hidden,
            tau = self.tau, #Causal dynamics support 1 time lag for now
            nonlinear_dynamics = self.nonlinear_dynamics,
        )

        if self.cuda:
            model.cuda()
            X_source = torch.tensor(X_source).cuda()
            X_target = torch.tensor(X_target).cuda()

        if self.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif self.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr)

        adj = torch.ones((3, 3))
        # for ep in range(self.epochs + 1):
        ep = 0
        while (adj > 0.5).sum() >= 6:
            adj, _ = self.train_step(X_source, X_target, model, self.optimizer)
            ep += 1
            if ep % 1000 == 0:
                print(f"prop of edges {adj.sum()/9}")
        print(f"Number of epochs {ep}")

        print("Adjacency matrix obtained")
        self.adj_matrix = adj > 0.5

class CDSD(nn.Module):

    def __init__(
        self,
        d_x: int, #Input dimension
        num_layers: int = 2, # number of layers for non linear dynamics
        num_hidden: int = 8,
        tau: int = 1, #Causal dynamics support 1 time lag for now
        nonlinear_dynamics: bool = False,
    ):
        
        super().__init__()

        self.d_x = d_x
        self.tau = tau
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        # Below params are set for causal discovery (and not causal representation learning)
        self.d_z = d_x
        self.nonlinear_dynamics = nonlinear_dynamics

        self.d = 1 # "One variable"

        self.distr_z0 = torch.normal
        self.distr_transition = distr.normal.Normal

        # Causal discovery not CRL
        self.transition_model = TransitionModel(
            self.d,
            self.d_z,
            self.tau,
            self.nonlinear_dynamics,
            self.num_layers,
            self.num_hidden,
        )

        self.mask = Mask(
            self.d,
            self.d_z,
            self.tau,
        )

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        """
        return self.mask.get_proba()

    def transition(self, z, mask):

        b = z.shape[0]
        mu = torch.zeros(b, self.d, self.d_z)
        std = torch.zeros(b, self.d, self.d_z)

        for i in range(self.d):
            pz_params = torch.zeros(b, self.d_z, 1)
            for k in range(self.d_z):
                pz_params[:, k] = self.transition_model(z, mask[:, :, :, i * self.d_z + k], i, k)
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model.logvar[i])

        return mu, std

    def forward(self, x, y):

        b = x.shape[0]


        mask = self.mask(b)
        pz_mu, pz_std = self.transition(x.clone(), mask)
        pz_mu = pz_mu.cuda()
        pz_std = pz_std.cuda()

        px_distr = self.distr_transition(pz_mu, pz_std)
        nll = -torch.mean(torch.sum(px_distr.log_prob(y), dim=[1, 2]))

        return nll

class Mask(nn.Module):
    def __init__(
        self,
        d: int,
        d_x: int,
        tau: int,
    ):
        super().__init__()

        self.d = d
        self.d_x = d_x
        self.tau = tau
        # Here we can just set what we want the output to be.
        self.uniform = distr.uniform.Uniform(0, 1)

        # Here we could change how the mask is instantiated in the causal graph.
        # initialize mask as log(mask_ij) = 1
        self.param = nn.Parameter(torch.ones((tau, d*d_x, d*d_x)) * 5)
        self.fixed_mask = torch.ones_like(self.param).cuda()

    def forward(self, b: int, tau: float = 1) -> torch.Tensor:
        """
        :param b: batch size
        :param tau: temperature constant for sampling
        """
        adj = gumbel_sigmoid(self.param, self.uniform, b, tau=tau, hard=False)
        adj = adj * self.fixed_mask
        return adj

    def get_proba(self) -> torch.Tensor:
        return torch.sigmoid(self.param) * self.fixed_mask


class MLP(nn.Module):
    def __init__(self, num_layers: int, num_hidden: int, num_input: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input

        module_dict = OrderedDict()

        # create model layer by layer
        in_features = num_input
        out_features = num_hidden
        if num_layers == 0:
            out_features = 1

        module_dict["lin0"] = nn.Linear(in_features, out_features)

        for layer in range(num_layers):
            in_features = num_hidden
            out_features = num_hidden

            if layer == num_layers - 1:
                out_features = 1

            module_dict[f"nonlin{layer}"] = nn.LeakyReLU()
            module_dict[f"lin{layer+1}"] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class TransitionModel(nn.Module):
    """Models the transitions between the latent variables Z with neural networks."""

    def __init__(
        self,
        d: int,
        d_z: int,
        tau: int,
        nonlinear_dynamics: bool,
        num_layers: int,
        num_hidden: int,
    ):
        """
        Args:
            d: number of features
            d_z: number of latent variables
            tau: size of the timewindow
            num_layers: number of layers for the neural networks
            num_hidden: number of hidden units
        """
        super().__init__()
        self.d = d  # number of variables
        self.d_z = d_z
        self.tau = tau

        # initialize NNs
        self.nonlinear_dynamics = nonlinear_dynamics
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        # self.logvar = torch.ones(1)  * 0. # nn.Parameter(torch.ones(d) * 0.1)
        # self.logvar = nn.Parameter(torch.ones(d) * -4)
        self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)
        if self.nonlinear_dynamics:
            print("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(num_layers, num_hidden, d * d_z * tau) for i in range(d * d_z))
        else:
            print("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, d * d_z * tau) for i in range(d * d_z))

    def forward(self, z, mask, i, k):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k NN(G_{tau-1} * z_{t-1},
        ..., G_{tau-k} * z_{t-k})"""

        z = z.view(mask.size())
        masked_z = (mask * z).view(z.size(0), -1)
        param_z = self.nn[i * self.d_z + k](masked_z)
        return param_z

def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)

def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)
    logistic_noise = logistic_noise.cuda()

    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft

    else:
        y = y_soft
    return y

