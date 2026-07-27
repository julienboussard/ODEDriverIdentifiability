"""
Structured ODE modelling
===========================
Predicts the last timestep from a window of T timesteps:
 
    X_hat[-1] = f( flatten( X @ W ) )
 
where
    X  : (T, N)     input window  (full history)
    W  : (T, N, N)  per-timestep causal driver matrices
 
Step by step:
    z1 = X * W          (T, N) @ (T, N, N)  -> (T, N, N)  
    out = f_n(z1.flatten())  (T*N, N) -> (,N)             predict X_t+1 - X_t
 
Constraints
-----------
- W  : L0 sparsity 
- Lipschitz function / gradient penalty
"""
 
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import math
import numpy as np

from tqdm import trange

# Require torchdiffeq for the ODE solvers
from torchdiffeq import odeint

# ─────────────────────────────────────────────────────────────────────────────
# Core Utilities & Networks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    MLP: in_dim -> hidden -> ... -> out_dim.
    Used for f which maps flattened (T*N) -> N.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()] # Here, needed to have a non constant function i.e. derivative is not 0
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LipschitzNet(nn.Module):
    """
    MLP used for f_i mapping masked N variables -> 1 derivative.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, n_layers: int = 3, L: float = 2.0):
        super().__init__()
        self.L = L
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [spectral_norm(nn.Linear(d, hidden_dim)), nn.Tanh()]
            d = hidden_dim
        layers += [spectral_norm(nn.Linear(d, out_dim))]
        self.net = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.L * self.net(x)

# ─────────────────────────────────────────────────────────────────────────────
# Structured ODE Function (The Vector Field)
# ─────────────────────────────────────────────────────────────────────────────

class StructuredODEFunction(nn.Module):
    """
    Computes dX/dt = f(X \odot M)
    
    Compatible with torchdiffeq's odeint signature: forward(t, X)
    """
    def __init__(self,
                 n: int,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 L_lipschitz: float = 2.0, 
                 ):
        super().__init__()
        self.n = n
        
        # M: Interaction matrix (N, N) where M[j, i] determines if variable j affects dX_i/dt
        self.M_logits = nn.Parameter(torch.ones(n, n) * 3)
        
        self.f = nn.ModuleList(
            LipschitzNet(in_dim=n, out_dim=1, hidden_dim=hidden_dim, n_layers=n_layers, L=L_lipschitz) 
            for _ in range(n)
        )
        self.current_M = None

    def forward(self, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:

        single = X.dim() == 1
        if single:
            X = X.unsqueeze(0)

        # Use the pre-computed mask for this batch!
        if self.current_M is None:
            raise ValueError("current_M must be set before calling odeint")
            
        M = self.current_M 

        z = X.unsqueeze(1) * M.T.unsqueeze(0) 
        out = []
        for i in range(self.n):
            out.append(self.f[i](z[:, i, :]))
            
        out = torch.stack(out, dim=-1).squeeze(1)
        return out.squeeze(0) if single else out


    def infer(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inference-time prediction with deterministic causal structure (no sampling).
        
        Uses hard_concrete_mean instead of sampling for reproducible predictions.
        Disables gradients for efficiency.
        
        Parameters
        ----------
        X : (batch, T, N) or (T, N)
            Input window of shape (batch, context_length, num_variables) or (context_length, num_variables)
        
        Returns
        -------
        X_hat : (batch, N) or (N,)
            Predicted next step, same shape as input minus time dimension
        """
        self.eval()
        with torch.no_grad():
            single = X.dim() == 2
            if single:
                X = X.unsqueeze(0)              # (1, T, N)

            # Use deterministic mean instead of sampling
            M = hard_concrete_mean(self.M_logits)
            z = X.unsqueeze(-1) * M.unsqueeze(0)

            # Predict X[-1]: (B, T, N*N) -> (B, N)
            out = []
            for i in range(self.n):
                out.append(self.f[i](z[:, :, :, i].flatten(start_dim=1)))
            out = torch.stack(out, dim=-1)  # (B, N)

            return out.squeeze(0) if single else out
 

# ─────────────────────────────────────────────────────────────────────────────
# ODE Discovery & Training Loop
# ─────────────────────────────────────────────────────────────────────────────

class StructuredNODEDiscovery():
    def __init__(self, 
                 n: int, 
                 device, 
                 solver: str = 'dopri5', # or 'rk4'
                 hidden_dim: int = 8, 
                 n_layers: int = 2, 
                 L_lipschitz: float = 2.0, 
                 normalize: bool = True):
        self.n = n
        self.device = device
        self.solver = solver
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers    
        self.L_lipschitz = L_lipschitz
        self.normalize = normalize

    def loss_fn(self,
                batch_y0: torch.Tensor,
                batch_t_list: list[torch.Tensor],
                batch_y: torch.Tensor,
                lambda_m: float = 0.01,
                lambda_grad: float = 0.01,
                bool_sparse: bool = False, 
                l0_l1_l2: str = "l0") -> tuple[torch.Tensor, dict]:
        """
        ODE Trajectory Loss & Regularizations
        """
# 1. Trajectory Reconstruction Loss

        if self.model.training:
            self.model.current_M = sample_hard_concrete(self.model.M_logits)
        else:
            self.model.current_M = hard_concrete_mean(self.model.M_logits)

        pred_y_list = []
        batch_size = batch_y0.shape[0]
        
        # Integrate each sample in the batch over its specific relative time grid
        for i in range(batch_size):
            y0_single = batch_y0[i].unsqueeze(0) # Shape: (1, N)
            t_single = batch_t_list[i]           # Shape: (seq_len,)
            
            # pred_single shape: (seq_len, 1, N)
            pred_single = odeint(self.model, y0_single, t_single, method=self.solver)
            pred_y_list.append(pred_single)
            
        # Combine back into (seq_len, batch_size, N)
        pred_y = torch.cat(pred_y_list, dim=1) 
        recon_loss = F.mse_loss(pred_y, batch_y)

        total = recon_loss

        # 2. Gradient Penalty (Lipschitz regularization on the vector field)
        # We compute dX/dt for the initial conditions to regularize the function's slope
        gradient_penalty = torch.tensor(0.0).to(self.device)
        if lambda_grad > 0:
            batch_y0_tracked = batch_y0.detach().requires_grad_(True)
            dX_dt = self.model(batch_t_list[0][0], batch_y0_tracked)
            
            # Sum over all outputs to get a scalar for backprop
            sum_dX = dX_dt.sum()
            input_grads = torch.autograd.grad(
                outputs=sum_dX,
                inputs=batch_y0_tracked,
                create_graph=True,
                retain_graph=True,
            )[0]
            
            grad_norm = input_grads.view(input_grads.shape[0], -1).norm(2, dim=1)
            gradient_penalty = (grad_norm ** 2).mean()
            total = total + lambda_grad * gradient_penalty

        # 3. Sparsity Constraint on M
        m_sparse = torch.tensor(0.0).to(self.device)
        if bool_sparse:
            if l0_l1_l2 == "l0":
                m_sparse = hard_concrete_sparsity(self.model.M_logits)
            else:
                m_val = torch.sigmoid(self.model.M_logits) 
                if l0_l1_l2 == "l1":
                    m_sparse = m_val.sum()
                elif l0_l1_l2 == "l2":
                    m_sparse = (m_val ** 2).sum()

            total = total + lambda_m * m_sparse / (self.model.n ** 2)

        return total, {
            "recon": recon_loss.item(),
            "m_sparse": m_sparse.item(),
            "gradient_penalty": gradient_penalty.item()
        }

    def plot_loss_components(self,
                             history: list[dict],
                             M: int = 1,
                             figsize: tuple[int, int] = (10, 6),
                             ax=None,
                             savefig: bool = True,
                             save_path: str | None = None):
        """
        Plot loss components every M outer epochs from training history.

        Parameters
        ----------
        history : list[dict]
            List of dictionaries produced by `run(..., return_history=True)`.
        M : int
            Plot every M outer epochs.
        figsize : tuple[int, int]
            Figure size for a new plot.
        ax : matplotlib.axes.Axes | None
            Optional axis to plot into.
        savefig : bool
            If True, save the figure as a PNG file.
        save_path : str | None
            Optional path to save the PNG file. Defaults to `loss_components.png`.
        """
        if not history:
            raise ValueError("history must contain at least one entry")

        xs = [entry.get("step", entry.get("outer", i + 1)) for i, entry in enumerate(history)]
        indices = list(range(0, len(history), max(1, M)))
        xs = [xs[i] for i in indices]

        keys = ["total", "recon", "m_sparse", "gradient_penalty"] #"gradient_penalty"
        available = [k for k in keys if any(k in entry for entry in history)]
        if not available:
            raise ValueError("history entries must contain at least one loss component")

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        for key in available:
            ys = [history[i][key] for i in indices]
            ax.plot(xs, ys, label=key)

        ax.set_xlabel("Outer epoch")
        ax.set_ylabel("Loss / penalty value")
        ax.set_title("Training loss components")
        ax.legend()
        ax.grid(True)

        if created_fig:
            fig.tight_layout()
            if savefig:
                fig.savefig(save_path or "loss_components.png")
        if savefig:
            ax.figure.savefig(save_path or "loss_components.png")
        plt.close()

    def get_batch(self, X: torch.Tensor, t: torch.Tensor, seq_len: int, batch_size: int):
        """Samples random continuous sub-trajectories with non-uniform time."""
        max_start = len(X) - seq_len
        starts = torch.randint(0, max_start, (batch_size,))
        
        batch_y0 = []
        batch_y = []
        batch_t_list = [] 
        
        for start in starts:
            end = start + seq_len
            
            # State extraction
            y_seq = X[start:end]
            batch_y0.append(y_seq[0])
            batch_y.append(y_seq)
            
            # Time extraction: Must be relative to 0 for the ODE solver
            t_seq = t[start:end]
            t_relative = t_seq - t_seq[0]
            batch_t_list.append(t_relative.to(self.device))
            
        # batch_y shape becomes (seq_len, batch_size, N)
        return torch.stack(batch_y0).to(self.device), batch_t_list, torch.stack(batch_y, dim=1).to(self.device)

    def run(self,
            X: torch.Tensor,
            t: torch.Tensor,
            seq_len: int = 10,
            n_inner: int = 1_000,
            lr: float = 1e-3,
            l0_l1_l2: str = "l0",
            lambda_m: float = 0.01,
            lambda_grad: float = 0.01,
            batch_size: int = 32,
            n_inner_min_sparse: int = 200, 
            patience: int = 50,
            plot_frequency: int = 500,
            save_fig_path: str | None = None
        ) -> tuple[StructuredODEFunction, torch.Tensor]:
        
        print(f"Dimensions: N={self.n}, Total Timesteps={X.shape[0]}")
        assert l0_l1_l2 in ["l0", "l1", "l2"], "l0_l1_l2 must be 'l0', 'l1', or 'l2'"
        assert self.n == X.shape[-1], f"Model n={self.n} must match data N={X.shape[-1]}"
        if X.dim() != 2:
            raise ValueError("X must be a 2D tensor with shape (T, N) for training")

        if self.normalize:
            X = X - X.mean(dim=0, keepdim=True)
            X = X / (X.std(dim=0, keepdim=True) + 1e-8)

        print(f"NaN values in X? {torch.isnan(X).any()}")
        print(f"Constant dimensions in X: {torch.where(X[:, 0].std(dim=0) == 0)}")

        self.model = StructuredODEFunction(
            n=self.n, 
            hidden_dim=self.hidden_dim, 
            n_layers=self.n_layers, 
            L_lipschitz=self.L_lipschitz
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_loss = float('inf')
        no_improve_steps = 0
        history: list[dict] = []

        for iter in trange(n_inner):
            self.model.train()
            bool_sparse = iter >= n_inner_min_sparse 
            
            gradient_penalty_all = 0
            recon_all = 0
            loss_all = 0
            m_sparse_all = 0

            batch_y0, batch_t, batch_y = self.get_batch(X, t, seq_len, batch_size)
            
            optimizer.zero_grad()
            loss, info = self.loss_fn(batch_y0, batch_t, batch_y, 
                                      lambda_m, lambda_grad, bool_sparse, l0_l1_l2)
            
            gradient_penalty_all += info["gradient_penalty"]
            recon_all += info["recon"]
            loss_all += loss.item()
            m_sparse_all += info["m_sparse"]

            loss.backward()
            optimizer.step()

            history.append({
                "inner": iter + 1,
                "step": iter + 1,
                "total": loss_all,
                "recon":  recon_all,
                "m_sparse": lambda_m * m_sparse_all / (self.model.n**2),
                "gradient_penalty": lambda_grad * gradient_penalty_all
            })

            if plot_frequency > 0 and (iter + 1) % plot_frequency == 0:
                self.plot_loss_components(history, M=1, save_path=save_fig_path)

            if iter % 100 == 0:
                print(f"Iter {iter:04d} | Total Loss: {loss.item():.6f} | "
                      f"Recon: {info['recon']:.6f} | "
                      f"M Sparse: {info['m_sparse']:.3f}")

            if bool_sparse:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    if no_improve_steps >= patience:
                        print(f"Early stopping at step {iter} (best={best_loss:.6f})")
                        break

        # Return model and the computed structure matrix
        return self.model, torch.sigmoid(self.model.M_logits)
    



def sample_hard_concrete(log_alpha, beta=0.33, zeta=-0.1, gamma=1.1):
    u = torch.rand_like(log_alpha).clamp(1e-8, 1 - 1e-8)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    z_bar = s * (gamma - zeta) + zeta
    z = z_bar.clamp(0, 1)
    return z

def hard_concrete_mean(log_alpha, zeta=-0.1, gamma=1.1):
    return (torch.sigmoid(log_alpha) * (gamma - zeta) + zeta).clamp(0, 1)

def hard_concrete_sparsity(log_alpha, beta=0.33, zeta=-0.1, gamma=1.1):
    return torch.sigmoid(log_alpha - beta * math.log(-zeta / gamma)).sum()
