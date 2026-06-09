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
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# MLP  (shared over full N-dim vector)
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
    MLP: in_dim -> hidden -> ... -> out_dim.
    Used for f which maps flattened (T*N) -> N.
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
        return self.L  * self.net(x)

    
# ─────────────────────────────────────────────────────────────────────────────
# NOTEARS acyclicity constraint
# ─────────────────────────────────────────────────────────────────────────────
 
def acyclicity(W: torch.Tensor) -> torch.Tensor:
    """
    NOTEARS differentiable acyclicity constraint (Zheng et al. 2018).
 
    h(W) = tr(exp(W ⊙ W)) - K  =  0   iff W is a DAG
 
    Operates on the K×K matrix W (cheap since K << N).
    """
    return torch.trace(torch.matrix_exp(W * W)) - W.shape[0]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Core model
# ─────────────────────────────────────────────────────────────────────────────
 
class StructuredDynamics(nn.Module):
    """
    X[-1] = f( flatten( X @ S @ W @ S.T ) )
 
    Parameters
    ----------
    n          : number of variables N
    t          : context window length T
    device     : device to run the model on
    hidden_dim : hidden size for f MLP
    n_layers   : number of hidden layers in f
    """
 
    def __init__(self,
                 n: int,
                 t: int,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 L_lipschitz: float = 2.0, 
                 is_lipschitz: bool = False,
                 ):
        super().__init__()
        self.n = n
        self.t = t
        self.is_lipschitz = is_lipschitz
        # ── Structural parameters ─────────────────────────────────────────────
 
        self.M_logits = nn.Parameter(torch.ones(t, n, n) * 3)
        
        # W: per-timestep cluster interactions (T, K, K)
        # acyclicity enforced on W[-1] only; L1 sparsity on all T slices
#         print(torch.sigmoid(self.M_logits))

        self.f = nn.ModuleList(LipschitzNet(in_dim=t * n, out_dim=1,
                    hidden_dim=hidden_dim, n_layers=n_layers, L=L_lipschitz) for _ in range(n))

    # ── Forward pass ──────────────────────────────────────────────────────────
 
    def forward(self, X: torch.Tensor, ste_th: float) -> torch.Tensor:
        """
        Predict X_{t+1} from X_t.
 
        Parameters
        ----------
        X_t : (batch, N) or (N,)
        X_t1 : (batch, N) or (N,)
 
        Returns
        -------
        X_t1_hat : same shape as X_t
        """
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)              # (1, T, N)

        # This for hard concrete distrinution i.e. L0 regularization 
        M = sample_hard_concrete(self.M_logits)
        # z = torch.einsum("btk, tkj -> btkj", X, M)
        z = X.unsqueeze(-1) * M.unsqueeze(0)

        # Step 4 — flatten N*N and predict X[-1]
        # (B, T, N*N)  ->  (B, N)
        out = []
        for i in range(self.n):
            out.append(self.f[i](z[:, :, :, i].flatten(start_dim=1)))
        out = torch.stack(out, dim=-1) # (B, N)
                
        return out.squeeze(0) if single else out
 
 

class StructuredODEDiscovery():

    def __init__(self, n: int, t: int, device, instantaneous: bool = False, hidden_dim: int = 8, n_layers: int = 2, L_lipschitz: float = 2.0, is_lipschitz: bool = False, normalize: bool = True):

        self.t = t
        self.n = n
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers    
        self.instantaneous = instantaneous
        self.is_lipschitz = is_lipschitz
        self.L_lipschitz = L_lipschitz
        self.normalize = normalize

    # ─────────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────────
    
    def loss_fn(self,
                X_windows: torch.Tensor,
                X_target: torch.Tensor,
                lambda_m: float = 0.01,
                lambda_grad: float = 0.01,
                bool_sparse: bool = False, 
                l0_l1_l2: str = "l0", 
                ste_th: float = 0.5) -> tuple[torch.Tensor, dict]:
        """
        Total training loss.
    
        Terms
        -----
        1. Reconstruction  : MSE( X_hat[-1],  X_target )
        2. DAG penalty     : augmented Lagrangian on h(W[-1])
        3. S entropy       : pushes S toward hard (one-hot) assignments
        4. W sparsity      : L1 on all T slices of W
    
        Parameters
        ----------
        X_windows : (batch, T, N)   input windows
        X_target  : (batch, 1, N)      ground truth X at t (i.e. the next step)
        lambda_dag : Lagrange multiplier for W[-1] acyclicity  (updated externally)
        mu_dag     : quadratic penalty coefficient             (updated externally)
        lambda_s   : entropy regularisation weight for S
        lambda_w   : L1 weight for W
        lambda_grad: gradient penalty weight
        """
        X_hat = self.model(X_windows, ste_th)                        # (batch, N)
    
        # 1. Reconstruction
        total = F.mse_loss(X_hat, X_target)

        input_grads = torch.autograd.grad(
            outputs=total,
            inputs=X_hat,
            create_graph=True,   # keeps the graph alive so we can backprop through this
            retain_graph=True,   # keeps the graph alive for the final .backward()
        )[0]


        recon = total.item()

        if bool_sparse:
            if l0_l1_l2 == "l0":
                m_sparse = hard_concrete_sparsity(self.model.M_logits)
            else:
                m_sparse = torch.sigmoid(self.model.M_logits) 
                if l0_l1_l2 == "l1":
                    m_sparse = m_sparse.sum()
                elif l0_l1_l2 == "l2":
                    m_sparse = (m_sparse ** 2).sum()

            # To change to have two parameters
            total += lambda_m * m_sparse / self.model.n**2
        
            # Gradient penalty — gradient norm should be <= L (Lipschitz)
            grad_norm = input_grads.view(input_grads.shape[0], -1).norm(2, dim=1)
            # gradient_penalty = (torch.clamp(grad_norm - self.L_lipschitz, min=0) ** 2).mean()
            gradient_penalty = (grad_norm ** 2).mean()
            total = total + lambda_grad * gradient_penalty

        # print(gradient_penalty)

        if bool_sparse:
            return total, {
                "recon":    recon,
                "m_sparse": m_sparse.item(),
                "gradient_penalty": gradient_penalty.item()
            }
        else:
            return total, {
                "recon":    recon,
                "m_sparse": 0.0,
                "gradient_penalty": 0.0
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

    def run(self,
            X: torch.Tensor,
            n_inner: int = 10_000,
            lr: float = 1e-3,
            l0_l1_l2: str = "l0",
            lambda_m: float = 0.01,
            lambda_grad: float = 0.01,
            batch_size: int | None = None,
            n_inner_min_sparse: int = 500, # before sparsification
            th: float = 0.5, 
            patience: int = 50,
            plot_frequency: int = 500,
            return_history: bool = False,
            save_fig_path: str | None = None
            ) -> tuple[StructuredDynamics, torch.Tensor] | tuple[StructuredDynamics, torch.Tensor, list[dict]]:
        """
        Fit the model to a time series X of shape (T, N).
    
        Uses an augmented Lagrangian outer loop for the NOTEARS constraint on W,
        and Adam for all inner gradient steps. Progressively sparsifies M coefficients
        by setting those below threshold to 0.
    
        Parameters
        ----------
        X          : (T, N) full time series
        n_outer    : max outer iterations (Lagrangian updates)
        n_inner    : inner gradient steps per outer iteration
        lr         : Adam learning rate
        mu_init    : initial quadratic penalty for DAG constraint
        mu_factor  : how much to grow mu each outer step
        lambda_dag_init : initial Lagrange multiplier (0 = start unconstrained)
        h_tol      : convergence threshold on h(W)
        lambda_s   : entropy regularisation weight for S
        lambda_w   : L1 weight for W
        lambda_grad: gradient penalty weight
        batch_size : optional minibatch size for training
        th         : threshold for sparsifying M coefficients
        l0_l1_l2   : whether to use L0, L1, or L2 regularization for M
        """
        # Build sliding training windows (B, T, N) and targets (B, N)

        print(f"Dimensions: N={self.n}, T={self.t}")

        assert l0_l1_l2 in ["l0", "l1", "l2"], "l0_l1_l2 must be 'l0', 'l1', or 'l2'"
        assert self.n == X.shape[-1], f"Model n={self.n} must match data N={X.shape[-1]}"
        if X.dim() != 3:
            raise ValueError("X must be a 3D tensor with shape (B, T, N) for training")

        print(f"Number of NaN values in X: {torch.isnan(X).any()}")

        if self.normalize:
            X = X - X.mean(dim=0, keepdim=True)
            X = X / (X.std(dim=0, keepdim=True) + 1e-8) # Small fix

        print(f"Number of NaN values after normalization in X: {torch.isnan(X).any()}")
        print(f"Constant dimensions in X: {torch.where(X[:, 0].std(dim=0) == 0)}")

        if self.t == 0:
            X_t_all = X
            X_last_all = X
        elif self.t == 1:
            X_t_all = X[:-1]
            # Important to predict the difference and not the next step to avoid predicting identity
            X_last_all = X[1:] - X[:-1]
        else:
            if X.shape[0] <= self.t:
                raise ValueError(f"Time series length T={X.shape[0]} must be larger than window length t={self.t}")
            X_t_all = torch.stack([X[i : i + self.t] for i in range(X.shape[0] - self.t)], dim=0)
            X_last_all = X[self.t:] - X[self.t - 1 : -1] if self.instantaneous else X[self.t:]

        X_t_all   = X_t_all.to(self.device)
        X_last_all = X_last_all.to(self.device)

        n_samples = X_t_all.shape[0]
        if n_samples == 0:
            raise ValueError("No training samples could be constructed from X and the chosen window length t")

        if batch_size is None:
            batch_size = n_samples
        elif batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        batch_size = min(batch_size, n_samples)

        self.model = StructuredDynamics(n=self.n, t=self.t, hidden_dim=self.hidden_dim, n_layers=self.n_layers, L_lipschitz=self.L_lipschitz, is_lipschitz=self.is_lipschitz).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Prepare a DataLoader to avoid Python-side indexing/permutation overhead
        dataset = TensorDataset(X_t_all, X_last_all)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        iter = 0
        best_loss = float('inf')
        no_improve_steps = 0
        history: list[dict] = []

        while iter < n_inner:

            bool_sparse = iter >= n_inner_min_sparse 
            gradient_penalty_all = 0

            recon_all = 0
            loss_all = 0
            m_sparse_all = 0

            n_batches = 0
            for X_t, X_last in loader:
                n_batches += 1

                # ensure X_t requires grad for gradient-penalty computation
                X_t = X_t.detach().requires_grad_(True)

                optimizer.zero_grad()

                loss, info = self.loss_fn(X_t, X_last,
                                    ste_th=th,
                                    lambda_m=lambda_m,
                                    lambda_grad=lambda_grad,
                                    bool_sparse=bool_sparse, 
                                    l0_l1_l2=l0_l1_l2
                )

                gradient_penalty_all += info["gradient_penalty"]
                recon_all += info["recon"]
                loss_all += loss.item()
                m_sparse_all += info["m_sparse"]

                loss.backward()
                optimizer.step()

            history.append({
                "inner": iter + 1,
                "step": iter + 1,
                "total": loss_all / n_batches,
                "recon":    recon_all / n_batches,
                "m_sparse": lambda_m * m_sparse_all / n_batches / self.model.n**2,
                "gradient_penalty": lambda_grad * gradient_penalty_all / n_batches
            })

            if plot_frequency > 0 and (iter + 1) % plot_frequency == 0:
                self.plot_loss_components(history, M=1, save_path=save_fig_path)

            if bool_sparse:
                # We start checking for convergence after we start sparsifying
                if loss_all / n_batches < best_loss:
                    best_loss = loss_all / n_batches
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    if no_improve_steps >= patience:
                        print(f"  → stopping inner loop at step {iter} after {no_improve_steps} no-improve steps (best={best_loss:.6f}, current={(loss_all / n_batches):.6f})")
                        break
            iter += 1

        
        print(f"recon={(recon_all / n_batches):.4f}  "
            f"|M|_1={(m_sparse_all / n_batches):.3f}  "
            f"grad_penalty={(gradient_penalty_all / n_batches):.4f}  ")

        self.plot_loss_components(history, M=1, save_path=save_fig_path)

        if return_history:
            return self.model, torch.sigmoid(self.model.M_logits)

        return self.model, torch.sigmoid(self.model.M_logits) #.transpose(2, 1) # Should we transpose here? a priori no
    
    
def sample_hard_concrete(log_alpha, beta=0.33, zeta=-0.1, gamma=1.1):
    # During training: sample
    # Use rand_like which is slightly faster and stays on the same device/dtype
    u = torch.rand_like(log_alpha).clamp(1e-8, 1 - 1e-8)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    z_bar = s * (gamma - zeta) + zeta   # stretch to [zeta, gamma] = [-0.1, 1.1]
    z = z_bar.clamp(0, 1)               # hard clamp → exact 0s and 1s
    return z

def hard_concrete_mean(log_alpha, zeta=-0.1, gamma=1.1):
    # During eval: use the mean (no sampling)
    return (torch.sigmoid(log_alpha) * (gamma - zeta) + zeta).clamp(0, 1)

# Sparsity loss: expected number of open gates (L0 approximation)
def hard_concrete_sparsity(log_alpha, beta=0.33, zeta=-0.1, gamma=1.1):
    return torch.sigmoid(log_alpha - beta * math.log(-zeta / gamma)).sum()
