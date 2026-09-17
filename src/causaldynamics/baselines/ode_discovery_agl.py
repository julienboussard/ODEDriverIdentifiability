"""
Structured ODE modelling with Adaptive Group Lasso (AGL) sparsity
===================================================================
Predicts the last timestep from a window of T timesteps:

    X_hat[-1] = f( flatten(X) )

where
    X  : (T, N)     input window  (full history)

Unlike ode_discovery_gs.py, there is no learned soft mask M — every input
(t, n) pair is passed to every output subnetwork f[i]. Structural sparsity
(which lagged variables drive which output) is instead recovered by an
adaptive group lasso proximal step applied to the columns of each f[i]'s
first-layer weight matrix, following the two-stage adaptive group lasso
scheme of https://arxiv.org/pdf/2105.02522 :

    1. Gradient step on the smooth loss (MSE + gradient penalty).
    2. Proximal group soft-threshold step on each f[i]'s first-layer weight,
       shrinking/zeroing whole input columns.

The proximal step runs from the very first iteration. During the pilot phase
(the first n_inner_min_sparse steps) it uses a single plain, non-adaptive
lambda_m for every column. At n_inner_min_sparse, column norms are snapshotted
once as the pilot fit, and the weights switch to fixed, adaptive
lam_k = lambda_m / (||pilot_col_k||_2**gamma + eps) for the remainder of
training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# MLP  (shared over full N-dim vector)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    MLP: in_dim -> hidden -> ... -> out_dim.
    Used for f which maps flattened (T*N) -> 1.
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


# ─────────────────────────────────────────────────────────────────────────────
# Core model
# ─────────────────────────────────────────────────────────────────────────────

class StructuredDynamicsAGL(nn.Module):
    """
    X[-1] = f( flatten(X) ), one independent subnetwork f[i] per output variable.

    Parameters
    ----------
    n          : number of variables N
    t          : context window length T
    hidden_dim : hidden size for f MLP
    n_layers   : number of hidden layers in f
    """

    def __init__(self,
                 n: int,
                 t: int,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 ):
        super().__init__()
        self.n = n
        self.t = t

        self.f = nn.ModuleList(MLP(in_dim=t * n, out_dim=1,
                    hidden_dim=hidden_dim, n_layers=n_layers) for _ in range(n))

        # self.bn = nn.BatchNorm1d(num_features=t * n)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict X_{t+1} from a window X_t.

        Parameters
        ----------
        X : (batch, T, N) or (T, N)

        Returns
        -------
        X_hat : same shape as X minus the time dimension.
        """
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)              # (1, T, N)

        # x_flat = self.bn(X.flatten(start_dim=1))   # (B, T*N)
        x_flat = X.flatten(start_dim=1)   # (B, T*N)

        out = [self.f[i](x_flat) for i in range(self.n)]
        out = torch.cat(out, dim=-1)  # (B, N)

        return out.squeeze(0) if single else out

    def infer(self, X: torch.Tensor) -> torch.Tensor:
        """Inference-time prediction (no gradients)."""
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def infer_diff(self, X: torch.Tensor) -> torch.Tensor:
        """Inference-time prediction (no gradients). here for lyapunov exp computation, no eval / nograd"""
        
        # self.eval()
        return self.forward(X)

    def column_norms(self) -> torch.Tensor:
        """
        Learned edge-strength matrix from the first-layer weight columns.

        Returns
        -------
        (T, N, N) tensor where entry [t, j, i] is ||W_i[:, t*N + j]||_2,
        the L2 norm of the column of f[i]'s first-layer weight connecting
        input (t, j) to output i.
        """
        norms = torch.stack([
            self.f[i].net[0].weight.norm(dim=0) for i in range(self.n)
        ], dim=-1)  # (T*N, N)
        return norms.reshape(self.t, self.n, self.n)


class StructuredODEDiscoveryAGL():

    def __init__(self, n: int, t: int, device, coupled=False, hidden_dim: int = 8, n_layers: int = 2, normalize: bool = True, normalize_grad: bool = False):

        self.t = t
        self.n = n
        self.device = device
        self.coupled = coupled
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.normalize = normalize
        self.normalize_grad = normalize_grad

    # ─────────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────────

    def loss_fn(self,
                X_windows: torch.Tensor,
                X_target: torch.Tensor,
                lambda_grad: float = 0.01,
            ) -> tuple[torch.Tensor, dict]:
        """
        Total training loss (smooth part only -- the AGL penalty is applied
        as a proximal step outside of this function, not via backprop).

        Terms
        -----
        1. Reconstruction  : MSE( X_hat, X_target )
        2. Gradient penalty: penalizes d(loss)/d(X_hat) norm

        Parameters
        ----------
        X_windows : (batch, T, N)   input windows
        X_target  : (batch, N)      ground truth X at t (i.e. the next step)
        lambda_grad: gradient penalty weight
        """
        X_hat = self.model(X_windows)

        total = F.mse_loss(X_hat, X_target)
        input_grads = torch.autograd.grad(
            outputs=total,
            inputs=X_hat,
            create_graph=True,   # keeps the graph alive so we can backprop through this
            retain_graph=True,   # keeps the graph alive for the final .backward()
        )[0]

        recon = total.item()

        grad_norm = input_grads.view(input_grads.shape[0], -1).norm(2, dim=1)
        gradient_penalty = (grad_norm ** 2).mean()
        total = total + lambda_grad * gradient_penalty

        return total, {
            "recon": recon,
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

        keys = ["total", "recon", "gradient_penalty", "col_norm_sum"]
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
            lambda_m: float = 0.01,
            lambda_init: float = 0.01,
            agl_gamma: float = 1.0,
            lambda_grad: float = 10,
            batch_size: int | None = None,
            n_inner_min_sparse: int = 0, # pilot phase, before AGL proximal step kicks in
            patience: int = 50,
            plot_frequency: int = 500,
            return_history: bool = False,
            save_fig_path: str | None = None
            ) -> tuple[StructuredDynamicsAGL, torch.Tensor] | tuple[StructuredDynamicsAGL, torch.Tensor, list[dict]]:
        """
        Fit the model to a time series X of shape (T, N) using adaptive group lasso.

        Two-stage adaptive group lasso (https://arxiv.org/pdf/2105.02522):
        1. Pilot phase (iter < n_inner_min_sparse): plain Adam gradient steps
           on the smooth loss only.
        2. At iter == n_inner_min_sparse: snapshot each f[i]'s first-layer
           column norms as the pilot estimate, and compute fixed adaptive
           weights lam_k = lambda_m / (pilot_norm_k**agl_gamma + eps).
        3. For iter >= n_inner_min_sparse: after every Adam step, apply a
           group soft-threshold proximal step (step size alpha = lr) to
           each f[i]'s first-layer weight using those fixed lam_k.

        Parameters
        ----------
        X          : (T, N) full time series
        n_inner    : number of training steps
        lr         : Adam learning rate, also used as the proximal step size alpha
        lambda_m   : overall scalar multiplier for the adaptive group lasso weights
        agl_gamma  : power applied to the pilot column norm when adapting weights
        lambda_grad: gradient penalty weight
        batch_size : optional minibatch size for training
        n_inner_min_sparse : number of pilot steps before the AGL proximal step starts
        """
        print(f"Dimensions: N={self.n}, T={self.t}")

        assert self.n == X.shape[-1], f"Model n={self.n} must match data N={X.shape[-1]}"
        if X.dim() != 3:
            raise ValueError("X must be a 3D tensor with shape (B, T, N) for training (T=1)")

        print(f"Number of NaN values in X: {torch.isnan(X).any()}")

        if self.normalize:
            X = X - X.mean(dim=0, keepdim=True)
            X = X / (X.std(dim=0, keepdim=True) + 1e-8) # Small fix

        print(f"Number of NaN values after normalization in X: {torch.isnan(X).any()}")
        print(f"Constant dimensions in X: {torch.where(X[:, 0].std(dim=0) == 0)}")

        if not self.coupled:
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
                X_last_all = X[self.t:] - X[self.t - 1 : -1]
            if self.t > 1:
                X_t_all = X_t_all.squeeze(2)
        else:
            if self.t == 0:
                X_t_all = X
                X_last_all = X
            elif self.t == 1:
                X_t_all = X[:-1]
                # Important to predict the difference and not the next step to avoid predicting identity
                X_last_all = X[1:]
            else:
                if X.shape[0] <= self.t:
                    raise ValueError(f"Time series length T={X.shape[0]} must be larger than window length t={self.t}")
                X_t_all = torch.stack([X[i : i + self.t] for i in range(X.shape[0] - self.t)], dim=0)
                X_last_all = X[self.t:]
            if self.t > 1:
                X_t_all = X_t_all.squeeze(2)

        print("X_t_all.shape:", X_t_all.shape)
        print("X_last_all.shape:", X_last_all.shape)

        if self.normalize_grad:
            std_grad = X_last_all.std((0, 1))
            std_grad[std_grad < 1e-6] = 1.0
            X_last_all -= X_last_all.mean((0, 1))
            X_last_all = X_last_all / std_grad

        X_t_all   = X_t_all.to(self.device)
        X_last_all = X_last_all.to(self.device)

        n_dataset_samples = X_t_all.shape[0]
        if n_dataset_samples == 0:
            raise ValueError("No training samples could be constructed from X and the chosen window length t")

        if batch_size is None:
            batch_size = n_dataset_samples
        elif batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        batch_size = min(batch_size, n_dataset_samples)

        self.model = StructuredDynamicsAGL(n=self.n, t=self.t, hidden_dim=self.hidden_dim, n_layers=self.n_layers).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Prepare a DataLoader to avoid Python-side indexing/permutation overhead
        dataset = TensorDataset(X_t_all, X_last_all)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        iter = 0
        best_loss = float('inf')
        no_improve_steps = 0
        history: list[dict] = []
        lam = lambda_init  # plain (non-adaptive, scalar) group lasso weight during the pilot phase

        while iter < n_inner:

            bool_sparse = iter >= n_inner_min_sparse

            # At the pilot/sparsification boundary, snapshot column norms and
            # switch (once) to fixed adaptive weights lam_k = lambda_m / (pilot_norm_k**agl_gamma + eps)
            if bool_sparse and not isinstance(lam, list):
                with torch.no_grad():
                    pilot_norms = [self.model.f[i].net[0].weight.norm(dim=0) for i in range(self.n)]
                    lam = [lambda_m / (pn ** agl_gamma + 1e-12) for pn in pilot_norms]

            gradient_penalty_all = 0
            recon_all = 0
            loss_all = 0

            n_batches = 0

            for X_t, X_last in loader:
                n_batches += 1

                # ensure X_t requires grad for gradient-penalty computation
                X_t = X_t.detach().requires_grad_(True)

                optimizer.zero_grad()

                loss, info = self.loss_fn(X_t, X_last, lambda_grad=lambda_grad)

                gradient_penalty_all += info["gradient_penalty"]
                recon_all += info["recon"]
                loss_all += loss.item()

                loss.backward()
                optimizer.step()

                for i in range(self.n):
                    lam_i = lam[i] if isinstance(lam, list) else lam
                    group_soft_threshold(self.model.f[i].net[0].weight, lam_i, alpha=lr)

            with torch.no_grad():
                col_norm_sum = sum(
                    self.model.f[i].net[0].weight.norm(dim=0).sum().item() for i in range(self.n)
                )

            history.append({
                "inner": iter + 1,
                "step": iter + 1,
                "total": loss_all / n_batches,
                "recon":    recon_all / n_batches,
                "gradient_penalty": lambda_grad * gradient_penalty_all / n_batches,
                "col_norm_sum": col_norm_sum
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
            f"grad_penalty={(gradient_penalty_all / n_batches):.4f}  ")

        self.plot_loss_components(history, M=1, save_path=save_fig_path)

        edge_strength = self.model.column_norms()

        if return_history:
            return self.model, edge_strength, history

        return self.model, edge_strength


def group_soft_threshold(W, lam, alpha):
    """
    Proximal step (step 2): group soft-thresholding on the columns of a
    first-layer weight matrix W (shape [hidden_dim, d_in]). Each column k
    is the group [A_1^j]_{.k} -- all weights connecting input x_k into
    this subnetwork's first layer.

    lam: scalar (plain group lasso) or per-column tensor of shape [d_in]
         (adaptive group lasso weights, e.g. 1/||pilot_col_k||_2^gamma)
    alpha: step size
    """
    with torch.no_grad():
        col_norm = W.norm(dim=0, keepdim=True)                  # ||[A_1^j]_{.k}||_2
        shrink = torch.clamp(1 - alpha * lam / (col_norm + 1e-12), min=0.0)
        W.mul_(shrink)                                          # zero/shrink whole columns
    return W
