"""
Structured Dynamical Model
===========================
Predicts the last timestep from a window of T timesteps:
 
    X_hat[-1] = f( flatten( X @ S @ W @ S.T ) )
 
where
    X  : (T, N)     input window  (full history)
    S  : (N, K)     soft cluster assignment, softmax over rows
    W  : (T, K, K)  per-timestep cluster interaction matrices
    S.T: (K, N)
 
Step by step:
    z1 = X @ S          (T, N) @ (N, K)  -> (T, K)   project to cluster space
    z2 = z1 @ W         (T, K) x (T,K,K) -> (T, K)   per-timestep cluster mixing
    z3 = z2 @ S.T       (T, K) @ (K, N)  -> (T, N)   project back to variable space
    out = f(z3.flatten())  (T*N,) -> (N,)             predict X[-1]
 
Constraints
-----------
- S  : softmax rows  →  each variable soft-assigned to one cluster
- W  : L1 sparsity + NOTEARS acyclicity on W[-1] only
- S  : entropy regularisation  →  push toward hard (one-hot) assignments
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt


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
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
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
            layers += [spectral_norm(nn.Linear(d, hidden_dim)), nn.SiLU()]
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
    k          : number of clusters K  (K << N)
    t          : context window length T
    device     : device to run the model on
    hidden_dim : hidden size for f MLP
    n_layers   : number of hidden layers in f
    """
 
    def __init__(self,
                 n: int,
                 k: int,
                 t: int,
                 device,
                 instantaneous: bool = False,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 L_lipschitz: float = 2.0, 
                 is_lipschitz: bool = False, 
                 M_mask: torch.Tensor | None = None):
        super().__init__()
        self.n = n
        self.k = k
        self.t = t
        self.instantaneous = instantaneous
        self.is_lipschitz = is_lipschitz
        # ── Structural parameters ─────────────────────────────────────────────
 
        # S: cluster assignment logits (N, K)
        # softmax over dim=-1: each variable's row is a prob. dist. over clusters
        if k > 1:
            self.log_S = nn.Parameter(torch.randn(n, k) * 0.1) #.to(device)
            self.W_logits = nn.Parameter(torch.ones((t+1), k, k) * 3) #.to(device)
        else:
            self.log_S =torch.ones((n, k), requires_grad=False).to(device)  # if K=1, all variables in one cluster, no need to learn it
            self.W_logits = torch.ones(((t+1), k, k), requires_grad = False).to(device) * 3

        self.M_logits = nn.Parameter(torch.ones((t+1), n, n) * 3) #.to(device)
        
        # Hard mask for M: initialized to all ones, gets progressively sparsified
        self.register_buffer('M_mask', M_mask if M_mask is not None else torch.ones((t+1, n, n)))
        
 
        # W: per-timestep cluster interactions (T, K, K)
        # acyclicity enforced on W[-1] only; L1 sparsity on all T slices
#         print(torch.sigmoid(self.M_logits))

        # f: flattened (T*N,) -> (N,)
        if self.is_lipschitz:
            self.f_lagged = nn.ModuleList(LipschitzNet(in_dim=t * n, out_dim=1,
                        hidden_dim=hidden_dim, n_layers=n_layers, L=L_lipschitz) for _ in range(n))
            if instantaneous:
                self.f_instantaneous = LipschitzNet(in_dim=n, out_dim=n,
                            hidden_dim=hidden_dim, n_layers=n_layers, L=L_lipschitz)
        else:
            self.f_lagged = nn.ModuleList(MLP(in_dim=t * n, out_dim=1,
                        hidden_dim=hidden_dim, n_layers=n_layers) for _ in range(n))
            if instantaneous:
                self.f_instantaneous = MLP(in_dim=n, out_dim=n,
                            hidden_dim=hidden_dim, n_layers=n_layers)
    
    # ── Forward pass ──────────────────────────────────────────────────────────
 
    def forward(self, X: torch.Tensor) -> torch.Tensor:
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

        if self.k > 1:
        
            S =  F.softmax(self.log_S)
    
            z1 = X @ S
    
            # Step 2 — per-timestep cluster mixing
            # einsum: z2[b,t,:] = z1[b,t,:] @ W[t,:,:]
            # (B, T, K) x (T, K, K)  ->  (B, T, K)
            
            #TODO: clarify: is this K*K i.e. one more dimension
            
            W = torch.sigmoid(self.W_logits)
            z2 = torch.einsum("btk, tkj -> btj", z1, W[:-1])

    
            # Step 3 — project back to variable space
            # (B, T, K) @ (K, N)  ->  (B, T, N)
            z3 = z2 @ S.T
            
            M = torch.sigmoid(self.M_logits) * self.M_mask
            #TODO: clarify: is this K*K i.e. one more dimension --> proper ,mask
            z4 = torch.einsum("btk, tkj -> btkj", z3, M[:-1]) # elementwise mask to sparsify the full T*N space
        else:
            M = torch.sigmoid(self.M_logits) * self.M_mask
            z4 = torch.einsum("btk, tkj -> btkj", X, M[:-1])

        # Step 4 — flatten N*N and predict X[-1]
        # (B, T, N*N)  ->  (B, N)
        out = []
        for i in range(self.n):
            out.append(self.f_lagged[i](z4[:, :, :, i]))
        out = torch.stack(out, dim=-1) # (B, N)

            
        if self.instantaneous:
            z5 = torch.einsum("bk, kj -> bkj", out, M[-1]) # residual connection from input to output of masked space
            out =  out + self.f_instantaneous(z5.flatten(start_dim=1))
                
        return out.squeeze(0) if single else out
 
 
    @torch.no_grad()
    def hard_assignment(self) -> torch.LongTensor:
        """Hard cluster label for each variable via argmax. Shape: (N,)."""
        return F.softmax(self.log_S).argmax(dim=-1)
 

class StructuredLowRankDiscovery():

    def __init__(self, n: int, k: int, t: int, device, instantaneous: bool = False, hidden_dim: int = 8, n_layers: int = 2, L_lipschitz: float = 2.0, is_lipschitz: bool = False):

        self.t = t
        self.n = n
        self.k = k
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers    
        self.instantaneous = instantaneous
        self.is_lipschitz = is_lipschitz
        self.L_lipschitz = L_lipschitz


    # ─────────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────────
    
    def loss_fn(self,
                X_windows: torch.Tensor,
                X_target: torch.Tensor,
                lambda_dag: float = 1.0,
                mu_dag: float = 1.0,
                lambda_s: float = 0.01,
                lambda_w: float = 0.01,
                bool_sparse: bool = False) -> tuple[torch.Tensor, dict]:
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
        """
        X_hat = self.model(X_windows)                        # (batch, N)
    
        # 1. Reconstruction
        recon = F.mse_loss(X_hat, X_target)

        total = recon

        if bool_sparse:
            if self.instantaneous:
                m_sparse = torch.sigmoid(self.model.M_logits).sum()
            else:
                m_sparse = torch.sigmoid(self.model.M_logits[:-1]).sum()

            # To change to have two parameters
            total += lambda_w * m_sparse / self.model.n**2

            if self.model.k > 1:

                # 2. W sparsity — L1 over all T slices
                w_sparse = torch.sigmoid(self.model.W_logits).sum()
                # 3. DAG acyclicity on W[-1] only  (augmented Lagrangian) + S entropy — push rows toward one-hot
                h   = acyclicity(torch.sigmoid(self.model.W_logits)[-1])     
                dag = lambda_dag * h + (mu_dag / 2) * h ** 2
                
                S = F.softmax(self.model.log_S)

                entropy = -(S * (S + 1e-8).log()).sum(dim=-1).mean()
                total += lambda_w * w_sparse  / self.model.k**2 + dag + lambda_s * entropy
        
    
        if self.model.k > 1 and bool_sparse:
            return total, {
                "recon":    recon.item(),
                "last_hw": h.item(),
                "dag":      dag.item(),
                "entropy":  entropy.item(),
                "w_sparse": w_sparse.item(),
                "m_sparse": m_sparse.item(),
            }
        elif bool_sparse:
            return total, {
                "recon":    recon.item(),
                "last_hw": 0.0,
                "dag":      0.0,
                "entropy":  0.0,
                "w_sparse": 0.0,
                "m_sparse": m_sparse.item(),
            }
        else:
            return total, {
                "recon":    recon.item(),
                "last_hw": 0.0,
                "dag":      0.0,
                "entropy":  0.0,
                "w_sparse": 0.0,
                "m_sparse": 1.0,
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

        keys = ["total", "recon", "dag", "entropy", "w_sparse", "m_sparse", "gradient_penalty"]
        available = [k for k in keys if any(k in entry for entry in history)]
        if not available:
            raise ValueError("history entries must contain at least one loss component")

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        for key in available:
            ys = [history[i][key] for i in indices]
            ax.plot(xs, ys, marker="o", label=key)

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
            n_outer: int = 50,
            n_inner: int = 200,
            lr: float = 1e-3,
            mu_init: float = 0.0001,
            mu_factor: float = 1.1,
            lambda_dag_init: float = 0.0,
            h_tol: float = 1e-4,
            lambda_s: float = 0.01,
            lambda_w: float = 0.01,
            batch_size: int | None = None,
            n_inner_min_sparse: int = 500, # before sparsification
            th: float = 0.5,
            patience: int = 5,
            plot_frequency: int = 10,
            return_history: bool = False) -> tuple[StructuredDynamics, torch.Tensor] | tuple[StructuredDynamics, torch.Tensor, list[dict]]:
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
        batch_size : optional minibatch size for training
        th         : threshold for sparsifying M coefficients
        """
        # Build sliding training windows (B, T, N) and targets (B, N)

        print(f"Dimensions: N={self.n}, K={self.k}, T={self.t}")

        assert self.n == X.shape[-1], f"Model n={self.n} must match data N={X.shape[-1]}"
        if X.dim() != 3:
            raise ValueError("X must be a 3D tensor with shape (B, T, N) for training")

        if self.t == 0:
            X_t_all = X
            X_last_all = X
        elif self.t == 1:
            X_t_all = X[:-1].unsqueeze(1)
            X_last_all = X[1:]
        else:
            if X.shape[0] <= self.t:
                raise ValueError(f"Time series length T={X.shape[0]} must be larger than window length t={self.t}")
            X_t_all = torch.stack([X[i : i + self.t] for i in range(X.shape[0] - self.t)], dim=0)
            X_last_all = X[self.t:]
        print(f"X_t_all.shape = {X_t_all.shape}")
        print(f"X_last_all.shape = {X_last_all.shape}")

        n_samples = X_t_all.shape[0]
        if n_samples == 0:
            raise ValueError("No training samples could be constructed from X and the chosen window length t")

        if batch_size is None:
            batch_size = n_samples
        elif batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        batch_size = min(batch_size, n_samples)

        lambda_dag = lambda_dag_init
        mu_dag = mu_init
    
        M_mask = torch.ones((self.t + 1, self.n, self.n), device=self.device)
        history: list[dict] = []
        current_loss = float('nan')

        outer = 0
        while outer < n_outer:
            # print the different values of info
            # recon_list = []
            # dag_list = []
            # entropy_list = []
            # w_sparse_list = []

            self.model = StructuredDynamics(n=self.n, k=self.k, t=self.t, device=self.device, hidden_dim=self.hidden_dim, n_layers=self.n_layers, L_lipschitz=self.L_lipschitz, is_lipschitz=self.is_lipschitz, M_mask=M_mask).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            inner_step = 0
            best_loss = float('inf')
            no_improve_steps = 0

            # batch_start = 0
            while inner_step < n_inner:

                perm = torch.randperm(n_samples, device=X_t_all.device)

                bool_sparse = inner_step >= n_inner_min_sparse 

                gradient_penalty_all = 0
                recon_all = 0
                last_hw_all = 0
                dag_all = 0
                entropy_all = 0
                w_sparse_all = 0
                m_sparse_all = 0

                n_batches = 0

                for batch_start in range(0, n_samples, batch_size):

                    n_batches += 1
                    
                    idx   = perm[batch_start : batch_start + batch_size]
                    X_t   = X_t_all[idx].squeeze(1).detach().requires_grad_(True)
                    X_last = X_last_all[idx].squeeze(1)

                    optimizer.zero_grad()

                    loss, info = self.loss_fn(X_t, X_last,
                                        lambda_dag=lambda_dag,
                                        mu_dag=mu_dag,
                                        lambda_s=lambda_s,
                                        lambda_w=lambda_w,
                                        bool_sparse=bool_sparse)

                    input_grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=X_t,
                        create_graph=True,   # keeps the graph alive so we can backprop through this
                        retain_graph=True,   # keeps the graph alive for the final .backward()
                    )[0]

                    grad_norm = input_grads.view(input_grads.shape[0], -1).norm(2, dim=1)
                    gradient_penalty = (torch.clamp(grad_norm - self.L_lipschitz, min=0) ** 2).mean()

                    loss = loss +  lambda_s * gradient_penalty

                    gradient_penalty_all += gradient_penalty
                    recon_all += info["recon"]
                    last_hw_all += info["last_hw"]
                    dag_all += info["dag"]
                    entropy_all += info["entropy"]          
                    w_sparse_all += info["w_sparse"]
                    m_sparse_all += info["m_sparse"]


                    loss.backward()
                    # Clip gradients — A is N×N and can have large raw gradients
    #                 nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()


                info["gradient_penalty"] = float(gradient_penalty_all.item() / n_batches)
                history.append({
                    "outer": outer + 1,
                    "inner": inner_step + 1,
                    "step": outer * n_inner + inner_step + 1,
                    "total": float(loss.item()),
                    "recon":    recon_all / n_batches,
                    "last_hw": last_hw_all / n_batches,
                    "dag":      dag_all / n_batches,
                    "entropy":  entropy_all / n_batches,
                    "w_sparse": w_sparse_all / n_batches,
                    "m_sparse": m_sparse_all / n_batches,
                })

                if plot_frequency > 0 and (inner_step + 1) % plot_frequency == 0:
                    self.plot_loss_components(history, M=1)

                if recon_all / n_batches < best_loss:
                    best_loss = recon_all / n_batches
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    if no_improve_steps >= patience:
                        print(f"  → stopping inner loop at step {inner_step} after {no_improve_steps} no-improve steps (best={best_loss:.6f}, current={(recon_all / n_batches):.6f})")
                        break


                # recon_list.append(info["recon"])
                # dag_list.append(info["dag"])
                # entropy_list.append(info["entropy"])    
                # w_sparse_list.append(info["w_sparse"])

                inner_step += 1
    
            h_val = info["last_hw"]
            
            print(f"outer={outer+1:02d}  h(W)={(last_hw_all / n_batches):.5f}  "
                f"recon={(recon_all / n_batches):.4f}  "
                f"entropy={(entropy_all / n_batches):.3f}  "
                f"|W|_1={(w_sparse_all / n_batches):.3f}  "
                f"|M|_1={(m_sparse_all / n_batches):.3f}  "
                f"lambda={lambda_dag:.3f}  mu={mu_dag:.2f}")
    
            if bool_sparse:
                # Progressive sparsification: zero out M coefficients below threshold
                with torch.no_grad():
                    M = torch.sigmoid(self.model.M_logits)
                    # Find coefficients that should be masked (currently active but below threshold)
                    currently_active = M_mask > th
                    below_th = M < th
                    to_mask = currently_active & below_th
                    n_zeroed = to_mask.sum().item()
                    
                    if n_zeroed > 0:
                        # Update the mask to zero out these coefficients
                        M_mask[to_mask] = 0.0
                        print(f"  → Zeroed {n_zeroed} M coefficients (threshold={th})")
                        print(f"M_mask {M_mask}")
                        print(M[0]) 
                    else:
                        # No more coefficients below threshold, convergence reached
                        print(f"  → Sparsification converged: no coefficients below threshold={th}")
                        break

            if self.instantaneous and self.k > 1:
                with torch.no_grad():
                    lambda_dag += mu_dag * acyclicity(torch.sigmoid(self.model.W_logits[-1])).item()
                mu_dag *= mu_factor
            
            outer += 1


    
        if return_history:
            return self.model, M_mask, history
        return self.model, M_mask
    
    