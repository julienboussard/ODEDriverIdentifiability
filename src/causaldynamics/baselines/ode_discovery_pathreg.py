"""Structured ODE discovery with PathReg feature sparsity.

PathReg follows https://github.com/theislab/PathReg: hard-concrete gates are
placed on dense-layer input features, and the absolute gated connectivity
matrices are multiplied through the MLP to penalize input-output paths.
"""
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ZETA, _GAMMA, _EPS = -0.1, 1.1, 1e-6


def sample_hard_concrete(log_alpha, beta=2.0 / 3.0):
    u = torch.rand_like(log_alpha).clamp(_EPS, 1 - _EPS)
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / beta)
    return (s * (_GAMMA - _ZETA) + _ZETA).clamp(0.0, 1.0)


def hard_concrete_mean(log_alpha):
    return (torch.sigmoid(log_alpha) * (_GAMMA - _ZETA) + _ZETA).clamp(0.0, 1.0)


def hard_concrete_sparsity(log_alpha, beta=2.0 / 3.0):
    return torch.sigmoid(log_alpha - beta * math.log(-_ZETA / _GAMMA)).sum()


class PathRegMLP(nn.Module):
    """MLP with one hard-concrete input-feature gate per dense layer."""
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers + [out_dim]
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.gate_logits = nn.ParameterList(
            nn.Parameter(torch.full((dim,), math.log(4.0))) for dim in dims[:-1]
        )

    def _gates(self, sample):
        return [sample_hard_concrete(p) if sample and self.training else hard_concrete_mean(p)
                for p in self.gate_logits]

    def forward(self, x, sample=True):
        for i, (linear, gate) in enumerate(zip(self.linears, self._gates(sample))):
            x = linear(x * gate)
            if i + 1 < len(self.linears):
                x = torch.tanh(x)
        return x

    def path_strength(self, sample=False):
        path = None
        for linear, gate in zip(self.linears, self._gates(sample)):
            connectivity = gate[:, None].expand(-1, linear.out_features)
            path = connectivity if path is None else path @ connectivity
        return path.abs()


class StructuredDynamicsPathReg(nn.Module):
    """One independently gated PathReg MLP per output variable."""
    def __init__(self, n, t, hidden_dim=64, n_layers=3):
        super().__init__()
        self.n, self.t = n, t
        self.f = nn.ModuleList(PathRegMLP(t * n, 1, hidden_dim, n_layers) for _ in range(n))

    def forward(self, X, sample=True):
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
        flat = X.flatten(start_dim=1)
        out = torch.cat([net(flat, sample=sample) for net in self.f], dim=-1)
        return out.squeeze(0) if single else out

    def infer(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X, sample=False)

    def infer_diff(self, X):
        return self.forward(X, sample=False)

    def path_strengths(self):
        strengths = torch.stack([net.path_strength()[:, 0] for net in self.f], dim=-1)
        return strengths.reshape(self.t, self.n, self.n)

    def input_output_probabilities(self, beta=1.0 / 3.0):
        """Return first-layer gate probabilities as an ``(n, n)`` mask.

        Entry ``[j, i]`` is the probability that input variable ``j`` is
        active for the MLP predicting output variable ``i``. This method is
        defined for the single-timestep model, where the first-layer inputs
        are the ``n`` variables.
        """
        if self.t != 1:
            raise ValueError("input_output_probabilities requires t=1")
        probabilities = torch.stack([
            torch.sigmoid(
                network.gate_logits[0]
                - beta * math.log(-_ZETA / _GAMMA)
            )
            for network in self.f
        ], dim=-1)
        return probabilities


class StructuredODEDiscoveryPathReg:
    """Fit normalized one-step increments with PathReg regularization."""
    def __init__(self, n, t, device, coupled=False, hidden_dim=8, n_layers=2,
                 normalize=True, normalize_grad=False):
        self.n, self.t, self.device = n, t, device
        self.coupled, self.hidden_dim, self.n_layers = coupled, hidden_dim, n_layers
        self.normalize, self.normalize_grad = normalize, normalize_grad

    def loss_fn(self, X_windows, X_target, lambda_path=0.01, lambda_grad=0.01):
        prediction = self.model(X_windows, sample=True)
        reconstruction = F.mse_loss(prediction, X_target)
        grad = torch.autograd.grad(reconstruction, prediction, create_graph=True,
                                   retain_graph=True)[0]
        gradient_penalty = (grad.flatten(start_dim=1).norm(2, dim=1) ** 2).mean()
        path_reg = torch.stack([net.path_strength(sample=True).mean() for net in self.model.f]).mean()
        total = reconstruction + lambda_path * path_reg + lambda_grad * gradient_penalty
        return total, {"recon": reconstruction.item(), "path_reg": path_reg.item(),
                       "gradient_penalty": gradient_penalty.item()}

    def plot_loss_components(self, history, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in ("total", "recon", "path_reg", "gradient_penalty"):
            ax.plot([h["step"] for h in history], [h[key] for h in history], label=key)
        ax.set_xlabel("Training step"); ax.set_ylabel("Loss / penalty value")
        ax.legend(); ax.grid(True); fig.tight_layout()
        fig.savefig(save_path or "pathreg_loss.png"); plt.close(fig)

    def run(self, X, n_inner=10_000, lr=1e-3, lambda_path=0.01, lambda_grad=10,
            batch_size=None, patience=50, plot_frequency=500, return_history=False,
            save_fig_path=None, n_inner_min_sparse=0):
        if X.dim() != 3:
            raise ValueError("X must have shape (time, 1, n) or (time, t, n)")
        if X.shape[-1] != self.n:
            raise ValueError(f"Model n={self.n} must match data n={X.shape[-1]}")
        X = X.to(self.device)
        if self.normalize:
            X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)
        if self.t == 0:
            windows, target = X, X
        elif self.t == 1:
            windows = X[:-1]; target = X[1:] - X[:-1] if not self.coupled else X[1:]
        else:
            if X.shape[0] <= self.t:
                raise ValueError("time series must be longer than the window length")
            windows = torch.stack([X[i:i + self.t] for i in range(X.shape[0] - self.t)])
            target = X[self.t:] - X[self.t - 1:-1] if not self.coupled else X[self.t:]
            windows = windows.squeeze(2)
        if self.normalize_grad:
            std = target.std((0, 1)).clamp_min(1e-6)
            target = (target - target.mean((0, 1))) / std
        if not windows.shape[0]:
            raise ValueError("No training samples could be constructed")
        batch_size = windows.shape[0] if batch_size is None else min(batch_size, windows.shape[0])
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = StructuredDynamicsPathReg(self.n, self.t, self.hidden_dim, self.n_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(windows, target), batch_size=batch_size, shuffle=True)
        history, best, stale = [], float("inf"), 0
        for step in range(n_inner):
            if step < n_inner_min_sparse:
                step_lambda_path, step_lambda_grad = 0., 0.
            else:
                step_lambda_path, step_lambda_grad = lambda_path, lambda_grad
            sums, batches = {"total": 0., "recon": 0., "path_reg": 0., "gradient_penalty": 0.}, 0
            for batch, truth in loader:
                batch = batch.detach().requires_grad_(True); optimizer.zero_grad()
                loss, info = self.loss_fn(batch, truth, step_lambda_path, step_lambda_grad)
                loss.backward(); optimizer.step(); batches += 1
                sums["total"] += loss.item()
                for key, value in info.items(): sums[key] += value
            entry = {"step": step + 1, **{k: v / batches for k, v in sums.items()}}
            history.append(entry)
            if plot_frequency > 0 and (step + 1) % plot_frequency == 0:
                self.plot_loss_components(history, save_fig_path)
            if entry["total"] < best: best, stale = entry["total"], 0
            else:
                stale += 1
                if stale >= patience: break
        self.plot_loss_components(history, save_fig_path)
        if self.t == 1:
            mask = self.model.input_output_probabilities()
        else:
            mask = self.model.path_strengths()
        result = (self.model, mask)
        return (*result, history) if return_history else result
