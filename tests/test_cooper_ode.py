import cooper
import torch

from causaldynamics.baselines.ode_discovery_gs import (
    StructuredDynamics,
    StructuredODEDiscovery,
)


def test_structured_ode_uses_installed_cooper_formulations(tmp_path):
    torch.manual_seed(0)

    formulations = (
        "penalty",
        "lagrangian",
        "augmented_lagrangian",
        {"sparsity": "lagrangian", "lipschitz": "penalty"},
        {"sparsity": "augmented_lagrangian", "lipschitz": "penalty"},
    )
    for idx, formulation in enumerate(formulations):
        discovery = StructuredODEDiscovery(
            n=3,
            t=1,
            device=torch.device("cpu"),
            hidden_dim=2,
            n_layers=1,
            L_lipschitz=0.0,
            normalize=False,
        )

        trained_model, mask, history = discovery.run(
            torch.randn(8, 1, 3),
            n_inner=2,
            n_inner_min_sparse=0,
            batch_size=4,
            plot_frequency=0,
            return_history=True,
            save_fig_path=tmp_path / f"loss_components_{idx}.png",
            constraint_formulation=formulation,
        )

        assert cooper.__version__ == "1.0.1"
        assert isinstance(trained_model, StructuredDynamics)
        assert mask.shape == (1, 3, 3)
        assert len(history) == 2
        assert any(abs(entry["sparsity_violation"]) > 0 for entry in history)
        assert any(abs(entry["grad_violation"]) > 0 for entry in history)
        assert any(abs(entry["constraint_penalty"]) > 0 for entry in history)


def test_structured_ode_keeps_notebook_compatibility(tmp_path):
    torch.manual_seed(0)

    discovery = StructuredODEDiscovery(
        n=3,
        t=1,
        device=torch.device("cpu"),
        coupled=False,
        hidden_dim=2,
        n_layers=1,
        L_lipschitz=0.0,
        normalize=False,
    )

    trained_model, mask = discovery.run(
        torch.randn(8, 1, 3),
        n_inner=1,
        n_inner_min_sparse=0,
        batch_size=4,
        plot_frequency=0,
        save_fig_path=tmp_path / "loss_components_notebook.png",
    )

    predictions = trained_model.infer(torch.randn(7, 1, 3))

    assert mask.shape == (1, 3, 3)
    assert predictions.shape == (7, 1, 3)
    assert torch.isfinite(predictions).all()


def test_frozen_lagrangian_recovers_linear_penalty(tmp_path):
    """Classic Lagrangian with dual_lr=0 freezes the multipliers at lambda_m /
    lambda_grad, so the Lagrangian collapses to the fixed linear penalty used
    before Cooper (constraint_penalty == lambda_m * sparsity_violation +
    lambda_grad * grad_violation). This does NOT hold for the quadratic penalty."""
    lambda_m, lambda_grad = 0.1, 0.05

    def history_for(formulation, dual_lr):
        torch.manual_seed(0)
        discovery = StructuredODEDiscovery(
            n=3, t=1, device=torch.device("cpu"), hidden_dim=2, n_layers=1,
            L_lipschitz=0.0, normalize=False,
        )
        _, _, history = discovery.run(
            torch.randn(8, 1, 3), n_inner=3, n_inner_min_sparse=0, batch_size=8,
            plot_frequency=0, return_history=True,
            save_fig_path=tmp_path / f"loss_{formulation}.png",
            constraint_formulation=formulation, lambda_m=lambda_m,
            lambda_grad=lambda_grad, dual_lr=dual_lr,
        )
        return history

    def linear_penalty(entry):
        return lambda_m * entry["sparsity_violation"] + lambda_grad * entry["grad_violation"]

    frozen = history_for("lagrangian", dual_lr=0.0)
    assert all(
        abs(entry["constraint_penalty"] - linear_penalty(entry)) < 1e-5
        for entry in frozen
    )

    # The quadratic penalty is one-sided and squared, so it must NOT match.
    quadratic = history_for("penalty", dual_lr=None)
    assert any(
        abs(entry["constraint_penalty"] - linear_penalty(entry)) > 1e-4
        for entry in quadratic
    )


def test_lipschitz_budget_shifts_grad_violation(tmp_path):
    """grad_violation = mean(||grad||) - lipschitz_budget, so at the first (shared)
    step, dropping the budget from L_lipschitz to 0 raises grad_violation by exactly
    L_lipschitz. lipschitz_budget=None falls back to L_lipschitz."""
    L = 2.0

    def grad_violation_step0(lipschitz_budget):
        torch.manual_seed(0)
        discovery = StructuredODEDiscovery(
            n=3, t=1, device=torch.device("cpu"), hidden_dim=2, n_layers=1,
            L_lipschitz=L, normalize=False,
        )
        _, _, history = discovery.run(
            torch.randn(8, 1, 3), n_inner=1, n_inner_min_sparse=0, batch_size=8,
            plot_frequency=0, return_history=True,
            save_fig_path=tmp_path / f"loss_{lipschitz_budget}.png",
            constraint_formulation="lagrangian", dual_lr=0.0,
            lipschitz_budget=lipschitz_budget,
        )
        return history[0]["grad_violation"]

    budget_free = grad_violation_step0(0.0)
    default_budget = grad_violation_step0(None)  # None -> L_lipschitz
    assert abs((budget_free - default_budget) - L) < 1e-4
