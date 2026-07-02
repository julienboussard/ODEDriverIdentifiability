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
        "augmented_lagrangian",
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
