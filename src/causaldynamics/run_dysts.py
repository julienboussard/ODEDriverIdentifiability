"""Train and evaluate the ODE discovery baselines on a dysts system."""

from pathlib import Path

import dysts.flows as flows
import matplotlib.pyplot as plt
import numpy as np
import torch

from .baselines import (
    StructuredODEDiscovery,
    StructuredODEDiscoveryAGL,
    StructuredODEDiscoveryPathReg,
)
from .metrics import (
    correlation_dimension_error,
    dstsp,
    kaplan_yorke_dimension,
    log_spectral_distance,
    nn_jacobian_mse,
    nn_lyapunov_exponents_delta,
    nrmse_vs_time,
    true_lyapunov_exponents,
    valid_prediction_time,
    wasserstein_distance,
)


def _new_system(system_name, rng):
    system = getattr(flows, system_name)()
    system.ic = np.asarray(system.ic) * rng.uniform(0.5, 1.5)
    return system


def _rollout(model, initial_state, length, state_mean, state_std, increment_mean,
             increment_std, device):
    state = torch.as_tensor(initial_state, dtype=torch.float32, device=device)
    result = torch.empty((length, state.shape[-1]), device=device)
    result[0] = state
    for step in range(1, length):
        normalized_state = (state - state_mean) / state_std
        increment = model.infer(normalized_state[None])[0]
        state = normalized_state + increment * increment_std + increment_mean
        state = state * state_std + state_mean
        result[step] = state
    return result.cpu().numpy()


def _one_step_predictions(model, trajectory, state_mean, state_std,
                          increment_mean, increment_std, device):
    prediction = np.empty_like(trajectory)
    prediction[0] = trajectory[0]
    for step in range(1, len(trajectory)):
        state = torch.as_tensor(trajectory[step - 1], dtype=torch.float32,
                                device=device)
        normalized_state = (state - state_mean) / state_std
        increment = model.infer(normalized_state[None])[0]
        normalized_next = normalized_state + increment * increment_std + increment_mean
        prediction[step] = (normalized_next * state_std + state_mean).cpu().numpy()
    return prediction

def train_gs(models, training_input, dimension, device,  name, lambda_m, lambda_grad, patience = 200, n_iter_after_fixed = 200):
    discovery = StructuredODEDiscovery(
        coupled=False, t=1, n=dimension, L_lipschitz=1,
        is_lipschitz=False, hidden_dim=8, n_layers=2, device=device,
        beta_init=0.33, beta_min=0.33, annealing_rate=0.95,
        annealing_epochs=10, normalize=True, normalize_grad=True,
    )
    models[name], _ = discovery.run(
        n_samples=5, lambda_grad=lambda_grad, X=training_input,
        l0_l1_l2="l1" if name == "C-NODE" else "l0", th=0.5,
        lr=0.001, lambda_m=lambda_m, n_inner_min_sparse=100,
        batch_size=128, n_inner=2000,
        patience=patience, n_iter_after_fixed=n_iter_after_fixed,
    )
    return models


def _train_models(train_trajectory, dimension, device):
    train_mean = torch.as_tensor(train_trajectory.mean(axis=0), dtype=torch.float32,
                                 device=device)
    train_std = torch.as_tensor(train_trajectory.std(axis=0), dtype=torch.float32,
                                device=device).clamp_min(1e-8)
    normalized = (torch.as_tensor(train_trajectory, dtype=torch.float32,
                                   device=device) - train_mean) / train_std
    increments = normalized[1:] - normalized[:-1]
    increment_mean = increments.mean(dim=(0, 1))
    increment_std = increments.std(dim=(0, 1)).clamp_min(1e-6)
    training_input = normalized[:, None, :]

    # print(f"train_trajectory.shape: {train_trajectory.shape}")

    models = {}
    print("Train AGL")
    agl = StructuredODEDiscoveryAGL(
        coupled=False, t=1, n=dimension, hidden_dim=8, n_layers=2,
        device=device, normalize=True, normalize_grad=True,
    )
    models["agl"], _ = agl.run(
        lambda_grad=0, lambda_init=0.0, lambda_m=0.05, lr=0.001,
        n_inner_min_sparse=50, batch_size=128, n_inner=1000,
        patience=1000, X=training_input,
    )

    print("train L0")
    first_name = "L0-NODE GP 0 pat. 10"
    models = train_gs(models, training_input, dimension, device, first_name, 0.025, 0, n_iter_after_fixed = 10)
    models = train_gs(models, training_input, dimension, device, "L0-NODE GP 0 pat. 50", 0.025, 0, n_iter_after_fixed = 50)
    models = train_gs(models, training_input, dimension, device, "L0-NODE gp 0 pat 100", 0.025, 0, n_iter_after_fixed = 100)
    models = train_gs(models, training_input, dimension, device, "L0-NODE gp 0 pat 200", 0.025, 0, n_iter_after_fixed = 200)
    models = train_gs(models, training_input, dimension, device, "L0-NODE gp 0 pat 500", 0.025, 0, n_iter_after_fixed = 500)
    # print("train l1")
    # models = train_gs(models, training_input, dimension, device, "C-NODE", 50.0, 10)
    print("train NODE")
    models = train_gs(models, training_input, dimension, device, "NODE gp 0 pat 0", 0.0, 0, n_iter_after_fixed = 0)
    # models = train_gs(models, training_input, dimension, device, "NODE gp 0 pat 0", 0.0, 1000, n_iter_after_fixed = 0)
    # print("train no grad")
    # models = train_gs(models, training_input, dimension, device, "No grad reg.", 0.025, 0)


    print("train pathreg")
    pathreg = StructuredODEDiscoveryPathReg(
        coupled=False, t=1, n=dimension, hidden_dim=8, n_layers=2,
        device=device, normalize=True, normalize_grad=True,
    )
    models["pathreg"], _ = pathreg.run(
        lambda_grad=0, lambda_path=0.05, X=training_input, lr=0.001,
        n_inner_min_sparse=100, batch_size=128, n_inner=2000, patience=200,
    )
    return models, train_mean, train_std, increment_mean, increment_std, first_name


def run_dysts(system_name, save_fig_path, device=None, seed=0):
    """Train and evaluate all requested discovery models on one dysts system.

    The metric files are written next to ``save_fig_path`` as one NPZ per
    method. The returned dictionary contains the exponents, trajectories, and
    metrics in addition to the paths of the saved files.
    """
    if not hasattr(flows, system_name):
        raise ValueError(f"Unknown dysts system: {system_name}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)
    reference_system = getattr(flows, system_name)()
    lyapunov_exponents = true_lyapunov_exponents(reference_system)
    lambda_1 = float(lyapunov_exponents[0])
    if lambda_1 <= 0:
        raise ValueError(f"The first Lyapunov exponent must be positive, got {lambda_1}")

    use_lyapunov_times = 1.0 / lambda_1 > 20
    train_steps = int(np.ceil(50 / lambda_1)) if use_lyapunov_times else 1000
    test_steps = int(np.ceil(250 / lambda_1)) if use_lyapunov_times else 5000
    metric_steps = int(np.ceil(5 / lambda_1)) if use_lyapunov_times else 50
    burn_steps = int(np.ceil(50 / lambda_1)) if use_lyapunov_times else 1000

    print("Make trajectories")
    train_tpts, train_trajectory = reference_system.make_trajectory(
        train_steps, return_times=True
    )
    train_trajectory = np.asarray(train_trajectory)
    # Actual sampling interval of `make_trajectory`'s output, which (like in
    # `true_lyapunov_exponents`) generally differs from `system.dt` due to
    # its default Fourier-timescale resampling -- needed to get exponents
    # from `nn_lyapunov_exponents_delta` in the same time units as
    # `true_lyapunov_exponents`.
    dt = float(np.diff(np.asarray(train_tpts)).mean())

    test_trajectories = np.stack([
        np.asarray(_new_system(system_name, trajectory_rng).make_trajectory(test_steps))
        for trajectory_rng in rng.spawn(3) # Here, using 3 test trajectories each time
    ])

    # trajectories_dimension = train_trajectory.shape[-1]
    # trajectories_colors = ["steelblue", "firebrick", "seagreen", "darkorange"]
    # trajectories_labels = ["train"] + [
    #     f"test {index}" for index in range(test_trajectories.shape[0])
    # ]
    # trajectories_figure = plt.figure(figsize=(8, 8))
    # if trajectories_dimension == 3:
    #     trajectories_axis = trajectories_figure.add_subplot(projection="3d")
    #     trajectories_axis.plot(*train_trajectory.T, lw=0.5, color=trajectories_colors[0],
    #                             label=trajectories_labels[0])
    #     for index, test_trajectory in enumerate(test_trajectories):
    #         trajectories_axis.plot(*test_trajectory.T, lw=0.5,
    #                                 color=trajectories_colors[index + 1],
    #                                 label=trajectories_labels[index + 1])
    # else:
    #     trajectories_axis = trajectories_figure.add_subplot()
    #     trajectories_axis.plot(train_trajectory, lw=0.5, color=trajectories_colors[0],
    #                             label=trajectories_labels[0])
    #     for index, test_trajectory in enumerate(test_trajectories):
    #         trajectories_axis.plot(test_trajectory, lw=0.5,
    #                                 color=trajectories_colors[index + 1],
    #                                 label=trajectories_labels[index + 1])
    # trajectories_axis.set_title(f"{system_name} train / test trajectories")
    # trajectories_axis.legend()
    # trajectories_figure.tight_layout()
    # trajectories_figure.savefig(
    #     Path(save_fig_path).with_name(f"{system_name}_trajectories.png")
    # )
    # plt.close(trajectories_figure)

    print("Training ")


    dimension = train_trajectory.shape[-1]
    models, state_mean, state_std, increment_mean, increment_std, first_name = _train_models(
        train_trajectory, dimension, device
    )

    train_prediction = _one_step_predictions(
        models[first_name], train_trajectory, state_mean, state_std, increment_mean,
        increment_std, device,
    )
    predictions = {
        name: np.stack([
            _rollout(model, trajectory[0], test_steps, state_mean, state_std,
                     increment_mean, increment_std, device)
            for trajectory in test_trajectories
        ])
        for name, model in models.items()
    }

    print("Evaluation")

    output_dir = Path(save_fig_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # `models[name]` predicts a *normalized* increment,
    # normalized_state_t + increment ~= (X_{t+1} - state_mean) / state_std,
    # i.e. model.infer_diff(x_norm) ~= (X_{t+1} - X_t - increment_mean) / increment_std
    # with x_norm = (X_t - state_mean) / state_std. So the model's estimate
    # of the *raw* one-step increment is
    #   state_std * (increment_std * model.infer_diff(x_norm) + increment_mean)
    #   = (state_std * increment_std) * model.infer_diff(x_norm) + (state_std * increment_mean)
    # which is exactly `nn_jacobian_mse`'s `std * model(x) + mean` form with
    # `model` set to `x -> model.infer_diff(normalize(x))` and
    # `std, mean = state_std * increment_std, state_std * increment_mean`.
    jacobian_std = state_std * increment_std
    jacobian_mean = state_std * increment_mean

    # One raw test trajectory (and its normalized counterpart) used for the
    # Jacobian-error and Lyapunov-exponent metrics, which -- unlike the
    # other metrics -- operate on a single (T, D) trajectory rather than
    # the full (K, T, D) batch of test trajectories.
    raw_traj = torch.as_tensor(test_trajectories[0], dtype=torch.float32, device=device)
    normalized_traj = (raw_traj - state_mean) / state_std

    # Lyapunov exponents are invariant to the constant per-dimension affine
    # state normalization (X - state_mean) / state_std, so
    # `nn_lyapunov_exponents_delta` is called on `normalized_traj` directly
    # -- only the increment normalization (`increment_mean`/`increment_std`)
    # needs to be undone inside it, matching its docstring.
    lyap_num_steps = max(1, min(test_steps - burn_steps - 1, 2000))
    dky_true = kaplan_yorke_dimension(lyapunov_exponents)

    metrics = {}
    for name, prediction in predictions.items():
        model = models[name]
        short_true = test_trajectories[:, :metric_steps]
        short_pred = prediction[:, :metric_steps]
        post_burn_true = test_trajectories[:, burn_steps:]
        post_burn_pred = prediction[:, burn_steps:]

        def model_field(x_raw, model=model):
            return model.infer_diff(((x_raw - state_mean) / state_std)[None])[0]

        jacobian_mse = nn_jacobian_mse(
            reference_system, raw_traj[:metric_steps], model_field,
            jacobian_mean, jacobian_std,
        )

        nn_exponents = nn_lyapunov_exponents_delta(
            model, normalized_traj, increment_mean, increment_std, dt,
            num_steps=lyap_num_steps, t_burn=burn_steps,
        )
        lambda1_error = float(np.abs(nn_exponents[0] - lambda_1))
        dky_error = float(np.abs(kaplan_yorke_dimension(nn_exponents) - dky_true))

        values = {
            "nrmse": nrmse_vs_time(short_true, short_pred).mean(),
            "vpt": valid_prediction_time(test_trajectories, prediction),
            "jacobian_mse": jacobian_mse,
            "lambda1_error": lambda1_error,
            "dky_error": dky_error,
            "dstsp": dstsp(test_trajectories, prediction),
            # "correlation_dimension_error": correlation_dimension_error(
            #     test_trajectories, prediction, t_burn=burn_steps,
            # ),
            # "lsd": log_spectral_distance(post_burn_true, post_burn_pred),
            # "wasserstein": wasserstein_distance(
            #     test_trajectories, prediction, t_burn=burn_steps,
            #     multi_dimensional=False,
            # ),
            "lyapunov_exponents": lyapunov_exponents,
            "nn_lyapunov_exponents": nn_exponents,
        }
        metrics[name] = values
        np.savez(output_dir / f"{system_name}_{name}_metrics.npz", **values)

    print("Figures")
    n_panels = len(predictions) + 1
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    figure = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    if dimension == 3:
        axes = [figure.add_subplot(n_rows, n_cols, index + 1, projection="3d")
                for index in range(n_rows * n_cols)]
        for axis, (name, prediction) in zip(axes, predictions.items()):
            for traj_index in range(test_trajectories.shape[0]):
                axis.plot(*test_trajectories[traj_index, :test_steps].T, lw=0.5,
                          color="steelblue")
                axis.plot(*prediction[traj_index].T, lw=0.5, color="firebrick")
            axis.set_title(name)
        axes[len(predictions)].plot(*train_trajectory.T, lw=0.5, color="steelblue")
        axes[len(predictions)].plot(*train_prediction.T, lw=0.5, color="firebrick")
        axes[len(predictions)].set_title("training / next-step prediction")
    else:
        axes = list(figure.subplots(n_rows, n_cols, squeeze=False).flat)
        for axis, (name, prediction) in zip(axes, predictions.items()):
            for traj_index in range(test_trajectories.shape[0]):
                axis.plot(test_trajectories[traj_index], lw=0.5, color="steelblue")
                axis.plot(prediction[traj_index], lw=0.5, color="firebrick")
            axis.set_title(name)
        axes[len(predictions)].plot(train_trajectory, lw=0.5, color="steelblue")
        axes[len(predictions)].plot(train_prediction, lw=0.5, color="firebrick")
        axes[len(predictions)].set_title("training / next-step prediction")
    for axis in axes[n_panels:]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(save_fig_path)
    plt.close(figure)

    return {
        "lyapunov_exponents": lyapunov_exponents,
        "train_trajectory": train_trajectory,
        "test_trajectories": test_trajectories,
        "predictions": predictions,
        "metrics": metrics,
    }