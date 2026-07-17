import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/network/scratch/j/julien.boussard/CausalDynamics/data/simple")
OUTPUT_ROOT = PROJECT_ROOT / "output" / "ground_truth_sparsity" / "simple"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ground-truth adjacency sparsity for simple datasets."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--confounder", default="False")
    parser.add_argument("--sort", choices=["density", "name"], default="density")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--show-counts", action="store_true")
    return parser.parse_args()


def parse_bool(token: str) -> bool:
    normalized = token.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {token}")


def display_name(dataset_path: Path) -> str:
    name = dataset_path.stem
    return name.removesuffix("_N10_T1000")


def read_sparsity(dataset_path: Path) -> dict[str, object]:
    with xr.open_dataset(dataset_path) as ds:
        adjacency = ds["adjacency_matrix"].to_numpy()
    n_possible = int(adjacency.size)
    n_edges = int(adjacency.sum())
    return {
        "dataset": dataset_path.stem,
        "name": display_name(dataset_path),
        "n_nodes_in": adjacency.shape[0],
        "n_nodes_out": adjacency.shape[1],
        "true_edges": n_edges,
        "possible_edges": n_possible,
        "density": float(adjacency.mean()),
        "zeros": n_possible - n_edges,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "name",
        "n_nodes_in",
        "n_nodes_out",
        "true_edges",
        "possible_edges",
        "density",
        "zeros",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path, show_counts: bool) -> None:
    names = [str(row["name"]) for row in rows]
    densities = np.array([float(row["density"]) for row in rows])
    y = np.arange(len(rows))

    height = max(8.0, 0.24 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(12, height))
    colors = ["#2f6f73" if name == "Lorenz" else "#7d8a91" for name in names]
    ax.barh(y, densities, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(
        "Ground-truth edge density (nonzero adjacency entries / total entries)"
    )
    ax.set_title("Ground-truth sparsity level by simple dataset")
    ax.set_xlim(0, max(1.0, densities.max() * 1.08))
    ax.grid(axis="x", alpha=0.25)

    if show_counts:
        for idx, row in enumerate(rows):
            label = f"{row['true_edges']}/{row['possible_edges']}"
            ax.text(
                densities[idx] + 0.01,
                idx,
                label,
                va="center",
                ha="left",
                fontsize=7,
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    confounder = parse_bool(args.confounder)
    data_dir = (
        args.data_root / f"noise={args.noise:.2f}_confounder={confounder}" / "data"
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    rows = [read_sparsity(path) for path in sorted(data_dir.glob("*.nc"))]
    if args.sort == "density":
        rows.sort(key=lambda row: (float(row["density"]), str(row["name"])))
    else:
        rows.sort(key=lambda row: str(row["name"]))
    if args.descending:
        rows.reverse()

    experiment = f"noise={args.noise:.2f}_confounder={confounder}"
    csv_path = args.output_root / experiment / "ground_truth_sparsity.csv"
    png_path = args.output_root / experiment / "ground_truth_sparsity.png"
    write_csv(rows, csv_path)
    plot(rows, png_path, show_counts=args.show_counts)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
