import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math


REWARD_KEYS = (
    "reward_sum",
    "reward_mean",
    "reward_pred_sum",
    "reward_pred_mean",
    "reward_env_sum",
    "reward_env_mean",
)
SOCIAL_ORDER = ("efficiency", "equality", "sustainability", "peace")


def _load_metrics(metrics_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _iter_records_with_episode(
    records: Iterable[Dict[str, Any]],
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for record in records:
        episode = record.get("episode")
        if isinstance(episode, int):
            yield episode, record
        elif isinstance(episode, float):
            yield int(episode), record


def _normalize_social_metrics(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items() if isinstance(v, (int, float))}
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return {
            name: float(metric)
            for name, metric in zip(SOCIAL_ORDER, value)
            if isinstance(metric, (int, float))
        }
    return {}


def _smooth_points_with_std(
    points: List[Tuple[int, float]], window: int
) -> Tuple[List[Tuple[int, float]], List[float]]:
    if window <= 1 or len(points) < 2:
        return points, []
    points = sorted(points, key=lambda item: item[0])
    values = [value for _, value in points]
    smoothed: List[Tuple[int, float]] = []
    stds: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        mean = sum(window_values) / len(window_values)
        variance = sum((val - mean) ** 2 for val in window_values) / len(window_values)
        std = math.sqrt(variance)
        smoothed.append((points[idx][0], mean))
        stds.append(std)
    return smoothed, stds


def _plot_series(
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    ylabel: str,
    output_path: str,
    smooth_window: int,
) -> bool:
    if not series:
        return False
    plt.figure(figsize=(10, 5))
    plotted = False
    for name, points in series.items():
        if not points:
            continue
        points, stds = _smooth_points_with_std(points, smooth_window)
        xs = [episode for episode, _ in points]
        ys = [value for _, value in points]
        plt.plot(xs, ys, label=name)
        if smooth_window > 1 and stds:
            lower = [y - s for y, s in zip(ys, stds)]
            upper = [y + s for y, s in zip(ys, stds)]
            plt.fill_between(xs, lower, upper, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def _normalize_series(series: Dict[str, List[Tuple[int, float]]]) -> Dict[str, List[Tuple[int, float]]]:
    normalized: Dict[str, List[Tuple[int, float]]] = {}
    for name, points in series.items():
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        values = [value for _, value in points]
        min_val = min(values)
        max_val = max(values)
        denom = max_val - min_val
        if denom == 0:
            normalized[name] = [(episode, 0.0) for episode, _ in points]
            continue
        normalized[name] = [(episode, (value - min_val) / denom) for episode, value in points]
    return normalized


def _load_run_context(run_dir: str) -> Tuple[str, str]:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return "unknown", ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return "unknown", ""
    algo_name = payload.get("algorithm", {}).get("name", "unknown")
    rm_phi = ""
    reward_model_cfg = payload.get("reward_model")
    if isinstance(reward_model_cfg, dict) and reward_model_cfg.get("enabled"):
        phi = reward_model_cfg.get("phi")
        if isinstance(phi, str) and phi:
            rm_phi = phi
    return algo_name, rm_phi


def _plot_social_subplots(
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    output_path: str,
    smooth_window: int,
) -> bool:
    ordered_names = [name for name in SOCIAL_ORDER if series.get(name)]
    if not ordered_names:
        ordered_names = [name for name, points in series.items() if points]
    if not ordered_names:
        return False

    if len(ordered_names) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        axes_list = axes.flatten()
    else:
        fig, axes = plt.subplots(
            len(ordered_names), 1, figsize=(10, 3 * len(ordered_names)), sharex=True
        )
        axes_list = [axes] if len(ordered_names) == 1 else list(axes)

    for ax, name in zip(axes_list, ordered_names):
        points, stds = _smooth_points_with_std(series[name], smooth_window)
        xs = [episode for episode, _ in points]
        ys = [value for _, value in points]
        ax.plot(xs, ys, label=name)
        if smooth_window > 1 and stds:
            lower = [y - s for y, s in zip(ys, stds)]
            upper = [y + s for y, s in zip(ys, stds)]
            ax.fill_between(xs, lower, upper, alpha=0.15)
        ax.set_ylabel(name)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

    if len(ordered_names) == 4:
        for ax in axes_list[-2:]:
            ax.set_xlabel("Episode")
    else:
        axes_list[-1].set_xlabel("Episode")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def generate_run_plots(run_dir: str, smooth_window: int, normalize: bool) -> Tuple[bool, bool]:
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")

    algo_name, rm_phi = _load_run_context(run_dir)
    reward_title = f"Rewards (algo={algo_name})"
    if rm_phi:
        reward_title = f"{reward_title}, reward_model_phi={rm_phi}"
    social_title = f"Social Metrics (algo={algo_name})"

    records = _load_metrics(metrics_path)
    rewards: Dict[str, List[Tuple[int, float]]] = {key: [] for key in REWARD_KEYS}
    social: Dict[str, List[Tuple[int, float]]] = {}

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for episode, record in _iter_records_with_episode(records):
        for key in REWARD_KEYS:
            value = record.get(key)
            if isinstance(value, (int, float)):
                rewards[key].append((episode, float(value)))
        social_metrics = _normalize_social_metrics(record.get("social_metrics"))
        for name, value in social_metrics.items():
            social.setdefault(name, []).append((episode, value))

    rewards_path = os.path.join(plots_dir, "rewards.png")
    social_path = os.path.join(plots_dir, "social_metrics.png")
    rewards_series = {k: v for k, v in rewards.items() if v}
    social_series = {k: social.get(k, []) for k in SOCIAL_ORDER if social.get(k)}
    if not social_series:
        social_series = {k: v for k, v in social.items() if v}
    if normalize:
        rewards_series = _normalize_series(rewards_series)
        social_series = _normalize_series(social_series)
        reward_title = f"{reward_title} (normalized)"
        social_title = f"{social_title} (normalized)"
        reward_ylabel = "Normalized value"
        social_ylabel = "Normalized value"
    else:
        reward_ylabel = "Reward"
        social_ylabel = "Metric"
    rewards_plotted = _plot_series(
        rewards_series,
        title=reward_title,
        ylabel=reward_ylabel,
        output_path=rewards_path,
        smooth_window=smooth_window,
    )
    if normalize:
        social_plotted = _plot_series(
            social_series,
            title=social_title,
            ylabel=social_ylabel,
            output_path=social_path,
            smooth_window=smooth_window,
        )
    else:
        social_plotted = _plot_social_subplots(
            social_series,
            title=social_title,
            output_path=social_path,
            smooth_window=smooth_window,
        )
    return rewards_plotted, social_plotted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reward and social metric plots from a run folder."
    )
    parser.add_argument(
        "run_dir",
        help="Path to a run folder containing metrics.json or a metrics.json file.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window (episodes). Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each metric series to [0, 1] and plot social metrics on one graph.",
    )
    args = parser.parse_args()

    path = args.run_dir
    if os.path.isfile(path):
        run_dir = os.path.dirname(path)
    else:
        run_dir = path

    rewards_plotted, social_plotted = generate_run_plots(
        run_dir,
        smooth_window=args.smooth,
        normalize=args.normalize,
    )
    if rewards_plotted:
        print(f"Saved rewards plot to {os.path.join(run_dir, 'plots', 'rewards.png')}")
    else:
        print("No reward metrics found to plot.")
    if social_plotted:
        print(f"Saved social metrics plot to {os.path.join(run_dir, 'plots', 'social_metrics.png')}")
    else:
        print("No social metrics found to plot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
