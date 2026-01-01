import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    """Load metrics from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_metric_series(
    records: List[Dict[str, Any]], metric_path: str
) -> List[Tuple[int, float]]:
    """
    Extract a metric series from records.
    metric_path can be a simple key like "reward_mean" or a nested path like "social_metrics.efficiency".
    """
    series: List[Tuple[int, float]] = []
    for record in records:
        episode = record.get("episode")
        if episode is None:
            continue
        if isinstance(episode, float):
            episode = int(episode)
        
        # Handle nested paths
        value = record
        for key in metric_path.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        
        if isinstance(value, (int, float)):
            series.append((episode, float(value)))
    return sorted(series, key=lambda x: x[0])


def _align_series(
    all_series: List[List[Tuple[int, float]]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align multiple series to the same episode indices and compute mean and std.
    Returns: (episodes, means, stds)
    """
    if not all_series:
        return np.array([]), np.array([]), np.array([])
    
    # Get all unique episode numbers
    all_episodes = set()
    for series in all_series:
        all_episodes.update(ep for ep, _ in series)
    episodes = sorted(all_episodes)
    
    if not episodes:
        return np.array([]), np.array([]), np.ndarray([])
    
    # Interpolate/extract values for each series at each episode
    values_matrix = []
    for series in all_series:
        series_dict = dict(series)
        values = []
        for ep in episodes:
            if ep in series_dict:
                values.append(series_dict[ep])
            else:
                # Use linear interpolation if episode is missing
                # Find nearest episodes
                series_eps = [e for e, _ in series]
                if not series_eps:
                    values.append(np.nan)
                elif ep < series_eps[0]:
                    values.append(series_dict[series_eps[0]])
                elif ep > series_eps[-1]:
                    values.append(series_dict[series_eps[-1]])
                else:
                    # Interpolate - find the two closest episodes
                    for i in range(len(series) - 1):
                        ep1, val1 = series[i]
                        ep2, val2 = series[i + 1]
                        if ep1 <= ep <= ep2:
                            if ep2 == ep1:
                                values.append(val1)
                            else:
                                t = (ep - ep1) / (ep2 - ep1)
                                values.append(val1 + t * (val2 - val1))
                            break
                    else:
                        values.append(np.nan)
        values_matrix.append(values)
    
    # Compute mean and std across runs
    values_array = np.array(values_matrix)
    means = np.nanmean(values_array, axis=0)
    stds = np.nanstd(values_array, axis=0, ddof=1)  # Sample std
    
    return np.array(episodes), means, stds


def _plot_averaged_series(
    metric_name: str,
    episodes: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    ylabel: str,
    output_path: str,
    smooth_window: int = 1,
) -> bool:
    """Plot a single averaged metric series with standard deviation."""
    if len(episodes) == 0:
        return False
    
    # Apply smoothing if requested
    if smooth_window > 1 and len(episodes) > 1:
        smoothed_means = []
        smoothed_stds = []
        for i in range(len(episodes)):
            start = max(0, i - smooth_window + 1)
            end = min(len(episodes), i + 1)
            window_means = means[start:end]
            window_stds = stds[start:end]
            smoothed_means.append(np.nanmean(window_means))
            # For smoothed std, we compute std of the window
            smoothed_stds.append(np.nanstd(window_means) if len(window_means) > 1 else 0.0)
        means = np.array(smoothed_means)
        stds = np.array(smoothed_stds)
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        episodes,
        means,
        yerr=stds,
        label=metric_name,
        linewidth=2,
        capsize=3,
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.7,
    )
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def _thin_error_bars(episodes: np.ndarray, means: np.ndarray, stds: np.ndarray, step: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin out error bars to show every Nth point."""
    if len(episodes) <= step:
        return episodes, means, stds
    indices = np.arange(0, len(episodes), step)
    return episodes[indices], means[indices], stds[indices]


def _plot_multiple_averaged_series(
    series_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    ylabel: str,
    output_path: str,
    smooth_window: int = 1,
    error_bar_step: int = 10,
    line_color: str = None,
    error_bar_color: str = None,
) -> bool:
    """Plot multiple averaged metric series on the same plot."""
    if not series_dict:
        return False
    
    plt.figure(figsize=(12, 6))
    
    for metric_name, (episodes, means, stds) in series_dict.items():
        if len(episodes) == 0:
            continue
        
        # Apply smoothing if requested
        if smooth_window > 1 and len(episodes) > 1:
            smoothed_means = []
            smoothed_stds = []
            for i in range(len(episodes)):
                start = max(0, i - smooth_window + 1)
                end = min(len(episodes), i + 1)
                window_means = means[start:end]
                smoothed_means.append(np.nanmean(window_means))
                smoothed_stds.append(np.nanstd(window_means) if len(window_means) > 1 else 0.0)
            means = np.array(smoothed_means)
            stds = np.array(smoothed_stds)
        
        # Plot the line (use specified color or default)
        plot_kwargs = {"label": metric_name, "linewidth": 2}
        if line_color:
            plot_kwargs["color"] = line_color
        plt.plot(episodes, means, **plot_kwargs)
        
        # For reward metrics, use shaded area; for social metrics, use error bars
        if error_bar_color == 'black':
            # Social metrics: use error bars with black color
            ep_thin, means_thin, stds_thin = _thin_error_bars(episodes, means, stds, error_bar_step)
            plt.errorbar(
                ep_thin,
                means_thin,
                yerr=stds_thin,
                fmt='none',  # Don't draw line or markers, just error bars
                color='black',
                capsize=3,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.7,
            )
        else:
            # Reward metrics: use shaded area
            plt.fill_between(
                episodes,
                means - stds,
                means + stds,
                alpha=0.2,
            )
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def _plot_social_subplots_averaged(
    series_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    output_path: str,
    smooth_window: int = 1,
    error_bar_step: int = 10,
) -> bool:
    """Plot social metrics as subplots with averaged values (black line with red error bars)."""
    ordered_names = [name for name in SOCIAL_ORDER if name in series_dict]
    if not ordered_names:
        ordered_names = [name for name in series_dict.keys() if series_dict[name][0].size > 0]
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
        episodes, means, stds = series_dict[name]
        if len(episodes) == 0:
            continue
        
        # Apply smoothing if requested
        if smooth_window > 1 and len(episodes) > 1:
            smoothed_means = []
            smoothed_stds = []
            for i in range(len(episodes)):
                start = max(0, i - smooth_window + 1)
                end = min(len(episodes), i + 1)
                window_means = means[start:end]
                smoothed_means.append(np.nanmean(window_means))
                smoothed_stds.append(np.nanstd(window_means) if len(window_means) > 1 else 0.0)
            means = np.array(smoothed_means)
            stds = np.array(smoothed_stds)
        
        # Plot line (default blue color)
        ax.plot(episodes, means, label=name, linewidth=2)
        
        # Plot black error bars (thinned)
        ep_thin, means_thin, stds_thin = _thin_error_bars(episodes, means, stds, error_bar_step)
        ax.errorbar(
            ep_thin,
            means_thin,
            yerr=stds_thin,
            fmt='none',  # Don't draw line or markers, just error bars
            color='black',
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
        )
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


def _load_run_context(run_dir: str) -> Tuple[str, str]:
    """Load algorithm name and reward model phi from config."""
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


def plot_multiple_runs(
    run_dirs: List[str],
    output_dir: str,
    smooth_window: int = 1,
    normalize: bool = False,
) -> Tuple[bool, bool]:
    """
    Plot averaged metrics across multiple runs with standard deviation.
    
    Args:
        run_dirs: List of run directory paths
        output_dir: Directory to save plots
        smooth_window: Moving average window size
        normalize: Whether to normalize metrics to [0, 1]
    
    Returns:
        Tuple of (rewards_plotted, social_plotted)
    """
    if not run_dirs:
        raise ValueError("No run directories provided")
    
    # Load metrics from all runs
    all_reward_series: Dict[str, List[List[Tuple[int, float]]]] = {
        key: [] for key in REWARD_KEYS
    }
    all_social_series: Dict[str, List[List[Tuple[int, float]]]] = {}
    
    algo_name = "unknown"
    rm_phi = ""
    
    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"Warning: metrics.json not found in {run_dir}, skipping")
            continue
        
        # Load context from first valid run
        if algo_name == "unknown":
            algo_name, rm_phi = _load_run_context(run_dir)
        
        records = _load_metrics(metrics_path)
        
        # Extract reward metrics
        for key in REWARD_KEYS:
            series = _extract_metric_series(records, key)
            if series:
                all_reward_series[key].append(series)
        
        # Extract social metrics
        for name in SOCIAL_ORDER:
            series = _extract_metric_series(records, f"social_metrics.{name}")
            if series:
                if name not in all_social_series:
                    all_social_series[name] = []
                all_social_series[name].append(series)
    
    # Compute averages and stds
    rewards_averaged: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, series_list in all_reward_series.items():
        if series_list:
            episodes, means, stds = _align_series(series_list)
            if len(episodes) > 0:
                rewards_averaged[key] = (episodes, means, stds)
    
    social_averaged: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, series_list in all_social_series.items():
        if series_list:
            episodes, means, stds = _align_series(series_list)
            if len(episodes) > 0:
                social_averaged[name] = (episodes, means, stds)
    
    # Normalize if requested
    if normalize:
        # Normalize each metric to [0, 1]
        for key in rewards_averaged:
            episodes, means, stds = rewards_averaged[key]
            min_val = np.nanmin(means)
            max_val = np.nanmax(means)
            if max_val > min_val:
                means = (means - min_val) / (max_val - min_val)
                stds = stds / (max_val - min_val)
            rewards_averaged[key] = (episodes, means, stds)
        
        for name in social_averaged:
            episodes, means, stds = social_averaged[name]
            min_val = np.nanmin(means)
            max_val = np.nanmax(means)
            if max_val > min_val:
                means = (means - min_val) / (max_val - min_val)
                stds = stds / (max_val - min_val)
            social_averaged[name] = (episodes, means, stds)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate titles
    reward_title = f"Average Rewards Across {len(run_dirs)} Runs (algo={algo_name})"
    if rm_phi:
        reward_title = f"{reward_title}, reward_model_phi={rm_phi}"
    social_title = f"Average Social Metrics Across {len(run_dirs)} Runs (algo={algo_name})"
    
    if normalize:
        reward_title = f"{reward_title} (normalized)"
        social_title = f"{social_title} (normalized)"
        reward_ylabel = "Normalized value"
        social_ylabel = "Normalized value"
    else:
        reward_ylabel = "Reward"
        social_ylabel = "Metric"
    
    # Plot rewards
    rewards_path = os.path.join(output_dir, "rewards_averaged.png")
    rewards_plotted = _plot_multiple_averaged_series(
        rewards_averaged,
        title=reward_title,
        ylabel=reward_ylabel,
        output_path=rewards_path,
        smooth_window=smooth_window,
        error_bar_step=10,
    )
    
    # Plot social metrics
    social_path = os.path.join(output_dir, "social_metrics_averaged.png")
    if normalize:
        social_plotted = _plot_multiple_averaged_series(
            social_averaged,
            title=social_title,
            ylabel=social_ylabel,
            output_path=social_path,
            smooth_window=smooth_window,
            error_bar_step=10,
            line_color=None,  # Use default blue color
            error_bar_color='black',
        )
    else:
        # Filter to only ordered social metrics
        ordered_social = {
            name: social_averaged[name]
            for name in SOCIAL_ORDER
            if name in social_averaged
        }
        if not ordered_social:
            ordered_social = social_averaged
        social_plotted = _plot_social_subplots_averaged(
            ordered_social,
            title=social_title,
            output_path=social_path,
            smooth_window=smooth_window,
            error_bar_step=10,
        )
    
    return rewards_plotted, social_plotted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate averaged plots with standard deviation across multiple runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Paths to run directories containing metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots_averaged",
        help="Output directory for plots (default: plots_averaged)",
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
    
    rewards_plotted, social_plotted = plot_multiple_runs(
        args.run_dirs,
        output_dir=args.output_dir,
        smooth_window=args.smooth,
        normalize=args.normalize,
    )
    
    if rewards_plotted:
        print(f"Saved averaged rewards plot to {os.path.join(args.output_dir, 'rewards_averaged.png')}")
    else:
        print("No reward metrics found to plot.")
    
    if social_plotted:
        print(f"Saved averaged social metrics plot to {os.path.join(args.output_dir, 'social_metrics_averaged.png')}")
    else:
        print("No social metrics found to plot.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

