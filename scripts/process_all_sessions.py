"""
Process all experiment sessions:
1. Average social metrics across all runs (one subplot per metric)
2. Each social metric showing all runs individually (not averaged)
3. Average predicted reward per condition for all runs (averaged across agents per run)
4. Average predicted reward per action for all runs (averaged across agents per run)
"""

import os
import sys
import json
import csv
import math
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from plot_multiple_runs import plot_multiple_runs, PUBLICATION_COLORS, _format_label

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

SOCIAL_ORDER = ("efficiency", "equality", "sustainability", "peace")

# Define sessions and their run directories
SESSIONS = {
    "input aggregation - efficiency x peace": [
        "logs/input aggregation - efficiency x peace/20260109-040340-mappo-map=medium-agents=5-rm=input_aggregation-seed=1497856192",
        "logs/input aggregation - efficiency x peace/20260109-045125-mappo-map=medium-agents=5-rm=input_aggregation-seed=1185191064",
        "logs/input aggregation - efficiency x peace/20260109-053859-mappo-map=medium-agents=5-rm=input_aggregation-seed=457152814",
        "logs/input aggregation - efficiency x peace/20260109-062647-mappo-map=medium-agents=5-rm=input_aggregation-seed=791109345",
        "logs/input aggregation - efficiency x peace/20260109-071425-mappo-map=medium-agents=5-rm=input_aggregation-seed=1525681617",
    ],
    "narrow view - efficiency": [
        "logs/narrow view - efficiency/20260109-093404-mappo-map=medium-agents=5-rm=narrow_view-seed=1469728708",
        "logs/narrow view - efficiency/20260109-102016-mappo-map=medium-agents=5-rm=narrow_view-seed=1691341236",
        "logs/narrow view - efficiency/20260109-110535-mappo-map=medium-agents=5-rm=narrow_view-seed=610808009",
        "logs/narrow view - efficiency/20260109-115113-mappo-map=medium-agents=5-rm=narrow_view-seed=1176278676",
        "logs/narrow view - efficiency/20260109-123713-mappo-map=medium-agents=5-rm=narrow_view-seed=936768364",
    ],
    "narrow view - efficiency x peace": [
        "logs/narrow view - efficiency x peace/20260109-000553-mappo-map=medium-agents=5-rm=narrow_view-seed=1565503637",
        "logs/narrow view - efficiency x peace/20260109-005300-mappo-map=medium-agents=5-rm=narrow_view-seed=901297705",
        "logs/narrow view - efficiency x peace/20260109-014028-mappo-map=medium-agents=5-rm=narrow_view-seed=1733104918",
        "logs/narrow view - efficiency x peace/20260109-022810-mappo-map=medium-agents=5-rm=narrow_view-seed=1349720220",
        "logs/narrow view - efficiency x peace/20260109-031545-mappo-map=medium-agents=5-rm=narrow_view-seed=1744625372",
    ],
    "input aggregation - efficiency": [
        "logs/input aggregation - efficiency/20260109-132350-mappo-map=medium-agents=5-rm=input_aggregation-seed=792766690",
        "logs/input aggregation - efficiency/20260109-141043-mappo-map=medium-agents=5-rm=input_aggregation-seed=1977920447",
        "logs/input aggregation - efficiency/20260109-145843-mappo-map=medium-agents=5-rm=input_aggregation-seed=525605494",
        "logs/input aggregation - efficiency/20260109-154749-mappo-map=medium-agents=5-rm=input_aggregation-seed=711072676",
        "logs/input aggregation - efficiency/20260109-163637-mappo-map=medium-agents=5-rm=input_aggregation-seed=278575429",
    ],
}


def _load_metrics(metrics_path: str) -> List[Dict[str, Any]]:
    """Load metrics from a JSONL file."""
    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_social_metrics_series(
    records: List[Dict[str, Any]]
) -> Dict[str, List[Tuple[int, float]]]:
    """Extract social metrics series from records."""
    series = {name: [] for name in SOCIAL_ORDER}
    
    for record in records:
        episode = record.get("episode")
        if episode is None:
            continue
        if isinstance(episode, float):
            episode = int(episode)
        
        social_metrics = record.get("social_metrics")
        if isinstance(social_metrics, dict):
            for name in SOCIAL_ORDER:
                value = social_metrics.get(name)
                if isinstance(value, (int, float)):
                    series[name].append((episode, float(value)))
    
    # Sort by episode
    for name in series:
        series[name] = sorted(series[name], key=lambda x: x[0])
    
    return series


def _load_agent_csv(agent_csv_path: str) -> List[Dict[str, Any]]:
    """Load agent episode CSV file."""
    rows = []
    if not os.path.isfile(agent_csv_path):
        return rows
    try:
        with open(agent_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except (csv.Error, IOError):
        return rows
    return rows


def _parse_csv_value(value: str, field: str) -> Any:
    """Parse CSV value based on field type."""
    if not value or value == "":
        return None
    if field in ("episode", "step", "action", "nearby_apples"):
        try:
            return int(value)
        except ValueError:
            return None
    if field in ("reward", "predicted_reward"):
        try:
            return float(value)
        except ValueError:
            return None
    if field in ("apple_eaten", "ate_last_apple_in_cluster"):
        value_lower = value.lower().strip()
        if value_lower in ("true", "1", "yes"):
            return True
        if value_lower in ("false", "0", "no", ""):
            return False
        return None
    return value


def _extract_predicted_reward_by_condition_for_run(
    run_dir: str
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract predicted reward series by condition for a run.
    Averages across all agents in the run.
    
    Returns dict with condition names as keys, values as list of (episode, avg_reward) tuples.
    """
    extended_info_dir = os.path.join(run_dir, "extended_info")
    if not os.path.isdir(extended_info_dir):
        return {}
    
    # Find all agent CSV files
    agent_files = []
    for filename in os.listdir(extended_info_dir):
        if filename.startswith("agent_") and filename.endswith("_episodes.csv"):
            agent_files.append(os.path.join(extended_info_dir, filename))
    
    if not agent_files:
        return {}
    
    # Aggregate data across all agents
    # Structure: {episode: {condition: [list of predicted rewards]}}
    episodes_data: Dict[int, Dict[str, List[float]]] = {}
    
    for agent_csv_path in agent_files:
        rows = _load_agent_csv(agent_csv_path)
        
        for row in rows:
            episode = _parse_csv_value(row.get("episode", ""), "episode")
            if episode is None:
                continue
            
            predicted_reward = _parse_csv_value(row.get("predicted_reward", ""), "predicted_reward")
            if predicted_reward is None:
                continue
            
            apple_eaten = _parse_csv_value(row.get("apple_eaten", ""), "apple_eaten")
            nearby_apples = _parse_csv_value(row.get("nearby_apples", ""), "nearby_apples")
            
            if episode not in episodes_data:
                episodes_data[episode] = {
                    "no_apple_eaten": [],
                    "zero_apples_nearby": [],
                    "four_plus_apples_nearby": [],
                }
            
            if apple_eaten is False:
                episodes_data[episode]["no_apple_eaten"].append(predicted_reward)
            
            if nearby_apples == 0:
                episodes_data[episode]["zero_apples_nearby"].append(predicted_reward)
            
            if nearby_apples is not None and nearby_apples >= 4:
                episodes_data[episode]["four_plus_apples_nearby"].append(predicted_reward)
    
    # Compute average per episode per condition
    series = {
        "no_apple_eaten": [],
        "zero_apples_nearby": [],
        "four_plus_apples_nearby": [],
    }
    
    for episode in sorted(episodes_data.keys()):
        for condition in series.keys():
            values = episodes_data[episode][condition]
            if values:
                mean = sum(values) / len(values)
                series[condition].append((episode, mean))
    
    return series


def _extract_predicted_reward_by_action_for_run(
    run_dir: str
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract predicted reward series by action for a run.
    Averages across all agents in the run.
    
    Returns dict with action names as keys, values as list of (episode, avg_reward) tuples.
    """
    extended_info_dir = os.path.join(run_dir, "extended_info")
    if not os.path.isdir(extended_info_dir):
        return {}
    
    # Find all agent CSV files
    agent_files = []
    for filename in os.listdir(extended_info_dir):
        if filename.startswith("agent_") and filename.endswith("_episodes.csv"):
            agent_files.append(os.path.join(extended_info_dir, filename))
    
    if not agent_files:
        return {}
    
    # Aggregate data across all agents
    # Structure: {episode: {action: [list of predicted rewards]}}
    episodes_data: Dict[int, Dict[int, List[float]]] = {}
    
    for agent_csv_path in agent_files:
        rows = _load_agent_csv(agent_csv_path)
        
        for row in rows:
            episode = _parse_csv_value(row.get("episode", ""), "episode")
            if episode is None:
                continue
            
            predicted_reward = _parse_csv_value(row.get("predicted_reward", ""), "predicted_reward")
            if predicted_reward is None:
                continue
            
            action = _parse_csv_value(row.get("action", ""), "action")
            if action is None or action not in (0, 1, 2, 3):
                continue
            
            if episode not in episodes_data:
                episodes_data[episode] = {0: [], 1: [], 2: [], 3: []}
            
            episodes_data[episode][action].append(predicted_reward)
    
    # Compute average per episode per action
    action_names = {0: "move_left", 1: "move_right", 2: "move_up", 3: "move_down"}
    series = {name: [] for name in action_names.values()}
    
    for episode in sorted(episodes_data.keys()):
        for action, name in action_names.items():
            values = episodes_data[episode][action]
            if values:
                mean = sum(values) / len(values)
                series[name].append((episode, mean))
    
    return series


def _plot_individual_social_metric_all_runs(
    all_runs_series: Dict[str, Dict[str, List[Tuple[int, float]]]],
    metric_name: str,
    title: str,
    output_path: str,
) -> bool:
    """
    Plot a single social metric showing all runs individually (not averaged).
    
    Args:
        all_runs_series: {run_name: {metric_name: [(episode, value), ...]}}
        metric_name: which metric to plot
        title: plot title
        output_path: where to save
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    color_cycle = iter(PUBLICATION_COLORS)
    plotted = False
    
    for run_name, series_dict in all_runs_series.items():
        if metric_name not in series_dict:
            continue
        points = series_dict[metric_name]
        if not points:
            continue
        
        episodes = [ep for ep, _ in points]
        values = [val for _, val in points]
        
        color = next(color_cycle)
        # Extract seed from run name for shorter label
        label = run_name.split("seed=")[-1] if "seed=" in run_name else run_name[-10:]
        ax.plot(episodes, values, label=f"Run {label}", color=color, linewidth=1.5, alpha=0.8)
        plotted = True
    
    if not plotted:
        plt.close()
        return False
    
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel(metric_name.capitalize(), fontweight='normal')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def _plot_predicted_reward_all_runs(
    all_runs_series: Dict[str, Dict[str, List[Tuple[int, float]]]],
    categories: List[str],
    category_labels: Dict[str, str],
    title: str,
    output_path: str,
    show_std: bool = True,
) -> bool:
    """
    Plot predicted reward for all runs, one line per category (condition or action).
    Each line represents the average across all runs.
    
    Args:
        all_runs_series: {run_name: {category: [(episode, value), ...]}}
        categories: list of category names to plot
        category_labels: display labels for categories
        title: plot title
        output_path: where to save
        show_std: whether to show standard deviation shading
    """
    # First, align all runs and compute mean per category
    # Get all episodes across all runs
    all_episodes = set()
    for run_series in all_runs_series.values():
        for cat in categories:
            if cat in run_series:
                for ep, _ in run_series[cat]:
                    all_episodes.add(ep)
    
    if not all_episodes:
        return False
    
    episodes = sorted(all_episodes)
    
    # For each category, compute mean across runs at each episode
    category_means: Dict[str, List[float]] = {cat: [] for cat in categories}
    category_stds: Dict[str, List[float]] = {cat: [] for cat in categories}
    
    for ep in episodes:
        for cat in categories:
            values_at_ep = []
            for run_series in all_runs_series.values():
                if cat in run_series:
                    # Find value at this episode (or interpolate)
                    series = run_series[cat]
                    series_dict = dict(series)
                    if ep in series_dict:
                        values_at_ep.append(series_dict[ep])
            
            if values_at_ep:
                category_means[cat].append(np.mean(values_at_ep))
                category_stds[cat].append(np.std(values_at_ep))
            else:
                category_means[cat].append(np.nan)
                category_stds[cat].append(np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    color_cycle = iter(PUBLICATION_COLORS)
    plotted = False
    
    for cat in categories:
        means = category_means[cat]
        stds = category_stds[cat]
        
        if all(np.isnan(m) for m in means):
            continue
        
        color = next(color_cycle)
        label = category_labels.get(cat, cat)
        
        ax.plot(episodes, means, label=label, color=color, linewidth=2.0, zorder=2)
        
        # Add shaded std region if requested
        if show_std:
            lower = [m - s if not np.isnan(m) else np.nan for m, s in zip(means, stds)]
            upper = [m + s if not np.isnan(m) else np.nan for m, s in zip(means, stds)]
            ax.fill_between(episodes, lower, upper, alpha=0.25, color=color, zorder=1)
        plotted = True
    
    if not plotted:
        plt.close()
        return False
    
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel("Average Predicted Reward", fontweight='normal')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def process_session(session_name: str, run_dirs: list, base_dir: str):
    """Process a single session."""
    print(f"\n{'='*80}")
    print(f"Processing session: {session_name}")
    print(f"{'='*80}")
    
    # Convert relative paths to absolute
    abs_run_dirs = [os.path.join(base_dir, d) for d in run_dirs]
    
    # Create output directory
    session_output_dir = os.path.join(base_dir, "logs", session_name, "plots_averaged")
    os.makedirs(session_output_dir, exist_ok=True)
    
    # 1. Run plot_multiple_runs for averaged metrics
    print(f"\n[1] Generating averaged plots -> {session_output_dir}")
    try:
        rewards_plotted, social_plotted, agent_pred_plotted = plot_multiple_runs(
            abs_run_dirs,
            output_dir=session_output_dir,
            smooth_window=1,
            normalize=False,
        )
        
        if rewards_plotted:
            print(f"  [OK] Saved averaged rewards plot")
        if social_plotted:
            print(f"  [OK] Saved averaged social metrics plot")
        if agent_pred_plotted:
            print(f"  [OK] Saved normalized per-agent predicted rewards plot")
    except Exception as e:
        print(f"  [ERR] Error in averaged plots: {e}")
    
    # Load data from all runs
    all_runs_social: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    all_runs_condition: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    all_runs_action: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    
    for run_dir in abs_run_dirs:
        run_name = os.path.basename(run_dir)
        
        # Load social metrics
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if os.path.isfile(metrics_path):
            records = _load_metrics(metrics_path)
            social_series = _extract_social_metrics_series(records)
            all_runs_social[run_name] = social_series
        
        # Load predicted rewards by condition
        condition_series = _extract_predicted_reward_by_condition_for_run(run_dir)
        if condition_series:
            all_runs_condition[run_name] = condition_series
        
        # Load predicted rewards by action
        action_series = _extract_predicted_reward_by_action_for_run(run_dir)
        if action_series:
            all_runs_action[run_name] = action_series
    
    # 2. Plot each social metric showing all runs individually
    print(f"\n[2] Generating individual social metric plots (all runs per graph)")
    for metric_name in SOCIAL_ORDER:
        output_path = os.path.join(session_output_dir, f"social_{metric_name}_all_runs.png")
        title = f"{metric_name.capitalize()} - All Runs ({_format_label(session_name)})"
        
        plotted = _plot_individual_social_metric_all_runs(
            all_runs_social, metric_name, title, output_path
        )
        
        if plotted:
            print(f"  [OK] Saved {metric_name} all runs plot")
        else:
            print(f"  [--] No data for {metric_name}")
    
    # 3. Plot predicted reward by condition (averaged across all runs)
    print(f"\n[3] Generating predicted reward by condition plots")
    condition_categories = ["no_apple_eaten", "zero_apples_nearby", "four_plus_apples_nearby"]
    condition_labels = {
        "no_apple_eaten": "No apple eaten",
        "zero_apples_nearby": "0 apples nearby",
        "four_plus_apples_nearby": "4+ apples nearby",
    }
    
    # With std
    output_path = os.path.join(session_output_dir, "predicted_reward_by_condition_with_std.png")
    title = f"Predicted Reward by Condition ({_format_label(session_name)})"
    plotted = _plot_predicted_reward_all_runs(
        all_runs_condition, condition_categories, condition_labels, title, output_path, show_std=True
    )
    if plotted:
        print(f"  [OK] Saved predicted reward by condition (with std)")
    
    # Without std
    output_path = os.path.join(session_output_dir, "predicted_reward_by_condition_no_std.png")
    plotted = _plot_predicted_reward_all_runs(
        all_runs_condition, condition_categories, condition_labels, title, output_path, show_std=False
    )
    if plotted:
        print(f"  [OK] Saved predicted reward by condition (no std)")
    else:
        print(f"  [--] No condition data found")
    
    # 4. Plot predicted reward by action (averaged across all runs)
    print(f"\n[4] Generating predicted reward by action plots")
    action_categories = ["move_left", "move_right", "move_up", "move_down"]
    action_labels = {
        "move_left": "Move Left",
        "move_right": "Move Right",
        "move_up": "Move Up",
        "move_down": "Move Down",
    }
    
    # With std
    output_path = os.path.join(session_output_dir, "predicted_reward_by_action_with_std.png")
    title = f"Predicted Reward by Action ({_format_label(session_name)})"
    plotted = _plot_predicted_reward_all_runs(
        all_runs_action, action_categories, action_labels, title, output_path, show_std=True
    )
    if plotted:
        print(f"  [OK] Saved predicted reward by action (with std)")
    
    # Without std
    output_path = os.path.join(session_output_dir, "predicted_reward_by_action_no_std.png")
    plotted = _plot_predicted_reward_all_runs(
        all_runs_action, action_categories, action_labels, title, output_path, show_std=False
    )
    if plotted:
        print(f"  [OK] Saved predicted reward by action (no std)")
    else:
        print(f"  [--] No action data found")


def main():
    # Get the base directory (project root)
    base_dir = os.path.dirname(script_dir)
    
    print("="*80)
    print("Processing all experiment sessions")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Number of sessions: {len(SESSIONS)}")
    
    for session_name, run_dirs in SESSIONS.items():
        process_session(session_name, run_dirs, base_dir)
    
    print("\n" + "="*80)
    print("All sessions processed!")
    print("="*80)
    
    # Print summary of output locations
    print("\nOutput locations:")
    for session_name in SESSIONS.keys():
        session_plots = os.path.join(base_dir, "logs", session_name, "plots_averaged")
        print(f"  - {session_name}: {session_plots}")
        print(f"      - rewards_averaged.png")
        print(f"      - social_metrics_averaged.png")
        print(f"      - social_<metric>_all_runs.png (per metric)")
        print(f"      - predicted_reward_by_condition_with_std.png / _no_std.png")
        print(f"      - predicted_reward_by_action_with_std.png / _no_std.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
