"""
Process all experiment sessions:
1. Average social metrics across all runs (one subplot per metric)
2. Each social metric showing all runs individually (not averaged)
3. Average predicted reward per condition for all runs (averaged across agents per run)
4. Average predicted reward per action for all runs (averaged across agents per run)
"""

import argparse
import os
import sys
import json
import csv
import math
from typing import Any, Dict, List, Tuple, Optional

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


# =============================================================================
# Session Metadata Parsing
# =============================================================================

def parse_session_name(session_name: str) -> Tuple[str, str]:
    """
    Parse session name into (approach, social_target).
    
    Convention: "{approach} - {social_target}"
    Example: "narrow view - efficiency x peace" -> ("narrow_view", "efficiency_x_peace")
    """
    parts = session_name.split(" - ", 1)
    if len(parts) != 2:
        # Fallback: use whole name as approach, empty target
        return session_name.replace(" ", "_"), "unknown"
    
    approach = parts[0].strip().replace(" ", "_")
    social_target = parts[1].strip().replace(" ", "_")
    return approach, social_target


def parse_all_sessions() -> Dict[str, Dict[str, Any]]:
    """
    Parse all sessions into metadata structure.
    
    Returns:
        {session_name: {"approach": str, "social_target": str, "runs": list}}
    """
    metadata = {}
    for session_name, runs in SESSIONS.items():
        approach, social_target = parse_session_name(session_name)
        metadata[session_name] = {
            "approach": approach,
            "social_target": social_target,
            "runs": runs,
        }
    return metadata


def get_unique_approaches() -> List[str]:
    """Get list of unique approaches across all sessions."""
    approaches = set()
    for session_name in SESSIONS.keys():
        approach, _ = parse_session_name(session_name)
        approaches.add(approach)
    return sorted(approaches)


def get_unique_social_targets() -> List[str]:
    """Get list of unique social targets across all sessions."""
    targets = set()
    for session_name in SESSIONS.keys():
        _, target = parse_session_name(session_name)
        targets.add(target)
    return sorted(targets)


def group_sessions_by_approach() -> Dict[str, List[str]]:
    """
    Group session names by approach.
    
    Returns:
        {approach: [session_name1, session_name2, ...]}
    """
    groups: Dict[str, List[str]] = {}
    for session_name in SESSIONS.keys():
        approach, _ = parse_session_name(session_name)
        if approach not in groups:
            groups[approach] = []
        groups[approach].append(session_name)
    return groups


def group_sessions_by_target() -> Dict[str, List[str]]:
    """
    Group session names by social target.
    
    Returns:
        {social_target: [session_name1, session_name2, ...]}
    """
    groups: Dict[str, List[str]] = {}
    for session_name in SESSIONS.keys():
        _, target = parse_session_name(session_name)
        if target not in groups:
            groups[target] = []
        groups[target].append(session_name)
    return groups


def get_session_by_approach_and_target(approach: str, target: str) -> Optional[str]:
    """Find session name matching given approach and target."""
    for session_name in SESSIONS.keys():
        sess_approach, sess_target = parse_session_name(session_name)
        if sess_approach == approach and sess_target == target:
            return session_name
    return None


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


# =============================================================================
# Cross-Session Comparison Plots
# =============================================================================

def _load_session_averaged_social_metrics(
    session_name: str, base_dir: str
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load and average social metrics across all runs in a session.
    
    Returns:
        {metric_name: [(episode, avg_value), ...]}
    """
    means, _ = _load_session_social_metrics_with_std(session_name, base_dir)
    return means


def _load_session_social_metrics_with_std(
    session_name: str, base_dir: str
) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[str, List[Tuple[int, float]]]]:
    """
    Load social metrics across all runs in a session, returning both mean and std.
    
    Returns:
        (means, stds) where each is {metric_name: [(episode, value), ...]}
    """
    run_dirs = SESSIONS.get(session_name, [])
    if not run_dirs:
        return {}, {}
    
    # Collect all series from all runs
    all_runs_series: Dict[str, Dict[int, List[float]]] = {name: {} for name in SOCIAL_ORDER}
    
    for run_path in run_dirs:
        abs_run_dir = os.path.join(base_dir, run_path)
        metrics_path = os.path.join(abs_run_dir, "metrics.jsonl")
        
        if not os.path.isfile(metrics_path):
            continue
        
        records = _load_metrics(metrics_path)
        series = _extract_social_metrics_series(records)
        
        for metric_name, points in series.items():
            for episode, value in points:
                if episode not in all_runs_series[metric_name]:
                    all_runs_series[metric_name][episode] = []
                all_runs_series[metric_name][episode].append(value)
    
    # Compute mean and std across runs
    means: Dict[str, List[Tuple[int, float]]] = {}
    stds: Dict[str, List[Tuple[int, float]]] = {}
    
    for metric_name, ep_values in all_runs_series.items():
        means[metric_name] = []
        stds[metric_name] = []
        for episode in sorted(ep_values.keys()):
            values = ep_values[episode]
            if values:
                means[metric_name].append((episode, np.mean(values)))
                stds[metric_name].append((episode, np.std(values)))
    
    return means, stds


def _load_session_averaged_rewards(
    session_name: str, base_dir: str
) -> List[Tuple[int, float]]:
    """
    Load and average episode rewards across all runs in a session.
    
    Returns:
        [(episode, avg_reward), ...]
    """
    run_dirs = SESSIONS.get(session_name, [])
    if not run_dirs:
        return []
    
    # Collect rewards from all runs
    ep_rewards: Dict[int, List[float]] = {}
    
    for run_path in run_dirs:
        abs_run_dir = os.path.join(base_dir, run_path)
        metrics_path = os.path.join(abs_run_dir, "metrics.jsonl")
        
        if not os.path.isfile(metrics_path):
            continue
        
        records = _load_metrics(metrics_path)
        
        for record in records:
            episode = record.get("episode")
            if episode is None:
                continue
            if isinstance(episode, float):
                episode = int(episode)
            
            reward = record.get("mean_reward")
            if reward is None:
                reward = record.get("episode_reward")
            if isinstance(reward, (int, float)):
                if episode not in ep_rewards:
                    ep_rewards[episode] = []
                ep_rewards[episode].append(float(reward))
    
    # Average across runs
    result = []
    for episode in sorted(ep_rewards.keys()):
        values = ep_rewards[episode]
        if values:
            result.append((episode, np.mean(values)))
    
    return result


def _load_session_averaged_predicted_rewards(
    session_name: str, base_dir: str, by_condition: bool = True
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load and average predicted rewards across all runs in a session.
    
    Args:
        by_condition: If True, group by condition; if False, group by action
        
    Returns:
        {category: [(episode, avg_value), ...]}
    """
    run_dirs = SESSIONS.get(session_name, [])
    if not run_dirs:
        return {}
    
    # Collect from all runs
    all_runs_data: Dict[str, Dict[int, List[float]]] = {}
    
    for run_path in run_dirs:
        abs_run_dir = os.path.join(base_dir, run_path)
        
        if by_condition:
            series = _extract_predicted_reward_by_condition_for_run(abs_run_dir)
        else:
            series = _extract_predicted_reward_by_action_for_run(abs_run_dir)
        
        for cat, points in series.items():
            if cat not in all_runs_data:
                all_runs_data[cat] = {}
            for episode, value in points:
                if episode not in all_runs_data[cat]:
                    all_runs_data[cat][episode] = []
                all_runs_data[cat][episode].append(value)
    
    # Average across runs
    averaged: Dict[str, List[Tuple[int, float]]] = {}
    for cat, ep_values in all_runs_data.items():
        averaged[cat] = []
        for episode in sorted(ep_values.keys()):
            values = ep_values[episode]
            if values:
                averaged[cat].append((episode, np.mean(values)))
    
    return averaged


def _format_approach_label(approach: str) -> str:
    """Format approach name for display."""
    return approach.replace("_", " ").title()


def _format_target_label(target: str) -> str:
    """Format social target name for display."""
    return target.replace("_", " ").title()


def plot_overlay_comparison(
    sessions_data: Dict[str, List[Tuple[int, float]]],
    title: str,
    ylabel: str,
    output_path: str,
    sessions_std: Optional[Dict[str, List[Tuple[int, float]]]] = None,
) -> bool:
    """
    Create overlay plot with multiple sessions on same axes.
    
    Args:
        sessions_data: {session_label: [(episode, value), ...]}
        title: plot title
        ylabel: y-axis label
        output_path: where to save
        sessions_std: optional {session_label: [(episode, std_value), ...]} for shading
    """
    if not sessions_data:
        return False
    
    fig, ax = plt.subplots(figsize=(8, 5))
    color_cycle = iter(PUBLICATION_COLORS)
    plotted = False
    
    for label, points in sessions_data.items():
        if not points:
            continue
        
        episodes = [ep for ep, _ in points]
        values = [val for _, val in points]
        
        color = next(color_cycle)
        ax.plot(episodes, values, label=label, color=color, linewidth=2.0, zorder=2)
        
        # Add std shading if provided
        if sessions_std and label in sessions_std:
            std_points = sessions_std[label]
            if std_points:
                std_dict = dict(std_points)
                lower = [v - std_dict.get(ep, 0) for ep, v in zip(episodes, values)]
                upper = [v + std_dict.get(ep, 0) for ep, v in zip(episodes, values)]
                ax.fill_between(episodes, lower, upper, alpha=0.2, color=color, zorder=1)
        
        plotted = True
    
    if not plotted:
        plt.close()
        return False
    
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel(ylabel, fontweight='normal')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def plot_grid_comparison(
    sessions_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
    metric_name: str,
    title: str,
    output_path: str,
) -> bool:
    """
    Create grid of subplots, one per session.
    
    Args:
        sessions_data: {session_name: {metric_name: [(episode, value), ...]}}
        metric_name: which metric to plot
        title: overall figure title
        output_path: where to save
    """
    sessions = list(sessions_data.keys())
    n_sessions = len(sessions)
    
    if n_sessions == 0:
        return False
    
    # Determine grid layout
    if n_sessions <= 2:
        nrows, ncols = 1, n_sessions
    elif n_sessions <= 4:
        nrows, ncols = 2, 2
    elif n_sessions <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = (n_sessions + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    
    color = PUBLICATION_COLORS[0]
    plotted = False
    
    for idx, session_name in enumerate(sessions):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        data = sessions_data.get(session_name, {})
        points = data.get(metric_name, [])
        
        if points:
            episodes = [ep for ep, _ in points]
            values = [val for _, val in points]
            ax.plot(episodes, values, color=color, linewidth=2.0)
            plotted = True
        
        # Format session name for subplot title
        approach, target = parse_session_name(session_name)
        subplot_title = f"{_format_approach_label(approach)}\n{_format_target_label(target)}"
        ax.set_title(subplot_title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(metric_name.capitalize(), fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(n_sessions, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)
    
    if not plotted:
        plt.close()
        return False
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def plot_bar_comparison(
    sessions_stats: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str,
    output_path: str,
    group_by: str = "session",  # "session" or "metric"
    show_error: bool = True,
    error_data: Optional[Dict[str, Dict[str, float]]] = None,
    normalize: bool = False,
) -> bool:
    """
    Create grouped bar chart comparing sessions.
    
    Args:
        sessions_stats: {session_label: {metric: value}}
        metrics: list of metrics to include
        title: plot title
        output_path: where to save
        group_by: how to group bars
        show_error: whether to show error bars
        error_data: {session_label: {metric: std_value}} for error bars
        normalize: if True, normalize each metric to [0, 1] across sessions
    """
    if not sessions_stats:
        return False
    
    sessions = list(sessions_stats.keys())
    n_sessions = len(sessions)
    n_metrics = len(metrics)
    
    if n_sessions == 0 or n_metrics == 0:
        return False
    
    # Normalize values per metric if requested
    if normalize:
        normalized_stats: Dict[str, Dict[str, float]] = {s: {} for s in sessions}
        normalized_errors: Optional[Dict[str, Dict[str, float]]] = None
        if error_data:
            normalized_errors = {s: {} for s in sessions}
        
        for metric in metrics:
            values = [sessions_stats[s].get(metric, 0) for s in sessions]
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            
            for s in sessions:
                raw_val = sessions_stats[s].get(metric, 0)
                normalized_stats[s][metric] = (raw_val - min_val) / range_val
                
                if error_data and normalized_errors is not None:
                    raw_err = error_data.get(s, {}).get(metric, 0)
                    # Scale error by the same factor
                    normalized_errors[s][metric] = raw_err / range_val
        
        sessions_stats = normalized_stats
        if normalized_errors is not None:
            error_data = normalized_errors
    
    fig, ax = plt.subplots(figsize=(max(8, n_sessions * 1.5), 5))
    
    if group_by == "session":
        # Group by session, bars for each metric
        x = np.arange(n_sessions)
        width = 0.8 / n_metrics
        
        for i, metric in enumerate(metrics):
            values = [sessions_stats[s].get(metric, 0) for s in sessions]
            errors = None
            if show_error and error_data:
                errors = [error_data.get(s, {}).get(metric, 0) for s in sessions]
            
            offset = (i - n_metrics / 2 + 0.5) * width
            color = PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)]
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), 
                         color=color, yerr=errors, capsize=3, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" - ", "\n") for s in sessions], fontsize=9)
    else:
        # Group by metric, bars for each session
        x = np.arange(n_metrics)
        width = 0.8 / n_sessions
        
        for i, session in enumerate(sessions):
            values = [sessions_stats[session].get(m, 0) for m in metrics]
            errors = None
            if show_error and error_data:
                errors = [error_data.get(session, {}).get(m, 0) for m in metrics]
            
            offset = (i - n_sessions / 2 + 0.5) * width
            color = PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)]
            # Format session label
            approach, target = parse_session_name(session)
            label = f"{_format_approach_label(approach)} - {_format_target_label(target)}"
            bars = ax.bar(x + offset, values, width, label=label,
                         color=color, yerr=errors, capsize=3, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=10)
    
    ax.set_ylabel("Value", fontweight='normal')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9, fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def generate_all_comparisons(base_dir: str) -> None:
    """
    Generate all cross-session comparison plots.
    
    Creates:
    - Overlay plots comparing approaches (same target)
    - Overlay plots comparing targets (same approach)
    - Overlay plots with all sessions
    - Overlay plots with std shading (in with_std subfolders)
    - Grid plots for each metric
    - Bar charts with summary statistics (raw and normalized)
    """
    print("\n" + "=" * 80)
    print("Generating cross-session comparisons")
    print("=" * 80)
    
    # Create output directories
    comparisons_dir = os.path.join(base_dir, "logs", "comparisons")
    by_approach_dir = os.path.join(comparisons_dir, "by_approach")
    by_target_dir = os.path.join(comparisons_dir, "by_target")
    all_sessions_dir = os.path.join(comparisons_dir, "all_sessions")
    summary_bars_dir = os.path.join(comparisons_dir, "summary_bars")
    
    # Subdirectories for plots with std shading
    by_approach_std_dir = os.path.join(by_approach_dir, "with_std")
    by_target_std_dir = os.path.join(by_target_dir, "with_std")
    all_sessions_std_dir = os.path.join(all_sessions_dir, "with_std")
    
    # Subdirectory for normalized bar charts
    summary_bars_normalized_dir = os.path.join(summary_bars_dir, "normalized")
    
    for d in [by_approach_dir, by_target_dir, all_sessions_dir, summary_bars_dir,
              by_approach_std_dir, by_target_std_dir, all_sessions_std_dir,
              summary_bars_normalized_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Get groupings
    approaches = get_unique_approaches()
    targets = get_unique_social_targets()
    by_approach = group_sessions_by_approach()
    by_target = group_sessions_by_target()
    
    print(f"\nApproaches: {approaches}")
    print(f"Social targets: {targets}")
    
    # Load all session data (means and stds)
    print("\n[1] Loading data from all sessions...")
    all_social_data: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    all_social_std: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    all_rewards_data: Dict[str, List[Tuple[int, float]]] = {}
    
    for session_name in SESSIONS.keys():
        means, stds = _load_session_social_metrics_with_std(session_name, base_dir)
        all_social_data[session_name] = means
        all_social_std[session_name] = stds
        all_rewards_data[session_name] = _load_session_averaged_rewards(session_name, base_dir)
        print(f"  Loaded: {session_name}")
    
    # =========================================================================
    # 1. Overlay plots: Compare by target (same approach, different targets)
    # =========================================================================
    print("\n[2] Generating overlay plots by target (same approach)...")
    
    for approach, session_names in by_approach.items():
        # Social metrics
        for metric_name in SOCIAL_ORDER:
            sessions_data = {}
            sessions_std = {}
            for session_name in session_names:
                _, target = parse_session_name(session_name)
                label = _format_target_label(target)
                data = all_social_data.get(session_name, {}).get(metric_name, [])
                std_data = all_social_std.get(session_name, {}).get(metric_name, [])
                if data:
                    sessions_data[label] = data
                    if std_data:
                        sessions_std[label] = std_data
            
            if sessions_data:
                # Without std
                output_path = os.path.join(
                    by_target_dir, 
                    f"compare_{approach}_{metric_name}.png"
                )
                title = f"{metric_name.capitalize()} - {_format_approach_label(approach)}"
                plotted = plot_overlay_comparison(
                    sessions_data, title, metric_name.capitalize(), output_path
                )
                if plotted:
                    print(f"  [OK] {approach} - {metric_name}")
                
                # With std
                output_path_std = os.path.join(
                    by_target_std_dir,
                    f"compare_{approach}_{metric_name}.png"
                )
                plotted_std = plot_overlay_comparison(
                    sessions_data, title, metric_name.capitalize(), output_path_std,
                    sessions_std=sessions_std
                )
                if plotted_std:
                    print(f"  [OK] {approach} - {metric_name} (with std)")
        
        # Rewards
        sessions_data = {}
        for session_name in session_names:
            _, target = parse_session_name(session_name)
            label = _format_target_label(target)
            data = all_rewards_data.get(session_name, [])
            if data:
                sessions_data[label] = data
        
        if sessions_data:
            output_path = os.path.join(by_target_dir, f"compare_{approach}_rewards.png")
            title = f"Episode Rewards - {_format_approach_label(approach)}"
            plotted = plot_overlay_comparison(sessions_data, title, "Mean Reward", output_path)
            if plotted:
                print(f"  [OK] {approach} - rewards")
    
    # =========================================================================
    # 2. Overlay plots: Compare by approach (same target, different approaches)
    # =========================================================================
    print("\n[3] Generating overlay plots by approach (same target)...")
    
    for target, session_names in by_target.items():
        # Social metrics
        for metric_name in SOCIAL_ORDER:
            sessions_data = {}
            sessions_std = {}
            for session_name in session_names:
                approach, _ = parse_session_name(session_name)
                label = _format_approach_label(approach)
                data = all_social_data.get(session_name, {}).get(metric_name, [])
                std_data = all_social_std.get(session_name, {}).get(metric_name, [])
                if data:
                    sessions_data[label] = data
                    if std_data:
                        sessions_std[label] = std_data
            
            if sessions_data:
                # Without std
                output_path = os.path.join(
                    by_approach_dir, 
                    f"compare_{target}_{metric_name}.png"
                )
                title = f"{metric_name.capitalize()} - {_format_target_label(target)}"
                plotted = plot_overlay_comparison(
                    sessions_data, title, metric_name.capitalize(), output_path
                )
                if plotted:
                    print(f"  [OK] {target} - {metric_name}")
                
                # With std
                output_path_std = os.path.join(
                    by_approach_std_dir,
                    f"compare_{target}_{metric_name}.png"
                )
                plotted_std = plot_overlay_comparison(
                    sessions_data, title, metric_name.capitalize(), output_path_std,
                    sessions_std=sessions_std
                )
                if plotted_std:
                    print(f"  [OK] {target} - {metric_name} (with std)")
        
        # Rewards
        sessions_data = {}
        for session_name in session_names:
            approach, _ = parse_session_name(session_name)
            label = _format_approach_label(approach)
            data = all_rewards_data.get(session_name, [])
            if data:
                sessions_data[label] = data
        
        if sessions_data:
            output_path = os.path.join(by_approach_dir, f"compare_{target}_rewards.png")
            title = f"Episode Rewards - {_format_target_label(target)}"
            plotted = plot_overlay_comparison(sessions_data, title, "Mean Reward", output_path)
            if plotted:
                print(f"  [OK] {target} - rewards")
    
    # =========================================================================
    # 3. Overlay plots: All sessions together
    # =========================================================================
    print("\n[4] Generating overlay plots with all sessions...")
    
    for metric_name in SOCIAL_ORDER:
        sessions_data = {}
        sessions_std = {}
        for session_name in SESSIONS.keys():
            approach, target = parse_session_name(session_name)
            label = f"{_format_approach_label(approach)} - {_format_target_label(target)}"
            data = all_social_data.get(session_name, {}).get(metric_name, [])
            std_data = all_social_std.get(session_name, {}).get(metric_name, [])
            if data:
                sessions_data[label] = data
                if std_data:
                    sessions_std[label] = std_data
        
        if sessions_data:
            # Without std
            output_path = os.path.join(all_sessions_dir, f"compare_all_{metric_name}.png")
            title = f"{metric_name.capitalize()} - All Sessions"
            plotted = plot_overlay_comparison(
                sessions_data, title, metric_name.capitalize(), output_path
            )
            if plotted:
                print(f"  [OK] all sessions - {metric_name}")
            
            # With std
            output_path_std = os.path.join(all_sessions_std_dir, f"compare_all_{metric_name}.png")
            plotted_std = plot_overlay_comparison(
                sessions_data, title, metric_name.capitalize(), output_path_std,
                sessions_std=sessions_std
            )
            if plotted_std:
                print(f"  [OK] all sessions - {metric_name} (with std)")
    
    # Rewards - all sessions
    sessions_data = {}
    for session_name in SESSIONS.keys():
        approach, target = parse_session_name(session_name)
        label = f"{_format_approach_label(approach)} - {_format_target_label(target)}"
        data = all_rewards_data.get(session_name, [])
        if data:
            sessions_data[label] = data
    
    if sessions_data:
        output_path = os.path.join(all_sessions_dir, "compare_all_rewards.png")
        plotted = plot_overlay_comparison(
            sessions_data, "Episode Rewards - All Sessions", "Mean Reward", output_path
        )
        if plotted:
            print(f"  [OK] all sessions - rewards")
    
    # =========================================================================
    # 4. Grid plots: One subplot per session
    # =========================================================================
    print("\n[5] Generating grid comparison plots...")
    
    for metric_name in SOCIAL_ORDER:
        output_path = os.path.join(all_sessions_dir, f"grid_{metric_name}.png")
        title = f"{metric_name.capitalize()} Comparison"
        plotted = plot_grid_comparison(all_social_data, metric_name, title, output_path)
        if plotted:
            print(f"  [OK] grid - {metric_name}")
    
    # =========================================================================
    # 5. Bar charts: Summary statistics
    # =========================================================================
    print("\n[6] Generating bar chart summaries...")
    
    # Compute summary statistics for each session
    final_stats: Dict[str, Dict[str, float]] = {}
    avg_stats: Dict[str, Dict[str, float]] = {}
    std_stats: Dict[str, Dict[str, float]] = {}
    
    for session_name in SESSIONS.keys():
        final_stats[session_name] = {}
        avg_stats[session_name] = {}
        std_stats[session_name] = {}
        
        social_data = all_social_data.get(session_name, {})
        for metric_name in SOCIAL_ORDER:
            points = social_data.get(metric_name, [])
            if points:
                values = [v for _, v in points]
                final_stats[session_name][metric_name] = values[-1] if values else 0
                avg_stats[session_name][metric_name] = np.mean(values)
                std_stats[session_name][metric_name] = np.std(values)
    
    # Bar chart: Final values (grouped by metric)
    output_path = os.path.join(summary_bars_dir, "bar_final_by_metric.png")
    plotted = plot_bar_comparison(
        final_stats, list(SOCIAL_ORDER), 
        "Final Episode Values", output_path,
        group_by="metric", show_error=False
    )
    if plotted:
        print(f"  [OK] bar - final values by metric")
    
    # Bar chart: Average values (grouped by metric)
    output_path = os.path.join(summary_bars_dir, "bar_average_by_metric.png")
    plotted = plot_bar_comparison(
        avg_stats, list(SOCIAL_ORDER),
        "Average Values Across Episodes", output_path,
        group_by="metric", show_error=True, error_data=std_stats
    )
    if plotted:
        print(f"  [OK] bar - average values by metric")
    
    # Bar chart: Final values (grouped by session)
    output_path = os.path.join(summary_bars_dir, "bar_final_by_session.png")
    plotted = plot_bar_comparison(
        final_stats, list(SOCIAL_ORDER),
        "Final Episode Values", output_path,
        group_by="session", show_error=False
    )
    if plotted:
        print(f"  [OK] bar - final values by session")
    
    # Bar chart: Average values (grouped by session)
    output_path = os.path.join(summary_bars_dir, "bar_average_by_session.png")
    plotted = plot_bar_comparison(
        avg_stats, list(SOCIAL_ORDER),
        "Average Values Across Episodes", output_path,
        group_by="session", show_error=True, error_data=std_stats
    )
    if plotted:
        print(f"  [OK] bar - average values by session")
    
    # -------------------------------------------------------------------------
    # Normalized bar charts (each metric scaled to [0, 1] for better comparison)
    # -------------------------------------------------------------------------
    print("\n[7] Generating normalized bar chart summaries...")
    
    # Normalized: Final values (grouped by metric)
    output_path = os.path.join(summary_bars_normalized_dir, "bar_final_by_metric_normalized.png")
    plotted = plot_bar_comparison(
        final_stats, list(SOCIAL_ORDER), 
        "Final Episode Values (Normalized)", output_path,
        group_by="metric", show_error=False, normalize=True
    )
    if plotted:
        print(f"  [OK] bar - final values by metric (normalized)")
    
    # Normalized: Average values (grouped by metric)
    output_path = os.path.join(summary_bars_normalized_dir, "bar_average_by_metric_normalized.png")
    plotted = plot_bar_comparison(
        avg_stats, list(SOCIAL_ORDER),
        "Average Values Across Episodes (Normalized)", output_path,
        group_by="metric", show_error=True, error_data=std_stats, normalize=True
    )
    if plotted:
        print(f"  [OK] bar - average values by metric (normalized)")
    
    # Normalized: Final values (grouped by session)
    output_path = os.path.join(summary_bars_normalized_dir, "bar_final_by_session_normalized.png")
    plotted = plot_bar_comparison(
        final_stats, list(SOCIAL_ORDER),
        "Final Episode Values (Normalized)", output_path,
        group_by="session", show_error=False, normalize=True
    )
    if plotted:
        print(f"  [OK] bar - final values by session (normalized)")
    
    # Normalized: Average values (grouped by session)
    output_path = os.path.join(summary_bars_normalized_dir, "bar_average_by_session_normalized.png")
    plotted = plot_bar_comparison(
        avg_stats, list(SOCIAL_ORDER),
        "Average Values Across Episodes (Normalized)", output_path,
        group_by="session", show_error=True, error_data=std_stats, normalize=True
    )
    if plotted:
        print(f"  [OK] bar - average values by session (normalized)")
    
    # Per-metric bar charts
    for metric_name in SOCIAL_ORDER:
        metric_final: Dict[str, Dict[str, float]] = {}
        metric_avg: Dict[str, Dict[str, float]] = {}
        metric_std: Dict[str, Dict[str, float]] = {}
        
        for session_name in SESSIONS.keys():
            approach, target = parse_session_name(session_name)
            label = f"{_format_approach_label(approach)}\n{_format_target_label(target)}"
            metric_final[label] = {"value": final_stats[session_name].get(metric_name, 0)}
            metric_avg[label] = {"value": avg_stats[session_name].get(metric_name, 0)}
            metric_std[label] = {"value": std_stats[session_name].get(metric_name, 0)}
        
        output_path = os.path.join(summary_bars_dir, f"bar_final_{metric_name}.png")
        plotted = plot_bar_comparison(
            metric_final, ["value"],
            f"Final {metric_name.capitalize()} Value", output_path,
            group_by="session", show_error=False
        )
        if plotted:
            print(f"  [OK] bar - final {metric_name}")
        
        output_path = os.path.join(summary_bars_dir, f"bar_average_{metric_name}.png")
        plotted = plot_bar_comparison(
            metric_avg, ["value"],
            f"Average {metric_name.capitalize()} Value", output_path,
            group_by="session", show_error=True, error_data=metric_std
        )
        if plotted:
            print(f"  [OK] bar - average {metric_name}")
    
    print("\n[DONE] Cross-session comparisons complete!")
    print(f"\nOutput directories:")
    print(f"  - By approach: {by_approach_dir}")
    print(f"    - With std: {by_approach_std_dir}")
    print(f"  - By target: {by_target_dir}")
    print(f"    - With std: {by_target_std_dir}")
    print(f"  - All sessions: {all_sessions_dir}")
    print(f"    - With std: {all_sessions_std_dir}")
    print(f"  - Summary bars: {summary_bars_dir}")
    print(f"    - Normalized: {summary_bars_normalized_dir}")


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process experiment sessions and generate comparison plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (per-session plots + comparisons)
  python scripts/process_all_sessions.py

  # Run ONLY cross-session comparisons (skip per-session plots)
  python scripts/process_all_sessions.py --comparisons-only

  # Run ONLY per-session plots (skip comparisons)
  python scripts/process_all_sessions.py --skip-comparisons
        """
    )
    
    parser.add_argument(
        "--comparisons-only",
        action="store_true",
        help="Generate only cross-session comparison plots (skip per-session plots)"
    )
    
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Skip cross-session comparison plots (only generate per-session plots)"
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.comparisons_only and args.skip_comparisons:
        parser.error("--comparisons-only and --skip-comparisons are mutually exclusive")
    
    return args


def main():
    args = parse_args()
    
    # Get the base directory (project root)
    base_dir = os.path.dirname(script_dir)
    
    print("=" * 80)
    print("Processing all experiment sessions")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Number of sessions: {len(SESSIONS)}")
    
    if args.comparisons_only:
        print("\nMode: Comparisons only (skipping per-session plots)")
    elif args.skip_comparisons:
        print("\nMode: Per-session plots only (skipping comparisons)")
    else:
        print("\nMode: Full processing (per-session + comparisons)")
    
    # Process individual sessions (unless --comparisons-only)
    if not args.comparisons_only:
        for session_name, run_dirs in SESSIONS.items():
            process_session(session_name, run_dirs, base_dir)
        
        print("\n" + "=" * 80)
        print("Per-session processing complete!")
        print("=" * 80)
        
        # Print summary of output locations
        print("\nPer-session output locations:")
        for session_name in SESSIONS.keys():
            session_plots = os.path.join(base_dir, "logs", session_name, "plots_averaged")
            print(f"  - {session_name}: {session_plots}")
            print(f"      - rewards_averaged.png")
            print(f"      - social_metrics_averaged.png")
            print(f"      - social_<metric>_all_runs.png (per metric)")
            print(f"      - predicted_reward_by_condition_with_std.png / _no_std.png")
            print(f"      - predicted_reward_by_action_with_std.png / _no_std.png")
    
    # Generate cross-session comparisons (unless --skip-comparisons)
    if not args.skip_comparisons:
        generate_all_comparisons(base_dir)
    
    print("\n" + "=" * 80)
    print("All processing complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
