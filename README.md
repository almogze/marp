# MARP Environment

This repo contains a grid-based multi-agent environment built on top of `gymnasium`.
Agents move on an ASCII map, collect apples for reward, and can optionally use a
`FIRE` action to penalize other agents. The environment tracks social metrics like
efficiency, equality, sustainability, and peace.

Key pieces:
- `src/env/commons_env.py`: Harvest commons environment (`HarvestCommonsEnv`).
- `src/env/map_env.py`: Core map simulation, movement, and rendering.
- `src/env/commons_agent.py`: Agent behavior and action space.
- `src/env/maps.py`: ASCII map layouts used for spawning walls/apples/agents.
- `src/reward_model/`: MARP-style preference-based reward model and training utilities.

## Training (configurable trainer)

The training entrypoint is `main.py`. Uncomment one of the template lines or use the
inline script below.

Template (edit `main.py`):

```bash
python main.py
```

Inline run (example: `configs/train_dqn.json`):

```bash
python - << 'PY'
import sys
sys.path.append('src')
from train import Trainer, load_config

config = load_config('configs/train_dqn.json')
trainer = Trainer(config)
trainer.train()
print('done')
PY
```

Logs are written to `logs/<run-name>/metrics.json` and `logs/<run-name>/config.json`.
Videos are written to `logs/<run-name>/videos/episode=XXXX.mp4`.
TensorBoard logs are written to `logs/<run-name>/tensorboard/`.
Run folders include a reward-model suffix, e.g. `...-rm=off` or `...-rm=narrow_view`.

View TensorBoard (live during training):

```bash
tensorboard --logdir logs
```

Config tips:
- `logging.video_every_n_episodes` defaults to 100; reduce it to record more frequently.
- `logging.video_max_steps` caps episode length in videos.
- `logging.video_enabled=false` disables video capture for faster training.
- The last episode is always recorded (if video is enabled), regardless of `video_every_n_episodes`.
- Example configs: `configs/train_dqn.json`, `configs/train_ppo.json`, `configs/train_mappo.json`.

## Run the environment script

Use `scripts/run_env.py` to launch training with CLI overrides:

```bash
python scripts/run_env.py --algo mappo --episodes 200 --reward-model --mode narrow_view --phi efficiency_x_peace
```

Windows example:

```bash
python scripts\run_env.py --algo mappo --episodes 200 --agents 5 --seed 0 --reward-model --mode narrow_view --phi efficiency_x_peace
```

Random seed example:

```bash
python scripts/run_env.py --algo dqn --episodes 100 --random-seed
```

### Running sequences of games

You can run multiple games sequentially in several ways:

**Multiple algorithms:**
```bash
python scripts/run_env.py --algo dqn ppo mappo --episodes 100
```

**Multiple maps:**
```bash
python scripts/run_env.py --algo dqn --map small large --episodes 100
```

**Multiple agent counts:**
```bash
python scripts/run_env.py --algo dqn --agents 3 5 7 --episodes 100
```

**Multiple seeds (including random):**
```bash
python scripts/run_env.py --algo dqn --seed 0 1 random 3 --episodes 100
```
This runs 4 games: seeds 0, 1, random, and 3.

**All random seeds:**
```bash
python scripts/run_env.py --algo dqn ppo --random-seed --episodes 100
```
This runs 2 games, each with a different randomly generated seed.

**All combinations:**
```bash
python scripts/run_env.py --algo dqn ppo --map small large --agents 3 5 --seed 0 1 --episodes 100
```
This will run all combinations: 2 algorithms × 2 maps × 2 agent counts × 2 seeds = 16 games total.

**Using a sequence file:**
Create a JSON file (e.g., `sequence.json`) with a list of game configurations:

```json
[
  {"algo": "dqn", "episodes": 100, "map_type": "small", "agents": 3, "seed": 0},
  {"algo": "ppo", "episodes": 200, "map_type": "large", "agents": 5, "reward_model": true, "seed": 1},
  {"algo": "mappo", "episodes": 150, "map_type": "small", "agents": 7, "seed": null},
  {"algo": "dqn", "episodes": 100, "map_type": "small", "agents": 3, "random_seed": true}
]
```
Note: Use `"seed": null` or `"random_seed": true` in sequence files to use random seeds for that game.

Then run:
```bash
python scripts/run_env.py --sequence-file sequence.json
```

Arguments:
- `--algo {dqn,ppo,mappo,random}` selects the algorithm (default: dqn). Can specify multiple values to run sequentially.
- `--episodes N` sets the number of training episodes.
- `--seed N` sets the random seed(s) (integer or 'random'). Can specify multiple values to run sequentially (e.g., `--seed 0 1 random 3`). Use 'random' to generate a random seed for that specific game.
- `--random-seed` uses a randomly generated seed for all games (overrides any `--seed` values). Each game will get a different random seed.
- `--map NAME` sets `env.map_type`. Can specify multiple values to run sequentially.
- `--agents N` sets `env.num_agents`. Can specify multiple values to run sequentially.
- `--reward-model` / `--no-reward-model` toggles reward modeling.
- `--mode MODE` sets `reward_model.mode`.
- `--phi PHI` sets `reward_model.phi`.
- `--sequence-file PATH` path to JSON file containing a list of game configurations to run sequentially.

Switch algorithms by changing `algorithm.name` in the config. Supported values:
`dqn`, `random`, `ppo` (SB3), `mappo` (native).

PPO requires `stable-baselines3` and `gymnasium` installed.
For multiple agents, PPO trains independent policies sequentially against random opponents.
MAPPO uses the native trainer loop with a shared actor and centralized critic.

## Preference-based reward modeling (MARP)

The trainer supports learning a reward model `r_hat(o, a)` from preferences derived
from social metrics (e.g., efficiency, peace). When enabled, the trainer uses the
learned reward instead of the environment reward for training, while still logging
environment rewards for analysis.

Key mechanics:
- `compute_social_metrics()` and `get_social_metrics()` provide metrics per episode.
- Preference pairs are generated from episode metrics (e.g., efficiency x peace).
- Reward model is trained via Bradley-Terry on trajectory pairs.
- DQN uses the external loop and swaps `rewards` with `r_hat` in `Trainer.train()`.
- PPO uses a Gym wrapper that overrides the reward returned by `step()`.

Enable in config:

```json
{
  "reward_model": {
    "enabled": true,
    "mode": "narrow_view",
    "phi": "efficiency_x_peace",
    "lr": 0.0001,
    "batch_pairs": 64,
    "train_steps_per_update": 50,
    "update_every_env_steps": 1000,
    "warmup_episodes": 50,
    "max_episodes_in_buffer": 5000,
    "device": "auto",
    "save_every_episodes": 200
  }
}
```

Logging:
- `reward_pred_*` tracks predicted rewards when RM is enabled.
- `reward_env_*` tracks environment rewards (PPO wrapper).
- `reward_model/*` in TensorBoard shows RM loss/accuracy/correlation.

Checkpoints:
- DQN: `logs/<run>/model_last.pt`, `logs/<run>/reward_model_last.pt`
- PPO: `logs/<run>/ppo_model_<agent_id>_last.zip`, `logs/<run>/reward_model_<agent_id>_last.pt`

## Plotting run metrics

Generate reward and social-metric plots from a run folder (expects `metrics.json`):

```bash
python scripts/plot_run_metrics.py logs/<run-name>
```

Outputs:
- `logs/<run-name>/plots/rewards.png`
- `logs/<run-name>/plots/social_metrics.png`

Options:
- `--smooth N`: moving average window (episodes); also adds a faded ±1 std band.
- `--normalize`: normalize each series to [0, 1] and plot social metrics on one graph.

## Running PPO with multiple agents

Set `env.num_agents` and keep `algorithm.ppo.multi_agent_mode` as `independent`. Each
agent is trained sequentially against random opponents:

```json
{
  "env": {"num_agents": 5},
  "algorithm": {
    "name": "ppo",
    "ppo": {
      "multi_agent_mode": "independent",
      "opponent_policy": "random",
      "per_agent_timesteps": 100000
    }
  }
}
```

## Running MAPPO

MAPPO uses a shared actor and centralized critic over concatenated observations.
Enable it by switching the algorithm name and configuring the `mappo` block:

```json
{
  "env": {"num_agents": 5},
  "algorithm": {
    "name": "mappo",
    "mappo": {
      "n_steps": 1024,
      "batch_size": 256,
      "update_epochs": 4,
      "flatten_obs": false,
      "normalize_obs": true
    }
  }
}
```
