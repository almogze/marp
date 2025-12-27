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

Logs are written to `logs/<run-name>/metrics.jsonl` and `logs/<run-name>/config.json`.
Videos are written to `logs/<run-name>/videos/episode=XXXX.mp4`.

Config tips:
- `logging.video_every_n_episodes` defaults to 100; reduce it to record more frequently.
- `logging.video_max_steps` caps episode length in videos.
- `logging.video_enabled=false` disables video capture for faster training.

Switch algorithms by changing `algorithm.name` in the config. Supported values:
`dqn`, `random`, `ppo` (SB3).

PPO requires `stable-baselines3` and `gymnasium` installed.
For multiple agents, PPO trains independent policies sequentially against random opponents.

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
