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

## Requirements

Python packages:
- `gymnasium`
- `pettingzoo`
- `opencv-python`
- `matplotlib`
- `tqdm`

## How to run a quick step

From the repo root:

```bash
python - << 'PY'
import sys
sys.path.append('src')
from env.commons_env import HarvestCommonsEnv, MAP

env = HarvestCommonsEnv(ascii_map=MAP['small'], num_agents=1, render=False)
obs, infos = env.reset()
actions = {agent_id: env.action_space.sample() for agent_id in env.agents}
next_obs, rewards, dones, infos = env.step(actions)
print('actions:', actions)
print('rewards:', rewards)
print('dones:', dones)
PY
```

## Render frames to `output/`

This saves a frame before and after one step.

```bash
python - << 'PY'
import os
import sys
sys.path.append('src')
from env.commons_env import HarvestCommonsEnv, MAP

out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)

env = HarvestCommonsEnv(ascii_map=MAP['small'], num_agents=1, render=True)
env.reset()
env.render(os.path.join(out_dir, 'map_step_0.png'), mod='human')

actions = {agent_id: env.action_space.sample() for agent_id in env.agents}
env.step(actions)
env.render(os.path.join(out_dir, 'map_step_1.png'), mod='human')
print('saved:', os.path.join(out_dir, 'map_step_0.png'))
print('saved:', os.path.join(out_dir, 'map_step_1.png'))
PY
```

## Random run + video (15 steps)

This renders 16 frames (t=0000..0015) and builds a video:

```bash
python - << 'PY'
import os
import sys
sys.path.append('src')
from env.commons_env import HarvestCommonsEnv, MAP
from env.utils import utility_funcs

out_dir = os.path.join('output', 'run_15_steps')
img_dir = os.path.join(out_dir, 'imgs')
os.makedirs(img_dir, exist_ok=True)

env = HarvestCommonsEnv(ascii_map=MAP['small'], num_agents=1, render=True)
env.reset()
env.render(os.path.join(img_dir, 't=0000.png'), mod='human')

for t in range(1, 16):
    actions = {agent_id: env.action_space.sample() for agent_id in env.agents}
    env.step(actions)
    env.render(os.path.join(img_dir, f't={t:04d}.png'), mod='human')

utility_funcs.make_video_from_image_dir(
    vid_path=out_dir,
    img_folder=img_dir,
    video_name='run_15_steps',
    fps=5,
)
print('video:', os.path.join(out_dir, 'run_15_steps.mp4'))
PY
```
