# gym_vrep

OpenAI gym-like reinforcement environment created by V-REP

# Dependency

* V-REP (3.5 or later)
    * Its directory name should be changed to V-REP_PRO_EDU.
    * Its directory should be on ...
        * ... home directory (~/) in the case of linux.
        * ... Applications (/Applications/) in the case of mac.
* numpy
* gym
* fasteners
* git-lfs (for developer)

# Installation

```bash
git lfs clone https://github.com/kbys-t/gym_vrep.git
cd gym_vrep
pip install -e .
```

# How to use

For example,

```python
import gym_vrep
env = gym_vrep.VrepEnv(scene=env_name, is_render=is_check, is_boot=is_boot, port=19997)
if is_record:
  env.monitor(save_dir, force=True)
for epi in range(n_episode):
  observation = env.reset()
  for stp in range(n_time):
    observation, reward, done, info = env.step(action)
env.close()
```

* By changing port number, multiple simulations can be processed.
    * Please be aware that the specified port is opened at the OS level.
* If vrep process is already running, please set `is_boot=False`.
    * In that case, a scene file is not required to be opened.
