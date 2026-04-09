import os
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
from datetime import datetime
import functools

import numpy as np
import mediapy as media
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display
import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from ml_collections import config_dict
from orbax import checkpoint as ocp


from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

matplotlib.use('TkAgg')
plt.ion()

np.set_printoptions(precision=3, suppress=True, linewidth=100)


ENVS = registry.locomotion.ALL_ENVS
print(f"Avaliable envs:\n{ENVS}")

env_name = 'Go2JoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
env_ppo_params = locomotion_params.brax_ppo_config(env_name)
registry.get_domain_randomizer(env_name)
print(f"PPO param for Go2 training:\n{env_ppo_params}")

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  #clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.clf()
  plt.xlim([0, env_ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  #display(plt.gcf())
  plt.draw()
  plt.gcf().canvas.draw()  # Dibujar explícitamente
  plt.gcf().canvas.flush_events()
  plt.pause(0.01)
  pwd = "/root/EL7009_projects"
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] 
  plt.savefig(f'{pwd}/training_progress_{timestamp}.png')
  print("plot")


randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(env_ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in env_ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **env_ppo_params.network_factory
  )


train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress
)

print("Almost ready to training")

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
plt.show()