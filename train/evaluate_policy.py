import os
import json
import functools
import shutil
from typing import TypeVar, Type
from pathlib import Path 

import numpy as np
import jax
from jax import numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import orbax.checkpoint as orbax
from flax import nnx
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import checkpoint

from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


T = TypeVar("T") 

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

matplotlib.use('TkAgg')
plt.ion()


# evaluate_policy.py
import pickle

from brax import envs
from brax.training import types, networks
from brax.training.distribution import NormalTanhDistribution



if __name__ == "__main__":

    path = "/root/EL7009_projects/go2_train_logs/000206438400"
    """IMPORTANTE:
    Por algun motivo se me gusrada en el checkpoint, 
    el archivo json con valiores null. 
    Deben eliminarse esas llaves con valor null. Brax no sabe manejarlas."""

    policy = checkpoint.load_policy(path)

    print("model loaded")