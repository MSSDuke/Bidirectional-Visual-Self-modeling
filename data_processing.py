import numpy as np
import torch
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax
import flax.nnx as nnx

import minari

dataset = minari.load_dataset("mujoco/ant/simple-v0", download=True)
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)

"""Data processing and loading"""
class DataProcessor():
    def __init__(self):

        pass
    pass