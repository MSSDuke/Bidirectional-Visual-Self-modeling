import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import torch
import torch.nn as nn


env = gym.make("Ant-v5", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()