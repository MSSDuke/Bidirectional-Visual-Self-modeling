import os, json, datetime, pathlib
import random
import time
import argparse
import yaml
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC
import numpy as np
import torch
import torch.nn as nn

env = gym.make("Ant-v5", xml_file="/home/mss131/SoundSM/ant_m.xml", render_mode="rgb_array", width=128, height=128)
model_path = load_from_hub(
	repo_id="farama-minari/Ant-v5-SAC-medium",
	filename="ant-v5-sac-medium.zip",
)
model = SAC.load(model_path)

class DataCollector():
    def __init__(self, env=env, model=model, steps_per_episode=512, n_steps=100000, Hz=10):
        """
        Data collection agent

        Args:
            env: robot environment used for data collection
            steps_per_episode: number of env steps per episode of data collection
            n_samples: number of samples to collect
            Hz: frequency at which to sample from env
        """
        self.env = env
        self.model = model
        self.n_steps = n_steps
        self.steps_per_episode = steps_per_episode
        self.Hz = Hz

        _, _ = self.env.reset(seed=0)
        u = self.env.unwrapped
        self.state_dim = int(u.data.qpos.size) + int(u.data.qvel.size)
        self.act_dim = int(np.prod(self.env.action_space.shape))

        try:
            self.dt = float(u.model.opt.timestep) * float(u.frame_skip)
        except Exception:
            self.dt = 1.0 / getattr(self.env, "metadata", {}).get("render_fps", 50)

        steps_per_second = 1.0 / self.dt
        self.capture_every = max(1, int(round(steps_per_second / self.Hz)))

        
        self.states = np.empty((self.n_steps, self.state_dim))
        self.actions = np.empty((self.n_steps, self.act_dim))
        self.specs = []
        self.specs_indices = []

        self.term = np.empty((self.n_steps,), dtype=bool)
        self.trun = np.empty((self.n_steps,), dtype=bool)

    def _get_qpos_qvel(self):
        u = self.env.unwrapped
        qpos = np.array(u.data.qpos, dtype=np.float32, copy=True)
        qvel = np.array(u.data.qvel, dtype=np.float32, copy=True)
        return np.concatenate([qpos, qvel], axis=0)

    def collect(self, seed=0):
        """
        Collects env data for: joint_states, actions, spectrograms (NOTE we don't
        actually use joint_states)
        """
        state_env, info = self.env.reset(seed=seed)

        episode_step = 0
        step = 0

        while step < self.n_steps:
            state = self._get_qpos_qvel()
            step_action, _ = self.model.predict(state_env, deterministic=True)
            state_env, _, terminated, truncated, info = self.env.step(step_action)

            self.states[step] = state
            self.actions[step] = step_action
            self.term[step] = bool(terminated)
            self.trun[step] = bool(truncated)

            if (step % 100 == 0):
                print(f"Step {step}/{self.n_steps}")

            if (episode_step % self.capture_every) == 0:
                frame = self.env.render()
                if frame is not None:
                    self.specs.append(frame)
                    self.specs_indices.append(step)
            
            episode_step += 1
            step += 1

            if terminated or truncated or (episode_step >= self.steps_per_episode):
                state_env, info = self.env.reset()
                episode_step = 0
        
        return {
            "states": self.states,
            "actions": self.actions,
            "spectrograms": self.specs
        }


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)




def make_run_dir(base="data", env_id="Ant-v5", seed=0):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = pathlib.Path(base) / env_id / f"{ts}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_dataset(run_dir, dataset, meta=None):
    run_dir = Path(run_dir)
    specs = dataset.get("spectrograms", [])

    if len(specs) > 0:
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(specs):
            img = Image.fromarray(frame.astype(np.uint8))
            img.save(frames_dir / f"frame_{i:06d}.png")

        specs_arr = np.stack(specs).astype(np.uint8)
    else:
        specs_arr = np.empty((0,), dtype=np.uint8)

    np.savez_compressed(
        run_dir / "rollout.npz",
        states=dataset["states"],
        actions=dataset["actions"],
        spectrograms=specs_arr,
    )

    if meta is not None:
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def load_dataset(run_dir):
    run_dir = pathlib.Path(run_dir)
    arrays = np.load(run_dir / "rollout.npz", allow_pickle=False)
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    data = {k: arrays[k] for k in arrays.files}
    return data, meta



def run_data_collection():
    seed = np.random.randint
    dc = DataCollector(env, model, steps_per_episode=512, n_steps=10_000, Hz=16)
    dataset = dc.collect(seed=seed)

    meta = {
        "env_id": "Ant-v5",
        "seed": seed,
        "state_dim": int(dc.state_dim),
        "act_dim": int(dc.act_dim),
        "dt": float(dc.dt),
        "capture_every": int(dc.capture_every),
        "n_steps": int(dc.n_steps),
        "steps_per_episode": int(dc.steps_per_episode),
        "gymnasium_version": getattr(gym, "__version__", "unknown"),
        "numpy_version": np.__version__,
    }

    out_dir = make_run_dir(base="data", env_id="Ant-v5", seed=seed)
    save_dataset(out_dir, dataset, meta=meta)
    print(f"Saved run to: {out_dir}")


if __name__ == "__main__":
    run_data_collection()