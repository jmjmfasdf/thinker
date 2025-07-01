#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to check actual GVGAI observation format
"""
import sys
import os
import numpy as np

# Add paths
thinker_path = os.path.abspath('.')
if thinker_path not in sys.path:
    sys.path.insert(0, thinker_path)

gvgai_path = os.path.join(thinker_path, 'thinker', 'gvgai_master')
if gvgai_path not in sys.path:
    sys.path.insert(0, gvgai_path)

import gymnasium as gym
import gym_gvgai

print("Testing raw GVGAI environment...")

# Create raw GVGAI environment (without thinker wrappers)
env = gym.make("gvgai-bait-lvl0-v0")

print(f"Raw environment type: {type(env)}")
print(f"Raw observation_space: {env.observation_space}")
print(f"Raw observation_space.shape: {env.observation_space.shape}")
print(f"Raw observation_space.dtype: {env.observation_space.dtype}")

# Reset and get actual observation
print("\nResetting environment to get actual observation...")
try:
    obs_result = env.reset()
    if isinstance(obs_result, tuple):
        obs, info = obs_result
    else:
        obs = obs_result
        
    print(f"Actual observation type: {type(obs)}")
    print(f"Actual observation shape: {obs.shape}")
    print(f"Actual observation dtype: {obs.dtype}")
    print(f"Observation min/max: {obs.min()}/{obs.max()}")
    
    # Check if it's really RGB
    if len(obs.shape) == 3:
        print(f"✓ 3D observation: {obs.shape[0]}x{obs.shape[1]}x{obs.shape[2]}")
        if obs.shape[2] == 3:
            print("✓ Looks like RGB (3 channels)")
        elif obs.shape[2] == 1:
            print("✓ Looks like grayscale (1 channel)")
        else:
            print(f"? Unusual channel count: {obs.shape[2]}")
    elif len(obs.shape) == 2:
        print(f"✗ 2D observation: {obs.shape[0]}x{obs.shape[1]} (missing channel dimension)")
        
        # Check if observation_space is wrong
        print("\n⚠️  Observation space says 2D but actual data might be 3D")
        print("This could be a bug in the GVGAI environment definition")
        
except Exception as e:
    print(f"Error during reset: {e}")

env.close()

print("\n" + "="*60)
print("Now testing with thinker wrapper...")

from thinker.gym_add.wrapper import create_env_fn

class Flags:
    def __init__(self):
        self.detect_dan_num = 0
        self.grayscale = False
        self.discrete_k = 0
        self.repeat_action_n = 0
        self.rand_action_eps = 0.0
        self.sokoban_pomdp = False
        self.atari = False

flags = Flags()
try:
    env_fn = create_env_fn("gvgai-bait-lvl0-v0", flags)
    env_wrapped = env_fn()
    
    print(f"Wrapped environment type: {type(env_wrapped)}")
    print(f"Wrapped observation_space: {env_wrapped.observation_space}")
    print(f"Wrapped observation_space.shape: {env_wrapped.observation_space.shape}")
    
    env_wrapped.close()
except Exception as e:
    print(f"Error with wrapped environment: {e}")

print("\nDone.") 