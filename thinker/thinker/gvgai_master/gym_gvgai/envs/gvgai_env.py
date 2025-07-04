#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import path
import numpy as np

dir = path.dirname(__file__)
gvgai_path = path.join(dir, "gvgai", "clients", "GVGAI-PythonClient", "src", "utils")
sys.path.append(gvgai_path)

import gymnasium as gym
from gymnasium import spaces
import ClientCommGYM as gvgai

class GVGAI_Env(gym.Env):
    """
    Define a VGDL environment.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, game, level, version):
        self.__version__ = "0.0.2"

        self.GVGAI = gvgai.ClientCommGYM(game, version, level, dir)
        self.game = game
        self.lvl = level
        self.version = version

        self.actions = self.GVGAI.actions()
        self.img = self.GVGAI.sso.image
        self.viewer = None

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.img.shape, dtype=np.uint8
        )

    def step(self, action):
        """
        Take an action in the environment.
        Returns:
            observation (np.array)
            reward (float)
            terminated (bool)
            truncated (bool)
            info (dict)
        """
        state, reward, isOver, info = self.GVGAI.step(action)
        self.img = state

        terminated = isOver  # Game over
        truncated = False    # No truncation criteria for now

        return state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        Returns:
            observation (np.array)
            info (dict)
        """
        super().reset(seed=seed)
        self.img = self.GVGAI.reset(self.lvl)
        info = {}
        return self.img, info

    def render(self, mode="rgb_array"):
        img = self.img[:, :, :3]
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gymnasium.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _setLevel(self, level):
        if isinstance(level, int):
            if level < 9:
                self.lvl = level
            else:
                print("Level doesn't exist, playing level 0")
                self.lvl = 0
        else:
            newLvl = path.realpath(level)
            ogLvls = [path.realpath(path.join(
                dir, 'games', f'{self.game}_v{self.version}', f'{self.game}_lvl{i}.txt'
            )) for i in range(9)]
            if newLvl in ogLvls:
                self.lvl = ogLvls.index(newLvl)
            elif path.exists(newLvl):
                self.GVGAI.addLevel(newLvl)
                self.lvl = 9
            else:
                print("Level doesn't exist, playing level 0")
                self.lvl = 0

    def get_action_meanings(self):
        return self.actions