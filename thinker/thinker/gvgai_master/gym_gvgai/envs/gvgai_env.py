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

    def __init__(self, game, level, version, max_episode_steps=200):
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
        
        # TimeLimit 설정
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

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

        # 스텝 카운터 증가
        self._elapsed_steps += 1
        
        terminated = isOver  # Game over
        truncated = False    # 기본값
        
        # 디버깅: 에피소드 종료 상태 추적
        if terminated or self._elapsed_steps >= self._max_episode_steps:
            print(f"[DEBUG] GVGAI_Env - Episode termination detected:")
            print(f"  - isOver: {isOver}")
            print(f"  - terminated: {terminated}")
            print(f"  - elapsed_steps: {self._elapsed_steps}")
            print(f"  - max_episode_steps: {self._max_episode_steps}")
            print(f"  - info keys: {list(info.keys())}")
        
        # TimeLimit 조건 체크 - 최대 스텝에 도달하면 강제 종료
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            terminated = True  # 강제 종료
            info["time_limit_reached"] = True
            info["elapsed_steps"] = self._elapsed_steps
            # 에피소드 종료 시 추가 정보
            info["episode_terminated"] = True
            info["episode_truncated"] = True
            info["real_done"] = True  # real_done 정보 추가
        else:
            info["time_limit_reached"] = False
            info["elapsed_steps"] = self._elapsed_steps
            # 게임 종료 시에도 에피소드 정보 추가
            if terminated:
                info["episode_terminated"] = True
                info["episode_truncated"] = False
                info["real_done"] = True  # real_done 정보 추가
            else:
                info["real_done"] = False  # 에피소드가 계속 진행 중

        # 디버깅: 최종 상태 확인
        if terminated or truncated:
            print(f"[DEBUG] GVGAI_Env - Final state:")
            print(f"  - terminated: {terminated}")
            print(f"  - truncated: {truncated}")
            print(f"  - info['real_done']: {info.get('real_done', 'Not set')}")

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
        
        # 스텝 카운터 리셋
        self._elapsed_steps = 0
        
        info = {}
        info["real_done"] = False  # 에피소드 시작 시 real_done 초기화
        return self.img, info

    def set_max_episode_steps(self, max_steps):
        """
        Set the maximum number of steps per episode.
        Args:
            max_steps (int): Maximum number of steps before episode termination
        """
        self._max_episode_steps = max_steps

    def get_elapsed_steps(self):
        """
        Get the current number of elapsed steps in the episode.
        Returns:
            int: Current step count
        """
        return self._elapsed_steps

    def get_max_episode_steps(self):
        """
        Get the maximum number of steps per episode.
        Returns:
            int: Maximum step limit
        """
        return self._max_episode_steps

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