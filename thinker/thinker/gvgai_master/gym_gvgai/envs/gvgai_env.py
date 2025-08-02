#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import path
import os
import random
import fcntl
import time
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

    def __init__(self, game, level, version, max_episode_steps=500):
        self.__version__ = "0.0.2"

        self.game = game
        self.original_lvl = level
        self.lvl = level
        self.version = version
        
        self.GVGAI = gvgai.ClientCommGYM(game, version, level, dir)
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

        # 파일 잠금을 위한 경로 설정
        self.lock_path = path.join(path.dirname(__file__), 'games', f'{self.game}_v{self.version}', 'level.lock')
        # 잠금 파일 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)

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
        return state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        This method now shuffles the level map at the beginning of each episode.
        Returns:
            observation (np.array)
            info (dict)
        """
        super().reset(seed=seed)

        # 파일 잠금 시작
        while True:
            try:
                with open(self.lock_path, 'w') as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # --- Start of Shuffling Logic ---
                    game_folder = path.join(dir, 'games', f'{self.game}_v{self.version}')
                    original_lvl_path = path.join(game_folder, f'{self.game}_lvl{self.original_lvl}.txt')
                    
                    level_to_load = self.lvl

                    if path.exists(original_lvl_path):
                        with open(original_lvl_path, 'r') as f:
                            lines = [line.strip() for line in f if line.strip()]
                        
                        map_grid = [list(line) for line in lines]
                        
                        print("\nOriginal Level Design:")
                        for line in lines:
                            print(line)

                        if len(map_grid) > 2 and all(len(row) > 2 for row in map_grid):
                            inner_content = []
                            for r in range(1, len(map_grid) - 1):
                                for c in range(1, len(map_grid[r]) - 1):
                                    inner_content.append(map_grid[r][c])

                            random.seed() # Use system time for true randomness
                            random.shuffle(inner_content)

                            content_idx = 0
                            for r in range(1, len(map_grid) - 1):
                                for c in range(1, len(map_grid[r]) - 1):
                                    map_grid[r][c] = inner_content[content_idx]
                                    content_idx += 1
                            
                            # 임시 파일에 셔플된 맵을 작성
                            temp_level_path = path.join(game_folder, f'{self.game}_lvl9.txt.tmp')
                            with open(temp_level_path, 'w') as f:
                                for row in map_grid:
                                    f.write("".join(row) + "\n")

                            print("\nShuffled Level Design:")
                            for row in map_grid:
                                print("".join(row))

                            # 원자적으로 파일 이름 변경
                            shuffled_level_path = path.join(game_folder, f'{self.game}_lvl9.txt')
                            os.rename(temp_level_path, shuffled_level_path)
                            
                            level_to_load = 9
                    # --- End of Shuffling Logic ---
                    
                    self.img = self.GVGAI.reset(level_to_load)
                    
                    # 잠금 해제
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    break  # 잠금 후 작업 완료, 루프 탈출
            except IOError:
                # 잠금을 얻지 못하면 잠시 대기
                time.sleep(0.1)

        self._elapsed_steps = 0
        info = {"real_done": False}
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
        # Clean up the shuffled level file
        shuffled_level_path = path.join(dir, 'games', f'{self.game}_v{self.version}', f'{self.game}_lvl9.txt')
        if path.exists(shuffled_level_path):
            # os.remove(shuffled_level_path) # Optional: remove if you don't want to keep the last shuffled map
            pass


    def _setLevel(self, level):
        if isinstance(level, int):
            if level < 10:
                self.lvl = level
            else:
                print("Level doesn't exist, playing level 0")
                self.lvl = 0
        else:
            newLvl = path.realpath(level)
            ogLvls = [path.realpath(path.join(
                dir, 'games', f'{self.game}_v{self.version}', f'{self.game}_lvl{i}.txt'
            )) for i in range(10)]
            if newLvl in ogLvls:
                self.lvl = ogLvels.index(newLvl)
            elif path.exists(newLvl):
                # This logic is now handled in reset()
                self.lvl = 9 
            else:
                print("Level doesn't exist, playing level 0")
                self.lvl = 0

    def get_action_meanings(self):
        return self.actions

