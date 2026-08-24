import gymnasium as gym
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Box
from .csokoban import cSokoban
import numpy as np
import pkg_resources
import os 

class SokobanEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, difficulty='unfiltered', small=True, dan_num=0, seed=0):
        if difficulty == 'unfiltered': 
            level_num = 900000                      
            path = '/'.join(('boxoban-levels', difficulty, 'train'))
        elif difficulty == 'test':     
            level_num = 1000                  
            path = '/'.join(('boxoban-levels', 'unfiltered', 'test'))
        elif difficulty == 'medium':            
            level_num = 50000           
            path = '/'.join(('boxoban-levels', difficulty, 'valid'))
        elif difficulty == 'hard': 
            level_num = 3332         
            path = '/'.join(('boxoban-levels', difficulty))
        else:
            raise Exception(f"difficulty {difficulty} not accepted.")

        level_dir = pkg_resources.resource_filename(__name__, path)
        img_dir = pkg_resources.resource_filename(__name__, 'surface')
        
        self.sokoban = cSokoban(small=small, 
                                level_dir=level_dir.encode('UTF-8'), 
                                img_dir=img_dir.encode('UTF-8'), 
                                level_num=level_num, 
                                dan_num=dan_num,
                                seed=seed)
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(self.sokoban.obs_x, self.sokoban.obs_y, 3), dtype=np.uint8)
        # Perfect-model search keeps one simulator snapshot per tree node.  A
        # singleton save_state cannot represent branches that are alive at the
        # same time, so snapshots are keyed by the slot supplied by cPerfect.
        self.save_states = {}
        # self.sokoban.reset()

    def step(self, action):
        obs, reward, done, truncated_done, info = self.sokoban.step(action)
        reward = round(reward, 2)
        return obs, reward, done, truncated_done, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if options is not None and "room_id" in options:
            return self.sokoban.reset_level(options["room_id"])   
        else:
            return self.sokoban.reset()
            
        
    def quick_save(self, slot_id=0):
        slot_id = int(slot_id)
        if slot_id < 0:
            raise ValueError(f"Snapshot slot must be non-negative, got {slot_id}.")
        self.save_states[slot_id] = self.sokoban.clone_state()

    def quick_load(self, slot_id=0):
        slot_id = int(slot_id)
        if slot_id not in self.save_states:
            raise ValueError(f"No state has been saved in slot {slot_id}.")
        self.sokoban.restore_state(self.save_states[slot_id])

    def quick_delete(self, slot_id=0):
        self.save_states.pop(int(slot_id), None)
        
    def clone_state(self):
        return self.sokoban.clone_state()

    def restore_state(self, state):
        return self.sokoban.restore_state(state)    

    def seed(self, seed): 
        self.sokoban.seed(seed)    

    @property
    def step_n(self):
        return self.sokoban.step_n

    @step_n.setter
    def step_n(self, step_n):
        self.sokoban.step_n = step_n
