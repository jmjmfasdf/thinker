import os
from collections import OrderedDict
import gymnasium as gym
from gymnasium import spaces
import vgdl
from vgdl.state import StateObserver
import numpy as np
from .list_space import list_space
import random

class VGDLEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 25
    }

    def __init__(self,
                 game_file = None,
                 level_file = None,
                 obs_type='image',
                 max_episode_steps=1000,
                 **kwargs):
        # For rendering purposes only
        self.render_block_size = kwargs.pop('block_size', 24)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # Variables
        self._obs_type = obs_type
        self.viewer = None
        self.game_args = kwargs
        self.notable_sprites = kwargs.get('notable_sprites', None)
        
        # GVGAI처럼 명시적으로 state 저장
        self.img = None

        # Load game description and level description
        if game_file is not None:
            with open (game_file, "r") as myfile:
                self.game_desc = myfile.read()
            with open (level_file, "r") as myfile:
                self.level_desc = myfile.read()
            self.level_name = os.path.basename(level_file).split('.')[0]
            self.loadGame(self.game_desc, self.level_desc)


    def loadGame(self, game_desc, level_desc, **kwargs):

        self.game_desc = game_desc
        self.level_desc = level_desc
        self.game_args.update(kwargs)

        # Need to build a sample level to get the available actions and screensize....
        self.domain = vgdl.VGDLParser().parse_game(self.game_desc, **self.game_args)
        self.game = self.domain.build_level(self.level_desc)

        self.score_last = self.game.score

        # Set action space and observation space
        self._action_set = OrderedDict(self.game.get_possible_actions())
        self.action_space = spaces.Discrete(len(self._action_set))

        self.screen_width, self.screen_height = self.game.screensize

        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255,
                    shape=(self.screen_height, self.screen_width, 3) )
        elif self._obs_type == 'objects':
            from .state import NotableSpritesObserver
            self.observer = NotableSpritesObserver(self.game, self.notable_sprites)
            self.observation_space = list_space( spaces.Box(low=-100, high=100,
                    shape=self.observer.observation_shape) )
        elif self._obs_type == 'features':
            from .state import AvatarOrientedObserver
            self.observer = AvatarOrientedObserver(self.game)
            self.observation_space = spaces.Box(low=0, high=100,
                    shape=self.observer.observation_shape)
        # elif isinstance(self._obs_type, type) and issubclass(self._obs_type, StateObserver):
        else:
            try:
                self.observer = self._obs_type(self.game)
                self.observation_space = spaces.Box(low=0, high=100,
                                            shape=self.observer.observation_shape)
            except:
                raise Exception('Unknown obs_type `{}`'.format(self._obs_type))

        # For rendering purposes, will be initialised by first `render` call
        self.renderer = None

        if self._obs_type == 'image':
            # Force a render to initialize the screen and get the true shape
            initial_render = self.render(mode='rgb_array')
            h, w, c = initial_render.shape
            
            # Re-define the observation space with the correct shape
            self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)



    @property
    def _n_actions(self):
        return len(self._action_set)

    @property
    def _action_keys(self):
        return list(self._action_set.values())

    def get_action_meanings(self):
        # In the spirit of the Atari environment, describe actions with strings
        return list(self._action_set.keys())

    def _get_obs(self):
        if self._obs_type == 'image':
            if self.renderer is None:
                self.render(mode='rgb_array')
            return self.renderer.get_image()
        else:
            observation = self.observer.get_observation()
            if hasattr(observation, 'as_array'):
                return observation.as_array()
            return observation

    def step(self, a):
        # 게임 상태만 업데이트
        self.game.tick(self._action_keys[a])
        
        # 렌더링하여 이미지 저장 (GVGAI 방식)
        if self._obs_type == 'image':
            if self.renderer is None:
                from vgdl.render import PygameRenderer
                self.renderer = PygameRenderer(self.game, self.render_block_size)
                self.renderer.init_screen(headless=True)
            # 렌더링하여 이미지 저장
            self.renderer.draw_all()
            self.img = self.renderer.get_image()
            state = self.img
        else:
            state = self._get_obs()
        
        reward = self.game.score - self.score_last
        self.score_last = self.game.score
        terminated = self.game.ended
        
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps

        return state, reward, terminated, truncated, {}

    def _randomize_level(self, level_desc):
        print("=== 셔플 전 레벨 ===")
        print(level_desc)
        print("==================")
        
        lines = level_desc.strip().split('\n')
        inner_cells = []
        # 테두리를 제외한 내부 셀 추출
        for r, row in enumerate(lines[1:-1]):
            for c, char in enumerate(row[1:-1]):
                inner_cells.append(char)
        
        # 내부 셀 섞기
        random.shuffle(inner_cells)
        
        # 새 레벨 생성
        new_level_lines = [list(lines[0])]
        cell_idx = 0
        for r, row in enumerate(lines[1:-1]):
            new_row = ['w']
            for c, char in enumerate(row[1:-1]):
                new_row.append(inner_cells[cell_idx])
                cell_idx += 1
            new_row.append('w')
            new_level_lines.append(new_row)
        new_level_lines.append(list(lines[-1]))
        
        randomized_level = '\n'.join(''.join(row) for row in new_level_lines)
        
        print("=== 셔플 후 레벨 ===")
        print(randomized_level)
        print("==================")
        
        return randomized_level

    def reset(self, seed=None, options=None):
        # Reset the game state, not the entire game object
        if self.game is not None:
            # Randomize the level description before resetting the game
            randomized_level_desc = self._randomize_level(self.level_desc)
            self.game = self.domain.build_level(randomized_level_desc)
            # Reset renderer to reflect the new level
            self.renderer = None
            # self.game.reset()
        else:
            # First time reset, build the level
            domain = vgdl.VGDLParser().parse_game(self.game_desc, **self.game_args)
            self.game = domain.build_level(self.level_desc)

        self._elapsed_steps = 0
        self.score_last = self.game.score
        
        # 초기 이미지 렌더링 및 저장
        if self._obs_type == 'image':
            if self.renderer is None:
                from vgdl.render import PygameRenderer
                self.renderer = PygameRenderer(self.game, self.render_block_size)
                self.renderer.init_screen(headless=True)
            self.renderer.draw_all()
            self.img = self.renderer.get_image()
            state = self.img
        else:
            state = self._get_obs()
            
        return state, {}

    def render(self, mode='human', close=False):
        headless = mode != 'human'

        if self.renderer is None:
            from vgdl.render import PygameRenderer
            self.renderer = PygameRenderer(self.game, self.render_block_size)
            self.renderer.init_screen(headless)

        # 저장된 이미지가 있으면 반환 (GVGAI 방식)
        if mode == 'rgb_array' and self.img is not None:
            return self.img

        # 필요할 때만 렌더링 수행
        self.renderer.draw_all()
        self.renderer.update_display()

        if close:
            self.renderer.close()
        if mode == 'rgb_array':
            img = self.renderer.get_image()
            return img
        elif mode == 'human':
            return True

    def close(self):
        if self.renderer:
            self.renderer.close()



class Padlist(gym.ObservationWrapper):
    def __init__(self, env=None, max_objs=200):
        self.max_objects = max_objs
        super(Padlist, self).__init__(env)
        env_shape = self.observation_space.shape
        env_shape[0] = self.max_objects
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=env_shape)

    def _observation(self, obs):
        return Padlist.process(obs, self.max_objects)

    @staticmethod
    def process(input_list, to_len):
        max_len = to_len
        item_len = len(input_list)
        if item_len < max_len:
          padded = np.pad(
              np.array(input_list,dtype=np.float32),
              ((0,max_len-item_len),(0,0)),
              mode='constant')
          return padded
        else:
          return np.array(input_list, dtype=np.float32)[:max_len]
