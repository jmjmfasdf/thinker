import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces, logger
import gymnasium.wrappers as wrappers
from gymnasium.core import ObsType
from gymnasium.vector.utils import batch_space
from gymnasium.utils.step_api_compatibility import (
    convert_to_terminated_truncated_step_api,
)
import torch


def _call_named_slot(env, method_name, slot_id):
    """Forward snapshots without breaking legacy no-argument slot zero."""
    method = env.get_wrapper_attr(method_name)
    slot_id = int(slot_id)
    return method() if slot_id == 0 else method(slot_id)


def create_envpool(name, flags, env_n=1):
    import envpool
    kwargs = dict(
        gray_scale=flags.grayscale,
        episodic_life=True,        
        stack_num=flags.frame_stack_n,
    )
    env = EnvPoolWrap(envpool.make(name, env_type="gymnasium", num_envs=env_n, **kwargs), num_envs=env_n, **kwargs)
    return env

def create_env_fn(name, flags):
    if "Sokoban" in name:
        import gym_sokoban
        fn = gym.make
        args = {"id": name, "dan_num": flags.detect_dan_num}
    elif "vgdl" in name or "fmri" in name:
        from vgdl.interfaces.gym import VGDLEnv
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))
        vgdl_games_path = os.path.join(dir_path, '..', 'py-vgdl-master', 'vgdl', 'games')

        parts = name.split('/')
        if len(parts) == 2:
            game_dir, level_name_part = parts
            domain_name = game_dir.split('_v')[0]
            level_num = level_name_part.split('_')[-1]
            
            domain_file = os.path.join(vgdl_games_path, game_dir, f"{domain_name}.txt")
            level_file = os.path.join(vgdl_games_path, game_dir, f"{domain_name}_{level_num}.txt")
        else:
            # Fallback for single file case
            level_file_name = name
            game_name_part = '_'.join(level_file_name.split('_')[:-1])
            domain_file = os.path.join(vgdl_games_path, f"{game_name_part}.txt")
            level_file = os.path.join(vgdl_games_path, level_file_name)

        fn = VGDLEnv
        args = {
            'game_file': domain_file,
            'level_file': level_file,
            'obs_type': 'image',
        }
    else:
        fn = gym.make
        args = {"id": name}

    def pre_wrap(env, name, flags):
        if "Sokoban" in name:
            return TransposeWrap(env)
        elif "atari" in name:
            return atari_wrap(env, flags.grayscale, flags.frame_stack_n)
        elif "vgdl" in name or "fmri" in name:
            env = ResizeAndPadWrapper(env)
            return TransposeWrap(env)
        return env

    env_fn = lambda: pre_wrap(
        fn(**args), 
        name=name, 
        flags=flags,
    )
    return env_fn


class ResizeAndPadWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=84, pad_color=(89, 89, 89)):
        super().__init__(env)
        self.size = size
        self.pad_color = pad_color
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

    def observation(self, obs):
        import cv2
        h, w, _ = obs.shape

        if h == w:
            # Already square, just resize
            resized_obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        else:
            # Pad to square
            top, bottom, left, right = 0, 0, 0, 0
            if h > w:
                diff = h - w
                left = diff // 2
                right = diff - left
            else: # w > h
                diff = w - h
                top = diff // 2
                bottom = diff - top
            
            padded_obs = cv2.copyMakeBorder(
                obs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_color
            )
            resized_obs = cv2.resize(padded_obs, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return resized_obs

       
def atari_wrap(env, grayscale=True, frame_stack_n=4, expose_ram=False):    
    env = AtariSaveLoad(env, expose_ram=expose_ram)
    env = TimeLimitExtended(env, max_episode_steps=108000)
    env = AtariPreprocessingExtended(
        env, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=84, 
        terminal_on_life_loss=True, 
        grayscale_obs=grayscale, 
        grayscale_newaxis=True, 
        scale_obs=False
        )
    env = TransposeWrap(env)
    env = FrameStackExtended(env, num_stack=frame_stack_n)
    env = SqueezeWrap(env)
    return env

class AtariSaveLoad(gym.Wrapper):
    def __init__(self, env, expose_ram=False):
        gym.Wrapper.__init__(self, env)
        self.save_states = {}
        self.expose_ram = expose_ram

    def quick_save(self, slot_id=0):
        self.save_states[int(slot_id)] = self.env.unwrapped.clone_state()

    def quick_load(self, slot_id=0):
        slot_id = int(slot_id)
        if slot_id not in self.save_states:
            raise ValueError(f"No state has been saved in slot {slot_id}.")
        
        self.env.unwrapped.restore_state(self.save_states[slot_id])

    def quick_delete(self, slot_id=0):
        self.save_states.pop(int(slot_id), None)

    def reset(self, *args, **kwargs):
        observation, info = super().reset(*args, **kwargs)
        if self.expose_ram:
            info["ram"] = self.env.ale.getRAM()
        return observation, info
    
    def step(self, action, *args, **kwargs):
        observation, total_reward, terminated, truncated, info = super().step(action, *args, **kwargs)        
        if self.expose_ram:
            info["ram"] = self.env.ale.getRAM()
        return observation, total_reward, terminated, truncated, info

class AtariPreprocessingExtended(wrappers.AtariPreprocessing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_states = {}

    def reset(self, *args, **kwargs):
        observation, info = super().reset(*args, **kwargs)        
        info['real_done'] = False
        return observation, info

    def step(self, action, *args, **kwargs):
        observation, total_reward, terminated, truncated, info = super().step(action, *args, **kwargs)        
        info['real_done'] = (self.lives == 0) | truncated
        return observation, total_reward, terminated, truncated, info

    def quick_save(self, slot_id=0):
        """Save the current state of the wrapper."""
        slot_id = int(slot_id)
        self.save_states[slot_id] = {
            'lives': self.lives,
            'game_over': self.game_over,
        }
        _call_named_slot(self.env, 'quick_save', slot_id)

    def quick_load(self, slot_id=0):
        """Load the previously saved state of the wrapper."""
        slot_id = int(slot_id)
        if slot_id not in self.save_states:
            raise ValueError(f"No state has been saved in slot {slot_id}.")
        
        self.lives = self.save_states[slot_id]['lives']
        self.game_over = self.save_states[slot_id]['game_over']
        
        _call_named_slot(self.env, 'quick_load', slot_id)

    def quick_delete(self, slot_id=0):
        slot_id = int(slot_id)
        self.save_states.pop(slot_id, None)
        _call_named_slot(self.env, 'quick_delete', slot_id)

class TimeLimitExtended(wrappers.TimeLimit):
    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env, max_episode_steps)
        self.save_states = {}

    def quick_save(self, slot_id=0):
        """Save the current state of the wrapper."""
        slot_id = int(slot_id)
        self.save_states[slot_id] = {
            '_elapsed_steps': self._elapsed_steps,
        }
        _call_named_slot(self.env, 'quick_save', slot_id)

    def quick_load(self, slot_id=0):
        """Load the previously saved state of the wrapper."""
        slot_id = int(slot_id)
        if slot_id not in self.save_states:
            raise ValueError(f"No state has been saved in slot {slot_id}.")
        
        self._elapsed_steps = self.save_states[slot_id]['_elapsed_steps']
        
        _call_named_slot(self.env, 'quick_load', slot_id)

    def quick_delete(self, slot_id=0):
        slot_id = int(slot_id)
        self.save_states.pop(slot_id, None)
        _call_named_slot(self.env, 'quick_delete', slot_id)

class FrameStackExtended(wrappers.FrameStack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_states = {}
        self.frame_stack_n = kwargs["num_stack"]

    def quick_save(self, slot_id=0):
        """Save the current state of the wrapper."""
        slot_id = int(slot_id)
        self.save_states[slot_id] = {
            'frames': list(self.frames),
        }
        _call_named_slot(self.env, 'quick_save', slot_id)

    def quick_load(self, slot_id=0):
        """Load the previously saved state of the wrapper."""
        slot_id = int(slot_id)
        if slot_id not in self.save_states:
            raise ValueError(f"No state has been saved in slot {slot_id}.")
        self.frames = deque(self.save_states[slot_id]['frames'], maxlen=self.num_stack)
        _call_named_slot(self.env, 'quick_load', slot_id)

    def quick_delete(self, slot_id=0):
        slot_id = int(slot_id)
        self.save_states.pop(slot_id, None)
        _call_named_slot(self.env, 'quick_delete', slot_id)

class TransposeWrap(gym.ObservationWrapper):
    """Image shape to channels x weight x height"""

    def __init__(self, env):
        super(TransposeWrap, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.transpose(2, 0, 1),
            high=self.observation_space.high.transpose(2, 0, 1),
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))
    
    def quick_save(self, slot_id=0):
        _call_named_slot(self.env, 'quick_save', slot_id)

    def quick_load(self, slot_id=0):
        _call_named_slot(self.env, 'quick_load', slot_id)

    def quick_delete(self, slot_id=0):
        _call_named_slot(self.env, 'quick_delete', slot_id)
    
class SqueezeWrap(gym.ObservationWrapper):
    """Wrapper that squeezes the first two dimensions of the observation."""

    def __init__(self, env):
        super(SqueezeWrap, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[0] * old_shape[1], *old_shape[2:])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.reshape(new_shape),
            high=self.observation_space.high.reshape(new_shape),
            shape=new_shape,
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        if isinstance(observation, wrappers.LazyFrames):
            observation = np.array(observation)
        return observation.reshape(self.observation_space.shape)
    
    def quick_save(self, slot_id=0):
        _call_named_slot(self.env, 'quick_save', slot_id)

    def quick_load(self, slot_id=0):
        _call_named_slot(self.env, 'quick_load', slot_id)

    def quick_delete(self, slot_id=0):
        _call_named_slot(self.env, 'quick_delete', slot_id)

# the following are all vectorized wrapper

class WrapperExtended(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def quick_save(self, slot_id=0):
        _call_named_slot(self.env, 'quick_save', slot_id)

    def quick_load(self, slot_id=0):
        _call_named_slot(self.env, 'quick_load', slot_id)

    def quick_delete(self, slot_id=0):
        _call_named_slot(self.env, 'quick_delete', slot_id)

    def load_ckp(self, data):
        return self.env.get_wrapper_attr('load_ckp')(data) 
    
    def save_ckp(self):
        return self.env.get_wrapper_attr('save_ckp')() 
    
class VectorWrap(WrapperExtended):
    def __init__(self, env, flags):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)        
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_step = np.zeros(self.num_envs, dtype=np.int64)
        self.obs_clip = flags.obs_clip
        self.reward_clip = flags.reward_clip   
        if flags.obs_norm or flags.reward_norm:
            raise NotImplemented()
        
        self.save_states = {}
        self.keys_to_keep = ["real_done", "cost"] # all other info will be discarded for efficiency

    def reset(self, *args, **kwargs):
        env_id = kwargs.get("env_id", None)     
        reset_stat = kwargs.pop("reset_stat", False)
        if reset_stat:
            if env_id is None:
                self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
                self.episode_step = np.zeros(self.num_envs, dtype=np.int64)
            else:
                self.episode_return[env_id] = 0.
                self.episode_step[env_id] = 0        
        observation, info = self.env.reset(*args, **kwargs)

        info = {key: info[key] for key in self.keys_to_keep if key in info}
        info["real_done"] = np.zeros(self.num_envs if env_id is None else len(env_id), dtype=bool) 
        info["episode_return"] = self.episode_return[env_id] if env_id is not None else self.episode_return
        info["episode_step"] = self.episode_step[env_id] if env_id is not None else self.episode_step
        return observation, info

    def step(self, action, *args, **kwargs):      
        env_id = kwargs.get("env_id", None)     
        observation, reward, terminated, truncated, info = self.env.step(action, *args, **kwargs)

        # 디버깅: 에피소드 종료 상태 추적
        if np.any(terminated) or np.any(truncated):
            print(f"[DEBUG] VectorWrap - Episode termination detected:")
            print(f"  - terminated: {terminated}")
            print(f"  - truncated: {truncated}")
            print(f"  - info keys: {list(info.keys())}")
            if 'real_done' in info:
                print(f"  - info['real_done']: {info['real_done']}")

        # real_done 설정 개선
        if "real_done" not in info: 
            info["real_done"] = terminated | truncated
        real_done = info["real_done"]
        
        # 디버깅: real_done 설정 후 확인
        if np.any(terminated) or np.any(truncated):
            print(f"[DEBUG] VectorWrap - After setting real_done:")
            print(f"  - info['real_done']: {info['real_done']}")
            print(f"  - terminated | truncated: {terminated | truncated}")
        
        # 에피소드 통계 업데이트
        if env_id is None:
            self.episode_return = self.episode_return + reward.astype(np.float32)
            self.episode_step = self.episode_step + 1
        else:
            self.episode_return[env_id] = self.episode_return[env_id] + reward.astype(np.float32)
            self.episode_step[env_id] = self.episode_step[env_id] + 1
        episode_return = self.episode_return
        episode_step = self.episode_step

        # 에피소드 종료 시 통계 리셋
        if np.any(real_done):
            episode_return = np.copy(episode_return)
            episode_step = np.copy(episode_step)

            if env_id is None:    
                self.episode_return[real_done] = 0.
                self.episode_step[real_done] = 0
            else:
                idx_b = np.zeros(self.num_envs, np.bool_)
                idx_b[env_id] = real_done
                self.episode_return[idx_b] = 0.
                self.episode_step[idx_b] = 0

        # Observation과 reward clipping
        if self.obs_clip > 0.:
            observation = np.clip(observation, -self.obs_clip, +self.obs_clip)
        if self.reward_clip > 0.:
            reward = np.clip(reward, -self.reward_clip, +self.reward_clip)

        # Info 정리
        info = {key: info[key] for key in self.keys_to_keep if key in info}
        info["episode_return"] = episode_return[env_id] if env_id is not None else episode_return
        info["episode_step"] = episode_step[env_id] if env_id is not None else episode_step

        return observation, reward, terminated, truncated, info
    
    def _ensure_stat_slot(self, slot_id):
        slot_id = int(slot_id)
        if slot_id < 0:
            raise ValueError("slot_id must be non-negative")
        if slot_id not in self.save_states:
            self.save_states[slot_id] = {
                "episode_return": np.zeros(self.num_envs, dtype=np.float32),
                "episode_step": np.zeros(self.num_envs, dtype=np.int64),
                "valid": np.zeros(self.num_envs, dtype=np.bool_),
            }
        return self.save_states[slot_id]

    def _normalize_snapshot_env_ids(self, env_id):
        if env_id is None:
            return list(range(self.num_envs))
        env_ids = [int(idx) for idx in list(env_id)]
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("env_id must not contain duplicates")
        if any(idx < 0 or idx >= self.num_envs for idx in env_ids):
            raise ValueError(f"env_id must be in [0, {self.num_envs - 1}]")
        return env_ids

    def _normalize_snapshot_pairs(self, env_id, slot_ids):
        env_ids = self._normalize_snapshot_env_ids(env_id)
        slot_ids = [int(slot_id) for slot_id in list(slot_ids)]
        if len(env_ids) != len(slot_ids):
            raise ValueError("env_id and slot_ids must have the same length")
        if any(slot_id < 0 for slot_id in slot_ids):
            raise ValueError("slot_ids must be non-negative")
        return env_ids, slot_ids

    def _require_stat_slot(self, slot_id, env_ids):
        slot_id = int(slot_id)
        slot = self.save_states.get(slot_id)
        missing = env_ids if slot is None else [
            idx for idx in env_ids if not slot["valid"][idx]
        ]
        if missing:
            raise ValueError(
                f"No vector state has been saved for env_id {missing} "
                f"in slot {slot_id}."
            )
        return slot

    def _snapshot_each(self, operation, env_ids, slot_ids):
        method = getattr(self.env, f"quick_{operation}_each", None)
        if method is not None:
            return method(env_id=env_ids, slot_ids=slot_ids)
        method = getattr(self.env, f"quick_{operation}")
        results = []
        for idx, slot_id in zip(env_ids, slot_ids):
            kwargs = {"env_id": [idx]}
            if slot_id != 0 or operation == "delete":
                kwargs["slot_id"] = slot_id
            results.append(method(**kwargs))
        return results

    def quick_save(self, env_id=None, slot_id=0):
        env_id = self._normalize_snapshot_env_ids(env_id)
        slot_id = int(slot_id)
        if slot_id < 0:
            raise ValueError("slot_id must be non-negative")
        kwargs = {"env_id": env_id}
        if slot_id != 0:
            kwargs["slot_id"] = slot_id
        self.env.quick_save(**kwargs)
        slot = self._ensure_stat_slot(slot_id)
        slot["episode_return"][env_id] = self.episode_return[env_id]
        slot["episode_step"][env_id] = self.episode_step[env_id]
        slot["valid"][env_id] = True

    def quick_load(self, env_id=None, slot_id=0):
        env_id = self._normalize_snapshot_env_ids(env_id)
        slot_id = int(slot_id)
        slot = self._require_stat_slot(slot_id, env_id)
        kwargs = {"env_id": env_id}
        if slot_id != 0:
            kwargs["slot_id"] = slot_id
        self.env.quick_load(**kwargs)
        self.episode_return[env_id] = slot["episode_return"][env_id]
        self.episode_step[env_id] = slot["episode_step"][env_id]

    def quick_delete(self, env_id=None, slot_id=0):
        env_id = self._normalize_snapshot_env_ids(env_id)
        slot_id = int(slot_id)
        if slot_id < 0:
            raise ValueError("slot_id must be non-negative")
        self.env.quick_delete(env_id=env_id, slot_id=slot_id)
        slot = self.save_states.get(slot_id)
        if slot is not None:
            slot["valid"][env_id] = False
            if not np.any(slot["valid"]):
                self.save_states.pop(slot_id)

    def quick_save_slots(self, env_id, slot_ids):
        env_id, slot_ids = self._normalize_snapshot_pairs(env_id, slot_ids)
        result = self._snapshot_each("save", env_id, slot_ids)
        for idx, slot_id in zip(env_id, slot_ids):
            slot = self._ensure_stat_slot(slot_id)
            slot["episode_return"][idx] = self.episode_return[idx]
            slot["episode_step"][idx] = self.episode_step[idx]
            slot["valid"][idx] = True
        return result

    def quick_load_slots(self, env_id, slot_ids):
        env_id, slot_ids = self._normalize_snapshot_pairs(env_id, slot_ids)
        slots = [
            self._require_stat_slot(slot_id, [idx])
            for idx, slot_id in zip(env_id, slot_ids)
        ]
        result = self._snapshot_each("load", env_id, slot_ids)
        for idx, slot in zip(env_id, slots):
            self.episode_return[idx] = slot["episode_return"][idx]
            self.episode_step[idx] = slot["episode_step"][idx]
        return result

    def quick_delete_slots(self, env_id, slot_ids):
        env_id, slot_ids = self._normalize_snapshot_pairs(env_id, slot_ids)
        result = self._snapshot_each("delete", env_id, slot_ids)
        for idx, slot_id in zip(env_id, slot_ids):
            slot = self.save_states.get(slot_id)
            if slot is not None:
                slot["valid"][idx] = False
                if not np.any(slot["valid"]):
                    self.save_states.pop(slot_id)
        return result

    def load_ckp(self, data):
        return 
    
    def save_ckp(self):
        return {}


class EnvPoolWrap(WrapperExtended):    
    def __init__(self, env, num_envs, **kwargs):
        super().__init__(env)
        self.num_envs = num_envs
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.observation_space = batch_space(self.single_observation_space, n=num_envs)
        self.action_space = batch_space(self.single_action_space, n=num_envs)
        self.frame_stack_n = kwargs.pop("frame_stack_n", 1)

    def reset(self, *args, **kwargs):
        env_id = kwargs.pop("env_id", None)      
        if type(env_id) == list: env_id = np.array(env_id, np.int32)
        kwargs = dict(env_id=env_id) if env_id is not None else dict()
        observation, info = self.env.reset(**kwargs)
        return observation, info
    
    def step(self, action, *args, **kwargs):      
        env_id = kwargs.pop("env_id", None)      
        if type(env_id) == list: env_id = np.array(env_id, np.int32)
        kwargs = dict(env_id=env_id) if env_id is not None else dict()
        observation, reward, terminated, truncated, info = self.env.step(action, **kwargs)

        if env_id is None: env_id = np.arange(len(terminated))            
        assert np.all(info["env_id"] == env_id), f"Wrong env_id: {env_id} vs {info['env_id']}"

        real_done = info["terminated"] if "terminated" in info else terminated
        real_done = real_done.astype(bool) | truncated
        if np.any(real_done):
            # this is to be consistent with gymnasium - reset upon the same step as done instead of the next step            
            reset_env_id = env_id[real_done]
            new_observation, _, _, _, _ = self.env.step(action, env_id=reset_env_id)
            observation[real_done] = new_observation
        info["real_done"] = real_done
        return observation, reward, terminated, truncated, info
    
    def quick_save(self, env_id=None, slot_id=0):
        if type(env_id) == list: env_id = np.array(env_id, np.int32)
        kwargs = dict(env_id=env_id) if env_id is not None else {}
        if int(slot_id) != 0:
            kwargs["slot_id"] = int(slot_id)
        self.env.quick_save(**kwargs)

    def quick_load(self, env_id=None, slot_id=0):
        if type(env_id) == list: env_id = np.array(env_id, np.int32)
        kwargs = dict(env_id=env_id) if env_id is not None else {}
        if int(slot_id) != 0:
            kwargs["slot_id"] = int(slot_id)
        self.env.quick_load(**kwargs)

    def quick_delete(self, env_id=None, slot_id=0):
        if hasattr(self.env, "quick_delete"):
            kwargs = dict(env_id=env_id, slot_id=slot_id) if env_id is not None else dict(slot_id=slot_id)
            self.env.quick_delete(**kwargs)

class PostWrapper(WrapperExtended):
    """Final wrapper that recorrds useful statistics"""
    def __init__(self, env, flags):
        super().__init__(env)
        self.reset_called = False        
        low = torch.tensor(self.env.observation_space["real_states"].low[0])
        high = torch.tensor(self.env.observation_space["real_states"].high[0])
        self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()
        self.norm_low = low
        self.norm_high = high

        self.disable_thinker = flags.wrapper_type == 1
        self.dynamic_search = bool(getattr(flags, "dynamic_search", False))
        if not self.disable_thinker:
            self.pri_action_space = self.env.action_space[0][0]            
        else:
            self.pri_action_space = self.env.action_space[0]        
    
    def reset(self, model_net, seed=None):
        state, info = self.env.reset(model_net, seed=seed)
        self.device = state["real_states"].device
        self.env_n = state["real_states"].shape[0]

        self.episode_step = torch.zeros(
            self.env_n, dtype=torch.long, device=self.device
        )

        self.episode_return = {}
        reward_prefixes = ["im", "cur"]
        if self.dynamic_search:
            reward_prefixes.append("think")
        for key in reward_prefixes:
            self.episode_return[key] = torch.zeros(
                self.env_n, dtype=torch.float, device=self.device
            )
        self.reset_called = True
        return state, info

    def step(self, action, model_net):
        assert self.reset_called, "need to call reset ONCE before step"

        state, reward, done, truncated_done, info = self.env.step(action, model_net)
        real_done = info["real_done"]        

        reward_prefixes = ["im", "cur"]
        if self.dynamic_search:
            reward_prefixes.append("think")
        for prefix in reward_prefixes:
            if prefix+"_reward" in info:
                r = info[prefix+"_reward"]
                if prefix == "im": r = r[:, 0]
                nan_mask = ~torch.isnan(r)
                self.episode_return[prefix][nan_mask] += r[nan_mask]
                info[prefix + "_episode_return"] = self.episode_return[prefix].clone()
                self.episode_return[prefix][real_done] = 0.
                if prefix in ["im", "think"]:
                    if self.dynamic_search and "stage_end" in info:
                        self.episode_return[prefix][info["stage_end"].bool()] = 0.
                    elif prefix == "im":
                        self.episode_return[prefix][info["step_status"] == 0] = 0.
        return state, reward, done, truncated_done, info
    
    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)    
    
    def unnormalize(self, x):
        assert x.dtype == torch.float or x.dtype == torch.float32
        if self.need_norm:
            ch = x.shape[-3]
            x = torch.clamp(x, 0, 1)
            x = x * (self.norm_high[-ch:] -  self.norm_low[-ch:]) + self.norm_low[-ch:]
        return x
    
    def normalize(self, x):
        if self.need_norm:    
            if self.norm_low.device != x.device or self.norm_high.device != x.device:
                self.norm_low = self.norm_low.to(x.device)
                self.norm_high = self.norm_high.to(x.device)
            x = (x.float() - self.norm_low) / (self.norm_high -  self.norm_low)
        return x

class DummyWrapper(gym.Wrapper):
    """DummyWrapper that represents the core wrapper for the real env;
    the only function is to convert returning var into tensor
    and reset the env when it is done.
    """
    def __init__(self, env, env_n, flags, model_net, device=None, timing=False):   
        gym.Wrapper.__init__(self, env)
        self.env_n = env_n
        self.flags = flags
        self.device = torch.device("cpu") if device is None else device 
        self.observation_space = spaces.Dict({
            "real_states": self.env.observation_space,
        })        
        if env.observation_space.dtype == 'uint8':
            self.state_dtype = torch.uint8
        elif env.observation_space.dtype == 'float32':
            self.state_dtype = torch.float32
        else:
            raise Exception(f"Unupported observation sapce", env.observation_space)

        self.train_model = self.flags.train_model
        self.tuple_action = type(env.action_space) in [spaces.tuple.Tuple, spaces.Box]

    def reset(self, model_net, seed=None):
        obs, info = self.env.reset(seed=seed)
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)                
        if self.train_model: 
            self.per_state = model_net.initial_state(batch_size=self.env_n, device=self.device)
            pri_action = torch.zeros_like(torch.tensor(self.action_space.sample()), device=self.device)
            done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                model_net_out = model_net(
                    env_state=obs_py, 
                    done=done,
                    actions=pri_action.unsqueeze(0), 
                    state=self.per_state,)       
            self.per_state = model_net_out.state
            self.baseline = model_net_out.vs[-1]
        states = {"real_states": obs_py}   

        info = dict_map(info, lambda x: torch.tensor(x, device=self.device))
        info["step_status"] = torch.full((self.env_n,), fill_value=0, dtype=torch.long, device=self.device)
        info["real_states_np"] = obs
        # real_done 초기화
        info["real_done"] = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)
        if self.train_model:             
            info["initial_per_state"] = self.per_state
            info["baseline"] = self.baseline

        return states, info

    def step(self, action, model_net):  
        # action in shape (B, *) or (B,)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()        

        obs, reward, done, truncated_done, info = self.env.step(action) 
        
        # 디버깅: 에피소드 종료 상태 추적
        if np.any(done) or np.any(truncated_done):
            print(f"[DEBUG] DummyWrapper - Episode termination detected:")
            print(f"  - done: {done}")
            print(f"  - truncated_done: {truncated_done}")
            print(f"  - info keys: {list(info.keys())}")
            if 'real_done' in info:
                print(f"  - info['real_done']: {info['real_done']}")
        
        # 에피소드가 종료되면 자동으로 리셋
        if np.any(done):
            done_idx = np.arange(self.env_n)[done]
            obs_reset, _ = self.env.reset(seed=None)
            obs[done] = obs_reset
        
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)        
        truncated_done = torch.tensor(truncated_done, dtype=torch.bool, device=self.device)        
        states = {
            "real_states": obs_py,
        }     

        info = dict_map(info, lambda x: torch.tensor(x, device=self.device))
        info["step_status"] = torch.full((self.env_n,), fill_value=3, dtype=torch.long, device=self.device)
        info["real_states_np"] = obs
        
        # real_done 정보 설정 - 에피소드 종료 조건
        info["real_done"] = done | truncated_done
        
        # 디버깅: real_done 설정 후 확인
        if torch.any(done) or torch.any(truncated_done):
            print(f"[DEBUG] DummyWrapper - After setting real_done:")
            print(f"  - info['real_done']: {info['real_done']}")
            print(f"  - done | truncated_done: {done | truncated_done}")
        
        if self.train_model:             
            info["initial_per_state"] = self.per_state
            info["baseline"] = self.baseline
            pri_action = torch.tensor(action, dtype=torch.long, device=self.device)
            if not self.tuple_action: pri_action = pri_action.unsqueeze(-1)          
            with torch.no_grad():
                model_net_out = model_net(
                    env_state=obs_py, 
                    done=done,
                    actions=pri_action.unsqueeze(0), 
                    state=self.per_state,)       
                self.per_state = model_net_out.state
                self.baseline = model_net_out.vs[-1]
        
        return states, reward, done, truncated_done, info
    
def dict_map(x, f):
    return {k:f(v) if v is not None else None for (k, v) in x.items()}    
