import os
import shutil
import time
from collections import namedtuple
import ray
import numpy as np
import torch
import gymnasium as gym

import thinker.util as util
from thinker.buffer import ModelBuffer, SModelBuffer, GeneralBuffer
from thinker.learn_model import ModelLearner, SModelLearner
from thinker.model_net import ModelNet

from thinker.gym_add.asyn_vector_env import AsyncVectorEnv
import thinker.gym_add.wrapper as wrapper
from thinker.cenv import cModelWrapper, cPerfectWrapper

def ray_init(flags=None, **kwargs):
    # initialize resources for Thinker wrapper
    if flags is None:
        flags = util.create_flags(filename='default_thinker.yaml',
                              **kwargs)
        flags.parallel=True

    if not ray.is_initialized(): 
        object_store_memory = int(flags.ray_mem * 1024**3) if flags.ray_mem > 0 else None
        ray.init(num_cpus=flags.ray_cpu if flags.ray_cpu > 0 else None,
                 num_gpus=flags.ray_gpu if flags.ray_gpu > 0 else None,
                 object_store_memory=object_store_memory)
    model_buffer = ModelBuffer.options(num_cpus=1).remote(
            buffer_n = flags.model_buffer_n,
            max_rank = flags.self_play_n,
            batch_size = flags.env_n,
            alpha = flags.priority_alpha,
            warm_up_n = flags.model_warm_up_n,
    )
    param_buffer = GeneralBuffer.options(num_cpus=1).remote()    
    param_buffer.set_data.remote("flags", flags)
    signal_buffer = GeneralBuffer.options(num_cpus=1).remote()   
    ray_obj = {"model_buffer": model_buffer,
               "param_buffer": param_buffer,
               "signal_buffer": signal_buffer}
    return ray_obj

class Env(gym.Wrapper):
    def __init__(self, 
                 name=None, 
                 env_fn=None, 
                 ray_obj=None, 
                 env_n=1, 
                 gpu=True,
                 load_net=True, 
                 timing=False,
                 core_wrapper=None,
                 **kwargs):
        assert name is not None or env_fn is not None, \
            "need either env or env-making function"        
        
        if ray_obj is None:
            self.flags = util.create_flags(filename='default_thinker.yaml',
                              **kwargs)
            if self.flags.parallel:
                ray_obj = ray_init(self.flags)       
        else:
            assert not kwargs, "Unexpected keyword arguments provided"
            self.flags = ray.get(ray_obj["param_buffer"].get_data.remote("flags"))
        
        self._logger = util.logger() 
        self.parallel = self.flags.parallel
                
        self.env_n = env_n
        self.device = torch.device("cuda") if gpu else torch.device("cpu")
        
        if self.parallel:
            self.model_buffer = ray_obj["model_buffer"]
            self.param_buffer = ray_obj["param_buffer"]
            self.signal_buffer = ray_obj["signal_buffer"]
            self.rank = ray.get(self.param_buffer.get_and_increment.remote("rank"))
        else:
            self.rank = 0
        self.counter = 0
        self.ckp_start_time = int(time.strftime("%M")) // 10
        self.ckp_env_path = os.path.join(self.flags.ckpdir, "ckp_env.npz")    

        self._logger.info(
            "Initializing env %d with device %s"
            % (
                self.rank,
                "cuda" if self.device == torch.device("cuda") else "cpu",
            )
        )
        if self.flags.envpool:
            env = wrapper.create_envpool(name, self.flags, env_n)
        else:
            if env_fn is None: 
                env_fn = wrapper.create_env_fn(name, self.flags)
            # initialize a single env to collect env information            
            env = AsyncVectorEnv([env_fn for _ in range(env_n)]) 
        env = wrapper.VectorWrap(env, self.flags)

        
        self.real_state_space  = env.get_wrapper_attr('single_observation_space')
        self.real_state_shape = self.real_state_space.shape

        assert len(self.real_state_shape) in [1, 3], \
            f"env.observation_space should be 1d or 3d, not {self.real_state_shape}"
        # assert type(env.action_space) in [gym.spaces.discrete.Discrete, gym.spaces.tuple.Tuple], \
        #    f"env.action_space should be Discrete or Tuple, not {type(env.action_space)}"  
        
        if self.real_state_space.dtype == 'uint8':
            self.state_dtype = 0
        elif self.real_state_space.dtype == 'float32':
            self.state_dtype = 1                

        self.pri_action_space = env.get_wrapper_attr('single_action_space')
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(self.pri_action_space)

        if isinstance(self.pri_action_space, gym.spaces.Box):
            assert len(self.pri_action_space) == 1, f"Invalid action space {self.pri_action_space}"

        self._logger.info(f"Init. environment with obs space \033[91m{self.real_state_space}\033[0m and action space \033[91m{self.pri_action_space}\033[0m")        

        if self.tuple_action:
            self.pri_action_shape = (self.env_n, self.dim_actions)
            if self.discrete_action:
                self.action_prob_shape = self.pri_action_shape + (self.num_actions,)
            else:
                self.action_prob_shape = self.pri_action_shape + (2,) # mean and var of Gaussian dist
        else:
            self.pri_action_shape = (self.env_n,)
            self.action_prob_shape = (self.env_n, self.num_actions,)

        try:
            self.frame_stack_n = env.get_wrapper_attr('frame_stack_n')
        except AttributeError as e:
            self.frame_stack_n = 1        
        if self.rank == 0 and self.frame_stack_n > 1:
            self._logger.info("Detected frame stacking with %d counts" % self.frame_stack_n)
        
        self.frame_ch = env.observation_space.shape[0] // self.frame_stack_n
        self.model_mem_unroll_len = self.flags.model_mem_unroll_len
        self.pre_len = self.frame_stack_n - 1 + self.model_mem_unroll_len
        self.post_len = self.flags.model_unroll_len + self.flags.model_return_n + 1

        # initalize model
        self.has_model = self.flags.has_model
        self.train_model = self.has_model and self.flags.train_model 
        self.require_prob = False
        if self.has_model:
            model_param = {
                "obs_space": self.real_state_space,                
                "action_space": self.pri_action_space, 
                "flags": self.flags,
                "frame_stack_n": self.frame_stack_n
            }
            self.model_net = ModelNet(**model_param)
            if self.rank == 0:
                self._logger.info(
                    "Model network size: %d"
                    % sum(p.numel() for p in self.model_net.parameters())
                )
            if load_net: self._load_net()            
            self.model_net.train(False)
            self.model_net.to(self.device)       
            if self.train_model and self.rank == 0:
                if self.parallel:
                    # init. the model learner thread
                    self.model_learner = ModelLearner.options(
                        num_cpus=1, num_gpus=self.flags.gpu_learn,
                    ).remote(name, ray_obj, model_param, self.flags)
                    # start learning
                    self.r_learner = self.model_learner.learn_data.remote()
                    self.model_buffer.set_frame_stack_n.remote(self.frame_stack_n)
                else:
                    self.model_learner = SModelLearner(name=name, ray_obj=None, model_param=model_param,
                        flags=self.flags, model_net=self.model_net, device=self.device)
                    self.model_buffer = SModelBuffer(
                        buffer_n = self.flags.model_buffer_n,
                        max_rank = self.flags.self_play_n,
                        batch_size = self.flags.env_n,
                        alpha = self.flags.priority_alpha,
                        warm_up_n = self.flags.model_warm_up_n,                        
                    )
                    self.model_buffer.set_frame_stack_n(self.frame_stack_n)
            if self.train_model: self.require_prob = self.flags.require_prob
            
            per_state = self.model_net.initial_state(batch_size=1)
            self.per_state_shape = {k:v.shape[1:] for k, v in per_state.items()}
        else:
            self.model_net = None            
        
        self.env_seed =  list(range(
            self.rank * env_n + self.flags.base_seed, 
            self.rank * env_n + self.flags.base_seed + env_n
        ))        

        if core_wrapper is None:
            if self.flags.wrapper_type == 0:
                core_wrapper = cModelWrapper
            elif self.flags.wrapper_type == 1:
                core_wrapper = wrapper.DummyWrapper
            elif self.flags.wrapper_type == 2:
                core_wrapper = cPerfectWrapper
            else:
                raise Exception(
                    f"wrapper_type can only be [0, 1, 2], not {self.flags.wrapper_type}")

        # wrap the env with core Cython wrapper that runs
        # the core Thinker algorithm
        env = core_wrapper(
            env=env, 
            env_n=env_n, 
            flags=self.flags, 
            model_net=self.model_net, 
            device=self.device, 
            timing=timing
        )        

        if self.flags.ckp and os.path.exists(self.ckp_env_path):
            with np.load(self.ckp_env_path, allow_pickle=True) as data:
                env.get_wrapper_attr('load_ckp')(data)                    

        env = wrapper.PostWrapper(env, self.flags) 
        self.tree_rep_meaning = None

        gym.Wrapper.__init__(self, env)                          

        if self.train_model:
            if self.flags.parallel:
                self.status_ptr = self.model_buffer.get_status.remote()        
                self._update_status()
                self.signal_ptr = self.signal_buffer.get_data.remote("self_play_signals")
            else:
                self.status = self.model_buffer.get_status()
        else:
            self.status = {"processed_n": 0,
                           "warm_up_n": 0,
                           "running": False,
                           "finish": True,
                            }

        
    def _load_net(self):
        if self.rank == 0:
            # load the network from preload or load_checkpoint  
            path = None
            if self.flags.ckp:
                path = os.path.join(self.flags.ckpdir, "ckp_model.tar")
            else:
                if self.flags.preload:
                    path = os.path.join(self.flags.preload, "ckp_model.tar")
                    shutil.copyfile(path, os.path.join(self.flags.ckpdir, "ckp_model.tar"))
            if path is not None:                
                checkpoint = torch.load(
                    path, map_location=torch.device("cpu")
                )
                self.model_net.set_weights(
                    checkpoint["model_net_state_dict"]
                )
                self._logger.info("Loaded model net from %s" % path)
            
            if self.has_model and self.parallel:
                self.param_buffer.set_data.remote(
                    "model_net", self.model_net.get_weights()
                )
        else:
            self._refresh_net()
        return
    
    def _refresh_net(self):
        while True:
            weights = ray.get(
                self.param_buffer.get_data.remote("model_net")
            )  
            if weights is not None:
                self.model_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  
    
    def _update_status(self):
        self.status = ray.get(self.status_ptr)
        self.status_ptr = self.model_buffer.get_status.remote()        

    def reset(self, seed=None):
        if seed is None: seed = self.env_seed
        state = self.env.reset(self.model_net, seed=seed)
        return state

    def step(self, primary_action, reset_action=None, action_prob=None, ignore=False):        

        assert primary_action.shape == self.pri_action_shape, \
                    f"primary_action should have shape {self.pri_action_shape} not {primary_action.shape}"  
        if self.flags.wrapper_type == 1:
            action = primary_action                
        else:
            assert reset_action.shape == (self.env_n,), \
                    f"reset should have shape {(self.env_n,)} not {reset_action.shape}"
            action = (primary_action, reset_action)            
                
        if self.require_prob and not ignore: 
            assert action_prob is not None
            assert action_prob.shape == self.action_prob_shape, \
                    f"action_prob should have shape {self.action_prob_shape} not {action_prob.shape}"
        
        with torch.set_grad_enabled(False):
            state, reward, done, truncated_done, info = self.env.step(action, self.model_net)  
        last_step_real = (info["step_status"] == 0) | (info["step_status"] == 3)
        
        # real_done 정보가 없으면 설정
        if "real_done" not in info:
            info["real_done"] = done | truncated_done
        
        # 디버깅: 에피소드 종료 상태 추적
        if torch.any(done) or torch.any(truncated_done):
            real_done = info.get("real_done", done | truncated_done)
            self._logger.info(f"[DEBUG] Episode termination detected:")
            self._logger.info(f"  - done: {done}")
            self._logger.info(f"  - truncated_done: {truncated_done}")
            self._logger.info(f"  - real_done: {real_done}")
            self._logger.info(f"  - step_status: {info.get('step_status', 'N/A')}")
            self._logger.info(f"  - counter: {self.counter}")
        
        if self.train_model and not ignore and torch.any(last_step_real): 
            self._write_send_model_buffer(state, reward, done, truncated_done, info, primary_action, action_prob)        
        if self.train_model:
            if self.parallel:
                if self.counter % 200 == 0: self._refresh_wait()     
            else:
                self.status = self._train_model()
            if self.status["finish"]:                 
                if self.rank == 0 and self.train_model: 
                    self._logger.info("Finish training model")
                self.train_model = False   

        if self.rank == 0 and int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_ckp()
            self.ckp_start_time = int(time.strftime("%M")) // 10
        
        info["model_status"] = self.status
        self.counter += 1
        return state, reward, done, truncated_done, info      

    def _write_send_model_buffer(self, state, reward, done, truncated_done, info, primary_action, action_prob):
        real_step_mask = (info["step_status"] == 0) | (info["step_status"] == 3)
        reward = reward.unsqueeze(-1)
        step_times = info.get("step_times", None)
        if step_times is not None:
            if isinstance(step_times, torch.Tensor):
                st = step_times
            else:
                st = torch.tensor(step_times)
            st = st[real_step_mask]
            step_times_np = st.detach().cpu().numpy().astype(np.float32)
        else:
            # keep key shape consistent even if timing is disabled
            b = int(real_step_mask.sum().item())
            step_times_np = np.zeros((b, 1), dtype=np.float32)
        data = {
                "baseline": info["baseline"][real_step_mask],
                "action": primary_action[real_step_mask],            
                "reward": reward[real_step_mask],
                "done": done[real_step_mask],
                "truncated_done": truncated_done[real_step_mask],
                "real_state": info["real_states_np"][real_step_mask.cpu().numpy()],
                "step_times": step_times_np,
            }       

        per_state = info["initial_per_state"] if "initial_per_state"  in info else {}
        for k in per_state.keys():
            if not k.startswith("per"): continue
            data[k] = per_state[k][real_step_mask]   

        if action_prob is not None:
            action_prob = action_prob[real_step_mask]
            if not self.tuple_action: action_prob = action_prob.unsqueeze(-2)        
            data["action_prob"] = action_prob
        
        data = util.dict_map(data, lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)    
        for k in ["baseline", "reward"]: data[k] = data[k].astype(np.float32)
        if self.frame_stack_n > 1:
            data["real_state"] = data["real_state"][:, -self.frame_ch:]
        idx = np.arange(self.env_n)[real_step_mask.detach().cpu().numpy()]
        self.model_buffer.write.remote(ray.put(data), rank=self.rank, idx=idx, priority=None) 

    def _refresh_wait(self):
        self._update_status()
        if self.status["running"]: self._refresh_net()
        if self.status["processed_n"] < self.status["warm_up_n"] * 2: return         
        while self.status["replay_ratio"] < self.flags.min_replay_ratio and not self.status["finish"]:
            time.sleep(0.01)
            self._update_status()
        return 

    def _train_model(self):
        with torch.set_grad_enabled(True):
            beta = self.model_learner.compute_beta()
            while True:            
                data = self.model_buffer.read(beta)            
                self.model_learner.init_psteps(data)                  
                if data is None: 
                    self.model_learner.log_preload(self.model_buffer.get_status())
                    break
                self.model_learner.update_real_step(data)                        
                if (self.model_learner.step_per_transition() > 
                    self.flags.max_replay_ratio):
                    break   
                self.model_learner.consume_data(data, model_buffer=self.model_buffer)
            if self.model_learner.real_step >= self.flags.total_steps:
                self.model_buffer.set_finish()
        return self.model_buffer.get_status()

    def normalize(self, x):
        if self.flags.wrapper_type == 1:
            return self.env.normalize(x)
        else:
            return self.model_net.normalize(x)

    def unnormalize(self, x):
        if self.flags.wrapper_type == 1:
            return self.env.unnormalize(x)
        else:
            return self.model_net.unnormalize(x)

    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)

    def close(self):
        if self.parallel:
            self.model_buffer.set_finish.remote()
        del self.model_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.env.close()
    
    def decode_tree_reps(self, tree_reps):
        if self.flags.wrapper_type in [3, 4, 5]:
            return self.env.get_wrapper_attr('decode_tree_reps')(tree_reps)
        return util.decode_tree_reps(
            tree_reps=tree_reps,
            num_actions=self.num_actions,
            dim_actions=self.dim_actions,
            rec_t=self.flags.rec_t,
            enc_type=self.flags.model_enc_type,
            f_type=self.flags.model_enc_f_type,
        )
    
    def get_tree_rep_meaning(self):
        if self.tree_rep_meaning is None:
            if self.flags.wrapper_type in [3, 4, 5]:
                self.tree_rep_meaning = self.env.get_wrapper_attr('tree_rep_meaning')
            elif self.flags.wrapper_type in [0, 2]:
                self.tree_rep_meaning = util.slice_tree_reps(self.num_actions, self.dim_actions, self.flags.rec_t)        
        return self.tree_rep_meaning
    
    def save_ckp(self):
        data = self.env.get_wrapper_attr('save_ckp')()
        if len(data) > 0:
            np.savez(self.ckp_env_path, **data)

def make(*args, **kwargs):
    return Env(*args, **kwargs)
