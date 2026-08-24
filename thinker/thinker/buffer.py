import numpy as np
import timeit
from operator import itemgetter
import os
import datetime
import time
import ray
import thinker.util as util
from thinker.core.file_writer import FileWriter
import torch
from datetime import datetime
AB_CAN_WRITE, AB_FULL, AB_FINISH = 0, 1, 2

@ray.remote
class ActorBuffer:
    def __init__(self, batch_size=32, buffer_save_size=10, buffer_interval=10000, save_dir=None):
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_state = []
        self.finish = False

        # Parameters for saving buffer data
        self.buffer_save_size = buffer_save_size  # Number of episodes to save when saving buffer data
        self.buffer_interval = buffer_interval  # Frequency in steps to save buffer data
        
        # If save_dir is None, it will be set later when the log directory is known
        # Otherwise, use the provided save_dir (for backward compatibility)
        self.save_dir = save_dir
        self.saved_buffers_count = 0
        
        # 에피소드 저장을 위한 변수들
        self.episode_buffer = []  # 완료된 에피소드들을 저장하는 버퍼
        self.current_step = 0  # 현재 step 카운터 (로컬 카운터)
        self.real_step = 0     # 실제 학습 스텝 (글로벌 카운터)
        self.last_save_step = 0  # 마지막으로 저장한 step

    def write(self, data, state):
        # Write data, a named tuple of numpy arrays each with shape (t, b, ...)
        # and state, a tuple of numpy arrays each with shape (*,b, ...)
        self.buffer.append(data)
        self.buffer_state.append(state)

    def available_for_read(self):
        # Return True if the total size of data in the buffer is larger than the batch size
        total_size = sum([data[0].shape[1] for data in self.buffer])
        return total_size >= self.batch_size

    def get_status(self):
        # Return AB_FULL if the number of data inside the buffer is larger than 3 * self.batch_size
        if self.finish:
            return AB_FINISH
        total_size = sum([data[0].shape[1] for data in self.buffer])
        if total_size >= 2 * self.batch_size:
            return AB_FULL
        else:
            return AB_CAN_WRITE

    def read(self):
        # Return a named tuple of numpy arrays, each with shape (t, batch_size, ...)
        # from the first n=batch_size data in the buffer and remove it from the buffer

        if not self.available_for_read():
            return None

        collected_data = []
        collected_state = []
        collected_size = 0
        n_tuple = type(self.buffer[0])
        while collected_size < self.batch_size and self.buffer:
            collected_data.append(self.buffer.pop(0))
            collected_state.append(self.buffer_state.pop(0))
            collected_size += collected_data[-1][0].shape[1]

        # Concatenate the named tuple of numpy arrays along the batch dimension
        output = n_tuple(
            *(
                np.concatenate([data[i] for data in collected_data], axis=1)
                if collected_data[0][i] is not None
                else None
                for i in range(len(collected_data[0]))
            )
        )
        output_state = tuple(
            np.concatenate([state[i] for state in collected_state], axis=0)
            for i in range(len(collected_state[0]))
        )

        # If the collected data size is larger than the batch size, store the extra data back to the buffer
        if collected_size > self.batch_size:
            extra_data = n_tuple(
                *(
                    data[:, -collected_size + self.batch_size :, ...]
                    if data is not None
                    else None
                    for data in output
                )
            )
            self.buffer.insert(0, extra_data)
            extra_state = tuple(
                state[-collected_size + self.batch_size :, ...]
                for state in output_state
            )
            self.buffer_state.insert(0, extra_state)

            # Trim the output to have the exact batch size
            output = n_tuple(
                *(
                    data[:, : self.batch_size, ...] if data is not None else None
                    for data in output
                )
            )
            output_state = tuple(
                state[: self.batch_size, ...] for state in output_state
            )

        return output, output_state

    def set_finish(self):
        self.finish = True

    def update_real_step(self, step):
        """
        외부에서 실제 학습 스텝을 업데이트하는 메서드
        ActorLearner나 ModelLearner에서 호출됨
        """
        print(f"[DEBUG] update_real_step called with step={step}, current real_step={self.real_step}")
        if step > self.real_step:
            old_step = self.real_step
            self.real_step = step
            print(f"[DEBUG] Real step updated from {old_step} to {self.real_step}")
        else:
            print(f"[DEBUG] Real step NOT updated: received {step} <= current {self.real_step}")


class SModelBuffer:
    def __init__(self, buffer_n, max_rank, batch_size, alpha=1., warm_up_n=0):
        self.buffer_n = buffer_n                
        self.max_rank = max_rank
        self.batch_size = batch_size        
        self.alpha = alpha
        self.warm_up_n = warm_up_n
        self.B = max_rank * batch_size
        self.T = buffer_n // self.B + 1
        self.processed_n = 0
        self.filled_t = np.zeros(self.B, dtype=np.int64)        
        self.idx = np.zeros(self.B, dtype=np.int64)
        self.initialized = False       
        self.finish = False 
        self.frame_stack_n = 1

    def init_buffer(self, data):
        self.keys = list(data.keys()) # all inputs are of shape (B, *)    
        self.buffer = {}
        for key in self.keys:            
            v = data[key]
            self.buffer[key] = np.zeros((self.T, self.B) + v.shape[1:], dtype=v.dtype,)                    
        self.priority = np.zeros((self.T, self.B))
        self.priority[0] = 1. # for computing max priority correctly in first loop
        self.read_time = np.full((self.T, self.B), np.nan)
        self.write_time = np.zeros((self.T, self.B))
        self.initialized = True

    def write(self, data, rank, idx=None, priority=None):
        # data is a dict of numpy array, each with shape (batch_size, *).          
        if idx is None:
            b = self.batch_size      
        else:
            b = len(idx)
        self.processed_n += b
        if not self.initialized: self.init_buffer(data)
        if idx is None:
            b_idx = np.arange(rank*self.batch_size, (rank+1)*self.batch_size)
        else:
            b_idx = rank*self.batch_size+idx
        for key in self.keys:                 
            assert data[key].shape[0] == b, f"{key} should have shape ({b}, *) instead of {data[key].shape} (idx: {idx}; self.batch_size:{self.batch_size})"
            self.buffer[key][self.idx[b_idx], b_idx] = data[key]                    
        
        if priority is not None:
            assert priority.shape == (b,), f"priority should have shape ({b},) instead of {priority.shape}"
            self.priority[self.idx[b_idx], b_idx] = priority
        else:
            self.priority[self.idx[b_idx], b_idx] = max(self.priority.max(), 1e-8)
        self.filled_t[b_idx] = np.minimum(self.filled_t[b_idx]+1, self.T)
        self.read_time[self.idx[b_idx], b_idx] = 0
        self.idx[b_idx] = (self.idx[b_idx] + 1) % self.T            

    def read(self, t, b, beta=1., add_t=0):
        # Sample a trajectory with shape (t, b, *)
        # items in add_keys will have shape (t+add_t, b)        
        add_keys = ["baseline", "reward", "done", "truncated_done", "action_prob", "action"] 

        if self.finish: return "FINISH"
        if not self.initialized: return None
        priority_ = self.priority.copy()        
        for i in range(1, t + add_t + self.frame_stack_n - 1): priority_[self.idx - i] = 0.
        sample_n = np.sum((priority_ > 0).astype(np.float32)) 
        if sample_n < b or self.processed_n < self.warm_up_n: return None

        flat_priority = priority_.flatten()
        p = (flat_priority**self.alpha)
        p = p / max(p.sum(), 1e-8)
        flat_idx = np.random.choice(flat_priority.size, size=b, p=p, replace=False)
        t_idx, b_idx = np.unravel_index(flat_idx, priority_.shape)
        idx = (t_idx, b_idx, self.write_time[t_idx, b_idx])

        weights = (sample_n * p[flat_idx]) ** (-beta)
        weights /= weights.max()               

        data = {}
        for key in self.keys:
            if key == "real_state" and self.frame_stack_n > 1: continue
            t_ = t if key not in add_keys else t + add_t
            data[key] = np.zeros((t_, b) + self.buffer[key].shape[2:], dtype=self.buffer[key].dtype)

        t_idx_ = t_idx
        for i in range(t + add_t):            
            for key in self.keys:
                if key == "real_state" and self.frame_stack_n > 1: continue
                if i >= t and key not in add_keys: continue
                data[key][i] = self.buffer[key][t_idx_, b_idx]          
            if i < t: self.read_time[t_idx_, b_idx] += 1
            t_idx_ =  (t_idx_ + 1) % self.T
        if self.frame_stack_n > 1:
            frame = np.zeros((t + self.frame_stack_n - 1, b) + self.buffer["real_state"].shape[2:], dtype=self.buffer["real_state"].dtype)
            stack_done = np.zeros((t + self.frame_stack_n - 1, b), dtype=bool)
            t_idx_ = (t_idx - self.frame_stack_n + 1) % self.T
            for i in range(t + self.frame_stack_n - 1):    
                frame[i] = self.buffer["real_state"][t_idx_, b_idx]
                stack_done[i] = self.buffer["done"][t_idx_, b_idx]
                t_idx_ =  (t_idx_ + 1) % self.T
            data["real_state"] = stack_frame(frame, self.frame_stack_n, done=stack_done)
        avg_replay_ratio = np.nanmean(self.read_time)
        
        return {"data": data, 
                "replay_ratio": avg_replay_ratio, 
                "processed_n": self.processed_n,
                "weights": weights,
                "idx": idx,
                }
    
    def check_avail(self, t, b):
        if not self.initialized: return False
        priority_ = self.priority.copy()
        for i in range(1, t): priority_[self.idx - i] = 0.
        sample_n = np.sum((priority_ > 0).astype(np.float32)) 
        return sample_n >= b
    
    def get_status(self):
        return {"processed_n": self.processed_n,
                "warm_up_n": self.warm_up_n,
                "replay_ratio": np.nanmean(self.read_time) if self.initialized else 0,
                "running": self.processed_n >= self.warm_up_n,
                "finish": self.finish,                
                 }    
    
    def set_frame_stack_n(self, frame_stack_n):
        self.frame_stack_n = frame_stack_n
    
    def set_finish(self):
        self.finish = True    

    def update_priority(self, idx, priority):
        """Update priority in the buffer"""
        t_idx, b_idx, write_time = idx
        mask = self.write_time[t_idx, b_idx] == write_time        
        t_idx, b_idx, priority = t_idx[mask], b_idx[mask], priority[mask]
        self.priority[t_idx, b_idx] = np.maximum(priority, 1e-8)

@ray.remote
class ModelBuffer(SModelBuffer):
    pass

def stack_frame(frame, frame_stack_n, done):
    T, B, C = frame.shape[0] - frame_stack_n + 1, frame.shape[1], frame.shape[2]
    assert done.shape[0] == T + frame_stack_n - 1
    y = np.zeros((T, B, C * frame_stack_n)+frame.shape[3:], dtype=frame.dtype)
    for s in range(frame_stack_n):
        y[:, :, s*C:(s+1)*C] = frame[s:T+s]
    s_done = np.zeros(B, dtype=bool)
    for s in range(frame_stack_n-2, -1, -1):        
        s_done = done[s+1:T+s+1] | s_done
        y[:, :, s*C:(s+1)*C][s_done] = y[:, :, (s+1)*C:(s+2)*C][s_done]
    return y

@ray.remote
class GeneralBuffer(object):
    def __init__(self):
        self.data = {}

    def extend_data(self, name, x):
        if name in self.data:
            self.data[name].extend(x)
        else:
            self.data[name] = x
        return self.data[name]

    def update_dict_item(self, name, key, value):
        if not name in self.data:
            self.data[name] = {}
        self.data[name][key] = value
        return True

    def set_data(self, name, x):
        self.data[name] = x
        return True

    def get_data(self, name):
        return self.data[name] if name in self.data else None
    
    def get_and_increment(self, name):
        if name in self.data:
            self.data[name] += 1            
        else:
            self.data[name] = 0
        return self.data[name]

@ray.remote
class ModelBuffer(SModelBuffer):
    pass

@ray.remote
class RecordBuffer(object):
    # simply for logging return from self play thread if actor learner is not running
    def __init__(self, flags):
        self.flags = flags
        self.flags.git_revision = util.get_git_revision_hash()
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )
        self.real_step = 0
        max_actor_id = (
            self.flags.gpu_num_actors * self.flags.gpu_num_p_actors
            + self.flags.cpu_num_actors * self.flags.cpu_num_p_actors
        )
        self.last_returns = RetBuffer(max_actor_id, mean_n=400)
        self._logger = util.logger()
        self.preload = flags.ckp
        self.preload_n = flags.model_buffer_n
        self.proc_n = 0

    def set_real_step(self, x):
        self.real_step = x

    def insert(self, episode_return, episode_step, real_done, actor_id):
        T, B, *_ = episode_return.shape
        self.proc_n += T*B
        if self.preload and self.proc_n < self.preload_n:
            self._logger.info("[%s] Preloading: %d / %d" % (self.flags.xpid, self.proc_n, self.preload_n,))
            return self.real_step
        
        self.real_step += T*B        
        if np.any(real_done):
            episode_returns = episode_return[real_done]
            episode_returns = tuple(episode_returns)
            episode_lens = episode_step[real_done]
            episode_lens = tuple(episode_lens)
            done_ids = np.broadcast_to(actor_id, real_done.shape)[
                real_done
            ]
            done_ids = tuple(done_ids)
        else:
            episode_returns, episode_lens, done_ids = (), (), ()
        self.last_returns.insert(episode_returns, done_ids)
        rmean_episode_return = self.last_returns.get_mean()
        stats = {
            "real_step": self.real_step,
            "rmean_episode_return": rmean_episode_return,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
        }
        self.plogger.log(stats)
        print_str = ("[%s] Steps %i @ Ret %f." % (self.flags.xpid, self.real_step, stats["rmean_episode_return"],))
        self._logger.info(print_str)
        return self.real_step

class RetBuffer:
    def __init__(self, max_actor_id, mean_n=400):
        """
        Compute the trailing mean return by storing the last return for each actor
        and average them;
        Args:
            max_actor_id (int): maximum actor id
            mean_n (int): size of data for computing mean
        """
        buffer_n = mean_n // max_actor_id + 1
        self.return_buffer = np.zeros((max_actor_id, buffer_n))
        self.return_buffer_pointer = np.zeros(
            max_actor_id, dtype=int
        )  # store the current pointer
        self.return_buffer_n = np.zeros(
            max_actor_id, dtype=int
        )  # store the processed data size
        self.max_actor_id = max_actor_id
        self.mean_n = mean_n
        self.max_return = 0.
        self.all_filled = False

    def _insert_tuple(self, returns, actor_ids):
        """
        Insert new returnss to the return buffer
        Args:
            returns (tuple): tuple of float, the return of each ended episode
            actor_ids (tuple): tuple of int, the actor id of the corresponding ended episode
        """
        # actor_id is a tuple of integer, corresponding to the returns
        if len(returns) == 0:
            return
        assert len(returns) == len(actor_ids)
        for r, actor_id in zip(returns, actor_ids):
            if actor_id >= self.max_actor_id:
                continue
            # Find the current pointer for the actor
            pointer = self.return_buffer_pointer[actor_id]
            # Update the return buffer for the actor with the new return
            self.return_buffer[actor_id, pointer] = r
            # Update the pointer for the actor
            self.return_buffer_pointer[actor_id] = (
                pointer + 1
            ) % self.return_buffer.shape[1]
            # Update the processed data size for the actor
            self.return_buffer_n[actor_id] = min(
                self.return_buffer_n[actor_id] + 1, self.return_buffer.shape[1]
            )
            self.max_return = max(r, self.max_return)

        if not self.all_filled:
            # check if all filled
            self.all_filled = np.all(
                self.return_buffer_n >= self.return_buffer.shape[1]
            )

    def insert(self, episode_returns, ind, actor_id, done):                
        if not torch.any(done): return (), ()          
        if torch.is_tensor(episode_returns):
            episode_returns = episode_returns.detach().cpu().numpy()
        if torch.is_tensor(actor_id):
            actor_id = actor_id.detach().cpu().numpy()
        if torch.is_tensor(done):
            done = done.detach().cpu().numpy()

        episode_returns = episode_returns[done][:, ind]
        episode_returns = tuple(episode_returns)
        done_ids = np.broadcast_to(actor_id, done.shape)[done]
        done_ids = tuple(done_ids)
        self._insert_tuple(episode_returns, done_ids)
        return episode_returns, done_ids

    def get_mean(self):
        """
        Compute the mean of the returns in the buffer;
        """
        if self.all_filled:
            overall_mean = np.mean(self.return_buffer)
        else:
            # Create a mask of filled items in the return buffer
            col_indices = np.arange(self.return_buffer.shape[1])
            # Create a mask of filled items in the return buffer
            filled_mask = (
                col_indices[np.newaxis, :] < self.return_buffer_n[:, np.newaxis]
            )
            if np.any(filled_mask):
                # Compute the sum of returns for each actor considering only filled items
                sum_returns = np.sum(self.return_buffer * filled_mask)
                # Compute the mean for each actor by dividing the sum by the processed data size
                overall_mean = sum_returns / np.sum(filled_mask.astype(float))
            else:
                overall_mean = 0.0
        return overall_mean

    def get_max(self):
        return self.max_return

@ray.remote
class SelfPlayBuffer:    
    def __init__(self, flags):
        # A ray actor tailored for logging across self-play worker; code mostly from learn_actor
        self.flags = flags
        self._logger = util.logger()
        self.dynamic_search = util.dynamic_search_enabled(flags)

        max_actor_id = (
            self.flags.self_play_n * self.flags.env_n
        )

        self.ret_buffers = {"re": RetBuffer(max_actor_id, mean_n=400)}
        if self.flags.im_cost > 0.:
            self.ret_buffers["im"] = RetBuffer(max_actor_id, mean_n=20000)
        if self.flags.cur_cost > 0.:
            self.ret_buffers["cur"] = RetBuffer(max_actor_id, mean_n=400) 
        if self.dynamic_search:
            self.ret_buffers["think"] = RetBuffer(max_actor_id, mean_n=20000)
        self.ret_buffers["len"] = RetBuffer(max_actor_id, mean_n=400)
        
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )

        self.rewards_ls = util.get_reward_names(flags)
        self.num_rewards = len(self.rewards_ls)

        self.step, self.real_step, self.tot_eps = 0, 0, 0        
        self.ckp_path = os.path.join(flags.ckpdir, "ckp_self_play.tar")
        if flags.ckp: 
            if not os.path.exists(self.ckp_path):
                self.ckp_path = os.path.join(flags.ckpdir, "ckp_actor.tar")
            if not os.path.exists(self.ckp_path):
                raise Exception(f"Cannot find checkpoint in {flags.ckpdir}/ckp_self_play.tar or {flags.ckpdir}/ckp_actor.tar")
            self.load_checkpoint(self.ckp_path)

        self.timer = timeit.default_timer
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps = 0
        self.sps_buffer_n = 0
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.ckp_start_time = int(time.strftime("%M")) // 10
        

    def insert(
            self, step_status, episode_return, episode_step, real_done, actor_id,
            real_transition=None, stage_end=None, forced_stop=None,
            search_steps=None, search_control=None, control_valid=None,
            policy_valid=None, phase=None):

        stats = {}

        T, B, *_ = episode_return.shape

        step_status = torch.tensor(step_status)
        episode_return = torch.tensor(episode_return)        
        episode_step = torch.tensor(episode_step)
        real_done = torch.tensor(real_done)        

        if self.dynamic_search:
            if (real_transition is None or stage_end is None
                    or search_steps is None):
                raise ValueError(
                    "dynamic_search logging requires real_transition, "
                    "stage_end and search_steps"
                )
            # Self-play unrolls overlap by one bootstrap row.  The learner
            # drops row zero, so logging must do the same to avoid double
            # counting real transitions, stage endings and episode returns.
            step_status = step_status[1:]
            episode_return = episode_return[1:]
            episode_step = episode_step[1:]
            real_done = real_done[1:]
            real_transition = torch.as_tensor(real_transition)[1:]
            stage_end = torch.as_tensor(stage_end)[1:]
            search_steps = torch.as_tensor(search_steps)[1:]
            if forced_stop is not None:
                forced_stop = torch.as_tensor(forced_stop)[1:]
            if search_control is not None:
                search_control = torch.as_tensor(search_control)[1:]
            if control_valid is not None:
                control_valid = torch.as_tensor(control_valid)[1:]
            if policy_valid is not None:
                policy_valid = torch.as_tensor(policy_valid)[1:]
            if phase is not None:
                phase = torch.as_tensor(phase)[1:]
            T = episode_return.shape[0]
            last_step_real = real_transition.bool()
            next_step_real = stage_end.bool()
        else:
            last_step_real = (step_status == 0) | (step_status == 3)
            next_step_real = (step_status == 2) | (step_status == 3)

        # extract episode_returns
        episode_returns, done_ids = self.ret_buffers["re"].insert(
            episode_return, ind=0, actor_id=actor_id, done=real_done
        )
        episode_lens, _ = self.ret_buffers["len"].insert(
            episode_step.unsqueeze(-1), ind=0, actor_id=actor_id, done=real_done
        )

        stats = {"rmean_episode_return": self.ret_buffers["re"].get_mean(),
                 "max_episode_return": self.ret_buffers["re"].get_max(),
                 "rmean_len": self.ret_buffers["len"].get_mean(),}

        for prefix in self.rewards_ls[1:]:
            if prefix in ["im", "think"]:
                done = next_step_real
            elif prefix == "cur":
                done = real_done
            
            if prefix in self.rewards_ls:            
                n = self.rewards_ls.index(prefix)
                self.ret_buffers[prefix].insert(
                    episode_return, ind=n, actor_id=actor_id, done=done,
                )
                r = self.ret_buffers[prefix].get_mean()
                stats["rmean_%s_episode_return" % prefix] = r

        if self.dynamic_search:
            stage_end_t = next_step_real
            stats.update(util.get_search_budget_stats(
                search_steps, stage_end_t
            ))

            if search_control is not None and control_valid is not None:
                control_valid_t = control_valid.bool()
                controls = search_control[control_valid_t]
                control_n = max(int(controls.numel()), 1)
                stats.update({
                    "search/proceed_ratio": (
                        (controls == util.PROCEED).sum().item() / control_n
                    ),
                    "search/reset_ratio": (
                        (controls == util.RESET).sum().item() / control_n
                    ),
                    "search/stop_ratio": (
                        (controls == util.STOP).sum().item() / control_n
                    ),
                })
            if forced_stop is not None:
                stage_n = max(int(stage_end_t.sum().item()), 1)
                stats["search/forced_stop_rate"] = (
                    forced_stop.bool().sum().item() / stage_n
                )
            if policy_valid is not None:
                # policy_valid is aligned with the action sampled on this
                # row.  The post-step phase may already be SEARCH on the call
                # that releases the barrier, even for environments whose
                # action on that call was a WAIT no-op.
                policy_valid_t = policy_valid.bool()
                stats["search/wait_fraction"] = (
                    (~policy_valid_t).float().mean().item()
                )
                stats["search/active_batch_fraction"] = (
                    policy_valid_t.float().mean().item()
                )
                stats["search/active_batch_size"] = (
                    policy_valid_t.float().sum(dim=1).mean().item()
                )
            elif phase is not None:
                stats["search/wait_fraction"] = (
                    (phase == util.WAIT_PHASE).float().mean().item()
                )
            stats["search/real_transition_fraction"] = (
                last_step_real.float().mean().item()
            )

        self.step += T * B
        self.real_step += torch.sum(last_step_real).item()
        self.tot_eps += torch.sum(real_done).item()

        stats.update({
            "step": self.step,
            "real_step": self.real_step,
            "tot_eps": self.tot_eps,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
        })

        # write to log file
        self.plogger.log(stats)

        # print statistics
        if self.timer() - self.start_time > 5:
            self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())

            self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
            self.sps = (
                self.sps_buffer[self.sps_buffer_n - 1][0]
                - self.sps_buffer[self.sps_buffer_n][0]
            ) / (
                self.sps_buffer[self.sps_buffer_n - 1][1]
                - self.sps_buffer[self.sps_buffer_n][1]
            )
            tot_sps = (self.step - self.sps_start_step) / (
                self.timer() - self.sps_start_time
            )
            print_str = (
                "[%s] Steps %i @ %.1f SPS (%.1f). Eps %i. Ret %f (%f/%f)."
                % (
                    self.flags.xpid,
                    self.real_step,
                    self.sps,
                    tot_sps,
                    self.tot_eps,
                    stats["rmean_episode_return"],
                    stats.get("rmean_im_episode_return", 0.),
                    stats.get("rmean_cur_episode_return", 0.),
                )
            )            
            if self.dynamic_search:
                print_str += " ThinkRet %.4f SearchLen %.2f." % (
                    stats.get("rmean_think_episode_return", 0.),
                    stats.get("search/mean_steps", 0.),
                )
            self._logger.info(print_str)
            self.start_time = self.timer()
            self.queue_n = 0     

        if int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_checkpoint()
            self.ckp_start_time = int(time.strftime("%M")) // 10     

        return self.real_step  
    
    def save_checkpoint(self):
        self._logger.info("Saving self-play checkpoint to %s" % self.ckp_path)
        d = {
                "step": self.step,
                "real_step": self.real_step,
                "tot_eps": self.tot_eps,
                "ret_buffers": self.ret_buffers,
                "actor_arch_version": 2 if self.dynamic_search else 1,
                "dynamic_search": self.dynamic_search,
                "reward_names": list(self.rewards_ls),
                "flags": vars(self.flags),
            }      
        try:
            torch.save(d, self.ckp_path + ".tmp")
            os.replace(self.ckp_path + ".tmp", self.ckp_path)
        except:       
            pass

    def load_checkpoint(self, ckp_path: str):
        # Self-play checkpoints contain RetBuffer objects in addition to
        # tensors.  PyTorch 2.6+ defaults to weights_only=True, which rejects
        # those trusted local checkpoint objects.
        train_checkpoint = torch.load(
            ckp_path, torch.device("cpu"), weights_only=False
        )
        checkpoint_flags = train_checkpoint.get("flags", {})
        checkpoint_dynamic = bool(train_checkpoint.get(
            "dynamic_search", checkpoint_flags.get("dynamic_search", False)
        ))
        if checkpoint_dynamic != self.dynamic_search:
            raise ValueError(
                "Cannot resume self-play statistics across dynamic_search modes."
            )
        checkpoint_arch = train_checkpoint.get("actor_arch_version")
        expected_arch = 2 if self.dynamic_search else 1
        if self.dynamic_search and checkpoint_arch is None:
            raise ValueError(
                "Dynamic self-play checkpoint is missing "
                "actor_arch_version metadata."
            )
        if checkpoint_arch is not None and checkpoint_arch != expected_arch:
            raise ValueError(
                f"Self-play checkpoint architecture {checkpoint_arch} does "
                f"not match expected version {expected_arch}."
            )
        checkpoint_rewards = train_checkpoint.get("reward_names")
        if self.dynamic_search and checkpoint_rewards is None:
            raise ValueError(
                "Dynamic self-play checkpoint is missing reward_names metadata."
            )
        if (checkpoint_rewards is not None
                and list(checkpoint_rewards) != list(self.rewards_ls)):
            raise ValueError(
                "Self-play checkpoint reward channels do not match this run: "
                f"{checkpoint_rewards} != {self.rewards_ls}."
            )
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.ret_buffers = train_checkpoint["ret_buffers"]
        self._logger.info("Loaded self-play checkpoint from %s" % ckp_path)
