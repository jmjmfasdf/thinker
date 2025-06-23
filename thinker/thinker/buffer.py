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
    def __init__(self, batch_size=32, buffer_save_size=10, save_dir=None):
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_state = []
        self.finish = False
        
        # Parameters for saving buffer data
        self.buffer_save_size = buffer_save_size  # Number of episodes to save when saving buffer data
        self.save_dir = save_dir
        self.max_episode_return = 0
        
        # 에피소드 버퍼
        self.episode_buffer = []  # 완료된 에피소드들을 저장하는 버퍼
        
        # 버퍼 크기 제한
        self.max_episode_buffer = self.buffer_save_size * 2  # cap for episode_buffer size
        
        # 디버깅을 위한 카운터 추가
        self.total_transitions_received = 0
        self.last_debug_time = time.time()
        self.debug_interval = 60  # 60초마다 디버그 정보 출력
        
        # 실제 학습 스텝 카운터
        self.real_step = 0

    # ------------------------------------------------------------------
    # Helper utilities (defined at class level right after __init__)
    # ------------------------------------------------------------------
    def _log(self, msg):
        """Simple print-based logger for ActorBuffer."""
        print(msg, flush=True)

    def _trim_buffers(self):
        """Ensure episode buffer does not grow beyond predefined limits."""
        # Trim completed episode buffer
        if hasattr(self, 'max_episode_buffer') and len(self.episode_buffer) > self.max_episode_buffer:
            excess = len(self.episode_buffer) - self.max_episode_buffer
            del self.episode_buffer[:excess]

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
        if step > self.real_step:
            self.real_step = step
            # 디버깅 정보 출력
            current_time = time.time()
            if current_time - self.last_debug_time > self.debug_interval:
                self._log(f"] Real step updated to {step}, total transitions: {self.total_transitions_received}")
                self.last_debug_time = current_time

    def save_buffer_data(self, data=None, state=None):
        """Save buffer data when a new max episode return is found"""
        try:
            # 저장 디렉토리 확인 및 생성
            if self.save_dir is None:
                # 학습 로그 디렉토리 안에 buffer_saves 폴더 생성
                log_dir = os.environ.get('THINKER_LOG_DIR', None)
                if log_dir is not None and os.path.exists(log_dir):
                    self.save_dir = os.path.join(log_dir, 'replay_buffer')
                    self._log(f"] Using log directory for buffer saves: {self.save_dir}")
                else:
                    self.save_dir = 'replay_buffer'
                    self._log(f"] THINKER_LOG_DIR not found, using default: {self.save_dir}")
                os.makedirs(self.save_dir, exist_ok=True)
            
            # data와 state가 제공되면 처리
            if data is not None and state is not None:
                # 디버깅 카운터 업데이트
                self.total_transitions_received += 1
                
                # 완료된 에피소드 탐지
                if hasattr(data, 'done'):
                    # done 필드의 차원에 따라 처리 방식 변경
                    done_indices = np.where(data.done)[1] if data.done.ndim > 1 else np.where(data.done)[0]
                    
                    # 완료된 에피소드 데이터 추출
                    for idx in done_indices:
                        # 현재 에피소드의 데이터 추출
                        episode_data = self._extract_episode_data(data, idx)
                        episode_state = self._extract_episode_state(state, idx)
                        if episode_data is not None and episode_state is not None:
                            # 에피소드 리턴 가져오기
                            ep_return = None
                            # 직접 episode_return 필드에서 값을 가져옴
                            if hasattr(episode_data, 'episode_return'):
                                try:
                                    episode_return_arr = getattr(episode_data, 'episode_return')
                                    if episode_return_arr is not None:
                                        # episode_return 배열의 구조 처리
                                        # 배열이 다차원일 수 있으므로 numpy로 변환하여 처리
                                        if hasattr(episode_return_arr, 'shape') and len(episode_return_arr.shape) > 0:
                                            # numpy 배열인 경우
                                            ep_return = float(episode_return_arr.item(0))
                                        elif isinstance(episode_return_arr, (list, tuple)) and len(episode_return_arr) > 0:
                                            # 리스트나 튜플인 경우
                                            ep_return = float(episode_return_arr[0])
                                        else:
                                            # 단일 값인 경우
                                            ep_return = float(episode_return_arr)
                                        self._log(f"] Using episode_return field: {ep_return}")
                                except Exception as e:
                                    self._log(f"] Error getting episode_return: {e}")
                                    ep_return = None
                            
                            # 만약 episode_return이 없으면 reward로 계산
                            if ep_return is None and hasattr(episode_data, 'reward'):
                                try:
                                    reward_arr = getattr(episode_data, 'reward')
                                    if reward_arr is not None:
                                        ep_return = float(np.sum(reward_arr))
                                        self._log(f"] Calculated from rewards: {ep_return}")
                                except Exception as e:
                                    self._log(f"] Error calculating from rewards: {e}")
                                    ep_return = None

                            # 새로운 최고 리턴을 발견하면 즉시 저장
                            self._log(f"] Processing episode with return {ep_return}, current max: {self.max_episode_return}")
                            if ep_return is not None and ep_return > self.max_episode_return:
                                old_max = self.max_episode_return
                                self.max_episode_return = ep_return
                                timestamp_best = datetime.now().strftime("%Y%m%d_%H%M%S")
                                best_filename = os.path.join(
                                    self.save_dir,
                                    f"best_ep_return_{ep_return:.2f}_{timestamp_best}.npz",
                                )
                                self._log(f"] New max episode return: {old_max} -> {ep_return} (improvement: {ep_return - old_max})")
                                try:
                                    best_data = {}
                                    # 데이터 필드 저장
                                    if hasattr(episode_data, '_fields'):
                                        for field_name in episode_data._fields:
                                            value = getattr(episode_data, field_name)
                                            if value is not None:
                                                best_data[field_name] = value
                                    # 상태 데이터 저장 (구성 요소별 이름 지정 포함)
                                    if episode_state is not None:
                                        if isinstance(episode_state, tuple):
                                            # ----- 구성 요소 이름 산출 로직 -----
                                            num_components = len(episode_state)
                                            base_lstm_components = 4
                                            extra_states = num_components % base_lstm_components
                                            has_attention = has_memory_slot_mask = has_global_memory = False
                                            if extra_states == 1:
                                                has_attention = True
                                            elif extra_states == 2:
                                                has_attention = True; has_memory_slot_mask = True
                                            elif extra_states == 3:
                                                has_attention = True; has_memory_slot_mask = True; has_global_memory = True
                                            num_lstm_layers = (
                                                num_components - (1 if has_attention else 0) - (1 if has_memory_slot_mask else 0) - (1 if has_global_memory else 0)
                                            ) // base_lstm_components
                                            state_component_names = []
                                            for layer in range(num_lstm_layers):
                                                state_component_names.extend([
                                                    f'lstm_layer_{layer+1}_hidden_state',
                                                    f'lstm_layer_{layer+1}_cell_state',
                                                    f'lstm_layer_{layer+1}_attention_keys',
                                                    f'lstm_layer_{layer+1}_attention_values',
                                                ])
                                            if has_attention:
                                                state_component_names.append('attention_mask')
                                            if has_memory_slot_mask:
                                                state_component_names.append('memory_slot_mask')
                                            if has_global_memory:
                                                state_component_names.append('global_memory_state')
                                            # fallback names if mismatch
                                            if len(state_component_names) != num_components:
                                                state_component_names = [f'state_component_{i}' for i in range(num_components)]
                                            # 실제 데이터 저장
                                            for comp_idx, comp in enumerate(episode_state):
                                                if comp is not None:
                                                    best_data[state_component_names[comp_idx]] = comp
                                        else:
                                            best_data['state'] = episode_state
                                    np.savez_compressed(best_filename, **best_data)
                                    self._log(f"] Saved new best episode with return {ep_return} to {best_filename}")
                                except Exception as e:
                                    self._log(f"] Error saving best episode: {e}")
                            # 버퍼에 추가
                            self.episode_buffer.append((episode_data, episode_state))
                            self._trim_buffers()
        except Exception as e:
            print(f"Error saving buffer data: {e}")
            import traceback
            traceback.print_exc()

    def _extract_episode_data(self, data, env_idx):
        """특정 환경 인덱스의 에피소드 데이터를 추출"""
        try:
            # 결과를 저장할 딕셔너리
            episode_data = {}
            
            # 원본 데이터와 동일한 타입의 객체 생성
            if hasattr(data, '_fields'):
                for field_name in data._fields:
                    field_value = getattr(data, field_name)
                    if field_value is not None:
                        if isinstance(field_value, np.ndarray):
                            # 해당 환경 인덱스의 데이터만 추출
                            if field_value.ndim > 1 and field_value.shape[1] > env_idx:
                                episode_data[field_name] = field_value[:, env_idx:env_idx+1]
                            else:
                                self._log(f"] Field {field_name} shape {field_value.shape} does not contain env_idx {env_idx}")
                
                # 원본과 동일한 named tuple 타입 생성
                from collections import namedtuple
                EpisodeData = namedtuple(type(data).__name__, data._fields)
                return EpisodeData(**{field: episode_data.get(field) for field in data._fields})
            
            return None
        except Exception as e:
            self._log(f"] Error extracting episode data: {e}")
            return None
    
    def _extract_episode_state(self, state, env_idx):
        try:
            if state is None:
                return None
            episode_state = []
            for state_component in state:
                if state_component is not None and isinstance(state_component, np.ndarray):
                    if state_component.ndim > 1 and state_component.shape[0] > env_idx:
                        episode_state.append(state_component[env_idx:env_idx+1])
                    else:
                        episode_state.append(None)
                else:
                    episode_state.append(None)
            
            return tuple(episode_state)
        except Exception as e:
            return None
            
    def write(self, data, state):
        """Write data to buffer"""
        # 원래 버퍼에 추가
        self.buffer.append(data)
        self.buffer_state.append(state)
        
        # 에피소드 처리 및 저장 로직 실행
        self.save_buffer_data(data, state)


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

        max_actor_id = (
            self.flags.self_play_n * self.flags.env_n
        )

        self.ret_buffers = {"re": RetBuffer(max_actor_id, mean_n=400)}
        if self.flags.im_cost > 0.:
            self.ret_buffers["im"] = RetBuffer(max_actor_id, mean_n=20000)
        if self.flags.cur_cost > 0.:
            self.ret_buffers["cur"] = RetBuffer(max_actor_id, mean_n=400) 
        self.ret_buffers["len"] = RetBuffer(max_actor_id, mean_n=400)
        
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )

        self.rewards_ls = ["re"]
        if flags.im_cost > 0.0:
            self.rewards_ls += ["im"]
        if flags.cur_cost > 0.0:
            self.rewards_ls += ["cur"]
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
        

    def insert(self, step_status, episode_return, episode_step, real_done, actor_id):

        stats = {}

        T, B, *_ = episode_return.shape

        step_status = torch.tensor(step_status)
        episode_return = torch.tensor(episode_return)        
        episode_step = torch.tensor(episode_step)
        real_done = torch.tensor(real_done)        

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

        for prefix in ["im", "cur"]:            
            if prefix == "im":
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
                "flags": vars(self.flags),
            }      
        try:
            torch.save(d, self.ckp_path + ".tmp")
            os.replace(self.ckp_path + ".tmp", self.ckp_path)
        except:       
            pass

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.ret_buffers = train_checkpoint["ret_buffers"]
        self._logger.info("Loaded self-play checkpoint from %s" % ckp_path)
