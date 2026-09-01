import os
import time, timeit
from collections import namedtuple
import numpy as np
import random
import traceback
import torch
import ray
from thinker.buffer import (
    AB_FINISH,
    AB_FULL,
    validate_schema7_model_buffer_status,
)
from thinker.actor_net import ActorNet, ActorOut
from thinker.learn_actor import ActorLearner, SActorLearner
from thinker.main import (
    Env,
    _resolve_model_input_seal_runtime,
    _resolve_model_input_seal_schema_version,
)
import thinker.util as util

from thinker.util import EnvOut
_fields = tuple(ActorOut._fields) + tuple(EnvOut._fields) + ("id",)
exc_list = ["action",
            "action_prob",
            "entropy_loss",
            "reg_loss", 
            "baseline_enc", 
            "misc",
            # Learner-only detached critic input.  Never serialize it through
            # self-play/Ray replay; the learner regenerates it in its one full
            # ActorNet loss forward.
            "voc_features",
            # Evaluation-only carry observability remains available on
            # EnvOut without widening the actor replay/Ray buffer schema.
            "carried_descendant_visit_count",
            "carried_descendant_expanded_count",
            "useful_carry",
            ]
_fields = (item for item in _fields if item not in exc_list)
TrainActorOut = namedtuple("TrainActorOut", _fields)
# Schema 6 alone carries a per-row policy epoch.  Keeping a distinct tuple
# type leaves schemas 1--5 replay/Ray interfaces byte-for-byte unchanged.
VersionedTrainActorOut = namedtuple(
    "VersionedTrainActorOut", tuple(TrainActorOut._fields) + ("policy_version",)
)

@ray.remote
class SelfPlayWorker:
    def __init__(self, ray_obj_actor, ray_obj_env, rank, env_n, flags):
        self._logger = util.logger()
        worker_seed = int(getattr(flags, "base_seed", 0)) + int(rank) * int(env_n)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)
        gpu = False
        if flags.gpu_self_play > 0 and torch.cuda.is_available():
            gpu = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.actor_buffer = ray_obj_actor["actor_buffer"]
        self.actor_param_buffer = ray_obj_actor["actor_param_buffer"]

        self.log = not flags.train_actor
        if self.log: 
            self.self_play_buffer = ray_obj_actor["self_play_buffer"]
            self.real_step_ptr = None

        self._logger.info(
            "Initializing actor %d with device %s"
            % (
                rank,
                "cuda" if gpu else "cpu",
            )
        )

        self.rank = rank
        self.env_n = env_n
        self.flags = flags
        self.dynamic_search = util.dynamic_search_enabled(flags)
        self.voc_actor_policy_version_barrier = bool(
            getattr(flags, "voc_actor_policy_version_barrier", False)
        )
        self.voc_actor_policy_barrier_runtime = bool(
            self.voc_actor_policy_version_barrier
            and bool(getattr(flags, "train_actor", False))
            and bool(getattr(flags, "parallel_actor", False))
        )
        self.voc_model_input_seal_schema_version = (
            _resolve_model_input_seal_schema_version(flags)
        )
        self.voc_model_input_seal_runtime = _resolve_model_input_seal_runtime(
            flags
        )
        raw_gate_schema = getattr(flags, "voc_gate_policy_schema_version", None)
        if self.voc_actor_policy_barrier_runtime:
            if (
                type(raw_gate_schema) is not int
                or raw_gate_schema not in (6, 7, 8, 9, 10, 11, 12, 13)
            ):
                raise ValueError(
                    "versioned actor policy barrier requires exact gate schema 6--13"
                )
            self.voc_gate_policy_schema_version = raw_gate_schema
        else:
            self.voc_gate_policy_schema_version = None
        self.voc_actor_policy_barrier_timeout_s = float(
            getattr(
                flags,
                "voc_actor_policy_barrier_timeout_s",
                util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
            )
        )
        self.voc_actor_policy_version = -1
        self.voc_actor_policy_heartbeat_count = 0
        self._monotonic = time.monotonic
        self._barrier_sleep = time.sleep
     
        self.timing = util.Timings()
        self.actor_id = (
            torch.arange(self.env_n, device=self.device)
            + self.rank * self.env_n
        ).unsqueeze(0)
        self.time = self.rank == 0 and flags.profile

        if self.flags.parallel:
            self.env = Env(
                name = flags.name, 
                ray_obj = ray_obj_env,
                env_n = env_n,
                gpu = gpu,
                timing = self.time,
            )
        else:
            self.env = Env(gpu = gpu, **vars(flags))
            
        obs_space = self.env.observation_space
        action_space = self.env.action_space        

        self.has_actor = True
        self.train_actor = self.has_actor and flags.train_actor

        if self.has_actor:
            actor_param = {
                "obs_space":obs_space,
                "action_space":action_space,
                "flags":flags,
                "tree_rep_meaning": self.env.get_tree_rep_meaning() if self.flags.wrapper_type != 1 else None,
            }
            self.actor_net = ActorNet(**actor_param)
            if self.rank == 0 and not self.flags.mcts:
                self._logger.info(
                    "Actor network size: %d"
                    % sum(p.numel() for p in self.actor_net.parameters())
                )
            if not self.flags.mcts: self._load_net()          
            self.actor_net.to(self.device)
            self.actor_net.train(False)
            if self.train_actor and self.rank == 0:
                if self.flags.parallel_actor:
                    # init. the actor learner thread
                    learner_options = {
                        "num_cpus": 1,
                        "num_gpus": self.flags.gpu_learn_actor,
                    }
                    if self.voc_actor_policy_barrier_runtime:
                        learner_options.update(
                            max_restarts=0, max_task_retries=0
                        )
                    self.actor_learner = ActorLearner.options(
                        **learner_options
                    ).remote(
                        ray_obj_actor,
                        actor_param,
                        self.flags,
                        runtime_action_meanings=(
                            self.env.get_primary_action_meanings()
                        ),
                    )
                    # start learning
                    self.r_learner = self.actor_learner.learn_data.remote()
                else:
                    self.actor_learner = SActorLearner(
                        None,
                        actor_param,
                        self.flags,
                        self.actor_net,
                        self.device,
                        runtime_action_meanings=(
                            self.env.get_primary_action_meanings()
                        ),
                    )

        self.disable_thinker = flags.wrapper_type == 1
        self.finish_train_actor = False

    def gen_data(self, verbose: bool = True):
        """Generate self-play data
        Args:
            verbose (bool): whether to print output
        """
        try:
            if verbose:
                self._logger.info("Actor %d started." % self.rank)
            if self.voc_actor_policy_barrier_runtime:
                terminal = self._refresh_policy_bundle(expected_version=0)
                if terminal:
                    raise RuntimeError("initial actor policy bundle is terminal")
            n = 0
            state, info = self.env.reset()
            env_out = self.init_env_out(state, info)                        
            actor_state = self.actor_net.initial_state(
                    batch_size=self.env_n, device=self.device
            )
            actor_out, actor_state, env_out, info = self.env_step(env_out, actor_state)
       
            timer = timeit.default_timer
            start_time = timer()

            self.actor_net.train(False)
            while True:
          
                if self.time: self.timing.reset()
                # prepare train_actor_out data to be written
                initial_actor_state = actor_state

                send_buffer = self.train_actor and not (self.flags.train_model and 
                        not info["model_status"]["running"] and
                        not info["model_status"]["finish"]
                        and self.flags.ckp) 
                if send_buffer or self.log:
                    self.write_actor_buffer(env_out, actor_out, 0, log_only = not send_buffer and self.log)
                if self.time: self.timing.time("misc1")
         
                with torch.set_grad_enabled(False):
                    for t in range(self.flags.actor_unroll_len):
                        # generate action
                        actor_out, actor_state, env_out, info = \
                            self.env_step(env_out, actor_state)
                        if self.time: self.timing.time("step env")
                        # write the data to the respective buffers
                        if send_buffer or self.log: self.write_actor_buffer(env_out, actor_out, t + 1, log_only = not send_buffer and self.log)
                        if self.time: self.timing.time("finish actor buffer")                      
   
                if send_buffer and self.flags.parallel_actor:
                    # send the data to remote actor buffer
                    initial_actor_state = util.tuple_map(
                        initial_actor_state, lambda x: x.detach().cpu().numpy()
                    )
                    status = 0
                    if self.time: self.timing.time("mics2")
                    status_deadline = (
                        self._monotonic()
                        + self.voc_actor_policy_barrier_timeout_s
                    )
                    while True:
                        data_full_ptr = self.actor_buffer.get_status.remote()
                        if self.voc_actor_policy_barrier_runtime:
                            status = self._barrier_ray_get(
                                data_full_ptr,
                                deadline=status_deadline,
                                label="ActorBuffer status",
                            )
                        else:
                            status = ray.get(data_full_ptr)
                        if status == AB_FULL:
                            time.sleep(0.1)
                        else:
                            if status == AB_FINISH:
                                # ActorBuffer FINISH is emitted from the learner's
                                # finally block on both success and failure.  Rank
                                # zero owns the learner future and must resolve it
                                # before treating FINISH as a normal termination.
                                if self.rank == 0 and hasattr(self, "r_learner"):
                                    if self.voc_actor_policy_barrier_runtime:
                                        learner_ok = self._barrier_ray_get(
                                            self.r_learner,
                                            deadline=(
                                                self._monotonic()
                                                + self.voc_actor_policy_barrier_timeout_s
                                            ),
                                            label="actor learner terminal health",
                                        )
                                    else:
                                        learner_ok = ray.get(self.r_learner)
                                    if learner_ok is not True:
                                        raise RuntimeError(
                                            "actor learner terminated without success"
                                        )
                                self.train_actor = False
                            break
                    if self.train_actor:
                        write_ref = self.actor_buffer.write.remote(
                            ray.put(self.actor_local_buffer),
                            ray.put(initial_actor_state),
                        )
                        if self.voc_actor_policy_barrier_runtime:
                            self._barrier_ray_get(
                                write_ref,
                                deadline=(
                                    self._monotonic()
                                    + self.voc_actor_policy_barrier_timeout_s
                                ),
                                label="ActorBuffer.write acknowledgement",
                            )
                            self._publish_policy_heartbeat("enqueue")
                    if self.time: self.timing.time("send actor buffer")     

                if self.log:
                    if self.real_step_ptr is not None: 
                        self.real_step = ray.get(self.real_step_ptr)                        
                        if self.flags.mcts:
                            self.actor_net.set_real_step(self.real_step)

                    log_kwargs = {
                        "step_status": ray.put(self.actor_local_buffer.step_status),
                        "episode_return": ray.put(self.actor_local_buffer.episode_return),
                        "episode_step": ray.put(self.actor_local_buffer.episode_step),
                        "real_done": ray.put(self.actor_local_buffer.real_done),
                        "actor_id": ray.put(self.actor_local_buffer.id),
                    }
                    if self.dynamic_search:
                        for field in [
                                "real_transition", "stage_end", "forced_stop",
                                "search_steps", "search_control", "control_valid",
                                "policy_valid", "phase"]:
                            value = getattr(self.actor_local_buffer, field, None)
                            if value is not None:
                                log_kwargs[field] = ray.put(value)
                    self.real_step_ptr = self.self_play_buffer.insert.remote(**log_kwargs)
  
                if self.time: self.timing.time("mics3")
      
                if send_buffer and not self.flags.parallel_actor and hasattr(self, "actor_local_buffer"):
                    initial_actor_state = util.tuple_map(
                        initial_actor_state, lambda x: x.detach()   
                    )
                    data = (self.actor_local_buffer, initial_actor_state)
                    self.actor_net.train(True)
                    self.train_actor = not self.actor_learner.consume_data(data)
                    self.actor_net.train(False)
                
                if send_buffer and self.flags.parallel_actor:
                    if self.voc_actor_policy_barrier_runtime:
                        terminal = self._refresh_policy_bundle(
                            expected_version=self.voc_actor_policy_version + 1
                        )
                        if terminal:
                            self.train_actor = False
                            self.finish_train_actor = True
                            return self._complete_terminal_policy(info)
                    else:
                        self._refresh_net()

                if self.time:
                    self.timing.time("update actor net weight")

                n += 1
                if self.time and timer() - start_time > 5:
                    self._logger.info(self.timing.summary())
                    start_time = timer()

                fin = True
                if self.train_actor: fin = False                
                if not info["model_status"]["finish"]: fin = False
                if fin: 
                    self._logger.info("Terminating self-play thread %d" % self.rank)
                    self.env.close()                    
                    self._logger.info("Terminated self-play thread %d" % self.rank)
                    return True

        except Exception as e:
            if getattr(self, "voc_model_input_seal_runtime", False):
                try:
                    self.env.abort_model_input_no_step(
                        timeout=self.voc_actor_policy_barrier_timeout_s
                    )
                except Exception as abort_error:
                    self._logger.error(
                        "Failed to acknowledge schema-7 model input abort: %s",
                        abort_error,
                    )
            if self.voc_actor_policy_barrier_runtime:
                self._publish_policy_barrier_abort()
            self._logger.error(f"Exception detected in self_play: {e}")
            self._logger.error(traceback.format_exc())
            return False
    
    def env_step(self, env_out, actor_state):
        actor_out, actor_state = self.actor_net(
                            env_out = env_out, 
                            core_state = actor_state, 
                            greedy = False,
                        )
        if not self.disable_thinker:
            primary_action, secondary_action = actor_out.action
        else:
            primary_action, secondary_action = actor_out.action, None
        step_kwargs = {
            "primary_action": primary_action,
            "action_prob": actor_out.action_prob[-1],
        }
        if self.dynamic_search:
            step_kwargs["search_control"] = secondary_action
        else:
            step_kwargs["reset_action"] = secondary_action
        state, reward, done, truncated_done, info = self.env.step(**step_kwargs)
        next_env_out = self.create_env_out(
            actor_out.action, state, reward, done, truncated_done, info
        )
        if self.dynamic_search:
            # Inputs supplied while an item is waiting at the real-step barrier
            # are ignored.  Preserve the last accepted action tokens so batching
            # cannot leak dummy WAIT actions into the recurrent actor state.
            accepted_primary = info.get("accepted_primary_action")
            real_transition = info.get("real_transition")
            if accepted_primary is not None:
                invalid = accepted_primary < 0
                if real_transition is not None:
                    invalid = invalid & ~real_transition.bool()
                if torch.any(invalid):
                    last_pri = next_env_out.last_pri.clone()
                    mask = invalid
                    if last_pri.ndim > mask.ndim + 1:
                        mask = mask.unsqueeze(-1)
                    last_pri[0] = torch.where(mask, env_out.last_pri[0], last_pri[0])
                    next_env_out = next_env_out._replace(last_pri=last_pri)

            accepted_control = info.get("accepted_control")
            if accepted_control is not None and next_env_out.last_search_control is not None:
                invalid = accepted_control < 0
                if torch.any(invalid):
                    last_control = next_env_out.last_search_control.clone()
                    last_control[0] = torch.where(
                        invalid, env_out.last_search_control[0], last_control[0]
                    )
                    next_env_out = next_env_out._replace(
                        last_search_control=last_control
                    )
        env_out = next_env_out
        return actor_out, actor_state, env_out, info

    def write_actor_buffer(self, env_out: EnvOut, actor_out: ActorOut, t: int, log_only: bool = False):
        # write to local buffer
        if log_only:
            include_fields = [
                "step_status", "episode_return", "episode_step", "real_done"
            ]
            if self.dynamic_search:
                include_fields += [
                    "real_transition", "stage_end", "forced_stop",
                    "search_steps", "search_control", "control_valid",
                    "policy_valid", "phase",
                ]

        tuple_type = (
            VersionedTrainActorOut
            if self.voc_actor_policy_barrier_runtime and not log_only
            else TrainActorOut
        )
        if t == 0:
            out = {}
            
            for field in tuple_type._fields:
                out[field] = None
                if log_only and field not in include_fields: continue
                if field in ["id"]: continue                
                if field == "policy_version":
                    if self.flags.parallel_actor:
                        out[field] = torch.empty(
                            (self.flags.actor_unroll_len + 1, self.env_n),
                            dtype=torch.int64,
                            device=self.device,
                        )
                    else:
                        out[field] = []
                    continue
                if field == "real_states" and not self.flags.see_real_state: continue
                val = getattr(env_out if field in EnvOut._fields else actor_out, field)                
                if val is None: continue
                if self.flags.parallel_actor:
                    size_t = self.flags.actor_unroll_len + 1
                    if (not self.disable_thinker and field == "real_states"
                            and not self.dynamic_search):
                        size_t = size_t // self.flags.rec_t + int((size_t % self.flags.rec_t) > 0)                        
                        self.real_state_t = 0
                    out[field] = torch.empty(
                        size=(size_t, self.env_n)
                        + val.shape[2:],
                        dtype=val.dtype,
                        device=self.device,
                    )
                else:
                    out[field] = []
                # each is in the shape of (T x B xdim_1 x dim_2 ...)
            
            if self.flags.parallel_actor:
                id = self.actor_id
            else:
                id = [self.actor_id[0]]
            out["id"] = id

            self.actor_local_buffer = tuple_type(**out)

        for field in tuple_type._fields:
            if log_only and field not in include_fields: continue
            v = getattr(self.actor_local_buffer, field)
            if v is not None and field not in ["id"]:                
                if field == "policy_version":
                    new_val = torch.full(
                        (self.env_n,),
                        -1 if t == 0 else self.voc_actor_policy_version,
                        dtype=torch.int64,
                        device=self.device,
                    )
                else:
                    new_val = getattr(
                        env_out if field in EnvOut._fields else actor_out, field
                    )
                assert new_val is not None, f"{field} cannot be None"
                if field != "policy_version":
                    new_val = new_val[0]
                if self.flags.parallel_actor:
                    if (not self.disable_thinker and field == "real_states"
                            and not self.dynamic_search):
                        if env_out.step_status[0, 0].item() in [0, 3]:
                            # assume uniform step status
                            v[self.real_state_t] = new_val
                            self.real_state_t += 1
                    else:                        
                        v[t] = new_val
                else:
                    v.append(new_val)

        if self.time:
            self.timing.time("write_actor_buffer")

        if t == self.flags.actor_unroll_len:
            # post-processing
            if self.flags.parallel_actor:
                map = lambda x: x.cpu().numpy()
            else:
                map = lambda x: torch.stack(x, dim=0)
            self.actor_local_buffer = util.tuple_map(
                self.actor_local_buffer,  map
            )
        if self.time:
            self.timing.time("move_actor_buffer_to_cpu")

    def init_env_out(self, *args, **kwargs):
        return util.init_env_out(*args, **kwargs, flags=self.flags, dim_actions=self.actor_net.dim_actions, tuple_action=self.actor_net.tuple_action)

    def create_env_out(self, *args, **kwargs):
        return util.create_env_out(*args, **kwargs, flags=self.flags)

    def _barrier_ray_get(self, object_ref, *, deadline, label):
        remaining = deadline - self._monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"actor policy barrier timed out before {label}")
        try:
            return ray.get(object_ref, timeout=remaining)
        except ray.exceptions.GetTimeoutError as error:
            raise TimeoutError(
                f"actor policy barrier RPC timed out during {label}"
            ) from error

    def _publish_policy_barrier_abort(self):
        """Make worker failure immediately visible to the learner poll."""

        try:
            deadline = self._monotonic() + self.voc_actor_policy_barrier_timeout_s
            self._barrier_ray_get(
                self.actor_param_buffer.update_dict_item.remote(
                    util.VOC_ACTOR_POLICY_ACKS_KEY,
                    int(self.rank),
                    {"abort": True},
                ),
                deadline=deadline,
                label="worker abort acknowledgement",
            )
        except Exception:
            self._logger.error("failed to publish actor-policy worker abort")

    def _publish_policy_heartbeat(self, phase):
        if phase not in ("load_ack", "enqueue"):
            raise ValueError("invalid actor policy heartbeat phase")
        self.voc_actor_policy_heartbeat_count += 1
        heartbeat = {
            "rank": int(self.rank),
            "policy_version": int(self.voc_actor_policy_version),
            "phase": phase,
            "count": int(self.voc_actor_policy_heartbeat_count),
        }
        self._barrier_ray_get(
            self.actor_param_buffer.update_dict_item.remote(
                util.VOC_ACTOR_POLICY_HEARTBEAT_KEY,
                int(self.rank),
                heartbeat,
            ),
            deadline=(
                self._monotonic()
                + self.voc_actor_policy_barrier_timeout_s
            ),
            label="policy heartbeat",
        )

    def _refresh_policy_bundle(self, *, expected_version):
        """Load and acknowledge exactly one next policy before any rollout."""

        deadline = self._monotonic() + self.voc_actor_policy_barrier_timeout_s
        while True:
            bundle = self._barrier_ray_get(
                self.actor_param_buffer.get_data.remote(
                    util.VOC_ACTOR_POLICY_BUNDLE_KEY
                ),
                deadline=deadline,
                label="policy bundle load",
            )
            if bundle is None:
                self._barrier_sleep(0.01)
                continue
            observed = (
                bundle.get("policy_version")
                if isinstance(bundle, dict) else None
            )
            if (
                isinstance(observed, (int, np.integer))
                and not isinstance(observed, (bool, np.bool_))
                and int(observed) < int(expected_version)
            ):
                self._barrier_sleep(0.01)
                continue
            validated = util.validate_actor_policy_bundle(
                bundle,
                expected_epoch=expected_version,
                expected_actor_state=self.actor_net.state_dict(),
                expected_gate_schema=self.voc_gate_policy_schema_version,
                label=f"worker {self.rank} actor policy bundle",
            )
            self.actor_net.set_weights(validated["actor_state_dict"])
            self.voc_actor_policy_version = validated["policy_version"]
            ack = util.make_actor_policy_ack(
                self.rank,
                self.voc_actor_policy_version,
                terminal=validated["terminal"],
                gate_schema=self.voc_gate_policy_schema_version,
            )
            self._barrier_ray_get(
                self.actor_param_buffer.update_dict_item.remote(
                    util.VOC_ACTOR_POLICY_ACKS_KEY, int(self.rank), ack
                ),
                deadline=deadline,
                label="policy load acknowledgement",
            )
            self._publish_policy_heartbeat("load_ack")
            return validated["terminal"]

    def _wait_for_model_finish_without_env_actions(self, info):
        """After terminal policy, poll only model health; never step the env."""

        deadline = self._monotonic() + self.voc_actor_policy_barrier_timeout_s
        status = info.get("model_status", {})
        if getattr(self, "voc_model_input_seal_runtime", False):
            status = self._validate_schema7_model_status(status)
        while status.get("finish") is not True:
            if (
                getattr(self, "voc_model_input_seal_runtime", False)
                and status.get("voc_model_input_aborted") is True
            ):
                raise RuntimeError("model input aborted before normal finish")
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "model did not report normal finish after terminal policy"
                )
            status = self.env.poll_model_status_no_step(timeout=remaining)
            if getattr(self, "voc_model_input_seal_runtime", False):
                status = self._validate_schema7_model_status(status)
            if status.get("finish") is True:
                break
            self._barrier_sleep(0.01)

    def _complete_terminal_policy(self, info):
        """Finish without any post-terminal environment action."""

        if self.rank == 0 and hasattr(self, "r_learner"):
            learner_ok = self._barrier_ray_get(
                self.r_learner,
                deadline=(
                    self._monotonic()
                    + self.voc_actor_policy_barrier_timeout_s
                ),
                label="terminal actor learner health",
            )
            if learner_ok is not True:
                raise RuntimeError("actor learner terminated without success")
        if getattr(self, "voc_model_input_seal_runtime", False):
            seal_status = self.env.seal_model_input_no_step(
                timeout=self.voc_actor_policy_barrier_timeout_s
            )
            seal_status = self._validate_schema7_model_status(seal_status)
            info = dict(info)
            info["model_status"] = seal_status
        self._wait_for_model_finish_without_env_actions(info)
        self._logger.info(
            "Terminating self-play thread %d after terminal actor policy",
            self.rank,
        )
        self.env.close()
        return True

    def _validate_schema7_model_status(self, status, *, require_sealed=False):
        return validate_schema7_model_buffer_status(
            status,
            total_steps=self.flags.total_steps,
            self_play_n=self.flags.self_play_n,
            warm_up_n=self.flags.model_warm_up_n,
            require_sealed=require_sealed,
            label=f"schema-7 SelfPlay rank {self.rank} ModelBuffer status",
        )

    def _load_net(self):
        if self.rank == 0:
            util.validate_voc_fresh_control_inputs(
                self.flags, label="Self-play fresh VoC control"
            )
            # load the network from preload or load_checkpoint  
            path = None
            if self.flags.ckp:
                path = os.path.join(self.flags.ckpdir, "ckp_actor.tar")
            else:
                voc_parent_checkpoint = util.resolve_voc_parent_checkpoint(
                    self.flags
                )
                if (
                    getattr(self.flags, "dynamic_voc_mode", "off")
                    == "control"
                    and voc_parent_checkpoint
                ):
                    path = voc_parent_checkpoint
                elif self.flags.preload_actor:
                    path = os.path.join(self.flags.preload_actor, "ckp_actor.tar")
            if path is not None:
                if (
                    not self.flags.ckp
                    and getattr(self.flags, "dynamic_voc_mode", "off")
                    == "control"
                ):
                    promotion = util.validate_voc_control_preload(
                        path, flags=self.flags
                    )
                    checkpoint = promotion["_validated_checkpoint"]
                    self.flags.voc_resolved_parent_checkpoint_sha256 = (
                        promotion["voc_parent_checkpoint_sha256"]
                    )
                else:
                    checkpoint = torch.load(
                        path,
                        map_location=torch.device("cpu"),
                        weights_only=False,
                    )
                if (
                    not self.flags.ckp
                    and getattr(self.flags, "dynamic_voc_mode", "off")
                    == "shadow"
                ):
                    # A new shadow lineage may migrate legacy/off weights only.
                    # Importing an active head while resetting its counters and
                    # EMA provenance would launder a prior shadow/control run.
                    util.validate_voc_shadow_preload(
                        checkpoint, label="Self-play shadow preload"
                    )
                if self.flags.ckp:
                    util.validate_voc_active_resume_checkpoint(
                        checkpoint,
                        self.flags,
                        label="Self-play actor resume checkpoint",
                    )
                    checkpoint_flags = checkpoint.get("flags", {})
                    checkpoint_dynamic = bool(checkpoint.get(
                        "dynamic_search",
                        checkpoint_flags.get("dynamic_search", False),
                    ))
                    if checkpoint_dynamic != self.dynamic_search:
                        raise ValueError(
                            "Cannot resume actor checkpoint across "
                            "dynamic_search modes."
                        )
                    checkpoint_factorized = bool(checkpoint.get(
                        "dynamic_factorized_control",
                        checkpoint_flags.get(
                            "dynamic_factorized_control", False
                        ),
                    ))
                    run_factorized = bool(getattr(
                        self.flags, "dynamic_factorized_control", False
                    ))
                    if (
                        self.dynamic_search
                        and checkpoint_factorized != run_factorized
                    ):
                        raise ValueError(
                            "Cannot resume actor checkpoint across Dynamic "
                            "control objectives."
                        )
                    checkpoint_arch = checkpoint.get("actor_arch_version")
                    expected_arch = 2 if self.dynamic_search else 1
                    if self.dynamic_search and checkpoint_arch is None:
                        raise ValueError(
                            "Dynamic actor checkpoint is missing "
                            "actor_arch_version metadata."
                        )
                    if (checkpoint_arch is not None
                            and checkpoint_arch != expected_arch):
                        raise ValueError(
                            f"Actor checkpoint architecture {checkpoint_arch} "
                            f"does not match expected version {expected_arch}."
                        )
                    checkpoint_rewards = checkpoint.get("reward_names")
                    expected_rewards = util.get_reward_names(self.flags)
                    if self.dynamic_search and checkpoint_rewards is None:
                        raise ValueError(
                            "Dynamic actor checkpoint is missing reward_names "
                            "metadata."
                        )
                    if (checkpoint_rewards is not None
                            and list(checkpoint_rewards) != expected_rewards):
                        raise ValueError(
                            "Actor checkpoint reward channels do not match "
                            f"this run: {checkpoint_rewards} != "
                            f"{expected_rewards}."
                        )
                # Resume is architecture-strict.  preload_actor is explicitly
                # weight-only initialization and may migrate a legacy head;
                # optimizer, scheduler, counters and return normalization are
                # never read on this path.
                # Fixed checkpoints retain their historical strict-load
                # contract.  The only permissive path is the explicitly
                # supported legacy-fixed -> Dynamic preload migration.
                strict = bool(
                    self.flags.ckp
                    or not self.dynamic_search
                    # Shadow and control have the same ActorNet schema.  A
                    # promotion must therefore be an exact weight load; the
                    # permissive path is reserved for explicit legacy/off to
                    # shadow initialization.
                    or getattr(self.flags, "dynamic_voc_mode", "off")
                    == "control"
                )
                self.actor_net.set_weights(
                    checkpoint["actor_net_state_dict"], strict=strict
                )
                self._logger.info("Loaded actor net from %s" % path)
            if self.flags.parallel_actor:            
                ref = self.actor_param_buffer.set_data.remote(
                    "actor_net", self.actor_net.get_weights()
                )
                if self.voc_actor_policy_barrier_runtime:
                    self._barrier_ray_get(
                        ref,
                        deadline=(
                            self._monotonic()
                            + self.voc_actor_policy_barrier_timeout_s
                        ),
                        label="raw actor bootstrap publication",
                    )
        else:
            self._refresh_net()
        return
    
    def _refresh_net(self):
        deadline = (
            self._monotonic() + self.voc_actor_policy_barrier_timeout_s
            if self.voc_actor_policy_barrier_runtime else None
        )
        while True:
            ref = self.actor_param_buffer.get_data.remote("actor_net")
            weights = (
                self._barrier_ray_get(
                    ref, deadline=deadline, label="raw actor bootstrap load"
                )
                if self.voc_actor_policy_barrier_runtime
                else ray.get(ref)
            )
            if weights is not None:
                self.actor_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  
