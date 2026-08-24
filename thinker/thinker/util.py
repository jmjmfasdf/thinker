__version__ = "1.3.0"
__project__ = "thinker"

import collections
import time
import timeit
import yaml
import argparse
import subprocess
from collections import namedtuple
import os
import re
import sys
import math
import logging
from matplotlib import pyplot as plt
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# Dynamic Thinker public action/phase contracts.  Keep these values in sync
# with cenv.pyx; tests assert the mapping at the Python/Cython boundary.
PROCEED = 0
RESET = 1
STOP = 2

SEARCH_PHASE = 0
NEED_REAL_ACTION_PHASE = 1
WAIT_PHASE = 2

POLICY_NONE = 0
POLICY_SEARCH = 1
POLICY_REAL = 2

_fields = ("real_states", "tree_reps", "xs", "hs")
_fields += ("reward", "episode_return", "episode_step")
_fields += ("done", "real_done", "truncated_done")
_fields += ("max_rollout_depth", "step_status")
_fields += ("last_pri", "last_reset", "cur_gate")
_fields += ("last_search_control", "phase", "legal_control_mask")
_fields += ("tree_token_valid", "search_state_reset", "real_transition")
_fields += ("stage_end", "forced_stop", "search_steps")
EnvOut = namedtuple("EnvOut", _fields)   


def dynamic_search_enabled(flags):
    """Return whether the opt-in Dynamic Thinker state machine is enabled."""
    return bool(getattr(flags, "dynamic_search", False))


def get_reward_names(flags):
    """Canonical actor reward-channel order.

    The Dynamic computation channel is appended so legacy reward indices stay
    stable during partial actor-checkpoint migration.
    """
    names = ["re"]
    if flags.im_cost > 0.0:
        names.append("im")
    if flags.cur_cost > 0.0:
        names.append("cur")
    if dynamic_search_enabled(flags):
        names.append("think")
    return names


def get_search_budget_stats(search_steps, stage_end):
    """Summarize imagination actions for completed Dynamic search stages.

    ``search_steps`` counts accepted PROCEED/RESET actions in a stage.  STOP,
    real-action storage, and WAIT calls are therefore excluded.  Incomplete
    stages are excluded as well so an unroll boundary cannot shorten a budget.
    """
    search_steps = torch.as_tensor(search_steps)
    stage_end = torch.as_tensor(stage_end, device=search_steps.device).bool()
    ended_steps = search_steps[stage_end].float()
    if ended_steps.numel() == 0:
        return {
            "max_budget": 0.0,
            "mean_budget": 0.0,
            "search/mean_steps": 0.0,
            "search/median_steps": 0.0,
            "search/p95_steps": 0.0,
        }

    mean_budget = ended_steps.mean().item()
    return {
        "max_budget": ended_steps.max().item(),
        "mean_budget": mean_budget,
        # Keep the existing names for dashboard/checkpoint-log compatibility.
        "search/mean_steps": mean_budget,
        "search/median_steps": ended_steps.median().item(),
        "search/p95_steps": torch.quantile(ended_steps, 0.95).item(),
    }

def init_env_out(state, info, flags, dim_actions, tuple_action):
    # minimum env_out for actor_net
    num_rewards = len(get_reward_names(flags))

    env_n = state["real_states"].shape[0]
    device = state["real_states"].device

    last_pri_shape = (env_n, dim_actions) if tuple_action else (env_n)
    out = {
        "last_pri": torch.zeros(last_pri_shape, dtype=torch.long, device=device),
        "last_reset": torch.zeros(env_n, dtype=torch.long, device=device),
        "last_search_control": torch.zeros(env_n, dtype=torch.long, device=device),
        "reward": torch.zeros((env_n, num_rewards), 
                            dtype=torch.float, device=device),
        "done": torch.zeros(env_n, dtype=torch.bool, device=device),
        "truncated_done": torch.zeros(env_n, dtype=torch.long, device=device),
    }

    if dynamic_search_enabled(flags):
        out.update({
            "phase": torch.full((env_n,), SEARCH_PHASE, dtype=torch.long, device=device),
            "legal_control_mask": torch.ones((env_n, 3), dtype=torch.bool, device=device),
            "tree_token_valid": torch.ones(env_n, dtype=torch.bool, device=device),
            "search_state_reset": torch.ones(env_n, dtype=torch.bool, device=device),
            "real_transition": torch.zeros(env_n, dtype=torch.bool, device=device),
            "stage_end": torch.zeros(env_n, dtype=torch.bool, device=device),
            "forced_stop": torch.zeros(env_n, dtype=torch.bool, device=device),
            "search_steps": torch.zeros(env_n, dtype=torch.long, device=device),
        })

    # State/info are authoritative.  In particular, the Dynamic defaults
    # above describe only the reset observation; they must not shadow phase
    # and mask values returned by the wrapper.
    for field in EnvOut._fields:
        if field in state and state[field] is not None:
            out[field] = state[field]
        if field in info and info[field] is not None:
            out[field] = info[field]
        if field not in out:
            out[field] = None

    for k, v in out.items():
        if v is not None:
            out[k] = torch.unsqueeze(v, dim=0)
    env_out = EnvOut(**out)        
    return env_out     

def create_env_out(action, state, reward, done, truncated_done, info, flags):
    
    aug_reward = [reward]
    if flags.im_cost > 0:
        aug_reward.append(info["im_reward"][:, 0])
    if flags.cur_cost > 0:
        aug_reward.append(info["cur_reward"])
    if dynamic_search_enabled(flags):
        think_reward = info.get("think_reward")
        if think_reward is None:
            think_reward = torch.zeros_like(reward)
        if think_reward.ndim > reward.ndim:
            think_reward = think_reward[..., 0]
        aug_reward.append(think_reward)
    aug_reward = torch.stack(aug_reward, dim=-1)

    if 'episode_return' in info:
        aug_epsoide_return = [info['episode_return']]
        if flags.im_cost > 0:
            aug_epsoide_return.append(info["im_episode_return"])
        if flags.cur_cost > 0:
            aug_epsoide_return.append(info["cur_episode_return"])
        if dynamic_search_enabled(flags):
            think_episode_return = info.get("think_episode_return")
            if think_episode_return is None:
                think_episode_return = torch.zeros_like(info["episode_return"])
            aug_epsoide_return.append(think_episode_return)
        aug_epsoide_return = torch.stack(aug_epsoide_return, dim=-1)
    else:
        aug_epsoide_return = None
    
    out = {
        "reward": aug_reward, 
        "episode_return": aug_epsoide_return,
        "done": done,
        "truncated_done": truncated_done,           
    }
    if not flags.wrapper_type == 1:    
        last_pri = action[0]
        if dynamic_search_enabled(flags):
            effective_primary = info.get("effective_primary_action")
            if effective_primary is not None:
                last_pri = effective_primary.to(last_pri.device)
            elif info.get("executed_primary_action") is not None:
                real_transition = info.get("real_transition")
                if real_transition is not None:
                    mask = real_transition.bool()
                    executed = info["executed_primary_action"].to(last_pri.device)
                    if last_pri.ndim > mask.ndim:
                        mask = mask.unsqueeze(-1)
                    last_pri = torch.where(mask, executed, last_pri)
        out["last_pri"] = last_pri
        out["last_reset"] = action[1]
        effective_control = info.get("effective_search_control")
        out["last_search_control"] = (
            effective_control.to(action[1].device)
            if dynamic_search_enabled(flags) and effective_control is not None
            else action[1]
        )
    else:
        out["last_pri"] = action

    for field in EnvOut._fields:    
        if field not in out:
            out[field] = None
        else:
            continue
        if field in state.keys():
            out[field] = state[field]
        if field in info.keys():
            out[field] = info[field]
    
    for k, v in out.items():
        if v is not None:
            out[k] = torch.unsqueeze(v, dim=0)
    env_out = EnvOut(**out)
    return env_out    

def process_flags(flags):
    # Defaults are needed when resuming a configuration written by an older
    # Thinker version.
    if not hasattr(flags, "dynamic_search"):
        flags.dynamic_search = False
    if not hasattr(flags, "max_search_steps"):
        flags.max_search_steps = -1
    if not hasattr(flags, "think_cost"):
        flags.think_cost = 0.002
    if not hasattr(flags, "think_cost_anneal"):
        flags.think_cost_anneal = False
    if not hasattr(flags, "dynamic_search_hidden_dim"):
        flags.dynamic_search_hidden_dim = 100

    if flags.dynamic_search:
        if flags.wrapper_type not in [0, 2]:
            raise ValueError(
                "dynamic_search supports wrapper_type 0 (learned) and "
                "2 (perfect)"
            )
        if flags.reset_mode != 0:
            raise ValueError(
                "dynamic_search currently preserves reset_mode=0 semantics only"
            )
        if not (flags.max_search_steps == -1 or flags.max_search_steps > 0):
            raise ValueError(
                "max_search_steps must be -1 (unbounded) or a positive integer"
            )
        # Dynamic history is represented along the temporal token axis.
        flags.has_action_seq = False

    if flags.wrapper_type == 1:
        flags.rec_t = 1
        # flags.train_model = False
        flags.im_enable = False
        flags.cur_enable = False
        flags.return_h = False
        flags.return_double = False

    if check_perfect_model(flags.wrapper_type):
        flags.dual_net = False
        flags.cur_enable = False
        flags.model_rs_loss_cost = 0
        flags.model_img_loss_cost = 0
        flags.model_done_loss_cost = 0

    assert flags.wrapper_type != 5, "wrapper-type 5 (meta-learning) not yet supported"

    return flags

def process_flags_actor(flags):    
    if flags.drc:
        flags.wrapper_type = 1

    if flags.wrapper_type == 1:
        flags.see_h = False
        flags.see_x = False
        flags.see_tree_rep = False
        flags.see_real_state = True
        flags.im_cost = 0.
        flags.cur_cost = 0.
        flags.policy_vis_freq = -1
        flags = process_flags(flags)

    if getattr(flags, "dynamic_search", False) and flags.mcts:
        raise ValueError("MCTS is not compatible with dynamic_search")

    if flags.mcts:
        flags.train_actor = False
        flags.policy_vis_freq = -1

    if "Safexp" in flags.name or flags.name.startswith("DM"):
        flags.policy_vis_freq = -1

    flags.return_h = flags.see_h
    flags.return_x = flags.see_x

    if check_perfect_model(flags.wrapper_type):
        flags.cur_cost = 0.
        flags.cur_enable = False    

    if not flags.has_model:
        flags.train_model = False
    
    return flags

def alloc_res(flags, gpu_n):
    if flags.auto_res:
        flags.self_play_n = [1, 1, 2, 2][gpu_n]
        flags.env_n = [64, 32, 32, 32][gpu_n]
        flags.gpu_self_play = [0.25, 0.5, 0.5, 1][gpu_n]
        flags.gpu_learn_actor = [0.25, 0.5, 1, 1][gpu_n]
        flags.gpu_learn = [0.5, 1, 1, 1][gpu_n]
        if not flags.train_model:
            flags.gpu_learn = 0
            flags.self_play_n = [2, 2, 2, 2][gpu_n]
            flags.gpu_self_play = [0.25, 0.5, 1, 1][gpu_n]
        if not flags.train_actor:
            flags.gpu_learn_actor = 0
            flags.self_play_n = [2, 2, 2, 3][gpu_n]
            flags.gpu_self_play = [0.25, 0.5, 1, 1][gpu_n]
        if not flags.parallel:
            flags.self_play_n = 1
            flags.env_n = 64
            flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn_actor = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn = 0
        if not flags.parallel_actor:
            flags.self_play_n = 1
            flags.env_n = flags.actor_batch_size
            flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn_actor = 0
            flags.gpu_learn = [0.5, 1, 1, 1][gpu_n]
        if not flags.parallel_actor and not flags.parallel:
            flags.self_play_n = 1
            flags.env_n = flags.actor_batch_size
            flags.gpu_self_play = 1
            flags.gpu_learn_actor = 0
            flags.gpu_learn = 0
    return flags

def add_parse(filename, parser=None, prefix=''):
    # Load default configuration
    if type(filename) is not list: 
        filename = [filename]
    config = {}
    for n in filename:
        default_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', n)
        with open(default_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # Set up command line argument parsing
    if parser is None:
        parser = argparse.ArgumentParser(description=f"{__project__} v{__version__}")
    try:
        parser.add_argument('--config', type=str, help="Path to user's thinker configuration file")
    except:
        # if there is dulplicate key, just ignore
        pass

    if prefix and prefix[-1] != "_": prefix = prefix + "_"
    # Dynamically add command line arguments based on the default config keys and their types
    for key, value in config.items():
        try:
            if isinstance(value, bool):
                parser.add_argument(f'--{prefix}{key}', type=lambda x: (str(x).lower() == 'true'), help=f"Override {key}")
            else:
                parser.add_argument(f'--{prefix}{key}', type=type(value), help=f"Override {key}")
        except:
            # if there is dulplicate key, just ignore
            pass
    return parser

def create_flags(filename, save_flags=True, post_fn=None, **kwargs):
    """create flags, a namespace object that contains the config; the load
       order is filename[0], filename[1], ..., kwargs['config'], kwargs       
       args:
            filename (str/list of str): the config file(s) to load
            save_flags (bool): weather to save the flags
            post_fn (function): a function that takes flags and output flags
            **kwargs: all other settings 
       return:
            flags (namespace): config                
    """
    if type(filename) is not list: 
        filename = [filename]

    config = {}
    for n in filename:
        default_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', n)
        with open(default_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # If user provided their own YAML configuration, load it and update defaults
    if "config" in kwargs and kwargs["config"]:
        with open(kwargs["config"], 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    # Check for command line argument overrides and apply them
    for key in config.keys():
        if key in kwargs and kwargs[key] is not None:
            config[key] = kwargs[key]            

    # Convert dictionary to named tuple
    flags = argparse.Namespace(**config)    

    # additional info
    if not flags.project: flags.project = __project__
    flags.savedir = flags.savedir.replace("__project__", flags.project)    
    flags.__version__ = __version__
    flags.cmd = " ".join(sys.argv) 

    try:
        flags.git_revision = get_git_revision_hash()
    except Exception:
        flags.git_revision = None

    if flags.ckp:
        # load setting from checkpoint yaml    
        xpid = 'latest' if not flags.xpid else flags.xpid
        config_path = os.path.join(flags.savedir, xpid, "config_c.yaml")        
        if os.path.islink(config_path): config_path = os.readlink(config_path)
        with open(config_path, 'r') as f:
            config_ = yaml.safe_load(f)
        for key, value in config_.items():
            if (key not in ['ckp', 'ray_mem', 'ray_gpu', 'savedir'] and
                not (key in kwargs and kwargs[key] is not None)):
                setattr(flags, key, value)
        print("Loaded config from %s" % config_path)

    if not flags.xpid:        
        flags.xpid = "%s-%s" % (flags.project, time.strftime("%Y%m%d-%H%M%S"))

    flags.ckpdir = os.path.join(flags.savedir, flags.xpid,)     

    flags = process_flags(flags)
    if post_fn is not None: flags = post_fn(flags)

    if save_flags and not flags.ckp:        
        ckpdir = full_path(flags.ckpdir)
        if not os.path.exists(ckpdir):   
            os.makedirs(ckpdir)
            
        # Set environment variable for the replay buffer to use
        os.environ['THINKER_LOG_DIR'] = ckpdir
        print(f"Set THINKER_LOG_DIR environment variable to: {ckpdir}")
            
        try:
            # create sym link for the latest run
            symlink = os.path.join(full_path(flags.savedir), "latest")
            if os.path.islink(symlink):
                os.remove(symlink)
            if not os.path.exists(symlink):
                os.symlink(flags.ckpdir, symlink)
                print("Symlinked log directory: %s" % symlink)
        except OSError:
            # os.remove() or os.symlink() raced. Don't do anything.
            pass

        config_path = os.path.join(full_path(flags.savedir), 
                                   flags.xpid, 
                                   "config_c.yaml")
        with open(config_path, 'w') as outfile:
            yaml.dump(vars(flags), outfile)
        print("Wrote config file to %s" % config_path)  

    fs = ["savedir", "preload", "ckpdir"]
    for f in fs:
        path = getattr(flags, f)
        if path:            
            setattr(flags, f, full_path(path))
    return flags

def create_setting(args=None, save_flags=True, **kwargs):
    filenames = ['default_thinker.yaml', 'default_actor.yaml']
    parser = add_parse(filenames)
    if args is not None:
        parse_flags = parser.parse_args(args)
    else:
        parse_flags = parser.parse_args()

    parse_dict = vars(parse_flags)
    for key in parse_dict.keys():
        if key in kwargs and kwargs[key] is not None:
            parse_dict[key] = kwargs[key]            

    flags = create_flags(filenames, 
                         save_flags=save_flags, 
                         post_fn=process_flags_actor, 
                         **parse_dict)
    return flags

def full_path(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.islink(path):
        path = os.readlink(path)
    return path

def tuple_map(x, f, skip_dict=False):
    def process_element(y):
        # Apply function to dictionary items
        if isinstance(y, dict):
            if not skip_dict:
                return {k: f(v) if v is not None else None for k, v in y.items()}
            else:
                return {}
        return f(y) if y is not None else None

    if type(x) == tuple:
        return tuple(process_element(y) for y in x)
    else:
        return type(x)(*(process_element(y) for y in x))

def dict_map(x, f):
    return {k:f(v) if v is not None else None for (k, v) in x.items()}

def safe_view(x, dims):
    if x is None:
        return None
    else:
        return x.view(*dims)
    
def safe_squeeze(x, dim=0):
    if x is None:
        return None
    else:
        return x.squeeze(dim)


def safe_unsqueeze(x, dim=0):
    if x is None:
        return None
    else:
        return x.unsqueeze(dim)


def safe_concat(xs, attr, dim=0):
    if len(xs) == 0:
        return None
    if getattr(xs[0], attr) is None:
        return None
    return torch.concat([getattr(i, attr).unsqueeze(dim) for i in xs], dim=dim)


def construct_tuple(x, **kwargs):
    return x(**{k: kwargs[k] if k in kwargs else None for k in x._fields})


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

def enc(x, f_type=0):
    if f_type == 0:
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + (0.001) * x
    else:
        return np.sign(x) * np.log(np.abs(x) + 1)

def dec(x, f_type=0):
    if f_type == 0:
        return np.sign(x) * (
            np.square(
                (np.sqrt(1 + 4 * 0.001 * (np.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001)
            )
            - 1
        )
    else:
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def copy_net(tar_net, net):
    for tar_module, new_module in zip(tar_net.modules(), net.modules()):
        if isinstance(tar_module, nn.modules.batchnorm._BatchNorm):
            # Copy BatchNorm running mean and variance
            tar_module.running_mean = new_module.running_mean.clone()
            tar_module.running_var = new_module.running_var.clone()
        for tar_param, new_param in zip(tar_module.parameters(), new_module.parameters()):
            tar_param.data = new_param.data.clone()

def load_optimizer(optimizer, optimizer_state_dict):
    # to not replacing lr
    current_lrs = [group['lr'] for group in optimizer.param_groups]
    for i, group in enumerate(optimizer_state_dict['param_groups']):
        if i < len(current_lrs):
            group['lr'] = current_lrs[i]
    optimizer.load_state_dict(optimizer_state_dict)

def load_scheduler(scheduler, scheduler_state_dict):
    if 'base_lrs' in scheduler_state_dict:
        del scheduler_state_dict['base_lrs']
    scheduler.load_state_dict(scheduler_state_dict)

def logger():
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("logs/out")
    if not logger.hasHandlers():
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        logger.addHandler(shandle)
    logger.setLevel(logging.INFO)
    return logger

class Timings:
    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self._mean_deques = {}
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

        if name not in self._mean_deques:
            self._mean_deques[name] = collections.deque(maxlen=5)
        self._mean_deques[name].append(x)

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v**0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        mean_deques = self._mean_deques
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms (last 5: %.6fms) +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * np.average(mean_deques[k]),
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result

class Wandb:
    def __init__(self, flags, subname=""):
        import wandb

        self.wandb = wandb
        xpid = flags.full_xpid if hasattr(flags, "full_xpid") else flags.xpid
        exp_name = xpid + subname
        tags = []
        if subname == "_model":
            tags.append("model")
        m = re.match(r"^v\d+", exp_name)
        if m:
            tags.append(m[0])
        self.wandb.init(
            project=flags.project,
            config=flags,
            entity=os.getenv("WANDB_USER", ""),
            reinit=True,
            # Restore parameters
            resume="allow",
            id=exp_name,
            name=exp_name,
            tags=tags,
        )
        self.wandb.config.update(flags, allow_val_change=True)

def compute_grad_norm(parameters, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
        norm_type,
    )
    return total_norm

def slice_tree_reps(num_actions, dim_actions, rec_t):
    idx1 = num_actions * 5 + 6
    idx2 = idx1
    idx3 = idx2 + num_actions * 5 + 3
    idx4 = idx3
    idx5 = idx4 + 2 + rec_t  
    tree_rep_map = [
        ["root_action", 0],
        ["root_r", num_actions],
        ["root_d", num_actions+1],
        ["root_v", num_actions+2],
        ["root_policy", num_actions+3],
        ["root_qs_mean", 2*num_actions+3],
        ["root_qs_max", 3*num_actions+3],
        ["root_ns", 4*num_actions+3],
        ["root_trail_r", 5*num_actions+3],
        ["rollout_return", 5*num_actions+4],
        ["max_rollout_return", 5*num_actions+5],
        ["root_raw_action", idx1],
        ["cur_action", idx2],
        ["cur_r", idx2+num_actions],
        ["cur_d", idx2+num_actions+1],
        ["cur_v", idx2+num_actions+2],
        ["cur_policy", idx2+num_actions+3],
        ["cur_qs_mean", idx2+2*num_actions+3],
        ["cur_qs_max", idx2+3*num_actions+3],
        ["cur_ns", idx2+4*num_actions+3],
        ["cur_raw_action", idx3],
        ["cur_reset", idx4],
        ["k", idx4+1],
        ["deprec", idx4+1+rec_t],
        ["action_seq", idx5]
        ]
    tree_rep_map_d = {}
    for n, (k, idx) in enumerate(tree_rep_map):
        next_idx = tree_rep_map[n+1][1] if n + 1 < len(tree_rep_map) else None
        tree_rep_map_d[k] = slice(idx, next_idx)    
    return tree_rep_map_d


def slice_dynamic_tree_reps(num_actions, dim_actions=1):
    """Return the budget-independent Dynamic Thinker token schema.

    A token contains the original detailed root and compact current-node
    statistics followed by five scalar search metadata values.  The feature
    width is therefore ``10 * num_actions + 14`` and is independent of both
    max_search_steps and max_depth.
    """
    del dim_actions  # the current wrappers expose discrete augmented actions
    root_end = num_actions * 5 + 6
    cur_end = root_end + num_actions * 5 + 3
    names = [
        ["root_action", 0],
        ["root_r", num_actions],
        ["root_d", num_actions + 1],
        ["root_v", num_actions + 2],
        ["root_policy", num_actions + 3],
        ["root_qs_mean", 2 * num_actions + 3],
        ["root_qs_max", 3 * num_actions + 3],
        ["root_ns", 4 * num_actions + 3],
        ["root_trail_r", 5 * num_actions + 3],
        ["rollout_return", 5 * num_actions + 4],
        ["max_rollout_return", 5 * num_actions + 5],
        ["cur_action", root_end],
        ["cur_r", root_end + num_actions],
        ["cur_d", root_end + num_actions + 1],
        ["cur_v", root_end + num_actions + 2],
        ["cur_policy", root_end + num_actions + 3],
        ["cur_qs_mean", root_end + 2 * num_actions + 3],
        ["cur_qs_max", root_end + 3 * num_actions + 3],
        ["cur_ns", root_end + 4 * num_actions + 3],
        ["tree_reset", cur_end],
        ["depth_discount", cur_end + 1],
        ["search_steps", cur_end + 2],
        ["rollout_depth", cur_end + 3],
        ["search_start", cur_end + 4],
    ]
    out = {}
    for i, (key, start) in enumerate(names):
        end = names[i + 1][1] if i + 1 < len(names) else cur_end + 5
        out[key] = slice(start, end)
    # Compatibility name used by visualizers for the reset event scalar.
    out["cur_reset"] = out["tree_reset"]
    return out


def get_tree_rep_meaning(num_actions, dim_actions, flags):
    if dynamic_search_enabled(flags):
        return slice_dynamic_tree_reps(num_actions, dim_actions)
    return slice_tree_reps(num_actions, dim_actions, flags.rec_t)

def _decode_tree_reps_with_schema(tree_reps, schema, enc_type=0, f_type=0):
    nd = [
            "root_r", "root_v", "root_qs_mean", "root_qs_max", 
            "root_trail_r", "rollout_return", "max_rollout_return", 
            "cur_r", "cur_v", "cur_qs_mean", "cur_qs_max"
        ]
    def dec_k(x, key):        
        if enc_type != 0 and key in nd:
            return dec(x, f_type)
        else:
            return x

    if len(tree_reps.shape) == 3:
        tree_reps = tree_reps[0]

    return {k: dec_k(tree_reps[:, v], k) for k, v in schema.items()}


def decode_tree_reps(tree_reps, num_actions, dim_actions, rec_t, enc_type=0, f_type=0):
    """Decode the legacy fixed-budget tree representation."""
    schema = slice_tree_reps(num_actions, dim_actions, rec_t)
    return _decode_tree_reps_with_schema(
        tree_reps, schema, enc_type=enc_type, f_type=f_type
    )


def decode_dynamic_tree_reps(
        tree_reps, num_actions, dim_actions=1, enc_type=0, f_type=0):
    """Decode a budget-independent Dynamic Thinker token."""
    schema = slice_dynamic_tree_reps(num_actions, dim_actions)
    return _decode_tree_reps_with_schema(
        tree_reps, schema, enc_type=enc_type, f_type=f_type
    )

def mask_tree_rep(tree_reps, num_actions, rec_t):
    # deprecated
    d = slice_tree_reps(num_actions, rec_t)  
    N, C = tree_reps.shape
    act_seq_len = C - (num_actions * 10 + 11 + rec_t)
    tree_reps_m = torch.zeros(N, 4+rec_t+act_seq_len, device=tree_reps.device)
    tree_reps_m[:, [0]] = tree_reps[:, d["reset"]]
    tree_reps_m[:, [1]] = tree_reps[:, d["cur_r"]] # imagainary reward
    tree_reps_m[:, [2]] = tree_reps[:, d["cur_d"]] # imagainary done
    tree_reps_m[:, [3]] = tree_reps[:, d["derec"]] # deprec
    tree_reps_m[:, 4:4+rec_t] = tree_reps[:, d["k"]] # time
    tree_reps_m[:, 4+rec_t:] = tree_reps[:, d["action_seq"]]
    return tree_reps_m

def encode_action(action, action_space, one_hot=False):
    if type(action_space) == spaces.discrete.Discrete:       
        if one_hot:
            return action
        else:
            return F.one_hot(action.squeeze(-1), num_classes=action_space.n).float()
    elif type(action_space) == spaces.tuple.Tuple:   
            if one_hot:
                action = torch.sum(action * torch.arange(action_space[0].n, device=action.device), dim=-1)   
            return action.float()/action_space[0].n
    elif type(action_space) == spaces.Box:  
            return action.float()
    
def process_action_space(action_space):
    if type(action_space) == spaces.discrete.Discrete:                        
        num_actions = action_space.n    
        dim_actions = 1
        dim_rep_actions = num_actions
        tuple_action = False        
        discrete_action = True
    elif type(action_space) == spaces.tuple.Tuple:              
        num_actions = action_space[0].n    
        dim_actions = len(action_space)    
        dim_rep_actions = dim_actions
        tuple_action = True
        discrete_action = True
    elif type(action_space) == spaces.Box:  
        num_actions = 1   
        dim_actions = action_space.shape[0] 
        dim_rep_actions = dim_actions
        tuple_action = True
        discrete_action = False
    else:
        raise AssertionError(f"Unsupported action space {action_space}")
    return num_actions, dim_actions, dim_rep_actions, tuple_action, discrete_action

def plot_raw_state(x, ax=None, title=None, savepath=None):
    if ax is None:
        _, ax = plt.subplots()
    if not isinstance(x, np.ndarray):
        x = x[-3:].detach().cpu().numpy()
    else:
        x = x[-3:]    
    # Swap axes
    x = np.swapaxes(np.swapaxes(x, 0, 2), 0, 1)
    if x.dtype in [float, np.float32]:
        x = np.clip(x, 0, 1)
    if x.dtype in [int, np.uint8]:
        x = np.clip(x, 0, 255)
    ax.imshow(x, interpolation="nearest", aspect="auto")
    if title is not None:
        ax.set_title(title)
    if savepath is not None:
        plt.savefig(os.path.join(savepath, title + ".png"))
        plt.close()

def check_perfect_model(wrapper_type):
    return wrapper_type in [2, 4, 5]        

class FifoBuffer:
    def __init__(self, size, device):
        self.size = size
        self.buffer = torch.empty(
            (self.size,), dtype=torch.float32, device=device
        ).fill_(float("nan"))
        self.current_index = 0
        self.num_elements = 0

    def push(self, data):
        num_entries = math.prod(data.shape)
        assert num_entries <= self.size, "Data too large for buffer"

        start_index = self.current_index
        end_index = (self.current_index + num_entries) % self.size

        if end_index < start_index:
            # The new data wraps around the buffer
            remaining_space = self.size - start_index
            self.buffer[start_index:] = data.flatten()[:remaining_space]
            self.buffer[:end_index] = data.flatten()[remaining_space:]
        else:
            # The new data fits within the remaining space
            self.buffer[start_index:end_index] = data.flatten()

        self.current_index = end_index
        self.num_elements = min(self.num_elements + num_entries, self.size)

    def get_percentile(self, percentile):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0:
            return None
        return torch.quantile(self.buffer[:num_valid_elements], q=percentile)

    def get_variance(self):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0:
            return None
        return torch.mean(torch.square(self.buffer[:num_valid_elements]))

    def get_mean(self):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0:
            return None
        return torch.mean(self.buffer[:num_valid_elements])

    def full(self):
        return self.num_elements >= self.size

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
    
def clone_bn_running_stats(module):
    """
    Traverse the module and its submodules to clone all BatchNorm layers' running mean and variance.
    
    Parameters:
    - module: The root module to traverse.
    
    Returns:
    - A dictionary containing the cloned running mean and variance for each BatchNorm layer.
    """
    cloned_stats = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
            # Use the module's name as a unique identifier
            cloned_stats[name] = {
                "running_mean": submodule.running_mean.clone(),
                "running_var": submodule.running_var.clone(),
            }
    return cloned_stats

def restore_bn_running_stats(module, cloned_stats):
    """
    Traverse the module and its submodules to restore BatchNorm layers' running mean and variance from cloned statistics.
    
    Parameters:
    - module: The root module to traverse.
    - cloned_stats: A dictionary containing the cloned running mean and variance for each BatchNorm layer.
    """
    for name, submodule in module.named_modules():
        if name in cloned_stats and isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Restore the running statistics from the cloned values
            submodule.running_mean = cloned_stats[name]["running_mean"]
            submodule.running_var = cloned_stats[name]["running_var"]
