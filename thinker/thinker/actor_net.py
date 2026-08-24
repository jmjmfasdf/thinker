from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from thinker import util
from thinker.core.rnn import ConvAttnLSTM, MaskedGRU
from thinker.core.module import MLP, OneDResBlock, tile_and_concat_tensors
from thinker.model_net import RVTran
from gymnasium import spaces

SEARCH_PHASE = getattr(util, "SEARCH_PHASE", 0)
NEED_REAL_ACTION_PHASE = getattr(util, "NEED_REAL_ACTION_PHASE", 1)
WAIT_PHASE = getattr(util, "WAIT_PHASE", 2)
PROCEED = getattr(util, "PROCEED", 0)
RESET = getattr(util, "RESET", 1)
STOP = getattr(util, "STOP", 2)
POLICY_NONE = getattr(util, "POLICY_NONE", 0)
POLICY_SEARCH = getattr(util, "POLICY_SEARCH", 1)
POLICY_REAL = getattr(util, "POLICY_REAL", 2)


ActorOut = namedtuple(
    "ActorOut",
    [     
        "pri", # sampled primiary action
        "pri_param", # parameter for primary action dist, can be logit or gaussian mean + log var        
        "reset", # sampled reset action
        "reset_logits", # parameter for reset dist, i.e. logit
        "action", # tuple of the above two actions 
        "action_prob", # prob of primary action 
        "c_action_log_prob", # log prob of chosen action
        "baseline", # baseline 
        "baseline_enc", # baseline encoding, only for non-scalar enc_type
        "entropy_loss", # entropy loss
        "reg_loss", # regularization loss
        "misc",
        # Dynamic-search fields.  The legacy reset fields above intentionally
        # remain in place so fixed-budget checkpoints and consumers retain
        # their original tuple contract.
        "search_control",
        "search_control_logits",
        "primary_valid",
        "control_valid",
        "policy_valid",
        "policy_type",
    ],
    defaults=[None] * 6,
)

def compute_discrete_log_prob(logits, actions):
    assert len(logits.shape) == len(actions.shape) + 1
    has_dim = len(actions.shape) == 3    
    end_dim = 2 if has_dim else 1
    log_prob = -torch.nn.CrossEntropyLoss(reduction="none")(
            input=torch.flatten(logits, 0, end_dim), target=torch.flatten(actions, 0, end_dim)
    )
    log_prob = log_prob.view_as(actions)
    if has_dim:
        log_prob = torch.sum(log_prob, dim=-1)
    return log_prob


def sample(logits, greedy, dim=-1):
    if not greedy:
        gumbel_noise = torch.empty_like(logits).uniform_().clamp(1e-10, 1).log().neg_().clamp(1e-10, 1).log().neg_()
        sampled_action = (logits + gumbel_noise).argmax(dim=dim)
        return sampled_action.detach()
    else:
        return torch.argmax(logits, dim=dim)

def atanh(x, eps=1e-6):
    x = torch.clamp(x, -1.0+eps, 1.0-eps)
    return 0.5 * (x.log1p() - (-x).log1p())

class AFrameEncoder(nn.Module):
    # processor for 3d inputs; can be applied to model's hidden state or predicted real state
    def __init__(self, 
                 input_shape, 
                 flags,
                 downpool=False, 
                 firstpool=False,    
                 out_size=256,
                 see_double=False,                 
                 ):
        super(AFrameEncoder, self).__init__()
        if see_double:
            input_shape = (input_shape[0] // 2,) + tuple(input_shape[1:])
        self.input_shape = input_shape        
        self.downpool = downpool
        self.firstpool = firstpool
        self.out_size = out_size
        self.see_double = see_double    
        self.enc_1d_shallow = getattr(flags, "enc_1d_shallow", False)
        self.flags = flags    

        self.oned_input = len(self.input_shape) == 1
        if self.enc_1d_shallow and self.oned_input: self.out_size = 64

        in_channels = input_shape[0]
        if not self.oned_input:
            # following code is from Torchbeast, which is the same as Impala deep model            
            conv_out_h = input_shape[1]
            conv_out_w = input_shape[2]

            self.feat_convs = []
            self.resnet1 = []
            self.resnet2 = []
            self.convs = []

            if firstpool:
                self.down_pool_conv = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                in_channels = 16
                conv_out_h = (conv_out_h - 1) // 2 + 1
                conv_out_w = (conv_out_w - 1) // 2 + 1

            num_chs = [16, 32, 32] if downpool else [64, 64, 32]
            for num_ch in num_chs:
                feats_convs = []
                feats_convs.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if downpool:
                    feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                    conv_out_h = (conv_out_h - 1) // 2 + 1
                    conv_out_w = (conv_out_w - 1) // 2 + 1
                self.feat_convs.append(nn.Sequential(*feats_convs))
                in_channels = num_ch
                for i in range(2):
                    resnet_block = []
                    resnet_block.append(nn.ReLU())
                    resnet_block.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=num_ch,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                    resnet_block.append(nn.ReLU())
                    resnet_block.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=num_ch,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                    if i == 0:
                        self.resnet1.append(nn.Sequential(*resnet_block))
                    else:
                        self.resnet2.append(nn.Sequential(*resnet_block))
            self.feat_convs = nn.ModuleList(self.feat_convs)
            self.resnet1 = nn.ModuleList(self.resnet1)
            self.resnet2 = nn.ModuleList(self.resnet2)

            # out shape after conv is: (num_ch, input_shape[1], input_shape[2])
            core_out_size = num_ch * conv_out_h * conv_out_w
        else:
            if not self.enc_1d_shallow:
                n_block = self.flags.enc_1d_block
                hidden_size = self.flags.enc_1d_hs
                self.hidden_size = hidden_size
                self.input_block = nn.Sequential(
                    nn.Linear(in_channels, hidden_size),
                    nn.ReLU()
                )            
                self.res = nn.Sequential(*[OneDResBlock(hidden_size, norm=self.flags.enc_1d_norm) for _ in range(n_block)])
                core_out_size = hidden_size
            else:
                self.input_block = nn.Sequential(nn.Linear(in_channels, 64), nn.Tanh())            
                self.res = nn.Identity()   
                core_out_size = 64      
        
        mlp_out_size = self.out_size if not self.see_double else self.out_size // 2
        self.fc = nn.Sequential(nn.Linear(core_out_size, mlp_out_size), nn.ReLU())
            

    def forward(self, x, record_state=False):
        if not self.see_double:
            return self.forward_single(x, record_state=record_state)
        else:
            out_1 = self.forward_single(x[:, :self.input_shape[0]], record_state=record_state)
            out_2 = self.forward_single(x[:, self.input_shape[0]:])            
            return torch.concat([out_1, out_2], dim=1)

    def forward_single(self, x, record_state=False):
        """encode the state or model's encoding inside the actor network
        args:
            x: input tensor of shape (B, C, H, W); can be state or model's encoding
        return:
            output: output tensor of shape (B, self.out_size)"""
        assert x.dtype in [torch.float, torch.float16]
        if not self.oned_input:
            if self.firstpool:
                x = self.down_pool_conv(x)
            if record_state: self.hidden_state = []
            for i, fconv in enumerate(self.feat_convs):                
                x = fconv(x)
                res_input = x
                x = self.resnet1[i](x)
                x += res_input
                res_input = x
                x = self.resnet2[i](x)
                x += res_input
                if record_state: self.hidden_state.append(x)
            x = torch.flatten(x, start_dim=1)
        else:
            x = self.input_block(x)
            x = self.res(x)
        x = self.fc(F.relu(x))
        if record_state: self.hidden_state = tile_and_concat_tensors(self.hidden_state)
        return x

class RNNEncoder(nn.Module):
    # RNN processor for 1d inputs; can be used directly on tree rep or encoded 3d input
    def __init__(self, 
                 in_size, # int; input size
                 flags            
                 ):
        super(RNNEncoder, self).__init__()  
        self.rnn_in_fc = nn.Sequential(
                    nn.Linear(in_size, flags.tran_dim), nn.ReLU()
        )  
        self.tran_layer_n = flags.tran_layer_n 
        if self.tran_layer_n > 0:
            self.rnn = ConvAttnLSTM(
                input_dim=flags.tran_dim,
                hidden_dim=flags.tran_dim,
                num_layers=flags.tran_layer_n,
                attn=not flags.tran_lstm_no_attn,
                mem_n=flags.tran_mem_n,
                num_heads=flags.tran_head_n,
                attn_mask_b=flags.tran_attn_b,
                tran_t=flags.tran_t,
            ) 
        self.rnn_out_fc = nn.Sequential(
            nn.Linear(flags.tran_dim, flags.tran_dim), nn.ReLU()
        )

    def initial_state(self, batch_size=1, device=None):
        if self.tran_layer_n > 0:
            return self.rnn.initial_state(batch_size, device=device)
        else:
            return ()

    def forward(self, x, done, core_state, record_state=False, update_mask=None):
        # input should have shape (T*B, C) 
        # done should have shape (T, B)
        T, B = done.shape
        x = self.rnn_in_fc(x)
        if self.tran_layer_n >= 1:
            x = x.view(*((T, B) + x.shape[1:])).unsqueeze(-1).unsqueeze(-1)            
            core_output, core_state = self.rnn(
                x,
                done,
                core_state,
                record_state,
                update_mask=update_mask,
            )
            core_output = torch.flatten(core_output, 0, 1)
            d = torch.flatten(core_output, 1)   
        else:
            d = x     
        d = self.rnn_out_fc(d)
        return d, core_state
    
class ActorBaseNet(nn.Module):
    # base class for all actor network
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=False, record_state=False):
        super(ActorBaseNet, self).__init__()
        self.disable_thinker = flags.wrapper_type == 1
        self.dynamic_search = bool(getattr(flags, "dynamic_search", False)) and not self.disable_thinker
        self.record_state = record_state        

        self.obs_space = obs_space        
        if not self.disable_thinker:
            self.pri_action_space = action_space[0][0]            
        else:
            self.pri_action_space = action_space[0]

        self.flags = flags      
        self.tree_rep_meaning = tree_rep_meaning

        self.float16 = flags.float16
        # Keep the actor/critic output channels in the single canonical order
        # used by rollout construction and the learner.  Dynamic search appends
        # the computation-cost channel without moving legacy reward rows.
        self.num_rewards = len(util.get_reward_names(flags))
        # The computation cost is a return/value target, not an observation.
        # Keeping actor inputs to the legacy reward prefix preserves every
        # downstream feature width during fixed -> dynamic checkpoint loading.
        self.num_input_rewards = 1
        self.num_input_rewards += int(flags.im_cost > 0.0)
        self.num_input_rewards += int(flags.cur_cost > 0.0)
        self.enc_type = flags.critic_enc_type  
        self.rv_tran = None
        self.critic_zero_init = flags.critic_zero_init     

        # action space processing
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(self.pri_action_space)
        
        self.ordinal = flags.actor_ordinal
        if self.ordinal:
            indices = torch.arange(self.num_actions).view(-1, 1)
            ordinal_mask = (indices + indices.T) <= (self.num_actions - 1)
            ordinal_mask = ordinal_mask.float()
            self.register_buffer("ordinal_mask", ordinal_mask)

        # state space processing
        self.see_tree_rep = flags.see_tree_rep and not self.disable_thinker
        if self.see_tree_rep:
            self.tree_reps_shape = obs_space["tree_reps"].shape[1:]   

        self.see_h = flags.see_h and not self.disable_thinker
        if self.see_h:
            self.hs_shape = obs_space["hs"].shape[1:]
        self.see_x = flags.see_x
        if self.see_x and not self.disable_thinker:
            self.xs_shape = obs_space["xs"].shape[1:]
        self.see_real_state = flags.see_real_state        
        
        if flags.see_real_state:
            assert obs_space["real_states"].dtype in ['uint8', 'float32'], f"Unupported observation sapce {obs_space['real_states']}"            
            low = torch.tensor(obs_space["real_states"].low[0])
            high = torch.tensor(obs_space["real_states"].high[0])
            self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()            
            if self.need_norm:
                self.register_buffer("norm_low", low)
                self.register_buffer("norm_high", high)
            self.real_states_shape = obs_space["real_states"].shape[1:]     

        self.autotune = flags.autotune
        if self.autotune:
            log_entropy_cost = nn.Parameter(torch.log(torch.full((1,), 0.1)))
            log_im_entropy_cost = nn.Parameter(torch.log(torch.full((1,), 0.1)))
            self.register_parameter("log_entropy_cost", log_entropy_cost)
            self.register_parameter("log_im_entropy_cost", log_im_entropy_cost)

        if flags.ppo_k > 1:
            kl_beta = torch.tensor(1.)
            self.register_buffer("kl_beta", kl_beta)

    def normalize(self, x):
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        if self.need_norm:
            x = (x.float() - self.norm_low) / \
                (self.norm_high -  self.norm_low)
        return x
    
    def ordinal_encode(self, logits):
        norm_softm = F.sigmoid(logits)
        norm_softm_tiled = torch.tile(norm_softm.unsqueeze(-1), [1,1,1,self.num_actions])
        return torch.sum(torch.log(norm_softm_tiled + 1e-8) * self.ordinal_mask + torch.log(1 - norm_softm_tiled + 1e-8) * (1 - self.ordinal_mask), dim=-1)

    def get_weights(self):
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}    

    def set_weights(self, weights, strict=True):
        device = next(self.parameters()).device
        tensor = isinstance(next(iter(weights.values())), torch.Tensor)
        incoming = (
            {k: torch.tensor(v, device=device) for k, v in weights.items()}
            if not tensor
            else {k: v.to(device) for k, v in weights.items()}
        )
        if not self.dynamic_search:
            self.load_state_dict(incoming, strict=strict)
            return

        # Resume must never fall through to the legacy partial-migration path.
        # A strict Dynamic checkpoint is required to match every key and shape.
        if strict:
            self.load_state_dict(incoming, strict=True)
            return

        # Dynamic search deliberately replaces the flattened tree MLP with a
        # masked GRU and expands the binary reset head to a ternary control
        # head.  PyTorch's strict=False still raises on same-key shape
        # mismatches, so filter/migrate them explicitly.  Exact dynamic
        # checkpoints still take the ordinary strict path.
        target = self.state_dict()
        if incoming.keys() == target.keys() and all(
            incoming[k].shape == target[k].shape for k in target
        ):
            self.load_state_dict(incoming, strict=False)
            return

        migrated = {}
        for key, value in incoming.items():
            if key not in target:
                continue
            target_value = target[key]
            if value.shape == target_value.shape:
                migrated[key] = value
                continue

            # Preserve the learned PROCEED/RESET rows and initialize the new
            # STOP row to zero.  This works for ActorNetSingle and nested
            # ActorNetSep keys alike.
            if (
                key.endswith("reset.weight")
                and value.ndim == 2
                and target_value.ndim == 2
                and value.shape[0] == 2
                and target_value.shape[0] == 3
                and value.shape[1] == target_value.shape[1]
            ):
                expanded = torch.zeros_like(target_value)
                expanded[:2] = value
                migrated[key] = expanded
                continue
            if (
                key.endswith("reset.bias")
                and value.ndim == 1
                and target_value.ndim == 1
                and value.shape[0] == 2
                and target_value.shape[0] == 3
            ):
                expanded = torch.zeros_like(target_value)
                expanded[:2] = value
                migrated[key] = expanded
                continue

            # Baseline output expansion (e.g. an added thinking-cost value
            # row): retain all legacy rows and leave new rows zero-initialized.
            if (
                (key.endswith("baseline.weight") or key.endswith("baseline.bias"))
                and value.ndim == target_value.ndim
                and value.shape[0] < target_value.shape[0]
                and value.shape[1:] == target_value.shape[1:]
            ):
                expanded = torch.zeros_like(target_value)
                expanded[: value.shape[0]] = value
                migrated[key] = expanded

        self.load_state_dict(migrated, strict=False)

class ActorNetSep(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False):
        super(ActorNetSep, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        self.actor = ActorNetSingle(obs_space, action_space, flags, tree_rep_meaning, record_state, actor=True, critic=False)
        self.critic = ActorNetSingle(obs_space, action_space, flags, tree_rep_meaning, record_state, actor=False, critic=True)
        self.initial_state(1)
        self.rv_tran = self.critic.rv_tran

    def initial_state(self, batch_size, device=None):
        actor_state = self.actor.initial_state(batch_size, device)
        critic_state = self.critic.initial_state(batch_size, device)
        self.state_idx = len(actor_state)
        return actor_state + critic_state
    
    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        actor_state = core_state[:self.state_idx]
        critic_state = core_state[self.state_idx:]
        actor_out, actor_state = self.actor(env_out, actor_state, clamp_action, compute_loss, greedy)
        critic_out, critic_state = self.critic(env_out, critic_state, clamp_action, compute_loss, greedy)
        misc = actor_out.misc
        actor_out = ActorOut(
            pri=actor_out.pri,
            pri_param=actor_out.pri_param,
            reset=actor_out.reset,
            reset_logits=actor_out.reset_logits,
            action=actor_out.action,
            action_prob=actor_out.action_prob,
            c_action_log_prob=actor_out.c_action_log_prob,            
            baseline=critic_out.baseline,
            baseline_enc=critic_out.baseline_enc,
            entropy_loss=actor_out.entropy_loss,
            reg_loss=actor_out.reg_loss,
            misc=misc,
            search_control=actor_out.search_control,
            search_control_logits=actor_out.search_control_logits,
            primary_valid=actor_out.primary_valid,
            control_valid=actor_out.control_valid,
            policy_valid=actor_out.policy_valid,
            policy_type=actor_out.policy_type,
        )
        core_state = actor_state + critic_state
        return actor_out, core_state

class ActorNetSingle(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False, actor=True, critic=True):
        super(ActorNetSingle, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)      
                  
        self.actor = actor
        self.critic = critic

        if not self.discrete_action and self.actor:
            min_log_var = 2 * torch.log(torch.tensor(flags.actor_min_std))
            max_log_var = 2 * torch.log(torch.tensor(flags.actor_max_std))
            self.register_buffer("min_log_var", min_log_var)
            self.register_buffer("max_log_var", max_log_var)   

        self.tran_dim = flags.tran_dim 
        self.tran_reset_mode = flags.tran_reset_mode
        self.dynamic_tree_rep = self.dynamic_search and self.see_tree_rep
        self.tree_rep_rnn = (
            flags.tree_rep_rnn and flags.see_tree_rep and not self.dynamic_tree_rep
        )
        self.se_lstm_table = (
            getattr(flags, "se_lstm_table", False)
            and flags.see_tree_rep
            and flags.wrapper_type in [3, 4]
            and not self.dynamic_tree_rep
        )
        self.x_rnn = flags.x_rnn and flags.see_x  
        self.h_rnn = flags.h_rnn and flags.see_h
        self.real_state_rnn = flags.real_state_rnn and flags.see_real_state         

        self.sep_im_head = flags.sep_im_head
        self.last_layer_n = flags.last_layer_n
          
        # encoder for state or encoding output
        last_out_size = self.dim_rep_actions + self.num_input_rewards

        if not self.disable_thinker:
            last_out_size += 2

        if self.see_h:
            FrameEncoder = AFrameEncoder 
            self.h_encoder = FrameEncoder(
                input_shape=self.hs_shape,                    
                flags=flags,                      
            )
            h_out_size = self.h_encoder.out_size
            if self.h_rnn:
                rnn_in_size = h_out_size
                self.h_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                h_out_size = flags.tran_dim            
            last_out_size += h_out_size   
        
        if self.see_x:
            FrameEncoder = AFrameEncoder 
            self.x_encoder_pre = FrameEncoder(
                input_shape=self.xs_shape,                 
                flags=flags,
                downpool=True,
                firstpool=True,
            )
            x_out_size = self.x_encoder_pre.out_size
            if self.x_rnn:
                rnn_in_size = x_out_size + self.dim_rep_actions + 2
                self.x_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                x_out_size = flags.tran_dim            
            last_out_size += x_out_size           

        if self.see_real_state:
            self.real_state_ch = getattr(flags, "real_state_ch", -1)
            if self.real_state_ch > 0:
                self.real_states_shape = list(self.real_states_shape)
                self.real_states_shape[0] = int(self.real_state_ch)
                self.real_states_shape = tuple(self.real_states_shape)            
            self.real_state_encoder =  AFrameEncoder(
                input_shape=self.real_states_shape,                 
                flags=flags,
                downpool=True,
                firstpool=True,
            )                        
            r_out_size = self.real_state_encoder.out_size
            self.pre_r_shape = (r_out_size,)
            if self.real_state_rnn:
                rnn_in_size = r_out_size
                self.r_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                r_out_size = flags.tran_dim   
            self.post_r_shape = (r_out_size,)
            last_out_size += r_out_size       

        if self.see_tree_rep:            
            self.tree_rep_meaning = tree_rep_meaning
            in_size = self.tree_reps_shape[0]
            if self.dynamic_tree_rep:
                tree_hidden_size = int(
                    getattr(flags, "dynamic_search_hidden_dim", 100)
                )
                tree_layer_n = int(getattr(flags, "dynamic_search_layer_n", 1))
                self.tree_rep_encoder = MaskedGRU(
                    input_size=in_size,
                    hidden_size=tree_hidden_size,
                    num_layers=tree_layer_n,
                )
                self.tree_rep_out_size = tree_hidden_size
                last_out_size += tree_hidden_size
            elif self.se_lstm_table:
                assert flags.se_query_cur == 2                
                root_table_mask = torch.zeros(in_size, dtype=torch.bool)
                root_query_keys = [k for k in tree_rep_meaning if k.startswith("root_query")]
                for i in root_query_keys:
                    root_table_mask[self.tree_rep_meaning[i]] = 1        
                # print("root_query_size: ", sum(root_table_mask).long().item())        
                cur_table_mask = torch.zeros(in_size, dtype=torch.bool)
                cur_query_keys = [k for k in tree_rep_meaning if k.startswith("cur_query")]
                for i in cur_query_keys:
                    cur_table_mask[self.tree_rep_meaning[i]] = 1
                # print("cur_query_size: ", sum(cur_table_mask).long().item())        
                non_table_mask = torch.logical_or(root_table_mask, cur_table_mask)
                non_table_mask = torch.logical_not(non_table_mask)
                self.register_buffer("root_table_mask", root_table_mask)
                self.register_buffer("cur_table_mask", cur_table_mask)
                self.register_buffer("non_table_mask", non_table_mask)
                input_size = (sum(root_table_mask) / flags.se_query_size).long().item()
                self.tree_rep_table_lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=3, batch_first=True)
                in_size = torch.sum(non_table_mask).long() + 64 * 2           

            if self.dynamic_tree_rep:
                pass
            elif self.tree_rep_rnn:
                self.tree_rep_encoder = RNNEncoder(
                    in_size=in_size,
                    flags=flags
                )
                last_out_size += flags.tran_dim
            else:
                self.tree_rep_encoder = MLP(
                    input_size=in_size,
                    layer_sizes=[200, 200, 200],
                    output_size=100,
                    norm=False,
                    skip_connection=True,
                )
                last_out_size += 100        

        if self.last_layer_n > 0:
            self.final_mlp =  MLP(
                input_size=last_out_size,
                layer_sizes=[200]*self.last_layer_n,
                output_size=100,
                norm=False,
                skip_connection=True,
            )
            last_out_size = 100

        if self.dynamic_search:
            # SEARCH, NEED_REAL_ACTION and WAIT can expose otherwise identical
            # observations but require different heads/semantics.  An additive
            # embedding makes the phase observable without changing any
            # downstream tensor width.  Zero initialization keeps migrated
            # fixed-budget policies behaviourally neutral at load time.
            self.phase_embedding = nn.Embedding(3, last_out_size)
            nn.init.zeros_(self.phase_embedding.weight)

        if self.actor:
            self.policy = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
            self.im_policy = self.policy
            if not self.discrete_action:
                self.tanh_action = flags.tanh_action
                self.policy_lvar = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
                self.im_policy_lvar = self.policy

            if not self.disable_thinker:
                if self.sep_im_head:
                    self.im_policy = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
                    if not self.discrete_action:
                        self.im_policy_lvar = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
                    
                self.reset = nn.Linear(last_out_size, 3 if self.dynamic_search else 2)

        if self.critic:
            self.rv_tran = None
            if self.enc_type == 0:
                self.baseline = nn.Linear(last_out_size, self.num_rewards)
                if self.flags.reward_clip > 0:
                    self.baseline_clamp = self.flags.reward_clip / (
                        1 - self.flags.discounting
                    )
            elif self.enc_type == 1:
                self.baseline = nn.Linear(last_out_size, self.num_rewards)
                self.rv_tran = RVTran(enc_type=self.enc_type, enc_f_type=flags.critic_enc_f_type)
            elif self.enc_type in [2, 3]:                        
                self.rv_tran = RVTran(enc_type=self.enc_type, enc_f_type=flags.critic_enc_f_type)
                self.out_n = self.rv_tran.encoded_n
                self.baseline = nn.Linear(last_out_size, self.num_rewards * self.out_n)            

            if self.critic_zero_init:
                nn.init.constant_(self.baseline.weight, 0.0)
                nn.init.constant_(self.baseline.bias, 0.0)                

        self.initial_state(batch_size=1) # initialize self.state_idx        

    def initial_state(self, batch_size, device=None):
        self.state_idx = {}
        idx = 0
        initial_state = ()
        
        conditions = [
            self.x_rnn,
            self.real_state_rnn,
            self.tree_rep_rnn or self.dynamic_tree_rep,
            self.h_rnn,
        ]
        rnn_names = ["x_encoder_rnn", "r_encoder_rnn", "tree_rep_encoder", "h_encoder_rnn"]
        state_names = ["x", "r", "tree_rep", "h"]

        for condition, rnn_name, state_name in zip(conditions, rnn_names, state_names):
            if condition:
                core_state = getattr(self, rnn_name).initial_state(batch_size, device=device)
                initial_state = initial_state + core_state
                self.state_idx[state_name] = slice(idx, idx+len(core_state))
                idx += len(core_state)

        if self.see_real_state:
            pre_encoded_real_state = torch.zeros((batch_size,) + self.pre_r_shape, device=device)
            initial_state = initial_state + (pre_encoded_real_state,)
            self.state_idx["pre_encoded_real_state"] = slice(idx, idx+1)
            idx += 1

            encoded_real_state = torch.zeros((batch_size,) + self.post_r_shape, device=device)
            initial_state = initial_state + (encoded_real_state,)
            self.state_idx["encoded_real_state"] = slice(idx, idx+1)
            idx += 1

        self.state_len = idx
        return initial_state
    
    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        """one-step forward for the actor;
        args:
            env_out (EnvOut):
                tree_reps (tensor): tree_reps output with shape (T x B x C)
                xs (tensor): optional - model predicted state with shape (T x B x C X H X W)                
                hs (tensor): optional - hidden state with shape (T x B x C X H X W)                
                real_states (tensor): optional - root's real state with shape (T x B x C X H X W)                
                done  (tensor): if episode ends with shape (T x B)
                step_status (tensor): current step status with shape (T x B)
                last_pri (tensor): last primiary action (non-one-hot) with shape (T x B)
                last_reset (tensor): last reset action (non-one-hot) with shape (T x B)
                and other environment output that is not used.
            core_state (tuple): rnn state of the actor network
            clamp_action (tuple): option - if not none, the sampled action will be set to this action;
                the main purpose is for computing c_action_log_prob
            compute_loss (boolean): wheather to return entropy loss and reg loss
            greedy (bool): whether to sample greedily
        return:
            ActorOut:
                see definition of ActorOut; this is a tuple with elements of 
                    shape (T x B x ...) except actor_out.action, which is a 
                    tuple of primiary and reset action, each with shape (B,),
                    selected on the last step
        """
        done = env_out.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape

        if self.dynamic_search:
            phase = getattr(env_out, "phase", None)
            if phase is None:
                # Transitional fallback for rollouts produced before phase was
                # added: legacy status 0/1 requests a search action and 2/3 a
                # real action.  Dynamic cenv always supplies phase explicitly.
                phase = torch.where(
                    env_out.step_status <= 1,
                    torch.full_like(env_out.step_status, SEARCH_PHASE),
                    torch.full_like(env_out.step_status, NEED_REAL_ACTION_PHASE),
                )
            phase = phase.to(device=done.device, dtype=torch.long)
            if phase.shape != (T, B):
                raise ValueError(
                    f"phase must have shape {(T, B)}, got {phase.shape}"
                )
            # Recurrent memories advance on an emitted tree/model token, not
            # merely because the *next* policy phase is SEARCH.  In
            # particular, the final computation at a positive safety cap
            # emits a valid token while transitioning directly to
            # NEED_REAL_ACTION.  STOP, action-store and ordinary WAIT calls
            # emit no token and therefore remain exact recurrent no-ops.
            recurrent_update_mask = getattr(env_out, "tree_token_valid", None)
            if recurrent_update_mask is None:
                recurrent_update_mask = phase == SEARCH_PHASE
            else:
                recurrent_update_mask = recurrent_update_mask.to(
                    device=done.device, dtype=torch.bool
                )
                if (recurrent_update_mask.ndim == 3
                        and recurrent_update_mask.shape[-1] == 1):
                    recurrent_update_mask = recurrent_update_mask.squeeze(-1)
                if recurrent_update_mask.shape != (T, B):
                    raise ValueError(
                        "tree_token_valid must have shape "
                        f"{(T, B)}, got {recurrent_update_mask.shape}"
                    )
        else:
            phase = None
            recurrent_update_mask = None

        assert len(core_state) == self.state_len, "core_state should have length %d" % self.state_len
        new_core_state = [None] * self.state_len

        if self.tran_reset_mode == 0:
            rnn_done = env_out.done
            if self.dynamic_search:
                rnn_done = rnn_done | env_out.truncated_done.bool()
        elif self.tran_reset_mode == 1:
            rnn_done = env_out.real_done
            if self.dynamic_search:
                rnn_done = rnn_done | env_out.truncated_done.bool()
        else:
            rnn_done = torch.zeros_like(env_out.done)

        final_out = []
        
        last_pri = torch.flatten(env_out.last_pri, 0, 1)
        if not self.tuple_action: last_pri = last_pri.unsqueeze(-1)
        last_pri = util.encode_action(last_pri, self.pri_action_space)

        last_control = None
        if self.dynamic_search:
            last_control = getattr(env_out, "last_search_control", None)
            if last_control is None:
                last_control = env_out.last_reset
            last_control = torch.flatten(last_control, 0, 1).long()
            # Dynamic rollout construction keeps last_pri at the most recent
            # accepted primary action when STOP/WAIT dummy inputs are ignored,
            # and replaces it with the stored real action in WAIT.  Consume
            # that stable token verbatim here; last_control can legitimately
            # remain STOP while last_pri advances to the stored real action.
        final_out.append(last_pri)

        if not self.disable_thinker:
            if self.dynamic_search:
                # Preserve the legacy two-feature width exactly.  STOP maps to
                # the zero vector; PROCEED/RESET retain their old one-hot
                # representation and therefore keep downstream weight shapes
                # checkpoint-compatible.
                last_reset = torch.stack(
                    [last_control == PROCEED, last_control == RESET], dim=-1
                ).to(dtype=last_pri.dtype)
            else:
                last_reset = torch.flatten(env_out.last_reset, 0, 1)
                last_reset = F.one_hot(last_reset, 2)
            final_out.append(last_reset)

        reward = env_out.reward[..., : self.num_input_rewards]
        reward = torch.where(torch.isnan(reward), torch.zeros_like(reward), reward)
        last_reward = torch.clamp(torch.flatten(reward, 0, 1), -1, +1).float()
        final_out.append(last_reward)

        if self.see_tree_rep:                
            tree_rep = env_out.tree_reps
            if self.dynamic_tree_rep:
                if tree_rep.shape[:2] != (T, B):
                    raise ValueError(
                        "dynamic tree_reps must have shape (T, B, C), got "
                        f"{tree_rep.shape}"
                    )
                tree_token_valid = getattr(env_out, "tree_token_valid", None)
                if tree_token_valid is None:
                    tree_token_valid = recurrent_update_mask
                else:
                    tree_token_valid = tree_token_valid.to(
                        device=tree_rep.device, dtype=torch.bool
                    )
                    if tree_token_valid.ndim == 3 and tree_token_valid.shape[-1] == 1:
                        tree_token_valid = tree_token_valid.squeeze(-1)
                search_state_reset = getattr(env_out, "search_state_reset", None)
                if search_state_reset is None:
                    search_state_reset = getattr(env_out, "real_transition", None)
                if search_state_reset is None:
                    search_state_reset = torch.zeros_like(tree_token_valid)
                else:
                    search_state_reset = search_state_reset.to(
                        device=tree_rep.device, dtype=torch.bool
                    )
                    if search_state_reset.ndim == 3 and search_state_reset.shape[-1] == 1:
                        search_state_reset = search_state_reset.squeeze(-1)
                core_state_ = core_state[self.state_idx['tree_rep']]
                encoded_tree_rep, core_state_ = self.tree_rep_encoder(
                    tree_rep,
                    update_mask=tree_token_valid,
                    reset_mask=search_state_reset,
                    state=core_state_,
                    record_state=self.record_state,
                )
                encoded_tree_rep = torch.flatten(encoded_tree_rep, 0, 1)
                new_core_state[self.state_idx['tree_rep']] = core_state_
            else:
                tree_rep = torch.flatten(tree_rep, 0, 1)

                if self.se_lstm_table:
                    root_table = tree_rep[:, self.root_table_mask]
                    root_table = torch.flip(tree_rep[:, self.root_table_mask].view(T*B, self.flags.se_query_size, -1), dims=[1])
                    root_table_rep, _ = self.tree_rep_table_lstm(root_table)
                    root_table_rep = root_table_rep[:, -1]
                    cur_table = tree_rep[:, self.cur_table_mask]
                    cur_table = torch.flip(cur_table.view(T*B, self.flags.se_query_size, -1), dims=[1])
                    cur_table_rep, _ = self.tree_rep_table_lstm(cur_table)
                    cur_table_rep = cur_table_rep[:, -1]
                    tree_rep = torch.concat([tree_rep[:, self.non_table_mask], root_table_rep, cur_table_rep], dim=-1)

            if self.dynamic_tree_rep:
                pass
            elif self.tree_rep_rnn:
                core_state_ = core_state[self.state_idx['tree_rep']]
                encoded_tree_rep, core_state_ = self.tree_rep_encoder(
                    tree_rep, rnn_done, core_state_)
                new_core_state[self.state_idx['tree_rep']] = core_state_
            else:
                encoded_tree_rep = self.tree_rep_encoder(tree_rep)
            final_out.append(encoded_tree_rep)
        
        if self.see_h:
            hs = torch.flatten(env_out.hs, 0, 1)
            encoded_h = self.h_encoder(hs)            

            if self.h_rnn:
                core_state_ = core_state[self.state_idx['h']]
                encoded_h, core_state_ = self.h_encoder_rnn(
                    encoded_h,
                    rnn_done,
                    core_state_,
                    update_mask=recurrent_update_mask,
                )
                new_core_state[self.state_idx['h']] = core_state_

            final_out.append(encoded_h)                

        if self.see_x:
            xs = torch.flatten(env_out.xs, 0, 1)
            with autocast(enabled=self.float16):                
                encoded_x = self.x_encoder_pre(xs)
            if self.float16: encoded_x = encoded_x.float()
                
            if self.x_rnn:
                encoded_x = torch.concat([encoded_x, last_pri, last_reset], dim=-1)
                core_state_ = core_state[self.state_idx['x']]                
                encoded_x, core_state_ = self.x_encoder_rnn(
                    encoded_x,
                    rnn_done,
                    core_state_,
                    update_mask=recurrent_update_mask,
                )
                new_core_state[self.state_idx['x']] = core_state_
            
            final_out.append(encoded_x)

        if self.see_real_state:
            pre_encoded_real_state, encoded_real_state, core_state_, pre_reg_loss = self.compute_encoded_real_state(env_out, core_state, rnn_done)
            if self.real_state_rnn:
                new_core_state[self.state_idx['r']] = core_state_
            new_core_state[self.state_idx['pre_encoded_real_state']] = (pre_encoded_real_state[-B:],)
            new_core_state[self.state_idx['encoded_real_state']] = (encoded_real_state[-B:],)
            final_out.append(encoded_real_state)        

        final_out = torch.concat(final_out, dim=-1)   

        if self.last_layer_n > 0:
            final_out = self.final_mlp(final_out)     

        if self.dynamic_search:
            final_out = final_out + self.phase_embedding(
                phase.flatten(0, 1)
            ).to(dtype=final_out.dtype)

        misc = {}
        if self.actor:
            # Compute both primary-policy branches before phase routing.
            real_logits = self.policy(final_out)
            real_logits = real_logits.view(T * B, self.dim_actions, self.num_actions)
            if self.ordinal:
                real_logits = self.ordinal_encode(real_logits)
            if not self.discrete_action:
                real_mean = real_logits[:, :, 0]
                real_log_var = torch.clamp(
                    self.policy_lvar(final_out), self.min_log_var, self.max_log_var
                )

            if not self.disable_thinker:
                search_logits = self.im_policy(final_out)
                search_logits = search_logits.view(
                    T * B, self.dim_actions, self.num_actions
                )
                if self.ordinal:
                    search_logits = self.ordinal_encode(search_logits)
                if not self.discrete_action:
                    search_mean = search_logits[:, :, 0]
                    search_log_var = torch.clamp(
                        self.im_policy_lvar(final_out),
                        self.min_log_var,
                        self.max_log_var,
                    )
                reset_logits = self.reset(final_out)
            else:
                search_logits = None
                reset_logits = None

            if self.dynamic_search:
                search_phase_mask = phase == SEARCH_PHASE
                real_phase_mask = phase == NEED_REAL_ACTION_PHASE
                wait_phase_mask = phase == WAIT_PHASE

                forced_stop = getattr(env_out, "forced_stop", None)
                if forced_stop is None:
                    forced_stop = torch.zeros_like(search_phase_mask)
                else:
                    forced_stop = forced_stop.to(
                        device=phase.device, dtype=torch.bool
                    )
                    if forced_stop.ndim == 3 and forced_stop.shape[-1] == 1:
                        forced_stop = forced_stop.squeeze(-1)
                control_valid = search_phase_mask & ~forced_stop

                legal_control_mask = getattr(env_out, "legal_control_mask", None)
                if legal_control_mask is None:
                    legal_control_mask = torch.ones(
                        (T, B, 3), dtype=torch.bool, device=phase.device
                    )
                else:
                    legal_control_mask = legal_control_mask.to(
                        device=phase.device, dtype=torch.bool
                    )
                    if legal_control_mask.shape != (T, B, 3):
                        raise ValueError(
                            "legal_control_mask must have shape "
                            f"{(T, B, 3)}, got {legal_control_mask.shape}"
                        )

                # Non-control phases receive a deterministic dummy PROCEED.
                # A malformed all-false SEARCH mask also falls back safely;
                # forced-stop rows fall back to STOP.
                fallback_control = torch.where(
                    forced_stop,
                    torch.full_like(phase, STOP),
                    torch.full_like(phase, PROCEED),
                )
                fallback_legal = F.one_hot(fallback_control, 3).bool()
                has_legal = torch.any(legal_control_mask, dim=-1, keepdim=True)
                legal_control_mask = torch.where(
                    has_legal, legal_control_mask, fallback_legal
                )
                legal_control_mask = torch.where(
                    search_phase_mask.unsqueeze(-1),
                    legal_control_mask,
                    F.one_hot(torch.full_like(phase, PROCEED), 3).bool(),
                )
                raw_control_logits = reset_logits.view(T, B, 3)
                search_control_logits = raw_control_logits.masked_fill(
                    ~legal_control_mask, -1e9
                )
                search_control = sample(
                    search_control_logits, greedy=greedy, dim=-1
                )
                search_control = torch.where(
                    search_phase_mask,
                    search_control,
                    torch.full_like(search_control, PROCEED),
                )
                search_control = torch.where(
                    forced_stop,
                    torch.full_like(search_control, STOP),
                    search_control,
                )

                if clamp_action is not None:
                    clamp_control = clamp_action[1]
                    search_control[: clamp_control.shape[0]] = clamp_control
                    search_control = torch.where(
                        forced_stop,
                        torch.full_like(search_control, STOP),
                        search_control,
                    )

                reset_uses_primary = self.flags.reset_mode == 0
                search_primary_valid = search_phase_mask & (
                    (search_control == PROCEED)
                    | ((search_control == RESET) & reset_uses_primary)
                )
                primary_valid = search_primary_valid | real_phase_mask
                # STOP is still a learned policy decision even though it has
                # no primary action.  Only forced controls and WAIT calls are
                # excluded from policy-gradient accounting.
                policy_valid = control_valid | real_phase_mask
                policy_type = torch.full_like(phase, POLICY_NONE)
                policy_type = torch.where(
                    control_valid,
                    torch.full_like(policy_type, POLICY_SEARCH),
                    policy_type,
                )
                policy_type = torch.where(
                    real_phase_mask,
                    torch.full_like(policy_type, POLICY_REAL),
                    policy_type,
                )

                route_search = search_phase_mask.flatten(0, 1)
                if self.discrete_action:
                    pri_logits = torch.where(
                        route_search.unsqueeze(-1).unsqueeze(-1),
                        search_logits,
                        real_logits,
                    )
                else:
                    pri_mean = torch.where(
                        route_search.unsqueeze(-1), search_mean, real_mean
                    )
                    pri_log_var = torch.where(
                        route_search.unsqueeze(-1), search_log_var, real_log_var
                    )

                # Sample the routed primary distribution.  Invalid samples are
                # harmless dummies and are removed from the joint log-prob,
                # entropy, regularization and downstream policy masks.
                if self.discrete_action:
                    pri = sample(pri_logits, greedy=greedy, dim=-1)
                    pri_logits = pri_logits.view(
                        T, B, self.dim_actions, self.num_actions
                    )
                    pri = pri.view(T, B, self.dim_actions)
                    pri_param = pri_logits
                else:
                    pri_mean = pri_mean.view(T, B, self.dim_actions)
                    pri_log_var = pri_log_var.view(T, B, self.dim_actions)
                    pri_std = torch.exp(pri_log_var / 2)
                    normal_dist = torch.distributions.Normal(pri_mean, pri_std)
                    pri_pre_tanh = pri_mean if greedy else normal_dist.sample()
                    pri = (
                        torch.tanh(pri_pre_tanh)
                        if self.tanh_action
                        else pri_pre_tanh
                    )
                    pri_param = torch.stack((pri_mean, pri_log_var), dim=-1)

                if clamp_action is not None:
                    clamp_primary = clamp_action[0]
                    pri[: clamp_primary.shape[0]] = clamp_primary
                    if not self.discrete_action:
                        pri_pre_tanh = atanh(pri) if self.tanh_action else pri

                if self.discrete_action:
                    primary_log_prob = compute_discrete_log_prob(pri_logits, pri)
                else:
                    primary_log_prob = normal_dist.log_prob(pri_pre_tanh)
                    if self.tanh_action:
                        primary_log_prob = primary_log_prob - torch.log(
                            1.0 - pri ** 2 + 1e-6
                        )
                    primary_log_prob = torch.sum(primary_log_prob, dim=-1)
                primary_log_prob = primary_log_prob * primary_valid.float()

                control_log_prob = compute_discrete_log_prob(
                    search_control_logits, search_control
                )
                control_log_prob = control_log_prob * control_valid.float()
                c_action_log_prob = primary_log_prob + control_log_prob

                if compute_loss:
                    if self.discrete_action:
                        primary_entropy_loss = -torch.nn.CrossEntropyLoss(
                            reduction="none"
                        )(
                            input=torch.flatten(pri_logits, 0, 2),
                            target=torch.flatten(
                                F.softmax(pri_logits, dim=-1), 0, 2
                            ),
                        )
                        primary_entropy_loss = primary_entropy_loss.view(
                            T, B, self.dim_actions
                        )
                        primary_entropy_loss = torch.sum(
                            primary_entropy_loss, dim=-1
                        )
                    else:
                        primary_entropy_loss = -torch.sum(pri_log_var, dim=-1)
                    # Entropy of the hierarchical SEARCH policy is
                    # H(control) + P(control != STOP) H(imaginary action).
                    # Using the sampled primary-valid bit here would be a
                    # biased Monte-Carlo entropy regularizer and would omit
                    # its probability-gradient term.
                    control_probs = F.softmax(search_control_logits, dim=-1)
                    non_stop_prob = 1.0 - control_probs[..., STOP]
                    primary_entropy_weight = (
                        real_phase_mask.float()
                        + control_valid.float() * non_stop_prob
                    )
                    primary_entropy_loss = (
                        primary_entropy_loss * primary_entropy_weight
                    )
                    control_entropy_loss = -torch.nn.CrossEntropyLoss(
                        reduction="none"
                    )(
                        input=torch.flatten(search_control_logits, 0, 1),
                        target=torch.flatten(
                            F.softmax(search_control_logits, dim=-1), 0, 1
                        ),
                    ).view(T, B)
                    control_entropy_loss = (
                        control_entropy_loss * control_valid.float()
                    )
                    entropy_loss = primary_entropy_loss + control_entropy_loss
                    misc["primary_entropy_loss"] = primary_entropy_loss
                    misc["control_entropy_loss"] = control_entropy_loss
                    misc["non_stop_prob"] = non_stop_prob
                else:
                    entropy_loss = None

                reset = search_control
                reset_logits = search_control_logits
                pri_env = pri[-1, :, 0] if not self.tuple_action else pri[-1]
                action = (pri_env, search_control[-1])
                if self.discrete_action:
                    action_prob = F.softmax(pri_logits, dim=-1)
                else:
                    action_prob = pri_param
                if not self.tuple_action:
                    action_prob = action_prob[:, :, 0]

                misc["primary_log_prob"] = primary_log_prob
                misc["control_log_prob"] = control_log_prob
                misc["wait_mask"] = wait_phase_mask
            else:
                # Original fixed-budget routing and loss are intentionally
                # unchanged.
                pri_logits = real_logits
                if not self.discrete_action:
                    pri_mean = real_mean
                    pri_log_var = real_log_var
                if not self.disable_thinker:
                    im_mask = env_out.step_status <= 1
                    if self.discrete_action:
                        im_mask = torch.flatten(im_mask, 0, 1).unsqueeze(-1).unsqueeze(-1)
                        pri_logits = torch.where(im_mask, search_logits, pri_logits)
                    else:
                        im_mask = torch.flatten(im_mask, 0, 1).unsqueeze(-1)
                        pri_mean = torch.where(im_mask, search_mean, pri_mean)
                        pri_log_var = torch.where(im_mask, search_log_var, pri_log_var)

                if compute_loss:
                    if self.discrete_action:
                        entropy_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                            input=torch.flatten(pri_logits, 0, 1),
                            target=torch.flatten(F.softmax(pri_logits, dim=-1), 0, 1),
                        )
                        entropy_loss = entropy_loss.view(T, B, self.dim_actions)
                        entropy_loss = torch.sum(entropy_loss, dim=-1)
                    else:
                        entropy_loss = -torch.sum(
                            pri_log_var.view(T, B, self.dim_actions), dim=-1
                        )
                    if not self.disable_thinker:
                        ent_reset_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                            input=reset_logits, target=F.softmax(reset_logits, dim=-1)
                        )
                        ent_reset_loss = ent_reset_loss.view(T, B) * (
                            env_out.step_status <= 1
                        ).float()
                        entropy_loss = entropy_loss + ent_reset_loss
                else:
                    entropy_loss = None

                if self.discrete_action:
                    pri = sample(pri_logits, greedy=greedy, dim=-1)
                    pri_logits = pri_logits.view(
                        T, B, self.dim_actions, self.num_actions
                    )
                    pri = pri.view(T, B, self.dim_actions)
                    pri_param = pri_logits
                else:
                    pri_mean = pri_mean.view(T, B, self.dim_actions)
                    pri_log_var = pri_log_var.view(T, B, self.dim_actions)
                    pri_std = torch.exp(pri_log_var / 2)
                    normal_dist = torch.distributions.Normal(pri_mean, pri_std)
                    pri_pre_tanh = pri_mean if greedy else normal_dist.sample()
                    pri = (
                        torch.tanh(pri_pre_tanh)
                        if self.tanh_action
                        else pri_pre_tanh
                    )
                    pri_param = torch.stack((pri_mean, pri_log_var), dim=-1)

                if not self.disable_thinker:
                    reset = sample(reset_logits, greedy=greedy, dim=-1)
                    reset_logits = reset_logits.view(T, B, 2)
                    reset = reset.view(T, B)
                else:
                    reset = None

                if clamp_action is not None:
                    if not self.disable_thinker:
                        pri[: clamp_action[0].shape[0]] = clamp_action[0]
                        reset[: clamp_action[1].shape[0]] = clamp_action[1]
                    else:
                        pri[: clamp_action.shape[0]] = clamp_action
                    if not self.discrete_action:
                        pri_pre_tanh = atanh(pri) if self.tanh_action else pri

                if self.discrete_action:
                    c_action_log_prob = compute_discrete_log_prob(pri_logits, pri)
                else:
                    c_action_log_prob = normal_dist.log_prob(pri_pre_tanh)
                    if self.tanh_action:
                        c_action_log_prob = c_action_log_prob - torch.log(
                            1.0 - pri ** 2 + 1e-6
                        )
                    c_action_log_prob = torch.sum(c_action_log_prob, dim=-1)

                if not self.disable_thinker:
                    c_reset_log_prob = compute_discrete_log_prob(reset_logits, reset)
                    c_reset_log_prob = c_reset_log_prob * (
                        env_out.step_status <= 1
                    ).float()
                    c_action_log_prob += c_reset_log_prob

                pri_env = pri[-1, :, 0] if not self.tuple_action else pri[-1]
                action = (pri_env, reset[-1]) if not self.disable_thinker else pri_env
                if self.discrete_action:
                    action_prob = F.softmax(pri_logits, dim=-1)
                else:
                    action_prob = pri_param
                if not self.tuple_action:
                    action_prob = action_prob[:, :, 0]

                primary_valid = torch.ones((T, B), dtype=torch.bool, device=done.device)
                policy_valid = primary_valid
                if not self.disable_thinker:
                    control_valid = env_out.step_status <= 1
                    policy_type = torch.where(
                        control_valid,
                        torch.full_like(env_out.step_status, POLICY_SEARCH),
                        torch.full_like(env_out.step_status, POLICY_REAL),
                    )
                    search_control = reset
                    search_control_logits = reset_logits
                else:
                    control_valid = torch.zeros_like(primary_valid)
                    policy_type = torch.full(
                        (T, B), POLICY_REAL, dtype=torch.long, device=done.device
                    )
                    search_control = None
                    search_control_logits = None

        if self.critic:
            # compute baseline
            if self.enc_type == 0:
                baseline = self.baseline(final_out)
                if self.flags.reward_clip > 0:
                    baseline = torch.clamp(
                        baseline, -self.baseline_clamp, +self.baseline_clamp
                    )
                baseline_enc = None
            elif self.enc_type == 1:
                baseline_enc_s = self.baseline(final_out)
                baseline = self.rv_tran.decode(baseline_enc_s)
                baseline_enc = baseline_enc_s
            elif self.enc_type in [2, 3]:
                baseline_enc_logit = self.baseline(final_out).reshape(
                    T * B, self.num_rewards, self.out_n
                )
                baseline_enc_v = F.softmax(baseline_enc_logit, dim=-1)
                baseline = self.rv_tran.decode(baseline_enc_v)
                baseline_enc = baseline_enc_logit

            baseline_enc = (
                baseline_enc.view((T, B) + baseline_enc.shape[1:])
                if baseline_enc is not None
                else None
            )
            baseline = baseline.view(T, B, self.num_rewards)

        if compute_loss:
            reg_loss = 1e-6 * torch.sum(final_out**2, dim=-1).view(T, B) / 2
            if self.dynamic_search and self.actor:
                reg_loss = reg_loss * policy_valid.float()
                if self.discrete_action:
                    reg_loss += (
                        1e-3
                        * torch.sum(pri_logits**2, dim=(-2, -1))
                        * primary_valid.float()
                        / 2
                    )
                reg_loss += (
                    1e-3
                    * torch.sum(raw_control_logits**2, dim=-1)
                    * control_valid.float()
                    / 2
                )
                if self.see_real_state:
                    reg_loss += (
                        1e-5 * pre_reg_loss * policy_valid.float()
                    )
            else:
                if self.discrete_action and self.actor:
                    reg_loss += 1e-3 * torch.sum(pri_logits**2, dim=(-2,-1)) / 2
                if not self.disable_thinker and self.actor:
                    reg_loss += (
                        + 1e-3 * torch.sum(reset_logits**2, dim=-1) / 2
                    )
                    if self.see_real_state:
                        reg_loss += 1e-5 * pre_reg_loss
        else:
            reg_loss = None
        
        actor_out = ActorOut(
            pri=pri if self.actor else None,
            pri_param=pri_param if self.actor else None,
            reset=reset if self.actor else None,
            reset_logits=reset_logits if self.actor else None,
            action=action if self.actor else None,
            action_prob=action_prob if self.actor else None,
            c_action_log_prob=c_action_log_prob if self.actor else None,            
            baseline=baseline if self.critic else None,
            baseline_enc=baseline_enc if self.critic else None,
            entropy_loss=entropy_loss if self.actor else None,
            reg_loss=reg_loss,
            misc=misc,
            search_control=search_control if self.actor else None,
            search_control_logits=search_control_logits if self.actor else None,
            primary_valid=primary_valid if self.actor else None,
            control_valid=control_valid if self.actor else None,
            policy_valid=policy_valid if self.actor else None,
            policy_type=policy_type if self.actor else None,
        )
        core_state = tuple(new_core_state)
        return actor_out, core_state    
    
    def compute_encoded_real_state(self, env_out, core_state, rnn_done):
        T, B = env_out.step_status.shape[:2]
        core_state_ = core_state[self.state_idx['r']] if self.real_state_rnn else None
        if self.dynamic_search and getattr(env_out, "real_transition", None) is not None:
            need_update = env_out.real_transition.to(dtype=torch.bool)
            # A fresh env.reset emits the first real root without crossing an
            # env.step boundary.  search_state_reset marks both that initial
            # root and every post-barrier root, so include it in the real
            # encoder clock while preserving real_transition as the strict
            # underlying-environment-step event used by learning/logging.
            search_state_reset = getattr(env_out, "search_state_reset", None)
            if search_state_reset is not None:
                need_update = need_update | search_state_reset.to(
                    device=need_update.device, dtype=torch.bool
                )
        else:
            need_update = (env_out.step_status == 0) | (env_out.step_status == 3)
        requires_grad = env_out.real_states.requires_grad
        if requires_grad: need_update[0] = True
        assert torch.all(need_update == need_update[:,[0]]), f"expect uniform step_status, not ({env_out.step_status.shape}) {env_out.step_status}"

        need_update = need_update[:, 0] # shape (T,)
        if torch.all(~need_update):
            last_pre_encoded_real_state = core_state[self.state_idx['pre_encoded_real_state']][0]            
            expand_shape = (T,) + (1,) * (len(last_pre_encoded_real_state.shape) - 1)
            pre_encoded_real_state = last_pre_encoded_real_state.repeat(*expand_shape)            
            last_encoded_real_state = core_state[self.state_idx['encoded_real_state']][0]            
            expand_shape = (T,) + (1,) * (len(last_encoded_real_state.shape) - 1)
            encoded_real_state = last_encoded_real_state.repeat(*expand_shape)            
            new_core_state = core_state_
            return pre_encoded_real_state, encoded_real_state, new_core_state, 0. # should only happen in self-play

        real_states = env_out.real_states
        if real_states.shape[0] == T:
            real_states = real_states[need_update]        
        else:
            real_states = real_states[:torch.sum(need_update).item()]
        real_states = self.normalize(real_states.float())
        with autocast(enabled=self.float16): 
            real_states = torch.flatten(real_states, 0, 1)
            if self.real_state_ch > 0:
                real_states = real_states[:, -self.real_state_ch:]
            pre_encoded_real_state = self.real_state_encoder(real_states)  

        if self.float16: pre_encoded_real_state = pre_encoded_real_state.float()        

        if self.real_state_rnn:
            rnn_done = rnn_done[need_update]
            encoded_real_state, new_core_state = self.r_encoder_rnn(
                pre_encoded_real_state, rnn_done, core_state_, record_state=self.record_state)
            if self.record_state: self.hidden_state = self.r_encoder_rnn.rnn.hidden_state
        else:
            if self.record_state: self.hidden_state = self.real_state_encoder.hidden_state
            new_core_state = None
            encoded_real_state = pre_encoded_real_state       

        last_x = core_state[self.state_idx['pre_encoded_real_state']][0]
        xs = pre_encoded_real_state        
        pre_encoded_real_state = self.repeat_for_no_update(last_x, xs, need_update, B)

        last_x = core_state[self.state_idx['encoded_real_state']][0]
        xs = encoded_real_state        
        encoded_real_state = self.repeat_for_no_update(last_x, xs, need_update, B)

        pre_reg_loss = torch.sum(torch.square(pre_encoded_real_state.view(T, B, -1)), dim=-1) / 2
        return pre_encoded_real_state, encoded_real_state, new_core_state, pre_reg_loss

    def repeat_for_no_update(self, last_x, xs, need_update, B):
        k = 0
        K = int(torch.sum(need_update).cpu().detach().item())
        T = need_update.shape[0]
        xs = xs.view((K, B) + xs.shape[1:])
        xs_ls = []
        for t in range(T):
            if need_update[t]:
                last_x = xs[k]
                k += 1
            xs_ls.append(last_x)        
        xs = torch.stack(xs_ls)
        return torch.flatten(xs, 0, 1)

class DRCNet(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False):
        super(DRCNet, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        assert flags.wrapper_type == 1

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_space["real_states"].shape[1], out_channels=32, kernel_size=8, stride=4, padding=2
            ),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
        )
        output_shape = lambda h, w, kernel, stride, padding: (
            ((h + 2 * padding - kernel) // stride + 1),
            ((w + 2 * padding - kernel) // stride + 1),
        )

        h, w = output_shape(self.real_states_shape[1], self.real_states_shape[2], 8, 4, 2)
        h, w = output_shape(h, w, 4, 2, 1)

        self.core = ConvAttnLSTM(            
            input_dim=32,
            hidden_dim=32,
            num_layers=3,
            attn=False,
            h=h,
            w=w,            
            kernel_size=3,
            mem_n=None,            
            num_heads=8,            
            attn_mask_b=None,
            tran_t=3,
            pool_inject=True,
        )
        last_out_size = 32 * h * w * 2
        self.final_layer = nn.Linear(last_out_size, 256)
        self.policy = nn.Linear(256, self.num_actions * self.dim_actions)
        self.baseline = nn.Linear(256, 1)

        if getattr(flags, "ppo_k", 1) > 1:
            kl_beta = torch.tensor(1.)
            self.register_buffer("kl_beta", kl_beta)

    def initial_state(self, batch_size, device=None):
        return self.core.initial_state(batch_size, device=device)

    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        done = env_out.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape
        x = self.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x_enc = self.encoder(x)
        core_input = x_enc.view(*((T, B) + x_enc.shape[1:]))
        core_output, core_state = self.core(core_input, done, core_state, record_state=self.record_state)
        if self.record_state: self.hidden_state = self.core.hidden_state
        core_output = torch.flatten(core_output, 0, 1)
        core_output = torch.cat([x_enc, core_output], dim=1)
        core_output = torch.flatten(core_output, 1)
        final_out = F.relu(self.final_layer(core_output))

        pri_logits = self.policy(final_out)
        pri_logits = pri_logits.view(T*B, self.dim_actions, self.num_actions)

        # compute entropy loss
        if compute_loss:
            entropy_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(pri_logits, 0, 1), 
                target=torch.flatten(F.softmax(pri_logits, dim=-1), 0, 1),
            )
            entropy_loss = entropy_loss.view(T, B, self.dim_actions)            
            entropy_loss = torch.sum(entropy_loss, dim=-1)
        else:
            entropy_loss = None

        # sample_action
        pri = sample(pri_logits, greedy=greedy, dim=-1)
        pri_logits = pri_logits.view(T, B, self.dim_actions, self.num_actions)
        pri = pri.view(T, B, self.dim_actions)      

        # clamp the action to clamp_action
        if clamp_action is not None:
            pri[:clamp_action.shape[0]] = clamp_action

        # compute chosen log porb
        c_action_log_prob = compute_discrete_log_prob(pri_logits, pri)    

        # pack last step's action and action prob        
        pri_env = pri[-1, :, 0] if not self.tuple_action else pri[-1]   
        action = pri_env  
        action_prob = F.softmax(pri_logits, dim=-1)
        if not self.tuple_action: action_prob = action_prob[:, :, 0]    

        baseline = self.baseline(final_out).view(T, B, 1)

        if compute_loss:
            reg_loss = (
                1e-3 * torch.sum(torch.square(pri_logits), dim=(-2, -1))
                + 1e-5 * torch.sum(torch.square(self.baseline.weight)) 
                + 1e-5 * torch.sum(torch.square(self.policy.weight))
            )
        else:
            reg_loss = None
            
        actor_out = ActorOut(
            pri=pri,
            pri_param=pri_logits,
            reset=None,
            reset_logits=None,
            action=action,
            action_prob=action_prob,
            c_action_log_prob=c_action_log_prob,            
            baseline=baseline,
            baseline_enc=None,
            entropy_loss=entropy_loss,
            reg_loss=reg_loss,
            misc={},
            search_control=None,
            search_control_logits=None,
            primary_valid=torch.ones(
                (T, B), dtype=torch.bool, device=done.device
            ),
            control_valid=torch.zeros(
                (T, B), dtype=torch.bool, device=done.device
            ),
            policy_valid=torch.ones(
                (T, B), dtype=torch.bool, device=done.device
            ),
            policy_type=torch.full(
                (T, B), POLICY_REAL, dtype=torch.long, device=done.device
            ),
        )
        return actor_out, core_state

class MCTS(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False):
        super(MCTS, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        assert flags.wrapper_type in [0, 2], "MCTS only support wrapper_type 0, 2"
        assert not flags.tree_carry, "MCTS does not support tree carry"
        assert type(action_space[0][0]) == spaces.discrete.Discrete, f"Unsupported action space f{action_space}"
        
        self.temp = 1
        self.dir_dist = None
        self.root_psa = None            

    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        tree_rep = env_out.tree_reps  
        T, B, C = tree_rep.shape
        assert T == 1
        tree_rep = tree_rep[0]

        assert torch.all(env_out.step_status == env_out.step_status[0, 0]), f"step_status should be the same for all item, not {env_out.step_status}."
        step_status = env_out.step_status[0, 0]
        last_real_step = step_status in [0, 3]
        next_real_step = step_status in [2, 3]                
        
        if last_real_step:
            # last step is real, re init. variables   
            root_logits = torch.clone(tree_rep[:, self.tree_rep_meaning["root_policy"]])  
            self.root_psa = F.softmax(root_logits, dim=-1)
            if self.dir_dist is None:
                con = torch.tensor([0.3]*self.num_actions, device=tree_rep.device)
                self.dir_dist = torch.distributions.dirichlet.Dirichlet(con, validate_args=None)
            self.dir_noise = self.dir_dist.sample((B,))
            self.root_psa = self.root_psa * 0.75 + self.dir_noise * 0.25

        if next_real_step:
            # real step
            root_nsa = tree_rep[:, self.tree_rep_meaning["root_ns"]] * self.flags.rec_t            
            if not greedy:
                root_nsa_temp = root_nsa ** (1 / self.temp)
                pri_prob = root_nsa_temp / torch.sum(root_nsa_temp, dim=-1, keepdim=True)
                pri = torch.multinomial(pri_prob, num_samples=1)[:, 0]
            else:
                pri = torch.argmax(root_nsa, dim=-1)
                pri_prob = F.one_hot(pri, self.num_actions)  

            reset = torch.ones_like(pri)      
        else:
            # imaginary step            
            reset_m = tree_rep[:, self.tree_rep_meaning["cur_reset"]].squeeze(-1) == 1

            if last_real_step:
                cur_psa = self.root_psa
            else:
                cur_logits = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_policy"]])  
                cur_psa = F.softmax(cur_logits, dim=-1)
                if self.root_psa is not None:
                    cur_psa[reset_m] = self.root_psa[reset_m]
                else:
                    print("Warning: root_psa is not initialized. Make sure the first state has step_status 0 or 3")

            cur_nsa = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_ns"]])    
            root_nsa = torch.clone(tree_rep[:, self.tree_rep_meaning["root_ns"]])    
            cur_nsa[reset_m] = root_nsa[reset_m]
            cur_nsa = cur_nsa * self.flags.rec_t
            
            # compute normalized q(s,a)
            cur_qsa = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_qs_mean"]])    
            root_qsa = torch.clone(tree_rep[:, self.tree_rep_meaning["root_qs_mean"]])    
            cur_qsa[reset_m] = root_qsa[reset_m]

            # normalization (see https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py#L87)
            cur_v = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_v"]])    
            root_v = torch.clone(tree_rep[:, self.tree_rep_meaning["root_v"]])    
            cur_v[reset_m] = root_v[reset_m]

            cur_qsa[cur_nsa==0] = cur_v.broadcast_to(B, self.num_actions)[cur_nsa==0]
            q_min = torch.minimum(cur_v.squeeze(-1), torch.min(cur_qsa, dim=-1)[0])
            q_max = torch.maximum(cur_v.squeeze(-1), torch.max(cur_qsa, dim=-1)[0])            
            cur_qsa = (cur_qsa - q_min.unsqueeze(-1)) / (q_max.unsqueeze(-1) - q_min.unsqueeze(-1) + 1e-8)
            cur_qsa[cur_nsa==0] = 0.

            assert torch.all((cur_qsa >= 0) & (cur_qsa <= 1)), f"normalized cur_qsa should range from [0, 1], not {cur_qsa}"

            c_1 = 1.25
            c_2 = 19652
            sum_cur_nsa = torch.sum(cur_nsa, dim=-1, keepdim=True)
            score = cur_qsa + cur_psa * (torch.sqrt(sum_cur_nsa)) / (1 + cur_nsa) * (
                c_1 + torch.log((sum_cur_nsa + c_2 + 1) / c_2)
            )
            pri = torch.argmax(score, dim=-1)
            pri_prob = F.one_hot(pri, self.num_actions)      

            reset = (torch.sum(cur_nsa, dim=-1) <= 0).long()
        
        pri = pri.view(T, B, 1)
        reset = reset.view(T, B)
        action = (pri[-1, :, 0], reset[-1])  
        action_prob = pri_prob.view(T, B, self.num_actions)

        actor_out = ActorOut(
            pri=pri,
            pri_param=None,
            reset=reset,
            reset_logits=None,
            action=action,
            action_prob=action_prob,
            c_action_log_prob=None,            
            baseline=None,
            baseline_enc=None,
            entropy_loss=None,
            reg_loss=None,
            misc={},
            search_control=reset,
            search_control_logits=None,
            primary_valid=torch.ones(
                (T, B), dtype=torch.bool, device=tree_rep.device
            ),
            control_valid=torch.full(
                (T, B), not next_real_step,
                dtype=torch.bool,
                device=tree_rep.device,
            ),
            policy_valid=torch.ones(
                (T, B), dtype=torch.bool, device=tree_rep.device
            ),
            policy_type=torch.full(
                (T, B), POLICY_REAL if next_real_step else POLICY_SEARCH,
                dtype=torch.long,
                device=tree_rep.device,
            ),
        )
        return actor_out, core_state    
    
    def set_real_step(self, real_step):
        if real_step < self.flags.total_steps * 0.5:
            self.temp = 1
        elif real_step < self.flags.total_steps * 0.75:
            self.temp = 0.5
        else:
            self.temp = 0.25
    
    def initial_state(self, batch_size, device=None):
        return ()
    
    def set_weights(self, weights):
        return
    
    def get_weights(self):
        return {}
    
    def to(self, device):
        return self
    
    def train(self, train):
        return     
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def ActorNet(*args, **kwargs):

    if getattr(kwargs["flags"], "drc", False):        
        Net = DRCNet
    elif getattr(kwargs["flags"], "mcts", False):  
        Net = MCTS
    elif not getattr(kwargs["flags"], "sep_actor_critic", False):
        Net = ActorNetSingle
    else:
        Net = ActorNetSep

    return Net(*args, **kwargs)
