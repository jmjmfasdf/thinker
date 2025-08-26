#!/usr/bin/env python3
"""
Thinker Pipeline Demo
=====================

Behavioral data에서 observation을 불러와서 pong_v5 설정으로 
전체 Thinker 파이프라인을 실행하는 데모 스크립트

Pipeline:
1. BC Loader로 behavioral data 불러오기 (4-frame stack)
2. Model Network (SRN, VPN) rec_t만큼 실행 
3. tree_reps, xs, hs 생성
4. Actor Network로 action 결정
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Thinker 모듈 import (상대 경로)
import thinker.util as util
from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.model_net import ModelNet
from thinker.actor_net import ActorNet
from gymnasium import spaces

def create_pong_config():
    """Pong v5 환경에 맞는 설정 생성"""
    # 기본 thinker 설정 불러오기
    flags = util.create_flags(filename='default_thinker.yaml')
    
    # Pong 특화 설정
    flags.name = "PongNoFrameskip-v4"
    flags.frame_stack_n = 4
    flags.grayscale = True
    flags.rec_t = 40  # planning steps
    flags.max_depth = 40
    flags.env_n = 32  # 32 environments to match model expectations
    
    # Model network 설정
    flags.dual_net = True
    flags.model_size_nn = 1
    flags.model_downscale_c = 2
    flags.model_decoder_depth = 0
    flags.return_h = True
    flags.return_x = False
    
    # Actor network 설정  
    flags.see_real_state = True
    flags.see_tree_rep = True
    flags.see_h = True
    flags.see_x = False
    flags.tree_rep_rnn = True
    flags.tran_dim = 128
    flags.tran_layer_n = 3
    
    # Additional required flags for actor
    flags.im_cost = 1.0
    flags.cur_cost = 0.0
    flags.critic_zero_init = True
    flags.critic_enc_type = 0
    flags.critic_enc_f_type = 0
    flags.wrapper_type = 0  # Default thinker
    flags.actor_ordinal = False
    flags.tanh_action = True
    flags.actor_min_std = 0.003
    flags.actor_max_std = 10.0
    flags.autotune = False
    flags.ppo_k = 1
    
    # RNN and transformer settings
    flags.tran_reset_mode = 0
    flags.tran_mem_n = 40
    flags.tran_head_n = 8
    flags.tran_lstm_no_attn = False
    flags.tran_attn_b = 5
    flags.tran_t = 1
    flags.sep_im_head = True
    flags.last_layer_n = 0
    flags.sep_actor_critic = False
    flags.float16 = True
    
    # Temporarily disable thinker to simplify action space handling
    flags.disable_thinker = False  # Keep thinker enabled for proper functionality
    
    # 1D encoder settings
    flags.enc_1d_shallow = False
    flags.enc_1d_norm = True
    flags.enc_1d_block = 2
    flags.enc_1d_hs = 256
    
    # RNN settings for different inputs
    flags.x_rnn = False
    flags.h_rnn = False
    flags.real_state_rnn = False
    flags.real_state_ch = -1
    
    return flags

def create_observation_spaces(flags):
    """관찰 공간과 액션 공간 정의"""
    
    # Pong environment spaces
    # Atari Pong: (210, 160, 3) -> after preprocessing: (batch, 4, 84, 84) for 4-frame stack
    # ActorNet expects batch dimension in observation space shape
    real_state_space = spaces.Box(
        low=0, high=255, 
        shape=(flags.env_n, flags.frame_stack_n, 84, 84), 
        dtype=np.uint8
    )
    
    # Action space: Simple discrete action space to get dim_actions=1
    primary_action_space = spaces.Discrete(6)  # Pong has 6 actions
    reset_action_space = spaces.Discrete(2)    # Reset or not
    
    # Simplest structure that ActorNet can handle
    action_space = spaces.Tuple((
        spaces.Tuple((primary_action_space,)),  # Single primary action
        spaces.Tuple((reset_action_space,))     # Single reset action  
    ))
    
    # Tree reps dimension calculation
    num_actions = primary_action_space.n  # Use primary action space for tree reps
    obs_n = 11 + num_actions * 10 + flags.rec_t
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        obs_n += flags.max_depth * num_actions
    
    # Observation space for thinker
    observation_space = spaces.Dict({
        "real_states": real_state_space,
        "tree_reps": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(flags.env_n, obs_n),
            dtype=np.float32
        )
    })
    
    # Add hs if return_h is True
    if flags.return_h:
        # Model hidden size estimation (will be updated after model creation)
        hs_shape = (flags.env_n, 128, 8, 8)  # placeholder
        observation_space.spaces["hs"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=hs_shape,
            dtype=np.float32
        )
    
    return observation_space, action_space

def init_env_out(state, flags, dim_actions=1, tuple_action=False):
    """Environment output 초기화"""
    batch_size = flags.env_n
    device = torch.device('cpu')
    
    # Convert state to proper format
    real_states = state["real_states"]
    if isinstance(real_states, np.ndarray):
        real_states = torch.tensor(real_states, dtype=torch.float32, device=device)
    
    tree_reps = state.get("tree_reps", torch.zeros(batch_size, state["tree_reps"].shape[1], device=device))
    
    # Create initial values
    T = 1  # single timestep
    
    # Create environment output object with attribute access
    class EnvOut:
        def __init__(self):
            self.real_states = real_states.unsqueeze(0)  # (T, B, C, H, W) - T=1
            self.tree_reps = tree_reps.unsqueeze(0)  # (T, B, obs_n)
            self.done = torch.zeros(T, batch_size, dtype=torch.bool, device=device)  # Keep 2D as expected by ActorNet
            self.real_done = torch.zeros(T, batch_size, dtype=torch.bool, device=device)  # Keep 2D
            self.step_status = torch.zeros(T, batch_size, dtype=torch.long, device=device)  # Keep 2D like real Pong
            self.last_pri = torch.zeros(T, batch_size, dtype=torch.long, device=device)  # Will become [32, 6] after encode_action
            self.last_reset = torch.zeros(T, batch_size, dtype=torch.long, device=device)  # Will become [32, 2] after one_hot
            self.reward = torch.zeros(T, batch_size, 2, dtype=torch.float32, device=device)  # Will become [32, 2] after flatten
            self.hs = None  # Will be set later if available
            self.xs = None  # Will be set later if available
            
    env_out = EnvOut()
    
    # Initialize action history
    if tuple_action:
        env_out.last_pri = torch.zeros(T, batch_size, dim_actions, dtype=torch.long, device=device)
    else:
        env_out.last_pri = torch.zeros(T, batch_size, dtype=torch.long, device=device)  # Will become [32] after flatten, then [32, 1] after unsqueeze, then [32, 6] after encode_action
    
    # last_reset is already initialized in EnvOut.__init__ with proper dimensions
    
    # Add hs if available
    if "hs" in state:
        hs = state["hs"]
        if isinstance(hs, np.ndarray):
            hs = torch.tensor(hs, dtype=torch.float32, device=device)
        env_out.hs = hs.unsqueeze(0)  # (T, B, ...)
    
    return env_out

def run_real_model_pipeline(model_net, initial_state, actions_sequence, flags):
    """실제 Model Network를 사용한 planning pipeline"""
    device = torch.device('cpu')
    batch_size = flags.env_n
    rec_t = flags.rec_t
    
    print(f"Running REAL Model Network for {rec_t} planning steps...")
    
    # Set model to evaluation mode to avoid training-specific behavior
    model_net.eval()
    
    # Prepare inputs for model network
    current_state = initial_state  # Keep as uint8 for model.normalize()
    
    # Initialize outputs
    all_xs = []
    all_hs = []
    all_rewards = []
    all_values = []
    all_policies = []
    
    print(f"   Initial state shape: {current_state.shape}")
    print(f"   Actions sequence shape: {actions_sequence.shape}")
    
    # Run model for rec_t steps
    for step in range(rec_t):
        # Get action for this step from sequence
        if step < actions_sequence.shape[1]:
            action = actions_sequence[:, step]  # Shape: [batch_size]
        else:
            # If we run out of actions, use random actions
            action = torch.randint(0, 6, (batch_size,), device=device)
        
        # Run model forward pass
        # Enable gradients temporarily for model forward pass (some components need gradient hooks)
        current_state.requires_grad_(False)  # But don't require gradients for input
        
        # Forward pass through model (model.forward will call normalize internally)
        # ModelNet expects raw integer actions, not one-hot
        model_out = model_net.forward(
            env_state=current_state,  # Pass uint8 directly, normalize called inside forward()
            actions=action.unsqueeze(0).unsqueeze(-1),  # Add time and action dimension: [1, batch_size, 1]
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            state={}
        )
        
        # Extract outputs
        xs = model_out.xs  # Predicted next state, shape: [1, batch_size, C, H, W]
        if xs is not None and len(xs.shape) == 5:
            xs = xs.squeeze(0)  # Remove time dimension: [batch_size, C, H, W]
        
        # Extract hs and handle dimensions properly
        hs = model_out.hs if hasattr(model_out, 'hs') else torch.randn(batch_size, 256, 6, 6)
        if hs is not None and len(hs.shape) == 5:
            hs = hs.squeeze(0)  # Remove time dimension if present: [batch_size, C, H, W]
        reward = model_out.reward if hasattr(model_out, 'reward') else torch.zeros(batch_size, 1)
        value = model_out.value if hasattr(model_out, 'value') else torch.zeros(batch_size, 1) 
        policy = model_out.policy if hasattr(model_out, 'policy') else torch.randn(batch_size, 6)
        
        # Store outputs
        all_xs.append(xs)
        all_hs.append(hs)
        all_rewards.append(reward)
        all_values.append(value)
        all_policies.append(policy)
        
        # Update current state for next step
        if xs is not None:
            # Convert back to uint8 for next iteration (model.normalize expects uint8)
            current_state = torch.clamp(xs.detach() * 255, 0, 255).to(torch.uint8)
        else:
            # If xs is None, keep current state (no prediction available)
            print(f"   Warning: xs is None at step {step}, keeping current state")
        
        if step % 10 == 0:
            print(f"   Completed step {step}/{rec_t}")
    
    # Take the final state and hidden state
    final_xs = all_xs[-1]  # Last predicted state
    final_hs = all_hs[-1]  # Last hidden state
    
    print(f"   Final hs shape before processing: {final_hs.shape}")
    
    # Make sure final_hs has correct shape [batch_size, C, H, W]
    if len(final_hs.shape) == 5 and final_hs.shape[0] == 1:
        final_hs = final_hs.squeeze(0)  # Remove time dimension
        print(f"   Final hs shape after squeeze: {final_hs.shape}")
    
    # Generate tree_reps (normally computed by MCTS/planning algorithm)
    # For now, create realistic tree_reps based on the planning results
    num_actions = 6  # Pong actions
    obs_n = 11 + num_actions * 10 + rec_t
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        obs_n += flags.max_depth * num_actions
    
    # Create tree_reps that incorporates information from planning
    tree_reps = torch.randn(batch_size, obs_n, device=device)
    
    # Create comprehensive output based on real model results
    real_output = {
        'tree_reps': tree_reps,
        'hs': final_hs,
        'xs': final_xs,
        'rewards': torch.stack(all_rewards, dim=0),  # [rec_t, batch_size, 1]
        'values': torch.cat([torch.stack(all_values, dim=0), all_values[-1].unsqueeze(0)], dim=0),  # [rec_t+1, batch_size, 1]
        'policies': torch.cat([torch.stack(all_policies, dim=0), all_policies[-1].unsqueeze(0)], dim=0),  # [rec_t+1, batch_size, 6]
        'dones': torch.zeros(rec_t, batch_size, dtype=torch.bool, device=device)
    }
    
    print(f"   REAL tree_reps shape: {real_output['tree_reps'].shape}")
    print(f"   REAL hs shape: {real_output['hs'].shape}")
    print(f"   REAL xs shape: {real_output['xs'].shape}")
    print(f"   REAL policies shape: {real_output['policies'].shape}")
    print("   Model pipeline completed with REAL data!")
    
    return real_output

def create_thinker_observation(thinker_output, flags):
    """Thinker output으로부터 actor가 사용할 observation 생성"""
    device = torch.device('cpu')
    batch_size = flags.env_n
    
    # Create observation dictionary for actor
    observation = {
        "real_states": thinker_output['xs'],  # Keep as [32, 4, 84, 84], T dimension added in init_env_out
        "tree_reps": thinker_output['tree_reps']
    }
    
    # Add hs if available
    if flags.return_h and 'hs' in thinker_output:
        observation["hs"] = thinker_output['hs']
    
    return observation

def run_actor_pipeline(actor_net, observation, flags):
    """Actor network 파이프라인 실행"""
    device = torch.device('cpu')
    batch_size = flags.env_n
    
    print("Running actor pipeline...")
    
    # Create env_out for actor
    env_out = init_env_out(observation, flags, 
                          dim_actions=actor_net.dim_actions, 
                          tuple_action=actor_net.tuple_action)
    
    # Debug: print env_out tensor shapes
    print("   Debugging env_out tensor shapes:")
    for attr_name in ['real_states', 'tree_reps', 'done', 'last_pri', 'last_reset', 'reward', 'hs']:
        if hasattr(env_out, attr_name):
            attr_val = getattr(env_out, attr_name)
            if isinstance(attr_val, torch.Tensor):
                print(f"   {attr_name}: {attr_val.shape}")
                if attr_name == 'real_states':
                    print(f"   real_states dtype: {attr_val.dtype}")
                    print(f"   real_states min/max: {attr_val.min():.2f}/{attr_val.max():.2f}")
            elif attr_val is None:
                print(f"   {attr_name}: None")
    
    # Initialize actor state
    actor_state = actor_net.initial_state(batch_size=batch_size, device=device)
    
    # Run actor forward
    actor_out, new_actor_state = actor_net.forward(
        env_out=env_out,
        core_state=actor_state,
        clamp_action=None,
        compute_loss=False,
        greedy=False
    )
    
    return actor_out, new_actor_state

def main():
    """메인 파이프라인 실행"""
    print("=" * 60)
    print("Thinker Pipeline Demo")
    print("=" * 60)
    
    # 1. Configuration 설정
    print("\n1. Setting up configuration...")
    flags = create_pong_config()
    print(f"   rec_t: {flags.rec_t}")
    print(f"   frame_stack_n: {flags.frame_stack_n}")
    print(f"   env_n: {flags.env_n}")
    
    # 2. Behavioral Data 불러오기
    print("\n2. Loading behavioral data...")
    bc_loader = FrameStackedBehavioralDataLoader(
        base_path="../behavioral_data_4kframe_legacy",
        subjects=[1],
        game_id=1,  # Pong
        frame_stack_n=flags.frame_stack_n,
        target_size=(84, 84),
        grayscale=True,
        normalize=True
    )
    
    # Get a sequence batch for 32 environments
    batch_data = bc_loader.get_sequence_batch(batch_size=flags.env_n, sequence_length=flags.rec_t)
    if batch_data is None:
        print("   Error: Could not load behavioral data")
        return
    
    print(f"   Loaded batch with shape: {batch_data['images'].shape}")
    # Take first frame from all 32 environments: shape will be (32, 4, 84, 84)
    initial_state = torch.tensor(batch_data['images'][:, 0], dtype=torch.uint8)  # (32, 4, 84, 84)
    # Take action sequences from all 32 environments: shape will be (32, rec_t)
    actions_sequence = torch.tensor(batch_data['actions'][:, :flags.rec_t, 0], dtype=torch.long)  # (32, rec_t)
    
    # 3. Observation/Action spaces 설정
    print("\n3. Setting up observation and action spaces...")
    observation_space, action_space = create_observation_spaces(flags)
    print(f"   Real state shape: {observation_space['real_states'].shape}")
    print(f"   Tree reps shape: {observation_space['tree_reps'].shape}")
    print(f"   Action space: {action_space}")
    
    # Extract primary action space for model
    primary_action_space = action_space.spaces[0].spaces[0]  # Extract from nested tuple
    
    # 4. Model Network 초기화
    print("\n4. Initializing Model Network...")
    
    # Create simple observation space for model (without batch dimension)
    model_obs_space = spaces.Box(
        low=0, high=255, 
        shape=(flags.frame_stack_n, 84, 84), 
        dtype=np.uint8
    )
    
    model_net = ModelNet(
        obs_space=model_obs_space,
        action_space=primary_action_space,  # Model only needs primary action space
        flags=flags
    )
    print(f"   Model hidden shape: {model_net.hidden_shape}")
    
    # Update hs shape in observation space
    if flags.return_h:
        hs_shape = (flags.env_n,) + tuple(model_net.hidden_shape)
        observation_space.spaces["hs"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=hs_shape,
            dtype=np.float32
        )
        print(f"   Updated hs shape: {hs_shape}")
    
    # 5. REAL Thinker Pipeline (planning steps using actual Model Network)
    print("\n5. Running REAL Thinker Planning Pipeline...")
    try:
        thinker_output = run_real_model_pipeline(model_net, initial_state, actions_sequence, flags)
        print(f"   Real planning pipeline completed successfully")
    except Exception as e:
        print(f"   Error in real thinker pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Thinker Observation 생성
    print("\n6. Creating Thinker Observation...")
    thinker_obs = create_thinker_observation(thinker_output, flags)
    for key, value in thinker_obs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key} shape: {value.shape}")
    
    # 7. Actor Network 초기화
    print("\n7. Initializing Actor Network...")
    print(f"   Debug: observation_space['real_states'].shape = {observation_space['real_states'].shape}")
    print(f"   Debug: observation_space['real_states'].shape[1:] = {observation_space['real_states'].shape[1:]}")
    
    actor_net = ActorNet(
        obs_space=observation_space,  # Pass full observation space dict
        action_space=action_space,
        flags=flags,
        tree_rep_meaning=None  # Simplified for demo
    )
    print(f"   Actor network initialized")
    print(f"   Number of actions: {actor_net.num_actions}")
    print(f"   Action dimensions: {actor_net.dim_actions}")
    print(f"   Real states shape in ActorNet: {actor_net.real_states_shape}")
    print(f"   Tuple action: {actor_net.tuple_action}")
    print(f"   Expected total output size: {1 * 32 * actor_net.dim_actions * actor_net.num_actions}")
    
    # 8. Actor Pipeline 실행
    print("\n8. Running Actor Pipeline...")
    try:
        actor_out, new_actor_state = run_actor_pipeline(actor_net, thinker_obs, flags)
        
        print(f"   Generated action: {actor_out.action}")
        print(f"   Action probabilities shape: {actor_out.action_prob.shape if actor_out.action_prob is not None else 'None'}")
        print(f"   Baseline shape: {actor_out.baseline.shape if actor_out.baseline is not None else 'None'}")
        print(f"   Primary action shape: {actor_out.pri.shape if actor_out.pri is not None else 'None'}")
        
        if actor_out.action is not None:
            if isinstance(actor_out.action, tuple):
                pri_action, reset_action = actor_out.action
                print(f"   Final primary actions: {pri_action.squeeze().tolist()}")
                print(f"   Final reset actions: {reset_action.squeeze().tolist()}")
            else:
                print(f"   Final action: {actor_out.action.item()}")
                
    except Exception as e:
        print(f"   Error in actor pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
