"""
Behavioral Cloning Trainer Module for Thinker
============================================

This module provides BC training functionality to be integrated with the main train.py
"""

import os
import time
import json
import yaml
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple

from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.model_net import ModelNet
from thinker.actor_net import ActorNet
from gymnasium import spaces


def create_bc_data_loader(flags):
    """Create BC data loader from flags"""
    # Parse subjects from comma-separated string
    subjects = [int(s.strip()) for s in flags.bc_subjects.split(',')]
    
    bc_loader = FrameStackedBehavioralDataLoader(
        base_path=flags.bc_data_path,
        subjects=subjects,
        game_id=flags.bc_game_id,
        frame_stack_n=flags.frame_stack_n,
        target_size=(84, 84),
        grayscale=flags.grayscale,
        normalize=True
    )
    
    return bc_loader


def compute_bc_losses(model_out, actor_out, target_actions, device):
    """
    Compute BC losses for ActorNet only (Model/VPN training disabled)
    
    Args:
        model_out: ModelNet output (not used for loss)
        actor_out: ActorNet output  
        target_actions: Target actions from BC data [B]
        device: torch device
        
    Returns:
        Dict with 'actor_loss' only
    """
    losses = {}
    
    # ActorNet loss: Cross entropy between Actor policy and target actions
    # IMPORTANT: Use pri_param (logits) instead of action_prob (softmax probabilities)
    # NOTE: Only primary action is used for BC loss - reset actions are NOT considered in real step
    if hasattr(actor_out, 'pri_param') and actor_out.pri_param is not None:
        # pri_param contains the raw logits [T, B, num_actions] or [T, B, num_actions, 1]
        actor_logits = actor_out.pri_param  # Shape: [T, B, num_actions] or [T, B, num_actions, 1]
        
        # Take the first timestep (T=1)
        print(f"[DEBUG] Raw pri_param shape: {actor_logits.shape}")
        if len(actor_logits.shape) == 4:  # [T, B, 1, num_actions] or [T, B, num_actions, 1] for tuple actions
            if actor_logits.shape[2] == 1:  # [T, B, 1, num_actions]
                actor_logits = actor_logits[0, :, 0, :]  # [B, num_actions]
            else:  # [T, B, num_actions, 1]
                actor_logits = actor_logits[0, :, :, 0]  # [B, num_actions]
        elif len(actor_logits.shape) == 3:  # [T, B, num_actions]
            actor_logits = actor_logits[0, :, :]  # [B, num_actions]
        elif len(actor_logits.shape) == 2:  # [B, num_actions] or [B, 1]
            if actor_logits.shape[1] == 1:  # This suggests wrong action space processing
                print(f"[WARNING] pri_param has shape [B, 1] - action space might be misconfigured")
                # For now, handle this case by checking target actions range
                max_target = target_actions.max().item()
                if max_target >= actor_logits.shape[1]:
                    print(f"[ERROR] Target action {max_target} >= logits dim {actor_logits.shape[1]}")
                    return {'actor_loss': torch.tensor(0.0, device=device)}
        elif len(actor_logits.shape) == 1:  # [num_actions] - single batch
            actor_logits = actor_logits.unsqueeze(0)  # [1, num_actions]
        
        # Cross entropy loss with validation
        target_actions_long = target_actions.long()
        
        # Validate target actions before loss computation
        max_target = target_actions_long.max().item()
        min_target = target_actions_long.min().item()
        num_classes = actor_logits.shape[1]
        
        print(f"[DEBUG] Using pri_param logits for BC loss, shape: {actor_logits.shape}")
        print(f"[DEBUG] Target actions range: [{min_target}, {max_target}], num_classes: {num_classes}")
        
        if min_target < 0 or max_target >= num_classes:
            print(f"[ERROR] Invalid target actions: range [{min_target}, {max_target}] vs num_classes {num_classes}")
            losses['actor_loss'] = torch.tensor(0.0, device=device)
        else:
            actor_loss = F.cross_entropy(actor_logits, target_actions_long)
            losses['actor_loss'] = actor_loss
    elif hasattr(actor_out, 'action_prob') and actor_out.action_prob is not None:
        # Fallback: Use action_prob with CORRECT loss function
        actor_probs = actor_out.action_prob  # Shape: [T, B, num_actions]
        
        # Take the first timestep (T=1)
        if len(actor_probs.shape) == 3:
            actor_probs = actor_probs[0, :, :]  # [B, num_actions]
        elif len(actor_probs.shape) == 2:
            pass  # Already [B, num_actions]
        
        # CORRECT: Use NLLLoss with log-probabilities, not CrossEntropy
        # CrossEntropy = LogSoftmax + NLLLoss, so applying it to log(probs) is wrong
        actor_log_probs = torch.log(actor_probs + 1e-8)  # Convert to log-probabilities
        
        # Use NLL loss instead of CrossEntropy
        target_actions_long = target_actions.long()
        actor_loss = F.nll_loss(actor_log_probs, target_actions_long)
        losses['actor_loss'] = actor_loss
        
        print(f"[WARNING] Using action_prob fallback with NLL loss - pri_param preferred")
    else:
        losses['actor_loss'] = torch.tensor(0.0, device=device)
    
    return losses


def compute_enhanced_bc_losses(outputs, batch_data, model_net, flags, device):
    """
    Enhanced BC losses using first imagine step outputs from actual Thinker rollout
    
    사용자가 제시한 올바른 손실 구조:
    
    Model Loss (첫 번째 imagine step 사용):
    1. VPN Policy Loss: imagine_step_1의 VPN policy vs current_obs의 action (cross-entropy)
    2. SRN Reward Loss: imagine_step_1의 SRN reward vs current_obs의 reward (MSE) 
    3. SRN Next State Loss: imagine_step_1의 hidden_state vs next_obs 인코딩 (MSE)
    
    Actor Loss (최종 출력 사용):
    4. Actor Policy Loss: final Actor policy vs current_obs의 action (cross-entropy)
    """
    import torch.nn.functional as F
    
    losses = {}
    device = torch.device(device) if isinstance(device, str) else device
    
    # Extract first imagine step outputs from actual rollout
    first_step = outputs.get('first_imagine_step', {})
    vpn_policy = first_step.get('vpn_policy')      # [T, B, num_actions] or [T, B, 1, num_actions]
    srn_reward = first_step.get('srn_reward')      # [T, B, 1]
    hidden_state = first_step.get('hidden_state')  # [T, B, ...]
    
    # Extract BC data - current action and reward
    target_actions = outputs.get('target_actions')
    
    model_total_loss = torch.tensor(0.0, device=device)
    
    print(f"[DEBUG] Enhanced BC Loss Computation using first imagine step:")
    print(f"  - VPN policy available: {vpn_policy is not None}")
    print(f"  - SRN reward available: {srn_reward is not None}")
    print(f"  - Hidden state available: {hidden_state is not None}")
    print(f"  - Target actions shape: {target_actions.shape if target_actions is not None else None}")
    
    # ========== Model Loss 1: VPN Policy Loss ==========
    # imagine_step_1의 VPN policy vs current_obs의 action
    if vpn_policy is not None and target_actions is not None:
        # Use first timestep policy (imagine step 1)
        if len(vpn_policy.shape) == 4:  # [T, B, 1, num_actions]
            vpn_logits = vpn_policy[0, :, 0, :]  # [B, num_actions] - first imagine step
        elif len(vpn_policy.shape) == 3:  # [T, B, num_actions]
            vpn_logits = vpn_policy[0, :, :]     # [B, num_actions] - first imagine step
        else:
            vpn_logits = vpn_policy              # [B, num_actions]
        
        target_actions_long = target_actions.long()
        vpn_policy_loss = F.cross_entropy(vpn_logits, target_actions_long)
        losses['vpn_policy_loss'] = vpn_policy_loss
        
        vpn_policy_weight = 0.5  # Hardcoded weight
        model_total_loss += vpn_policy_weight * vpn_policy_loss
        print(f"[DEBUG] VPN Policy Loss (imagine_step_1): {vpn_policy_loss.item():.4f} (weight: {vpn_policy_weight})")
        print(f"[DEBUG] Comparing imagine_step_1 VPN policy vs current_obs action")
    else:
        losses['vpn_policy_loss'] = torch.tensor(0.0, device=device)
        print("[DEBUG] No VPN policy output for first imagine step")
    
    # ========== Model Loss 2: SRN Reward Loss ==========
    # imagine_step_1의 SRN reward vs current_obs의 reward
    if srn_reward is not None and 'rewards' in batch_data:
        # Use first timestep reward prediction (imagine step 1)
        predicted_rewards = srn_reward  # [T, B, 1]
        target_rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32, device=device)  # [B]
        pred_reward = predicted_rewards[0, :, 0]  # [B] - 첫 번째 imagine step의 예측
        
        reward_loss = F.mse_loss(pred_reward, target_rewards)
        losses['srn_reward_loss'] = reward_loss
        
        srn_reward_weight = 1.0  # Hardcoded weight
        model_total_loss += srn_reward_weight * reward_loss
        print(f"[DEBUG] SRN Reward Loss (imagine_step_1): {reward_loss.item():.4f} (weight: {srn_reward_weight})")
        print(f"[DEBUG] Comparing imagine_step_1 SRN reward vs current_obs reward")
        print(f"[DEBUG] Predicted rewards: {pred_reward[:5].detach().cpu().numpy()}")
        print(f"[DEBUG] Actual rewards: {target_rewards[:5].cpu().numpy()}")
    else:
        losses['srn_reward_loss'] = torch.tensor(0.0, device=device)
        if srn_reward is None:
            print("[DEBUG] No SRN reward output for first imagine step")
        if 'rewards' not in batch_data:
            print("[DEBUG] No reward data in batch_data")
    
    # ========== Model Loss 3: SRN Next State Loss ==========
    # imagine_step_1의 hidden_state vs next_obs 인코딩
    if hidden_state is not None and 'next_obs' in batch_data:
        # Use first timestep hidden state (imagine step 1)
        predicted_next_hs = hidden_state  # [T, B, ...]
        next_obs_data = batch_data['next_obs']
        
        # Encode next_obs to get target hidden state
        # ModelNet expects uint8 data (will be normalized internally)
        if next_obs_data.dtype == np.float32:
            # Convert back to uint8 if it's already normalized
            next_obs = torch.tensor((next_obs_data * 255.0).astype(np.uint8), dtype=torch.uint8, device=device)
        else:
            # Already uint8
            next_obs = torch.tensor(next_obs_data, dtype=torch.uint8, device=device)
        
        with torch.no_grad():
            target_model_out = model_net.forward(
                env_state=next_obs,
                actions=torch.zeros(1, next_obs.shape[0], 1, dtype=torch.long, device=device),  # dummy action
                done=torch.zeros(next_obs.shape[0], dtype=torch.bool, device=device),
                state=model_net.initial_state(batch_size=next_obs.shape[0], device=device)
            )
            target_next_hs = target_model_out.hs  # [T, B, ...]
        
        pred_hs = predicted_next_hs[0]  # 첫 번째 imagine step [B, ...]
        target_hs = target_next_hs[0]   # 첫 번째 타임스텝 [B, ...]
        
        if pred_hs.shape == target_hs.shape:
            next_state_loss = F.mse_loss(pred_hs, target_hs)
            losses['srn_next_state_loss'] = next_state_loss
            
            srn_next_state_weight = 1.0  # Hardcoded weight
            model_total_loss += srn_next_state_weight * next_state_loss
            print(f"[DEBUG] SRN Next State Loss (imagine_step_1): {next_state_loss.item():.6f} (weight: {srn_next_state_weight})")
            print(f"[DEBUG] Comparing imagine_step_1 hidden_state vs next_obs encoded")
            print(f"[DEBUG] Predicted HS shape: {pred_hs.shape}, Target HS shape: {target_hs.shape}")
        else:
            print(f"[WARNING] Hidden state shape mismatch: pred={pred_hs.shape}, target={target_hs.shape}")
            losses['srn_next_state_loss'] = torch.tensor(0.0, device=device)
    else:
        losses['srn_next_state_loss'] = torch.tensor(0.0, device=device)
        if hidden_state is None:
            print("[DEBUG] No hidden state output for first imagine step")
        if 'next_obs' not in batch_data:
            print("[DEBUG] No next_obs data in batch_data")
    
    losses['model_total_loss'] = model_total_loss
    
    # ========== Actor Loss ==========
    # final Actor policy vs current_obs의 action
    actor_out = outputs.get('actor_out')
    if actor_out is not None and target_actions is not None:
        actor_losses = compute_bc_losses(None, actor_out, target_actions, device)
        losses.update(actor_losses)
        print(f"[DEBUG] Actor Loss (final policy): {actor_losses.get('actor_loss', 0):.4f}")
        print(f"[DEBUG] Comparing final actor policy vs current_obs action")
    else:
        losses['actor_loss'] = torch.tensor(0.0, device=device)
        print("[DEBUG] No actor output available")
    
    return losses


def run_bc_training_step(model_net, actor_net, batch_data, flags, device):
    """
    Run one BC training step with proper Thinker imaginary rollout
    
    This implements the actual Thinker planning process:
    1. Start with real observation
    2. Perform rec_t-1 imaginary steps with SRN, VPN, ActorNet
    3. Update tree_reps progressively during rollout
    4. Use final ActorNet policy for BC loss
    """
    # Extract data
    # ModelNet expects specific data types based on state_dtype_n
    # state_dtype_n == 0: expects uint8, state_dtype_n == 1: expects float32
    obs_data = batch_data['obs']
    actions_raw = batch_data['actions']  # Should be [B, 6] one-hot
    
    # Convert one-hot actions to scalar indices
    if len(actions_raw.shape) == 2 and actions_raw.shape[1] > 1:
        # One-hot encoded actions: [B, num_actions] -> [B]
        target_actions = torch.from_numpy(np.argmax(actions_raw, axis=1)).to(device)
        print(f"[DEBUG] Converted one-hot actions {actions_raw.shape} to scalar indices {target_actions.shape}")
    else:
        # Already scalar actions
        target_actions = torch.from_numpy(actions_raw).to(device)
    
    print(f"[DEBUG] BC batch - target_actions: {target_actions}")
    print(f"[DEBUG] BC batch - obs shape: {obs_data.shape}")
    print(f"[DEBUG] BC batch - actions unique values: {torch.unique(target_actions)}")
    
    # Check if batch images are actually different
    print(f"[DEBUG] Image diversity check:")
    batch_size = obs_data.shape[0]
    for i in range(batch_size):
        img = obs_data[i]  # [C*stack_n, H, W]
        img_mean = np.mean(img)
        img_std = np.std(img)
        img_hash = hash(img.tobytes())
        print(f"  Batch[{i}]: mean={img_mean:.4f}, std={img_std:.4f}, hash={img_hash}")
    
    # Check for identical images
    unique_hashes = set()
    duplicate_count = 0
    for i in range(batch_size):
        img_hash = hash(obs_data[i].tobytes())
        if img_hash in unique_hashes:
            duplicate_count += 1
        unique_hashes.add(img_hash)
    print(f"[DEBUG] Unique images: {len(unique_hashes)}/{batch_size}, duplicates: {duplicate_count}")
    
    # Check ModelNet's expected input type
    if hasattr(model_net, 'state_dtype_n') and model_net.state_dtype_n == 0:
        # ModelNet expects uint8 input (will normalize internally)
        if obs_data.dtype == np.float32:
            # Convert float32 [0,1] back to uint8 [0,255]
            obs_uint8 = (obs_data * 255.0).astype(np.uint8)
            obs = torch.tensor(obs_uint8, dtype=torch.uint8, device=device)
            print(f"[DEBUG] Converted float32 to uint8 for ModelNet")
        else:
            # Already uint8
            obs = torch.tensor(obs_data, dtype=torch.uint8, device=device)
            print(f"[DEBUG] Using uint8 observations for ModelNet")
    else:
        # ModelNet expects float32 input
        if obs_data.dtype == np.float32:
            obs = torch.tensor(obs_data, dtype=torch.float32, device=device)
            print(f"[DEBUG] Using float32 observations for ModelNet")
        else:
            # Convert uint8 to float32
            obs = torch.tensor(obs_data, dtype=torch.uint8, device=device).float() / 255.0
            print(f"[DEBUG] Converted uint8 to float32 for ModelNet")
    
    # Note: target_actions already properly converted above from one-hot/indices
    # No need to re-assign here - this was causing the double assignment bug
    
    batch_size = obs.shape[0]
    rec_t = flags.rec_t
    num_actions = 6
    
    print(f"[DEBUG] Starting Thinker rollout: rec_t={rec_t}, batch_size={batch_size}")
    
    # IMPORTANT: Initialize ModelNet state with actual observation encoding
    # This ensures xs/hs are conditioned on real observation
    with torch.no_grad():
        # ModelNet.forward expects: env_state, actions, done, state
        # We use a dummy action [T=1, B, 1] to get initial encoding of the real observation
        initial_model_out = model_net.forward(
            env_state=obs,  # [B, C, H, W] normalized observation
            actions=torch.zeros(1, batch_size, 1, dtype=torch.long, device=device),  # [T=1, B, 1] dummy action
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),  # [B] no episodes done
            state=model_net.initial_state(batch_size=batch_size, device=device)  # Fresh state
        )
        model_state = initial_model_out.state
        initial_model_state = {k: v.clone() if hasattr(v, 'clone') else v for k, v in model_state.items()} if isinstance(model_state, dict) else model_state
        
        # Get initial xs and hs from real observation
        initial_xs = initial_model_out.xs[0] if initial_model_out.xs is not None else None
        initial_hs = initial_model_out.hs[0] if initial_model_out.hs is not None else None
    
    print(f"[DEBUG] Initialized ModelNet with real observation encoding")
    
    # Initialize actor state
    actor_core_state = actor_net.initial_state(batch_size=batch_size, device=device)
    
    # Initialize tree_reps with proper structure FIRST
    tree_reps = initialize_tree_reps(batch_size, num_actions, flags, device)
    
    # Set initial root action to match actual target actions for each batch
    tree_reps[:, :num_actions] = 0.0  # Clear default
    for i in range(batch_size):
        if target_actions[i] < num_actions:
            tree_reps[i, target_actions[i]] = 1.0
    
    # Update root statistics with initial ModelNet outputs
    if initial_model_out.policy is not None and initial_model_out.vs is not None:
        # root_policy from initial VPNet output
        initial_policy = initial_model_out.policy[0] if len(initial_model_out.policy.shape) > 1 else initial_model_out.policy
        # Handle potential extra dimensions in policy output
        if len(initial_policy.shape) == 3:  # [B, 1, num_actions]
            initial_policy = initial_policy.squeeze(1)  # [B, num_actions]
        elif len(initial_policy.shape) == 1:  # [num_actions] for single batch
            initial_policy = initial_policy.unsqueeze(0)  # [1, num_actions]
        
        print(f"[DEBUG] Initial policy shape: {initial_policy.shape}")
        tree_reps[:, num_actions+3:2*num_actions+3] = initial_policy
        
        # root_v from initial VPNet output
        initial_value = initial_model_out.vs[0] if len(initial_model_out.vs.shape) > 1 else initial_model_out.vs
        # Handle potential extra dimensions in value output
        if len(initial_value.shape) == 2:  # [B, 1]
            initial_value = initial_value.squeeze(1)  # [B]
        elif len(initial_value.shape) == 0:  # scalar for single batch
            initial_value = initial_value.unsqueeze(0)  # [1]
            
        print(f"[DEBUG] Initial value shape: {initial_value.shape}")
        tree_reps[:, num_actions+2] = initial_value
        
        # root_qs estimates from initial outputs
        root_base_value = initial_value.unsqueeze(-1)  # [B, 1]
        root_policy_normalized = initial_policy - initial_policy.mean(dim=-1, keepdim=True)
        root_estimated_qs = root_base_value + root_policy_normalized * 0.1
        tree_reps[:, 2*num_actions+3:3*num_actions+3] = root_estimated_qs  # root_qs_mean
        tree_reps[:, 3*num_actions+3:4*num_actions+3] = root_estimated_qs + 0.05  # root_qs_max
        
        # root_ns from initial policy (higher prob = more visits)
        root_estimated_visits = initial_policy * 50.0 + 10.0  # Scale to reasonable range
        tree_reps[:, 4*num_actions+3:5*num_actions+3] = root_estimated_visits
        
        print(f"[DEBUG] Updated root stats with initial VPNet outputs")
    
    # Track rollout state
    current_obs = obs.float()  # [B, C, H, W]
    current_xs = initial_xs  # Start with encoded real observation
    current_hs = initial_hs
    cur_t = torch.zeros(batch_size, dtype=torch.long, device=device)
    rollout_depth = torch.zeros(batch_size, dtype=torch.long, device=device)
    xs_history = [initial_xs]
    hs_history = [initial_hs]
    
    # Initialize last_reset tracking for ActorNet input  
    last_reset = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Use actual BC target action for first imaginary step (not dummy zeros!)
    last_action = target_actions.clone()  # Use real BC action as starting point
    
    print(f"[DEBUG] Using BC target actions {target_actions[:5]} as initial actions")
    
    print(f"[DEBUG] Starting {rec_t-1} imaginary steps...")
    
    # Variables to collect first imagine step info for enhanced BC
    first_imagine_state = None
    first_imagine_action = None
    
    # Perform rec_t-1 imaginary steps
    for step in range(rec_t - 1):
        print(f"[DEBUG] Imaginary step {step+1}/{rec_t-1}")
        
        # IMPORTANT: Update cur_t and rollout_depth per batch CORRECTLY
        # cenv.pyx logic: cur_t[i] += 1 for ALL batches in each imaginary step
        # rollout_depth[i] += 1 for each batch, but resets to 0 on reset
        cur_t += 1  # All batches increment cur_t independently
        # rollout_depth increments for all batches, but may be reset individually later
        rollout_depth += 1
        
        # 1. ModelNet forward pass FIRST (get SRN + VPN outputs)
        # This follows cenv.pyx order: ModelNet → tree_reps update → ActorNet → next_action
        
        # **ENHANCED BC**: Always use no_grad for rollout, collect info separately
        # Standard rollout: No gradients for ModelNet during rollout
        with torch.no_grad():
            model_out = model_net.forward_single(
                state=model_state,
                action=last_action,
                training=False
            )
        
        # **COLLECT FIRST IMAGINE STEP INFO for Enhanced BC (without gradients during rollout)**
        if step == 0:
            print(f"[DEBUG] Collecting first imagine step info for enhanced BC")
            # Store the state and action for later separate forward pass
            first_imagine_state = {k: v.clone() if hasattr(v, 'clone') else v for k, v in model_state.items()} if isinstance(model_state, dict) else model_state.clone()
            first_imagine_action = last_action.clone()
        
        # Extract model outputs
        xs = model_out.xs[0] if model_out.xs is not None else None  # [B, C, H, W]
        hs = model_out.hs[0] if model_out.hs is not None else None  # [B, ...]
        
        # Update current xs/hs (will be selectively reset later if needed)
        if xs is not None:
            current_xs = xs.clone()  # Use clone to avoid reference issues
        if hs is not None:
            current_hs = hs.clone()  # Use clone to avoid reference issues
            
        xs_history.append(current_xs)
        hs_history.append(current_hs)
        
        # Update model state (will be selectively reset later if needed)
        if isinstance(model_out.state, dict):
            # Deep copy to avoid reference issues
            for key in model_out.state:
                if hasattr(model_out.state[key], 'clone'):
                    model_state[key] = model_out.state[key].clone()
                else:
                    model_state[key] = model_out.state[key]
        else:
            model_state = model_out.state
        
        # 2. Update tree_reps with complete ModelNet outputs (SRN + VPN)
        # This creates the complete tree_reps that ActorNet will use
        tree_reps = update_tree_reps_step(
            tree_reps, cur_t, rollout_depth, last_action, num_actions, flags, device, 
            model_out=model_out, actor_out=None
        )
        
        # 3. Create env_out for ActorNet
        from types import SimpleNamespace
        env_out = SimpleNamespace()
        
        # Use current_xs if available, otherwise current_obs
        visual_input = current_xs if current_xs is not None else current_obs
        env_out.real_states = visual_input.unsqueeze(0)  # [T=1, B, C, H, W]
        env_out.tree_reps = tree_reps.unsqueeze(0)  # [T=1, B, obs_n]
        
        if current_hs is not None:
            env_out.hs = current_hs.unsqueeze(0)  # [T=1, B, ...]
        if current_xs is not None:
            env_out.xs = current_xs.unsqueeze(0)  # [T=1, B, C, H, W]
        
        # Set step status following cenv.pyx logic
        # step_status: 0=real action taken, 1=im action taken, 2=im action taken; next is real, 3=real action taken and next is real
        if step == rec_t - 2:  # Last imaginary step
            step_status = 2  # im action just taken; next action is real action
        else:
            step_status = 1  # im action just taken
        env_out.step_status = torch.full((1, batch_size), step_status, dtype=torch.long, device=device)
        
        # Other required fields
        env_out.done = torch.zeros(1, batch_size, dtype=torch.bool, device=device)
        env_out.real_done = torch.zeros(1, batch_size, dtype=torch.bool, device=device)
        env_out.last_pri = last_action.unsqueeze(0)
        env_out.last_reset = last_reset.unsqueeze(0)  # [T=1, B] pass actual reset history
        env_out.reward = torch.zeros(1, batch_size, 2, device=device)
        
        # 4. ActorNet forward pass to get next action (no gradients for imaginary steps)
        # This action will be used for the NEXT imaginary step's ModelNet input
        with torch.no_grad():
            actor_out, actor_core_state = actor_net.forward(
                env_out=env_out,
                core_state=actor_core_state
            )
        
        # Sample action for next step (using argmax for consistency)
        action_probs = actor_out.action_prob[0]  # [B, num_actions] - this is softmax probabilities
        next_action = torch.argmax(action_probs, dim=-1)  # [B]
        
        # Check for reset action from ActorNet (ONLY in imaginary steps)
        # NOTE: Reset actions are only valid during imagination, NOT in real steps
        reset_action = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if hasattr(actor_out, 'reset') and actor_out.reset is not None:
            reset_probs = actor_out.reset[0]
            reset_action = reset_probs > 0.5  # [B] boolean mask
            if reset_action.any():
                print(f"[DEBUG] Step {step+1}: Reset action detected - probs: {reset_probs.cpu().numpy()}")
                print(f"[DEBUG] Step {step+1}: Reset triggered for batches: {reset_action.nonzero().flatten().tolist()}")
        
        # Check for force reset conditions (following cenv.pyx logic)
        max_depth = getattr(flags, 'max_depth', 40)
        force_reset = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if max_depth > 0:
            force_reset = rollout_depth >= max_depth
        
        # Apply reset logic: reset OR force_reset
        should_reset = reset_action | force_reset
        
        # Handle reset: return to root state (following cenv.pyx status == 5 logic)
        if should_reset.any():
            print(f"[DEBUG] Reset triggered for batches: {should_reset.nonzero().flatten().tolist()}")
            # Reset rollout_depth to 0 for reset batches (but keep cur_t progressing)
            rollout_depth[should_reset] = 0
            # Record reset action for next ActorNet call
            last_reset[should_reset] = 1
            
            # Update tree_reps reset flag for batches that reset
            # Following cenv.pyx: result[i, idx4] = reset[i]  
            idx4 = num_actions * 5 + 6 + num_actions * 5 + 3  # Reset flag position
            tree_reps[should_reset, idx4] = 1.0
            
            # Return to root states
            if initial_xs is not None:
                current_xs[should_reset] = initial_xs[should_reset]
            if initial_hs is not None:
                current_hs[should_reset] = initial_hs[should_reset]
            # Reset model state to initial state for reset batches
            if isinstance(model_state, dict) and isinstance(initial_model_state, dict):
                for key in model_state:
                    if key in initial_model_state and hasattr(model_state[key], 'clone'):
                        # Reset specific batches to initial model state
                        reset_indices = should_reset.nonzero().flatten()
                        if len(reset_indices) > 0:
                            model_state[key][reset_indices] = initial_model_state[key][reset_indices].clone()
        else:
            # No reset occurred
            last_reset.fill_(0)
        
        last_action = next_action
        
        # Debug: Check if all batches predict same action
        unique_actions = torch.unique(next_action)
        if len(unique_actions) == 1:
            print(f"[WARNING] Step {step+1}: All batches predict same action {unique_actions[0].item()}")
            # Show first few action probabilities for debugging
            for i in range(min(3, batch_size)):
                probs = action_probs[i].cpu().numpy()
                print(f"  Batch[{i}] action_probs: {probs}")
                
                # Also show reset action probability if available
                if hasattr(actor_out, 'reset') and actor_out.reset is not None:
                    reset_prob = actor_out.reset[0][i].cpu().numpy()
                    print(f"  Batch[{i}] reset_prob: {reset_prob:.6f}")
                    

        else:
            print(f"[DEBUG] Step {step+1}: Diverse actions predicted: {unique_actions.cpu().numpy()}")
            
            # Still show reset info for diverse actions case
            if hasattr(actor_out, 'reset') and actor_out.reset is not None:
                reset_probs = actor_out.reset[0].cpu().numpy()
                print(f"  Reset probabilities: {reset_probs}")
        
        print(f"[DEBUG] Step {step+1}: Generated action distribution, next_action shape: {next_action.shape}")
    
    print(f"[DEBUG] Completed {rec_t-1} imaginary steps")
    
    # Final step: Real step for BC loss calculation
    print(f"[DEBUG] Final real step for BC loss")
    
    # Final tree_reps update for real step
    # In cenv.pyx, real step resets: cur_t[i] = 0, rollout_depth[i] = 0
    # All batches reset to 0 for real step (this is correct behavior)
    cur_t.zero_()  # Real step: all batches reset
    rollout_depth.zero_()  # Real step: all batches reset
    tree_reps = update_tree_reps_step(
        tree_reps, cur_t, rollout_depth, last_action, num_actions, flags, device,
        model_out=None, actor_out=None  # No new outputs for final step
    )
    
    # Create final env_out for BC loss
    env_out = SimpleNamespace()
    
    # Use final current_xs if available, otherwise current_obs
    final_visual_input = current_xs if current_xs is not None else current_obs
    env_out.real_states = final_visual_input.unsqueeze(0)  # [T=1, B, C, H, W]
    env_out.tree_reps = tree_reps.unsqueeze(0)  # [T=1, B, obs_n]
    
    # Use final hs and xs
    if current_hs is not None:
        env_out.hs = current_hs.unsqueeze(0)
    if current_xs is not None:
        env_out.xs = current_xs.unsqueeze(0)
    
    # Set step status (0 = real step)
    env_out.step_status = torch.zeros(1, batch_size, dtype=torch.long, device=device)
    env_out.done = torch.zeros(1, batch_size, dtype=torch.bool, device=device)
    env_out.real_done = torch.zeros(1, batch_size, dtype=torch.bool, device=device)
    env_out.last_pri = last_action.unsqueeze(0)
    env_out.last_reset = torch.zeros(1, batch_size, dtype=torch.long, device=device)
    env_out.reward = torch.zeros(1, batch_size, 2, device=device)
    
    # Final actor forward pass for BC loss
    # NOTE: This is the REAL step - no reset actions are processed here
    # Only primary actions are considered for BC loss computation
    final_actor_out, _ = actor_net.forward(
        env_out=env_out,
        core_state=actor_core_state
    )
    
    print(f"[DEBUG] Thinker rollout completed, ready for BC loss")
    
    # **Enhanced BC**: Perform separate forward pass for first imagine step with gradients
    enhanced_imagine_step = None
    if hasattr(flags, 'bc_enhanced') and flags.bc_enhanced and 'first_imagine_state' in locals() and 'first_imagine_action' in locals():
        print(f"[DEBUG] Enhanced BC: Performing separate forward pass for first imagine step")
        # Separate forward pass with gradients enabled
        first_imagine_model_out = model_net.forward_single(
            state=first_imagine_state,
            action=first_imagine_action,
            training=True  # Enable gradients for this forward pass
        )
        
        enhanced_imagine_step = {
            'vpn_policy': first_imagine_model_out.policy if hasattr(first_imagine_model_out, 'policy') else None,
            'srn_reward': first_imagine_model_out.rs if hasattr(first_imagine_model_out, 'rs') else None,
            'hidden_state': first_imagine_model_out.hs if hasattr(first_imagine_model_out, 'hs') else None,
            'model_out': first_imagine_model_out
        }
        print(f"[DEBUG] Enhanced BC: Separate forward pass completed")
    
    return {
        'model_out': None,  # Not used for standard BC loss
        'actor_out': final_actor_out,
        'target_actions': target_actions,
        'rollout_info': {
            'xs_history': xs_history,
            'hs_history': hs_history,
            'steps_completed': rec_t - 1
        },
        # **Enhanced BC**: First imagine step outputs for ModelNet training
        'first_imagine_step': enhanced_imagine_step
    }


def initialize_tree_reps(batch_size, num_actions, flags, device):
    """
    Initialize empty tree_reps structure that will be filled during rollout
    Based on compute_tree_reps in cenv.pyx and slice_tree_reps in util.py
    """
    rec_t = flags.rec_t
    
    # Calculate dimensions based on util.py slice_tree_reps
    idx1 = num_actions * 5 + 6  # root stats
    idx2 = idx1
    idx3 = idx2 + num_actions * 5 + 3  # current stats
    idx4 = idx3
    idx5 = idx4 + 2 + rec_t  # reset + time + deprec
    
    obs_n = idx5
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        obs_n += flags.max_depth * num_actions
        if hasattr(flags, 'reset_mode') and flags.reset_mode == 0:
            obs_n += num_actions
    
    # Initialize with zeros - will be filled during rollout
    tree_reps = torch.zeros(batch_size, obs_n, device=device)
    
    # Initialize root stats following util.py slice_tree_reps mapping:
    # root_action, root_r, root_d, root_v, root_policy, root_qs_mean, root_qs_max, root_ns, root_trail_r, rollout_return, max_rollout_return
    
    # root_action (one-hot encoding) - mock initial action
    tree_reps[:, :num_actions] = 0.0
    tree_reps[:, 0] = 1.0  # Default to action 0
    
    # root_r (root reward) - mock value
    tree_reps[:, num_actions] = torch.randn(batch_size, device=device) * 0.1
    
    # root_d (root done) - usually 0
    tree_reps[:, num_actions+1] = 0.0
    
    # root_v (root value) - mock value
    tree_reps[:, num_actions+2] = torch.randn(batch_size, device=device) * 0.1
    
    # root_policy (root policy distribution) - will be updated with initial VPNet output
    tree_reps[:, num_actions+3:2*num_actions+3] = 1.0/num_actions  # Start with uniform
    
    # root_qs_mean, root_qs_max (Q-value statistics) - will be updated with initial estimates
    tree_reps[:, 2*num_actions+3:4*num_actions+3] = torch.zeros(batch_size, 2*num_actions, device=device)
    
    # root_ns (root visit counts) - will be updated with initial estimates
    tree_reps[:, 4*num_actions+3:5*num_actions+3] = torch.ones(batch_size, num_actions, device=device) * 1.0
    
    # root_trail_r (trailing reward) - mock value
    tree_reps[:, 5*num_actions+3] = torch.randn(batch_size, device=device) * 0.1
    
    # rollout_return - mock value
    tree_reps[:, 5*num_actions+4] = torch.randn(batch_size, device=device) * 0.1
    
    # max_rollout_return - mock value  
    tree_reps[:, 5*num_actions+5] = torch.randn(batch_size, device=device) * 0.1
    
    # Initialize current node stats as well (these will be updated each step)
    tree_reps[:, idx2:idx3] = torch.randn(batch_size, idx3-idx2, device=device) * 0.1
    
    # Initialize reset flag to 0 (no reset initially)
    tree_reps[:, idx4] = 0.0
    
    # Initialize time encoding to 0 (will be set in first step)
    tree_reps[:, idx4+1:idx4+1+rec_t] = 0.0
    
    # Initialize depreciation factor
    tree_reps[:, idx4 + rec_t + 1] = 1.0  # No discounting initially
    
    return tree_reps


def update_tree_reps_step(tree_reps, cur_t, rollout_depth, last_action, num_actions, flags, device, model_out=None, actor_out=None):
    """
    Update tree_reps for current step following actual Thinker logic
    Based on compute_tree_reps time encoding and action sequence logic
    """
    batch_size = tree_reps.shape[0]
    rec_t = flags.rec_t
    
    # Calculate index positions
    idx1 = num_actions * 5 + 6  # root stats
    idx2 = idx1
    idx3 = idx2 + num_actions * 5 + 3  # current stats
    idx4 = idx3
    idx5 = idx4 + 2 + rec_t  # reset + time + deprec
    
    # Update current node stats (these change with each step)
    # Following util.py slice_tree_reps mapping:
    # cur_action, cur_r, cur_d, cur_v, cur_policy, cur_qs_mean, cur_qs_max, cur_ns, cur_raw_action
    
    # Extract real values from ModelNet and ActorNet outputs
    
    # cur_action (one-hot encoding of current action)
    tree_reps[:, idx2:idx2+num_actions] = 0.0  # Clear previous
    for i in range(batch_size):
        if last_action[i] < num_actions:
            tree_reps[i, idx2 + last_action[i]] = 1.0
    
    # Extract actual values from model outputs
    if model_out is not None:
        # cur_r (current reward) - from ModelNet
        if hasattr(model_out, 'rs') and model_out.rs is not None:
            # model_out.rs shape: [T, B] or [B] or [B, 1]
            cur_rewards = model_out.rs[-1] if len(model_out.rs.shape) > 1 else model_out.rs
            # Handle potential extra dimensions
            if len(cur_rewards.shape) == 2:  # [B, 1]
                cur_rewards = cur_rewards.squeeze(1)  # [B]
            tree_reps[:, idx2+num_actions] = cur_rewards.float()
        else:
            tree_reps[:, idx2+num_actions] = 0.0
        
        # cur_d (current done) - from ModelNet 
        if hasattr(model_out, 'dones') and model_out.dones is not None:
            cur_dones = model_out.dones[-1] if len(model_out.dones.shape) > 1 else model_out.dones
            # Handle potential extra dimensions
            if len(cur_dones.shape) == 2:  # [B, 1]
                cur_dones = cur_dones.squeeze(1)  # [B]
            tree_reps[:, idx2+num_actions+1] = cur_dones.float()
        else:
            tree_reps[:, idx2+num_actions+1] = 0.0
        
        # cur_v (current value) - from ModelNet
        if hasattr(model_out, 'vs') and model_out.vs is not None:
            cur_values = model_out.vs[-1] if len(model_out.vs.shape) > 1 else model_out.vs
            # Handle potential extra dimensions
            if len(cur_values.shape) == 2:  # [B, 1]
                cur_values = cur_values.squeeze(1)  # [B]
            tree_reps[:, idx2+num_actions+2] = cur_values.float()
        else:
            tree_reps[:, idx2+num_actions+2] = 0.0
    else:
        # Fallback values when no model output
        tree_reps[:, idx2+num_actions] = 0.0  # reward
        tree_reps[:, idx2+num_actions+1] = 0.0  # done
        tree_reps[:, idx2+num_actions+2] = 0.0  # value
    
    # Extract policy and Q-statistics from ActorNet output
    if actor_out is not None:
        # cur_policy (current policy distribution) - from ActorNet
        if hasattr(actor_out, 'pri_param') and actor_out.pri_param is not None:
            # Use raw logits converted to probabilities
            actor_logits = actor_out.pri_param[0] if len(actor_out.pri_param.shape) > 2 else actor_out.pri_param
            if len(actor_logits.shape) == 3:  # [B, num_actions, 1]
                actor_logits = actor_logits[:, :, 0]
            cur_policy = F.softmax(actor_logits, dim=-1)
            tree_reps[:, idx2+num_actions+3:idx2+2*num_actions+3] = cur_policy
        elif hasattr(actor_out, 'action_prob') and actor_out.action_prob is not None:
            # Use probabilities directly
            actor_probs = actor_out.action_prob[0] if len(actor_out.action_prob.shape) > 2 else actor_out.action_prob
            tree_reps[:, idx2+num_actions+3:idx2+2*num_actions+3] = actor_probs
        else:
            # Uniform distribution fallback
            tree_reps[:, idx2+num_actions+3:idx2+2*num_actions+3] = 1.0/num_actions
        
        # Q-value statistics - calculate from VPNet outputs
        # Following cenv.pyx logic: child_rollout_qs_mean, child_rollout_qs_max
        if model_out is not None and hasattr(model_out, 'vs') and model_out.vs is not None and hasattr(model_out, 'policy') and model_out.policy is not None:
            # Get VPNet outputs: policy and value
            vp_policy = model_out.policy[-1] if len(model_out.policy.shape) > 1 else model_out.policy  # [B, num_actions]
            vp_value = model_out.vs[-1] if len(model_out.vs.shape) > 1 else model_out.vs  # [B]
            
            # Handle potential extra dimensions in VPNet outputs
            if len(vp_policy.shape) == 3:  # [B, 1, num_actions]
                vp_policy = vp_policy.squeeze(1)  # [B, num_actions]
            if len(vp_value.shape) == 2:  # [B, 1]
                vp_value = vp_value.squeeze(1)  # [B]
            
            # Estimate Q-values for each action using: Q(s,a) ≈ V(s) + advantage
            # For BC, we can use policy as a proxy for advantage: higher prob = higher advantage
            base_value = vp_value.unsqueeze(-1)  # [B, 1]
            
            # Simple Q-value estimation: Q(s,a) = V(s) + policy_advantage
            # Normalize policy to get relative advantages
            policy_normalized = vp_policy - vp_policy.mean(dim=-1, keepdim=True)  # Center around mean
            estimated_qs = base_value + policy_normalized * 0.1  # [B, num_actions] scaled advantage
            
            # cur_qs_mean - use estimated Q-values as both mean and individual values
            tree_reps[:, idx2+2*num_actions+3:idx2+3*num_actions+3] = estimated_qs
            
            # cur_qs_max - use slightly higher values as "max" rollout Q-values 
            estimated_qs_max = estimated_qs + 0.05  # Small boost for max values
            tree_reps[:, idx2+3*num_actions+3:idx2+4*num_actions+3] = estimated_qs_max
            
            print(f"[DEBUG] Extracted Q-stats from VPNet: value={vp_value[0]:.3f}, policy_max={vp_policy[0].max():.3f}")
        else:
            # Fallback when no VPNet outputs available
            tree_reps[:, idx2+2*num_actions+3:idx2+4*num_actions+3] = torch.randn(batch_size, 2*num_actions, device=device) * 0.1
            print(f"[DEBUG] Using fallback Q-stats (no VPNet outputs)")
        
        # Visit counts - estimate from policy (higher policy = more visits)
        if actor_out is not None and hasattr(actor_out, 'action_prob') and actor_out.action_prob is not None:
            actor_probs = actor_out.action_prob[0] if len(actor_out.action_prob.shape) > 2 else actor_out.action_prob
            # Convert probabilities to visit counts (scaled by a factor)
            estimated_visits = actor_probs * 10.0 + 1.0  # Scale to reasonable visit count range
            tree_reps[:, idx2+4*num_actions+3:idx3] = estimated_visits
        else:
            # Uniform visit counts fallback
            tree_reps[:, idx2+4*num_actions+3:idx3] = torch.ones(batch_size, num_actions, device=device) * 1.0
    else:
        # Fallback values when no actor output
        tree_reps[:, idx2+num_actions+3:idx2+2*num_actions+3] = 1.0/num_actions  # uniform policy
        tree_reps[:, idx2+2*num_actions+3:idx2+4*num_actions+3] = 0.0  # Q-stats
        tree_reps[:, idx2+4*num_actions+3:idx3] = 1.0  # visit counts
    
    # Reset flag (0 for normal steps)
    # NOTE: Reset flag is updated later in main loop if reset occurs
    tree_reps[:, idx4] = 0.0
    
    # Clear previous time encoding per batch
    tree_reps[:, idx4+1:idx4+1+rec_t] = 0.0
    
    # Set current time encoding (one-hot)
    # cur_t[i] < rec_t condition from cenv.pyx line 449
    for i in range(batch_size):
        if cur_t[i] < rec_t:
            tree_reps[i, idx4 + 1 + cur_t[i]] = 1.0
    
    # Depreciation factor based on rollout depth
    # Following cenv.pyx line 452: (discounting ** rollout_depth)
    discounting = getattr(flags, 'discounting', 0.99)
    tree_reps[:, idx4 + rec_t + 1] = discounting ** rollout_depth.float()
    
    # Action sequence (if enabled)
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        # Following cenv.pyx logic: accumulate full path from root to current depth
        # cenv.pyx: for j in range(rollout_depth[i] + 1): 
        #               result[i, idx5+(rollout_depth[i] - j)*num_actions+node[0].action] = 1
        
        # Update action sequence based on current rollout
        for i in range(batch_size):
            current_depth = rollout_depth[i].item()
            if current_depth >= 0:
                # Set action at current depth position
                # Each depth gets its own "slot" in the action sequence
                action = last_action[i].item()
                action_idx = idx5 + current_depth * num_actions + action
                if action_idx < tree_reps.shape[1]:
                    tree_reps[i, action_idx] = 1.0
                    
                    # print(f"[DEBUG] Set action_seq[{i}][depth={current_depth}][action={action}] at idx={action_idx}")  # Reduced logging
        
        # Debug: Print current action sequence state
        if hasattr(flags, 'debug_action_seq') and flags.debug_action_seq and batch_size > 0:
            action_seq = tree_reps[0, idx5:idx5+flags.max_depth*num_actions]
            print(f"[DEBUG] Full action_seq for batch[0]: {action_seq.nonzero().squeeze().tolist() if action_seq.sum() > 0 else []}")
    
    return tree_reps


def create_mock_tree_reps(batch_size, num_actions, flags, device):
    """
    Create mock tree_reps following the actual Thinker structure
    Based on compute_tree_reps in cenv.pyx and slice_tree_reps in util.py
    (This function is kept for backward compatibility but not used in new rollout)
    """
    rec_t = flags.rec_t
    
    # Calculate dimensions based on util.py slice_tree_reps
    idx1 = num_actions * 5 + 6  # root stats
    idx2 = idx1
    idx3 = idx2 + num_actions * 5 + 3  # current stats
    idx4 = idx3
    idx5 = idx4 + 2 + rec_t  # reset + time + deprec
    
    obs_n = idx5
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        obs_n += flags.max_depth * num_actions
        if hasattr(flags, 'reset_mode') and flags.reset_mode == 0:
            obs_n += num_actions
    
    tree_reps = torch.zeros(batch_size, obs_n, device=device)
    
    # Fill root node stats (idx 0:idx1)
    # root_action, root_r, root_d, root_v, root_policy, etc.
    tree_reps[:, :idx1] = torch.randn(batch_size, idx1, device=device) * 0.1
    
    # Fill current node stats (idx2:idx3)
    # cur_action, cur_r, cur_d, cur_v, cur_policy, etc.
    tree_reps[:, idx2:idx3] = torch.randn(batch_size, idx3-idx2, device=device) * 0.1
    
    # Reset flag (usually 0 for non-reset)
    tree_reps[:, idx4] = 0.0
    
    # Time encoding (one-hot for current timestep)
    # For BC, we'll set a random timestep
    time_step = torch.randint(0, rec_t, (batch_size,), device=device)
    for i in range(batch_size):
        if time_step[i] < rec_t:
            tree_reps[i, idx4 + 1 + time_step[i]] = 1.0
    
    # Depreciation factor
    tree_reps[:, idx4 + rec_t + 1] = 0.99  # Default discount
    
    # Action sequence (if enabled)
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        # Fill with random action sequence
        action_seq_start = idx5
        action_seq_len = flags.max_depth * num_actions
        if hasattr(flags, 'reset_mode') and flags.reset_mode == 0:
            action_seq_len += num_actions
        
        # Sparse encoding of action sequence
        for i in range(batch_size):
            depth = torch.randint(1, min(5, flags.max_depth), (1,)).item()
            for d in range(depth):
                action = torch.randint(0, num_actions, (1,)).item()
                idx = action_seq_start + d * num_actions + action
                if idx < obs_n:
                    tree_reps[i, idx] = 1.0
    
    return tree_reps


def save_bc_checkpoint(model_net, actor_net, optimizers, epoch, losses, save_path, flags):
    """Save BC training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_net_state_dict': model_net.state_dict(),
        'actor_net_state_dict': actor_net.state_dict(),
        'model_optimizer_state_dict': optimizers['model'].state_dict(),
        'actor_optimizer_state_dict': optimizers['actor'].state_dict(),
        'losses': losses,
        'flags': flags
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved BC checkpoint: {save_path}")


def run_bc_training(flags, model_net, actor_net, logger=None):
    """
    Run behavioral cloning training
    
    Args:
        flags: Configuration flags
        model_net: ModelNet instance
        actor_net: ActorNet instance
        logger: Optional logger
    """
    print("=== Starting Behavioral Cloning Training ===")
    print(f"Epochs: {flags.bc_epochs}")
    print(f"Learning Rate: {flags.bc_lr}")
    print(f"Batch Size: {flags.bc_batch_size}")
    
    # Device setup
    device = next(model_net.parameters()).device
    print(f"Device: {device}")
    
    # Parse subjects
    subjects = [int(s.strip()) for s in flags.bc_subjects.split(',')]
    print(f"Subjects: {subjects}")
    print(f"Game ID: {flags.bc_game_id}")
    
    # Create data loader
    print("\nLoading behavioral data...")
    bc_loader = create_bc_data_loader(flags)
    
    # Test data loading
    test_batch = bc_loader.get_paired_batch(batch_size=flags.bc_batch_size)
    if test_batch is None:
        print("ERROR: Could not load test batch from BC data")
        return
    
    print(f"Test batch loaded successfully:")
    for key, value in test_batch.items():
        print(f"  {key}: {value.shape}")
    
    # Set networks to training mode
    # **Enhanced BC**: Enable ModelNet training if enhanced mode
    if hasattr(flags, 'bc_enhanced') and flags.bc_enhanced:
        print("=== Enhanced BC Mode: Training both ModelNet and ActorNet ===")
        
        # Anomaly detection no longer needed - issue resolved
        # torch.autograd.set_detect_anomaly(True)
        print("Enhanced BC mode - gradient issues resolved")
        
        model_net.train()  # Enable ModelNet training
        
        # Disable half-gradient hook to avoid hook registration errors in BC
        if hasattr(model_net, 'vp_net') and hasattr(model_net.vp_net, 'RNN'):
            model_net.vp_net.RNN.disable_half_grad = True
            print("Disabled half-gradient hook for BC training")
            
        actor_net.train()  # Train actor as well
        
        # Enable all model parameters for training
        print("Enabling ModelNet parameters for training (SRN + VPN)...")
        for param in model_net.parameters():
            param.requires_grad = True
        
        total_model_params = sum(p.numel() for p in model_net.parameters())
        print(f"Trainable ModelNet parameters: {total_model_params:,}")
        
        # Setup optimizers for both networks
        model_optimizer = optim.Adam(model_net.parameters(), lr=flags.bc_lr)
        actor_optimizer = optim.Adam(actor_net.parameters(), lr=flags.bc_lr)
        
        optimizers = {
            'model': model_optimizer,
            'actor': actor_optimizer
        }
        
        print(f"Model parameters: {sum(p.numel() for p in model_net.parameters()):,} (trainable)")
        print(f"Actor parameters: {sum(p.numel() for p in actor_net.parameters()):,} (trainable)")
    else:
        print("=== Standard BC Mode: Training ActorNet only ===")
        model_net.eval()  # Keep model in eval mode (frozen, only used for xs/hs generation)
        actor_net.train()  # Only train actor
        
        # Freeze all model parameters (no model training in BC)
        print("Freezing all Model parameters (SRN + VPN)...")
        for param in model_net.parameters():
            param.requires_grad = False
        
        total_model_params = sum(p.numel() for p in model_net.parameters())
        print(f"Frozen {total_model_params:,} Model parameters")
        
        # Setup optimizer for Actor only
        actor_optimizer = optim.Adam(actor_net.parameters(), lr=flags.bc_lr)
        
        optimizers = {
            'model': None,  # No model optimizer for standard BC
            'actor': actor_optimizer
        }
        
        print(f"Model parameters: {sum(p.numel() for p in model_net.parameters()):,} (frozen)")
        print(f"Actor parameters: {sum(p.numel() for p in actor_net.parameters()):,} (trainable)")
    
    print(f"Model parameters: {sum(p.numel() for p in model_net.parameters()):,}")
    print(f"Actor parameters: {sum(p.numel() for p in actor_net.parameters()):,}")
    
    # Training loop - Each epoch is now 1 batch (32 transitions)
    print(f"\nStarting BC training for {flags.bc_epochs} steps...")
    print(f"Each step processes {flags.bc_batch_size} transitions")
    print(f"Logging every step, saving every 100 steps")
    
    # Get the log directory from flags and ensure it's at the project root level
    log_dir = getattr(flags, 'savedir', './logs/thinker')
    # Get absolute path and go up from /thinker/thinker to /thinker
    current_dir = os.path.abspath('.')  # /home/jmme425/thinker/thinker
    project_root = os.path.dirname(current_dir)  # /home/jmme425/thinker
    bc_log_dir = os.path.join(project_root, 'logs', 'thinker', 'bc_checkpoints')
    os.makedirs(bc_log_dir, exist_ok=True)
    print(f"BC checkpoints will be saved to: {bc_log_dir}")
    
    # Save config file to bc_checkpoints directory
    config_path = os.path.join(bc_log_dir, 'config_c.yaml')
    with open(config_path, 'w') as outfile:
        yaml.dump(vars(flags), outfile)
    print(f"Wrote BC config file to {config_path}")
    
    # Save meta.json for BC training
    meta_info = {
        'training_type': 'behavioral_cloning',
        'model_net_params': sum(p.numel() for p in model_net.parameters()),
        'actor_net_params': sum(p.numel() for p in actor_net.parameters()),
        'bc_epochs': flags.bc_epochs,
        'bc_batch_size': flags.bc_batch_size,
        'bc_lr': flags.bc_lr,
        'bc_subjects': flags.bc_subjects,
        'bc_game_id': flags.bc_game_id,
        'preload_path': flags.preload,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    meta_path = os.path.join(bc_log_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f"Wrote BC meta file to {meta_path}")
    
    # Save meta_model.json for model-specific info
    meta_model_info = {
        'model_type': 'ModelNet + ActorNet',
        'model_parameters': sum(p.numel() for p in model_net.parameters()),
        'actor_parameters': sum(p.numel() for p in actor_net.parameters()),
        'frame_stack_n': flags.frame_stack_n,
        'grayscale': flags.grayscale,
        'env_n': flags.env_n,
        'rec_t': flags.rec_t,
        'has_action_seq': getattr(flags, 'has_action_seq', False),
        'max_depth': flags.max_depth,
        'reset_mode': getattr(flags, 'reset_mode', 0),
        'tree_rep_size': 11 + 6 * 10 + flags.rec_t + (flags.max_depth * 6 + 6 if getattr(flags, 'has_action_seq', False) and getattr(flags, 'reset_mode', 0) == 0 else 0)
    }
    meta_model_path = os.path.join(bc_log_dir, 'meta_model.json')
    with open(meta_model_path, 'w') as f:
        json.dump(meta_model_info, f, indent=2)
    print(f"Wrote BC model meta file to {meta_model_path}")
    
    best_loss = float('inf')
    epoch_losses = []
    
    for step in range(flags.bc_epochs):
        step_start = time.time()
        
        # Get one batch (= one step in new definition)
        print(f"[DEBUG] Requesting batch {step+1} with size {flags.bc_batch_size}")
        batch_data = bc_loader.get_paired_batch(batch_size=flags.bc_batch_size)
        if batch_data is None:
            # Reset data loader if we've exhausted all data
            bc_loader.reset()
            batch_data = bc_loader.get_paired_batch(batch_size=flags.bc_batch_size)
            if batch_data is None:
                print("ERROR: Could not load any batch data")
                break
        
        # Zero gradients
        if hasattr(flags, 'bc_enhanced') and flags.bc_enhanced:
            # Enhanced BC: Zero gradients for both networks
            optimizers['model'].zero_grad()
            optimizers['actor'].zero_grad()
        else:
            # Standard BC: Zero gradients for Actor only
            actor_optimizer.zero_grad()
            
        try:
            # Forward pass
            outputs = run_bc_training_step(model_net, actor_net, batch_data, flags, device)
            
            # Compute losses
            if hasattr(flags, 'bc_enhanced') and flags.bc_enhanced:
                # Enhanced BC: Use enhanced loss computation
                print(f"[DEBUG] Using enhanced BC loss computation")
                losses = compute_enhanced_bc_losses(
                    outputs,        # Contains first_imagine_step data
                    batch_data,     # Contains rewards, next_obs  
                    model_net,      # For encoding next_obs
                    flags,
                    device
                )
                
                model_loss = losses['model_total_loss']
                actor_loss = losses['actor_loss'] 
                total_loss = model_loss + actor_loss
                
                print(f"[DEBUG] Enhanced BC losses - Model: {model_loss.item():.4f}, Actor: {actor_loss.item():.4f}, Total: {total_loss.item():.4f}")
                
                # Backward pass for both networks
                total_loss.backward()
            else:
                # Standard BC: Use standard loss computation (Actor only)
                print(f"[DEBUG] Using standard BC loss computation")
                losses = compute_bc_losses(
                    outputs['model_out'],
                    outputs['actor_out'],
                    outputs['target_actions'],
                    device
                )
                
                actor_loss = losses['actor_loss']
                
                # Backward pass (Actor only)
                actor_loss.backward()
            
            # Gradient clipping and optimizer steps
            if hasattr(flags, 'bc_enhanced') and flags.bc_enhanced:
                # Enhanced BC: Gradient clipping for both networks
                torch.nn.utils.clip_grad_norm_(model_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
                
                # Optimizer steps for both networks
                optimizers['model'].step()
                optimizers['actor'].step()
                
                # Record losses
                model_loss_val = model_loss.item()
                actor_loss_val = actor_loss.item()
                total_loss_val = total_loss.item()
                epoch_losses.append(total_loss_val)
                
                if total_loss_val < best_loss:
                    best_loss = total_loss_val
                
                # Log every step
                step_time = time.time() - step_start
                print(f"Step {step+1:4d}/{flags.bc_epochs}: "
                      f"Model: {model_loss_val:.4f}, Actor: {actor_loss_val:.4f}, "
                      f"Total: {total_loss_val:.4f}, Best: {best_loss:.4f} "
                      f"({step_time:.3f}s)")
                
                # Additional detailed logging for enhanced BC
                if 'vpn_policy_loss' in losses:
                    print(f"  VPN Policy: {losses['vpn_policy_loss'].item():.4f}")
                if 'srn_reward_loss' in losses:
                    print(f"  SRN Reward: {losses['srn_reward_loss'].item():.4f}")
                if 'srn_next_state_loss' in losses:
                    print(f"  SRN Next State: {losses['srn_next_state_loss'].item():.6f}")
                    
            else:
                # Standard BC: Gradient clipping for Actor only
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
                
                # Optimizer step (Actor only)
                actor_optimizer.step()
                
                # Record losses
                actor_loss_val = actor_loss.item()
                total_loss_val = actor_loss_val  # Only actor loss
                epoch_losses.append(total_loss_val)
                
                if total_loss_val < best_loss:
                    best_loss = total_loss_val
                
                # Log every step
                step_time = time.time() - step_start
                print(f"Step {step+1:4d}/{flags.bc_epochs}: "
                      f"Actor: {actor_loss_val:.4f}, "
                      f"Best: {best_loss:.4f} "
                      f"({step_time:.3f}s)")
            
            # Save every 100 steps
            if (step + 1) % 100 == 0:
                if hasattr(flags, 'bc_enhanced') and flags.bc_enhanced:
                    # Enhanced BC: Save both ModelNet and ActorNet checkpoints
                    print(f"Saving Enhanced BC checkpoints at step {step + 1}...")
                    
                    # Save ModelNet checkpoint
                    model_save_path = os.path.join(bc_log_dir, 'ckp_model.tar')
                    model_checkpoint_data = {
                        'step': step + 1,
                        'real_step': step + 1,
                        'model_net_state_dict': model_net.state_dict(),  # 올바른 키 이름
                        'model_optimizer_state_dict': optimizers['model'].state_dict(),
                        'flags': vars(flags)
                    }
                    torch.save(model_checkpoint_data, model_save_path)
                    print(f"Saved ModelNet checkpoint: {model_save_path}")
                    
                    # Save ActorNet checkpoint
                    actor_save_path = os.path.join(bc_log_dir, 'ckp_actor.tar')
                    actor_checkpoint_data = {
                        'step': step + 1,
                        'real_step': step + 1,
                        'actor_net_state_dict': actor_net.state_dict(),
                        'actor_net_optimizer_state_dict': optimizers['actor'].state_dict(),
                        'flags': vars(flags)
                    }
                else:
                    # Standard BC: Model is frozen, no need to save model checkpoint
                    
                    # Save actor checkpoint (following original learn_actor.py structure)
                    actor_save_path = os.path.join(bc_log_dir, 'ckp_actor.tar')
                    actor_checkpoint_data = {
                        'step': step + 1,
                        'real_step': step + 1,
                        'actor_net_state_dict': actor_net.state_dict(),
                        'actor_net_optimizer_state_dict': actor_optimizer.state_dict(),
                        'flags': vars(flags)
                    }
                
                # Save with temporary file then rename (atomic operation)
                torch.save(actor_checkpoint_data, actor_save_path + ".tmp")
                os.replace(actor_save_path + ".tmp", actor_save_path)
                
                # Also save step-specific checkpoint
                step_actor_save_path = os.path.join(bc_log_dir, f'ckp_actor.tar_step_{step+1}')
                torch.save(actor_checkpoint_data, step_actor_save_path + ".tmp")
                os.replace(step_actor_save_path + ".tmp", step_actor_save_path)
                
                print(f"  → Saved BC Actor checkpoint: ckp_actor.tar (step {step+1}) [Model frozen]")
                
        except Exception as e:
            print(f"  Error in step {step+1}: {e}")
            traceback.print_exc()
            continue

    # Unfreeze model parameters after BC training (for potential future use)
    print("\nUnfreezing Model parameters...")
    for param in model_net.parameters():
        param.requires_grad = True
    print(f"Unfrozen {sum(p.numel() for p in model_net.parameters()):,} Model parameters")
    
    # Save final checkpoint
    print(f"\nSaving final BC checkpoint...")
    
    if hasattr(flags, 'bc_enhanced') and flags.bc_enhanced:
        # Enhanced BC: Save both ModelNet and ActorNet final checkpoints
        print("Enhanced BC: Saving both ModelNet and ActorNet final checkpoints...")
        
        # Save final ModelNet checkpoint
        model_save_path = os.path.join(bc_log_dir, 'ckp_model.tar')
        model_checkpoint_data = {
            'step': flags.bc_epochs,
            'real_step': flags.bc_epochs,
            'model_net_state_dict': model_net.state_dict(),  # 올바른 키 이름
            'model_optimizer_state_dict': optimizers['model'].state_dict(),
            'flags': vars(flags)
        }
        torch.save(model_checkpoint_data, model_save_path + ".tmp")
        os.replace(model_save_path + ".tmp", model_save_path)
        
        # Save final step-specific ModelNet checkpoint
        final_model_save_path = os.path.join(bc_log_dir, f'ckp_model.tar_step_{flags.bc_epochs}_final')
        torch.save(model_checkpoint_data, final_model_save_path + ".tmp")
        os.replace(final_model_save_path + ".tmp", final_model_save_path)
        
        # Save final ActorNet checkpoint
        actor_save_path = os.path.join(bc_log_dir, 'ckp_actor.tar')
        actor_checkpoint_data = {
            'step': flags.bc_epochs,
            'real_step': flags.bc_epochs,
            'actor_net_state_dict': actor_net.state_dict(),
            'actor_net_optimizer_state_dict': optimizers['actor'].state_dict(),
            'flags': vars(flags)
        }
        torch.save(actor_checkpoint_data, actor_save_path + ".tmp")
        os.replace(actor_save_path + ".tmp", actor_save_path)
        
        # Save final step-specific ActorNet checkpoint
        final_actor_save_path = os.path.join(bc_log_dir, f'ckp_actor.tar_step_{flags.bc_epochs}_final')
        torch.save(actor_checkpoint_data, final_actor_save_path + ".tmp")
        os.replace(final_actor_save_path + ".tmp", final_actor_save_path)
        
        print(f"Enhanced BC final checkpoints saved: ckp_model.tar, ckp_actor.tar (both trained)")
        
    else:
        # Standard BC: Save ActorNet only (Model unchanged)
        actor_save_path = os.path.join(bc_log_dir, 'ckp_actor.tar')
        actor_checkpoint_data = {
            'step': flags.bc_epochs,
            'real_step': flags.bc_epochs,
            'actor_net_state_dict': actor_net.state_dict(),
            'actor_net_optimizer_state_dict': actor_optimizer.state_dict(),
            'flags': vars(flags)
        }
        torch.save(actor_checkpoint_data, actor_save_path + ".tmp")
        os.replace(actor_save_path + ".tmp", actor_save_path)
        
        # Save final step-specific actor checkpoint
        final_actor_save_path = os.path.join(bc_log_dir, f'ckp_actor.tar_step_{flags.bc_epochs}_final')
        torch.save(actor_checkpoint_data, final_actor_save_path + ".tmp")
        os.replace(final_actor_save_path + ".tmp", final_actor_save_path)
        
        print(f"Standard BC final checkpoints saved: ckp_actor.tar (Model unchanged - frozen during BC)")
    
    # Final summary
    print(f"\nBC Training completed!")
    print(f"Total steps: {len(epoch_losses)}")
    if epoch_losses:
        print(f"Final loss: {epoch_losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"All checkpoints saved in: {bc_log_dir}")
    
    return best_loss, epoch_losses


def run_bc_training_with_existing_networks(flags, ray_obj_env, logger=None):
    """
    Run BC training using existing networks from Thinker initialization
    
    Args:
        flags: Configuration flags
        ray_obj_env: Ray object environment containing initialized networks
        logger: Optional logger
        
    Returns:
        best_loss: Best loss achieved
    """
    print("=== Starting BC Training with Existing Thinker Networks ===")
    logger.info("Starting BC training with existing networks")
    
    # Extract networks from ray environment
    # We need to get the actual network instances from the ray workers
    # For now, let's create new networks using the same flags
    
    from thinker.model_net import ModelNet
    from thinker.actor_net import ActorNet
    from gymnasium import spaces
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create observation and action spaces for BC (same as original logic)
    # Real state observation space
    real_state_space = spaces.Box(
        low=0, high=255,
        shape=(flags.env_n, flags.frame_stack_n, 84, 84),
        dtype=np.uint8
    )
    
    # Tree representation space
    num_actions = 6
    obs_n = 11 + num_actions * 10 + flags.rec_t
    tree_reps_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(flags.env_n, obs_n),
        dtype=np.float32
    )
    
    # Combined observation space
    observation_space = spaces.Dict({
        "real_states": real_state_space,
        "tree_reps": tree_reps_space
    })
    
    # Action space
    primary_action_space = spaces.Discrete(6)
    reset_action_space = spaces.Discrete(2)
    action_space = spaces.Tuple((
        spaces.Tuple((primary_action_space,)),
        spaces.Tuple((reset_action_space,))
    ))
    
    # Initialize networks (same as in regular training)
    logger.info("Initializing networks for BC training...")
    
    # Model Network
    model_obs_space = spaces.Box(
        low=0, high=255,
        shape=(flags.frame_stack_n, 84, 84),
        dtype=np.uint8
    )
    
    model_net = ModelNet(
        obs_space=model_obs_space,
        action_space=primary_action_space,
        flags=flags
    ).to(device)
    
    # Actor Network  
    actor_net = ActorNet(
        obs_space=observation_space,
        action_space=action_space,
        flags=flags
    ).to(device)
    
    # Load pretrained weights if specified
    if flags.preload:
        logger.info(f"Loading pretrained weights from {flags.preload}...")
        # TODO: Implement preloading logic
        pass
    
    logger.info("Networks initialized - starting BC training...")
    
    # Run BC training
    best_loss, epoch_losses = run_bc_training(flags, model_net, actor_net, logger)
    
    return best_loss
