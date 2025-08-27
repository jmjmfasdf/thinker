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
    if hasattr(actor_out, 'action_prob') and actor_out.action_prob is not None:
        actor_logits = actor_out.action_prob  # Shape: [T, B, num_actions]
        
        # Take the first timestep (T=1)
        if len(actor_logits.shape) == 3:
            actor_logits = actor_logits[0, :, :]  # [B, num_actions]
        elif len(actor_logits.shape) == 2:
            pass  # Already [B, num_actions]
        
        # Cross entropy loss
        target_actions_long = target_actions.long()
        actor_loss = F.cross_entropy(actor_logits, target_actions_long)
        losses['actor_loss'] = actor_loss
    else:
        losses['actor_loss'] = torch.tensor(0.0, device=device)
    
    return losses


def run_bc_training_step(model_net, actor_net, batch_data, flags, device):
    """Run one BC training step"""
    # Extract data
    obs = torch.tensor(batch_data['obs'], dtype=torch.uint8, device=device)  # [B, 4, 84, 84]
    target_actions = torch.tensor(batch_data['actions'][:, 0], dtype=torch.long, device=device)  # [B]
    
    batch_size = obs.shape[0]
    
    # Run Model Network
    actions_for_model = target_actions.unsqueeze(0).unsqueeze(-1)  # [1, B, 1]
    
    model_out = model_net.forward(
        env_state=obs,
        actions=actions_for_model,
        done=torch.zeros(batch_size, dtype=torch.bool, device=device),
        state={}
    )
    
    # Extract model outputs
    xs = model_out.xs
    if xs is not None and len(xs.shape) == 5:
        xs = xs.squeeze(0)  # Remove time dimension
    
    hs = model_out.hs
    if hs is not None and len(hs.shape) == 5:
        hs = hs.squeeze(0)  # Remove time dimension
    
    # Create mock tree_reps for actor (in real training, this comes from MCTS)
    # Use the same calculation as in train.py for consistency with pretrained model
    num_actions = 6
    obs_n = 11 + num_actions * 10 + flags.rec_t
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        obs_n += flags.max_depth * num_actions
        if hasattr(flags, 'reset_mode') and flags.reset_mode == 0:
            obs_n += num_actions
    
    tree_reps = torch.randn(batch_size, obs_n, device=device)
    
    # Create Actor input
    from types import SimpleNamespace
    
    env_out = SimpleNamespace()
    env_out.real_states = obs.unsqueeze(0).float()  # [T, B, C, H, W]
    env_out.tree_reps = tree_reps.unsqueeze(0)  # [T, B, obs_n]
    env_out.done = torch.zeros(1, batch_size, dtype=torch.bool, device=device)
    env_out.real_done = torch.zeros(1, batch_size, dtype=torch.bool, device=device)
    env_out.step_status = torch.zeros(1, batch_size, dtype=torch.long, device=device)
    env_out.last_pri = torch.zeros(1, batch_size, dtype=torch.long, device=device)
    env_out.last_reset = torch.zeros(1, batch_size, dtype=torch.long, device=device)
    env_out.reward = torch.zeros(1, batch_size, 2, device=device)
    
    # Add hs if available
    if hs is not None:
        env_out.hs = hs.unsqueeze(0)  # [T, B, C, H, W]
    
    # Add xs if available
    if xs is not None:
        env_out.xs = xs.unsqueeze(0)  # [T, B, C, H, W]
    
    # Run Actor Network with proper core_state initialization
    core_state = actor_net.initial_state(batch_size=batch_size, device=device)
    
    actor_out, _ = actor_net.forward(
        env_out=env_out,
        core_state=core_state
    )
    
    return {
        'model_out': model_out,
        'actor_out': actor_out,
        'target_actions': target_actions
    }


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
    test_batch = bc_loader.get_paired_batch(batch_size=4)
    if test_batch is None:
        print("ERROR: Could not load test batch from BC data")
        return
    
    print(f"Test batch loaded successfully:")
    for key, value in test_batch.items():
        print(f"  {key}: {value.shape}")
    
    # Set networks to training mode
    model_net.eval()  # Keep model in eval mode (no training)
    actor_net.train()  # Only train actor
    
    # Freeze all model parameters (no model training in BC)
    print("Freezing all Model parameters (SRN + VPN)...")
    for param in model_net.parameters():
        param.requires_grad = False
    
    total_model_params = sum(p.numel() for p in model_net.parameters())
    print(f"Frozen {total_model_params:,} Model parameters")
    
    # Setup optimizer for Actor only
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=flags.bc_lr)
    
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
        batch_data = bc_loader.get_paired_batch(batch_size=flags.bc_batch_size)
        if batch_data is None:
            # Reset data loader if we've exhausted all data
            bc_loader.reset()
            batch_data = bc_loader.get_paired_batch(batch_size=flags.bc_batch_size)
            if batch_data is None:
                print("ERROR: Could not load any batch data")
                break
        
        # Zero gradients (Actor only)
        actor_optimizer.zero_grad()
            
        try:
            # Forward pass
            outputs = run_bc_training_step(model_net, actor_net, batch_data, flags, device)
            
            # Compute losses (Actor only)
            losses = compute_bc_losses(
                outputs['model_out'],
                outputs['actor_out'],
                outputs['target_actions'],
                device
            )
            
            actor_loss = losses['actor_loss']
            
            # Backward pass (Actor only)
            actor_loss.backward()
            
            # Gradient clipping (Actor only)
            torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
            
            # Optimizer step (Actor only)
            actor_optimizer.step()
            
            # Record losses
            actor_loss_val = actor_loss.item()
            total_loss_val = actor_loss_val  # Only actor loss now
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
                # Model is frozen, no need to save model checkpoint
                
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
    
    # Save final checkpoint (Actor only, Model unchanged)
    print(f"\nSaving final BC checkpoint...")
    
    # Save final actor checkpoint
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
    
    # Final summary
    print(f"\nBC Training completed!")
    print(f"Total steps: {len(epoch_losses)}")
    if epoch_losses:
        print(f"Final loss: {epoch_losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final checkpoints saved: ckp_actor.tar (Model unchanged - frozen during BC)")
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
