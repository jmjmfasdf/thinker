"""
Behavioral Cloning Trainer Module for Thinker
============================================

This module provides BC training functionality to be integrated with the main train.py
"""

import os
import time
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
    Compute BC losses for VPN and ActorNet
    
    Args:
        model_out: ModelNet output
        actor_out: ActorNet output  
        target_actions: Target actions from BC data [B]
        device: torch device
        
    Returns:
        Dict with 'vpn_loss' and 'actor_loss'
    """
    losses = {}
    
    # VPN loss: Cross entropy between VPN policy and target actions
    if hasattr(model_out, 'policy') and model_out.policy is not None:
        vpn_policy = model_out.policy  # Shape: [k+1, B, 1, num_actions] or similar
        
        # Take the last timestep policy
        if len(vpn_policy.shape) == 4:
            vpn_logits = vpn_policy[-1, :, 0, :]  # [B, num_actions]
        elif len(vpn_policy.shape) == 3:
            vpn_logits = vpn_policy[-1, :, :]  # [B, num_actions]
        else:
            vpn_logits = vpn_policy  # [B, num_actions]
        
        # Cross entropy loss
        target_actions_long = target_actions.long()
        vpn_loss = F.cross_entropy(vpn_logits, target_actions_long)
        losses['vpn_loss'] = vpn_loss
    else:
        losses['vpn_loss'] = torch.tensor(0.0, device=device)
    
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
    model_net.train()
    actor_net.train()
    
    # Setup optimizers
    model_optimizer = optim.Adam(model_net.parameters(), lr=flags.bc_lr)
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=flags.bc_lr)
    
    optimizers = {
        'model': model_optimizer,
        'actor': actor_optimizer
    }
    
    print(f"Model parameters: {sum(p.numel() for p in model_net.parameters()):,}")
    print(f"Actor parameters: {sum(p.numel() for p in actor_net.parameters()):,}")
    
    # Training loop
    print(f"\nStarting training for {flags.bc_epochs} epochs...")
    
    best_loss = float('inf')
    epoch_losses = []
    
    for epoch in range(flags.bc_epochs):
        epoch_start = time.time()
        
        # Reset data loader
        bc_loader.reset()
        
        epoch_vpn_losses = []
        epoch_actor_losses = []
        epoch_total_losses = []
        
        batch_count = 0
        
        # Training batches for this epoch
        while True:
            batch_data = bc_loader.get_paired_batch(batch_size=flags.bc_batch_size)
            if batch_data is None:
                break
            
            batch_count += 1
            
            # Zero gradients
            model_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = run_bc_training_step(model_net, actor_net, batch_data, flags, device)
                
                # Compute losses
                losses = compute_bc_losses(
                    outputs['model_out'],
                    outputs['actor_out'],
                    outputs['target_actions'],
                    device
                )
                
                vpn_loss = losses['vpn_loss']
                actor_loss = losses['actor_loss']
                total_loss = vpn_loss + actor_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
                
                # Optimizer steps
                model_optimizer.step()
                actor_optimizer.step()
                
                # Record losses
                epoch_vpn_losses.append(vpn_loss.item())
                epoch_actor_losses.append(actor_loss.item())
                epoch_total_losses.append(total_loss.item())
                
                # Print progress every batch (as requested)
                print(f"  Epoch {epoch+1:3d}, Batch {batch_count:3d}: "
                      f"VPN: {vpn_loss.item():.4f}, "
                      f"Actor: {actor_loss.item():.4f}, "
                      f"Total: {total_loss.item():.4f}")
                
            except Exception as e:
                print(f"  Error in batch {batch_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        
        if epoch_vpn_losses:
            avg_vpn_loss = np.mean(epoch_vpn_losses)
            avg_actor_loss = np.mean(epoch_actor_losses)
            avg_total_loss = np.mean(epoch_total_losses)
            
            epoch_losses.append({
                'epoch': epoch + 1,
                'vpn_loss': avg_vpn_loss,
                'actor_loss': avg_actor_loss,
                'total_loss': avg_total_loss,
                'batches': batch_count,
                'time': epoch_time
            })
            
            print(f"Epoch {epoch+1:3d}/{flags.bc_epochs} ({epoch_time:.1f}s, {batch_count} batches): "
                  f"VPN: {avg_vpn_loss:.4f}, Actor: {avg_actor_loss:.4f}, Total: {avg_total_loss:.4f}")
            
            # Log to logger if available
            if logger:
                logger.info(f"BC Epoch {epoch+1}: VPN={avg_vpn_loss:.4f}, Actor={avg_actor_loss:.4f}, Total={avg_total_loss:.4f}")
            
            # Save best model
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                best_save_path = os.path.join(flags.savedir, 'bc_checkpoints', 'best_bc_model.pt')
                save_bc_checkpoint(model_net, actor_net, optimizers, epoch + 1, epoch_losses, best_save_path, flags)
                print(f"  → New best model saved (loss: {best_loss:.4f})")
        else:
            print(f"Epoch {epoch+1:3d}/{flags.bc_epochs}: No valid batches")
        
        # Save periodic checkpoint
        if (epoch + 1) % flags.bc_save_interval == 0:
            save_path = os.path.join(flags.savedir, 'bc_checkpoints', f'epoch_{epoch+1:03d}.pt')
            save_bc_checkpoint(model_net, actor_net, optimizers, epoch + 1, epoch_losses, save_path, flags)
    
    # Final save
    final_save_path = os.path.join(flags.savedir, 'bc_checkpoints', 'final_bc_model.pt')
    save_bc_checkpoint(model_net, actor_net, optimizers, flags.bc_epochs, epoch_losses, final_save_path, flags)
    
    print(f"\n=== BC Training Complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model: {final_save_path}")
    
    if logger:
        logger.info(f"BC Training completed. Best loss: {best_loss:.4f}")
    
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
