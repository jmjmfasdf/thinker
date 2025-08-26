#!/usr/bin/env python3
"""
Thinker Behavioral Cloning Training
===================================

Based on thinker_pipeline_demo.py, this script implements behavioral cloning training
for both VPN and ActorNet using real human gameplay data.

Usage:
    python train.py --name Pong-v5 --reward_clip 1 --model_size_nn 2 --discounting 0.99 \
                    --envpool True --preload ../logs/thinker/pong_0.0005 --savedir ./logs/thinker \
                    --bc_clone True --bc_epochs 100 --bc_lr 0.0001 --bc_subjects 1,2,3 --bc_game_id 1
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple

# Thinker 모듈 import
import thinker.util as util
from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.model_net import ModelNet
from thinker.actor_net import ActorNet
from gymnasium import spaces


def parse_bc_args():
    """Parse BC-specific command line arguments"""
    parser = argparse.ArgumentParser(description="Thinker Behavioral Cloning Training")
    
    # Standard Thinker args (compatibility with existing train.py)
    parser.add_argument('--name', type=str, default='Pong-v5', help='Environment name')
    parser.add_argument('--reward_clip', type=float, default=1.0, help='Reward clipping')
    parser.add_argument('--model_size_nn', type=int, default=2, help='Model size')
    parser.add_argument('--discounting', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--envpool', type=bool, default=True, help='Use envpool')
    parser.add_argument('--preload', type=str, default=None, help='Preload model path')
    parser.add_argument('--savedir', type=str, default='./logs/thinker', help='Save directory')
    
    # BC-specific args
    parser.add_argument('--bc_clone', action='store_true', help='Enable BC training')
    parser.add_argument('--bc_epochs', type=int, default=100, help='Number of BC training epochs')
    parser.add_argument('--bc_lr', type=float, default=0.0001, help='BC learning rate')
    parser.add_argument('--bc_batch_size', type=int, default=32, help='BC batch size')
    parser.add_argument('--bc_subjects', type=str, default='1,2,3', help='Comma-separated subject IDs')
    parser.add_argument('--bc_game_id', type=int, default=1, help='Game ID (0: Enduro, 1: Pong, 2: Space Invaders)')
    parser.add_argument('--bc_data_path', type=str, default='../behavioral_data_4kframe_legacy', help='Path to BC data')
    parser.add_argument('--bc_save_interval', type=int, default=10, help='Save model every N epochs')
    
    return parser.parse_args()


def create_pong_config_for_bc():
    """Create Pong configuration for BC training"""
    # Load default thinker config to get all flags
    flags = util.create_flags(filename='default_thinker.yaml')
    
    # Also load actor config to get actor-specific flags
    actor_flags = util.create_flags(filename='default_actor.yaml')
    
    # Merge actor flags into thinker flags
    for attr in dir(actor_flags):
        if not attr.startswith('_') and not hasattr(flags, attr):
            setattr(flags, attr, getattr(actor_flags, attr))
    
    # Override with Pong-specific settings
    flags.name = "PongNoFrameskip-v4"
    flags.frame_stack_n = 4
    flags.grayscale = True
    flags.rec_t = 40
    flags.max_depth = 40
    flags.env_n = 32
    
    # Model network settings
    flags.dual_net = True
    flags.model_size_nn = 1
    flags.model_downscale_c = 2
    flags.model_decoder_depth = 0
    flags.return_h = True
    flags.return_x = False
    
    # Actor network settings
    flags.dim_actions = 1
    flags.tuple_action = False
    flags.enc_1d_shallow = True
    flags.disable_thinker = False
    
    # BC training specific settings
    flags.training = True
    
    # Ensure critical flags are set
    if not hasattr(flags, 'critic_zero_init'):
        flags.critic_zero_init = False
    if not hasattr(flags, 'cur_cost'):
        flags.cur_cost = 0.0
    if not hasattr(flags, 'critic_enc_type'):
        flags.critic_enc_type = 1
    
    return flags


def create_observation_spaces_for_bc(flags):
    """Create observation and action spaces for BC training"""
    # Real state observation space
    real_state_space = spaces.Box(
        low=0, high=255,
        shape=(flags.env_n, flags.frame_stack_n, 84, 84),
        dtype=np.uint8
    )
    
    # Tree representation space
    num_actions = 6
    obs_n = 11 + num_actions * 10 + flags.rec_t
    if hasattr(flags, 'has_action_seq') and flags.has_action_seq:
        obs_n += flags.max_depth * num_actions
    
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
    
    return observation_space, action_space


def compute_bc_loss(model_out, actor_out, target_actions, device):
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
        
        # Ensure target actions are Long tensor
        target_actions_long = target_actions.long()
        
        # Cross entropy loss
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
    """
    Run one BC training step
    
    Args:
        model_net: ModelNet instance
        actor_net: ActorNet instance
        batch_data: Batch from BC loader
        flags: Configuration flags
        device: torch device
        
    Returns:
        Dict with model_out, actor_out, and target_actions
    """
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
    num_actions = 6
    obs_n = 11 + num_actions * 10 + flags.rec_t
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
    
    # Run Actor Network
    actor_out, _ = actor_net.forward(
        env_out=env_out,
        core_state=None,
        rnn_done=torch.zeros(batch_size, dtype=torch.bool, device=device)
    )
    
    return {
        'model_out': model_out,
        'actor_out': actor_out,
        'target_actions': target_actions
    }


def save_bc_checkpoint(model_net, actor_net, optimizers, epoch, losses, save_path):
    """Save BC training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_net_state_dict': model_net.state_dict(),
        'actor_net_state_dict': actor_net.state_dict(),
        'model_optimizer_state_dict': optimizers['model'].state_dict(),
        'actor_optimizer_state_dict': optimizers['actor'].state_dict(),
        'losses': losses
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")


def main():
    args = parse_bc_args()
    
    if not args.bc_clone:
        print("BC training not enabled. Use --bc_clone flag.")
        return
    
    print("Starting Behavioral Cloning Training for Thinker...")
    print(f"BC Epochs: {args.bc_epochs}")
    print(f"BC Learning Rate: {args.bc_lr}")
    print(f"BC Batch Size: {args.bc_batch_size}")
    
    # Parse subjects
    subjects = [int(s.strip()) for s in args.bc_subjects.split(',')]
    print(f"Using subjects: {subjects}")
    print(f"Game ID: {args.bc_game_id}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration
    print("\n1. Creating configuration...")
    flags = create_pong_config_for_bc()
    
    # Update from args
    flags.model_size_nn = args.model_size_nn
    
    # Create data loader
    print("\n2. Loading behavioral data...")
    bc_loader = FrameStackedBehavioralDataLoader(
        base_path=args.bc_data_path,
        subjects=subjects,
        game_id=args.bc_game_id,
        frame_stack_n=flags.frame_stack_n,
        target_size=(84, 84),
        grayscale=True,
        normalize=True
    )
    
    # Create observation and action spaces
    print("\n3. Setting up observation and action spaces...")
    observation_space, action_space = create_observation_spaces_for_bc(flags)
    primary_action_space = action_space.spaces[0].spaces[0]
    
    # Initialize networks
    print("\n4. Initializing networks...")
    
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
    
    # Set networks to training mode
    model_net.train()
    actor_net.train()
    
    # Load pretrained weights if specified
    if args.preload:
        print(f"\n5. Loading pretrained weights from {args.preload}...")
        # TODO: Implement preloading logic
        pass
    
    # Setup optimizers
    print("\n6. Setting up optimizers...")
    optimizers = {
        'model': optim.Adam(model_net.parameters(), lr=args.bc_lr),
        'actor': optim.Adam(actor_net.parameters(), lr=args.bc_lr)
    }
    
    # Training loop
    print(f"\n7. Starting BC training for {args.bc_epochs} epochs...")
    
    epoch_losses = []
    
    for epoch in range(args.bc_epochs):
        epoch_start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        epoch_end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        if epoch_start_time:
            epoch_start_time.record()
        
        # Reset data loader for each epoch
        bc_loader.reset()
        
        epoch_vpn_loss = 0.0
        epoch_actor_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        while True:
            # Get batch
            batch_data = bc_loader.get_paired_batch(batch_size=args.bc_batch_size)
            if batch_data is None:
                break
                
            num_batches += 1
            
            # Zero gradients
            optimizers['model'].zero_grad()
            optimizers['actor'].zero_grad()
            
            # Forward pass
            outputs = run_bc_training_step(model_net, actor_net, batch_data, flags, device)
            
            # Compute losses
            losses = compute_bc_loss(
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
            
            # Optimizer steps
            optimizers['model'].step()
            optimizers['actor'].step()
            
            # Accumulate losses
            epoch_vpn_loss += vpn_loss.item()
            epoch_actor_loss += actor_loss.item()
            epoch_total_loss += total_loss.item()
            
            # Print batch progress
            if num_batches % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.bc_epochs}, Batch {num_batches}: "
                      f"VPN Loss: {vpn_loss.item():.4f}, "
                      f"Actor Loss: {actor_loss.item():.4f}, "
                      f"Total Loss: {total_loss.item():.4f}")
        
        if epoch_end_time:
            epoch_end_time.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0  # Convert to seconds
        else:
            epoch_time = 0.0
        
        # Calculate average losses for epoch
        if num_batches > 0:
            avg_vpn_loss = epoch_vpn_loss / num_batches
            avg_actor_loss = epoch_actor_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
        else:
            avg_vpn_loss = avg_actor_loss = avg_total_loss = 0.0
        
        epoch_losses.append({
            'epoch': epoch + 1,
            'vpn_loss': avg_vpn_loss,
            'actor_loss': avg_actor_loss,
            'total_loss': avg_total_loss,
            'num_batches': num_batches,
            'time': epoch_time
        })
        
        print(f"Epoch {epoch+1}/{args.bc_epochs} completed in {epoch_time:.2f}s: "
              f"Avg VPN Loss: {avg_vpn_loss:.4f}, "
              f"Avg Actor Loss: {avg_actor_loss:.4f}, "
              f"Avg Total Loss: {avg_total_loss:.4f}, "
              f"Batches: {num_batches}")
        
        # Save checkpoint
        if (epoch + 1) % args.bc_save_interval == 0:
            save_path = os.path.join(args.savedir, 'bc_checkpoints', f'bc_checkpoint_epoch_{epoch+1}.pt')
            save_bc_checkpoint(model_net, actor_net, optimizers, epoch + 1, epoch_losses, save_path)
    
    # Save final model
    final_save_path = os.path.join(args.savedir, 'bc_checkpoints', 'bc_final_model.pt')
    save_bc_checkpoint(model_net, actor_net, optimizers, args.bc_epochs, epoch_losses, final_save_path)
    
    print(f"\nBC Training completed! Final model saved to {final_save_path}")
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Total epochs: {args.bc_epochs}")
    if epoch_losses:
        print(f"Final VPN Loss: {epoch_losses[-1]['vpn_loss']:.4f}")
        print(f"Final Actor Loss: {epoch_losses[-1]['actor_loss']:.4f}")
        print(f"Final Total Loss: {epoch_losses[-1]['total_loss']:.4f}")


if __name__ == "__main__":
    main()
