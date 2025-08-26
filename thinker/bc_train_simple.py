#!/usr/bin/env python3
"""
Simple BC Training Script
========================

A simple script to run behavioral cloning training without Ray dependencies.
This can be used as a standalone BC trainer or integrated into the main train.py.

Usage:
    python bc_train_simple.py --bc_clone --bc_epochs 100 --bc_lr 0.0001 --bc_subjects 1,2,3
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

# Add thinker to path
sys.path.append('/home/jmme425/thinker/thinker')

# Import BC training code
from thinker_bc_train import (
    parse_bc_args, create_pong_config_for_bc, create_observation_spaces_for_bc,
    compute_bc_loss, run_bc_training_step, save_bc_checkpoint
)
from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.model_net import ModelNet
from thinker.actor_net import ActorNet
from gymnasium import spaces


def run_simple_bc_training():
    """Run simple BC training without Ray"""
    args = parse_bc_args()
    
    # Override some settings for simple training
    if not hasattr(args, 'bc_clone') or not args.bc_clone:
        print("Enable BC training with --bc_clone flag")
        return
    
    print("=== Thinker Behavioral Cloning Training ===")
    print(f"Epochs: {args.bc_epochs}")
    print(f"Learning Rate: {args.bc_lr}")
    print(f"Batch Size: {args.bc_batch_size}")
    
    # Parse subjects
    subjects = [int(s.strip()) for s in args.bc_subjects.split(',')]
    print(f"Subjects: {subjects}")
    print(f"Game ID: {args.bc_game_id}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configuration
    print("\n1. Setting up configuration...")
    flags = create_pong_config_for_bc()
    flags.model_size_nn = args.model_size_nn
    flags.training = True
    
    # Data loader
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
    
    # Test data loading
    test_batch = bc_loader.get_paired_batch(batch_size=4)
    if test_batch is None:
        print("ERROR: Could not load test batch from BC data")
        return
    
    print(f"Test batch shapes:")
    for key, value in test_batch.items():
        print(f"  {key}: {value.shape}")
    
    # Observation and action spaces
    print("\n3. Setting up spaces...")
    observation_space, action_space = create_observation_spaces_for_bc(flags)
    primary_action_space = action_space.spaces[0].spaces[0]
    
    # Networks
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
    
    # Set to training mode
    model_net.train()
    actor_net.train()
    
    print(f"Model parameters: {sum(p.numel() for p in model_net.parameters()):,}")
    print(f"Actor parameters: {sum(p.numel() for p in actor_net.parameters()):,}")
    
    # Optimizers
    print("\n5. Setting up optimizers...")
    model_optimizer = optim.Adam(model_net.parameters(), lr=args.bc_lr)
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=args.bc_lr)
    
    optimizers = {
        'model': model_optimizer,
        'actor': actor_optimizer
    }
    
    # Training loop
    print(f"\n6. Starting training for {args.bc_epochs} epochs...")
    
    best_loss = float('inf')
    epoch_losses = []
    
    for epoch in range(args.bc_epochs):
        epoch_start = time.time()
        
        # Reset data loader
        bc_loader.reset()
        
        epoch_vpn_losses = []
        epoch_actor_losses = []
        epoch_total_losses = []
        
        batch_count = 0
        
        # Training batches for this epoch
        while True:
            batch_data = bc_loader.get_paired_batch(batch_size=args.bc_batch_size)
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
                
                # Gradient clipping (optional)
                torch.nn.utils.clip_grad_norm_(model_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
                
                # Optimizer steps
                model_optimizer.step()
                actor_optimizer.step()
                
                # Record losses
                epoch_vpn_losses.append(vpn_loss.item())
                epoch_actor_losses.append(actor_loss.item())
                epoch_total_losses.append(total_loss.item())
                
                # Print progress
                if batch_count % 20 == 0:
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
            
            print(f"Epoch {epoch+1:3d}/{args.bc_epochs} ({epoch_time:.1f}s, {batch_count} batches): "
                  f"VPN: {avg_vpn_loss:.4f}, Actor: {avg_actor_loss:.4f}, Total: {avg_total_loss:.4f}")
            
            # Save best model
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                best_save_path = os.path.join(args.savedir, 'bc_checkpoints', 'best_bc_model.pt')
                save_bc_checkpoint(model_net, actor_net, optimizers, epoch + 1, epoch_losses, best_save_path)
                print(f"  → New best model saved (loss: {best_loss:.4f})")
        else:
            print(f"Epoch {epoch+1:3d}/{args.bc_epochs}: No valid batches")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.bc_save_interval == 0:
            save_path = os.path.join(args.savedir, 'bc_checkpoints', f'epoch_{epoch+1:03d}.pt')
            save_bc_checkpoint(model_net, actor_net, optimizers, epoch + 1, epoch_losses, save_path)
    
    # Final save
    final_save_path = os.path.join(args.savedir, 'bc_checkpoints', 'final_bc_model.pt')
    save_bc_checkpoint(model_net, actor_net, optimizers, args.bc_epochs, epoch_losses, final_save_path)
    
    print(f"\n=== Training Complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model: {final_save_path}")
    
    if epoch_losses:
        print(f"Loss progression:")
        for i in [0, len(epoch_losses)//4, len(epoch_losses)//2, 3*len(epoch_losses)//4, -1]:
            loss_info = epoch_losses[i]
            print(f"  Epoch {loss_info['epoch']:3d}: {loss_info['total_loss']:.4f}")


if __name__ == "__main__":
    run_simple_bc_training()
