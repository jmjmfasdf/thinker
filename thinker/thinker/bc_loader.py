import os
import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import cv2


class FrameStackedBehavioralDataLoader:
    def __init__(self, 
                 base_path: str = "behavioral_data_4kframe_legacy",
                 subjects: List[int] = [1, 2, 3, 4, 5, 6],
                 game_id: int = 1,
                 frame_stack_n: int = 4,
                 target_size: Tuple[int, int] = (84, 84),
                 grayscale: bool = True,
                 normalize: bool = True):
        """
        Behavioral cloning data loader with frame stacking
        
        Args:
            base_path: Path to behavioral data directory
            subjects: List of subject IDs to use
            game_id: Game ID (0: Enduro, 1: Pong, 2: Space Invaders)
            frame_stack_n: Number of frames to stack
            target_size: Target image size (H, W)
            grayscale: Whether to convert to grayscale
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.base_path = Path(base_path)
        self.subjects = subjects
        self.game_id = game_id
        self.frame_stack_n = frame_stack_n
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        
        # Load all data files
        self.data_files = self._load_data_files()
        self.current_file_idx = 0
        self.current_pos = 0
        self.current_data = None  # Initialize current_data attribute
        self.num_actions = 6
        self.action_distribution = self._compute_action_distribution()
        
        print(f"Loaded {len(self.data_files)} data files")
        print(f"Human action prior: {self.action_distribution}")
        
    def _compute_action_distribution(self) -> np.ndarray:
        """Aggregate action frequency across dataset"""
        if len(self.data_files) == 0:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float64)

        action_counts = np.zeros(self.num_actions, dtype=np.float64)
        total = 0.0

        for file_path in self.data_files:
            try:
                data = np.load(file_path)
                actions = data['action']
            except Exception as e:
                print(f"[WARNING] Failed to load actions from {file_path}: {e}")
                continue

            if actions.ndim >= 2 and actions.shape[-1] == self.num_actions:
                flat = actions.reshape(-1, self.num_actions)
                counts = flat.sum(axis=0)
            else:
                flat = actions.reshape(-1)
                counts = np.bincount(flat.astype(np.int64), minlength=self.num_actions)

            action_counts += counts
            total += counts.sum()

        if total == 0:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float64)

        return action_counts / total

    def _load_data_files(self) -> List[str]:
        """Load all .npz files from specified subjects and game"""
        data_files = []
        
        for subject in self.subjects:
            subject_path = self.base_path / f"sub_{subject}" / f"game_{self.game_id}"
            print(f"Checking path: {subject_path}")
            if not subject_path.exists():
                print(f"Warning: {subject_path} does not exist")
                continue
                
            # Find all day and block directories
            for day_dir in subject_path.iterdir():
                if not day_dir.is_dir():
                    continue
                print(f"  Found day: {day_dir}")
                for block_dir in day_dir.iterdir():
                    if not block_dir.is_dir():
                        continue
                    print(f"    Found block: {block_dir}")
                    # Find all .npz files
                    for npz_file in block_dir.glob("*.npz"):
                        print(f"      Found file: {npz_file}")
                        data_files.append(str(npz_file))
        
        return data_files
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image"""
        # Resize
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if self.grayscale and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[..., np.newaxis]  # Add channel dimension
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _create_frame_stack(self, images: np.ndarray, start_idx: int) -> np.ndarray:
        """Create a frame stack starting from start_idx"""
        T = images.shape[0]
        
        # Create frame stack
        stacked_frames = []
        for i in range(self.frame_stack_n):
            frame_idx = start_idx - (self.frame_stack_n - 1 - i)
            
            if frame_idx < 0:
                # Pad with zeros for frames before the sequence
                if self.grayscale:
                    zero_frame = np.zeros((1, *self.target_size), dtype=np.float32)
                else:
                    zero_frame = np.zeros((3, *self.target_size), dtype=np.float32)
                stacked_frames.append(zero_frame)
            else:
                # Preprocess the frame
                frame = self._preprocess_image(images[frame_idx])
                stacked_frames.append(frame)
        
        # Concatenate frames along channel dimension
        stacked = np.concatenate(stacked_frames, axis=0)
        return stacked
    
    def get_sequence_batch(self, batch_size: int = 1, sequence_length: int = 40) -> Optional[Dict[str, np.ndarray]]:
        """
        Get a batch of sequences for BC training
        
        Args:
            batch_size: Number of sequences to return (default: 1)
            sequence_length: Length of each sequence (default: 40 for rec_t)
            
        Returns:
            Dictionary with keys: 'images', 'actions', 'rewards', 'is_first', 'is_terminal'
            Images shape: (B, T, C*stack_n, H, W) = (1, 40, 4, 84, 84)
            Actions shape: (B, T, 6)
            Rewards shape: (B, T)
            is_first shape: (B, T)
            is_terminal shape: (B, T)
        """
        if len(self.data_files) == 0:
            return None
        
        # Load current file if needed
        if not hasattr(self, 'current_data') or self.current_data is None:
            self._load_current_file()
        
        if self.current_data is None:
            return None
        
        images = self.current_data['image']
        actions = self.current_data['action']
        rewards = self.current_data['reward']
        is_first = self.current_data['is_first']
        is_terminal = self.current_data['is_terminal']
        
        T = images.shape[0]
        
        # Find valid sequence start positions
        valid_starts = []
        for i in range(self.frame_stack_n - 1, T - sequence_length + 1):
            # Check if sequence doesn't cross episode boundaries
            if not np.any(is_first[i:i+sequence_length]) and not np.any(is_terminal[i:i+sequence_length]):
                valid_starts.append(i)
        
        if len(valid_starts) < batch_size:
            return None
        
        # Sample batch_size sequences
        selected_starts = np.random.choice(valid_starts, size=batch_size, replace=False)
        
        # Create batch
        batch_images = []
        batch_actions = []
        batch_rewards = []
        batch_is_first = []
        batch_is_terminal = []
        
        for start_idx in selected_starts:
            sequence_images = []
            sequence_actions = []
            sequence_rewards = []
            sequence_is_first = []
            sequence_is_terminal = []
            
            for t in range(sequence_length):
                # Create frame stack for this timestep
                stacked_image = self._create_frame_stack(images, start_idx + t)
                sequence_images.append(stacked_image)
                
                # Get corresponding data
                sequence_actions.append(actions[start_idx + t])
                sequence_rewards.append(rewards[start_idx + t])
                sequence_is_first.append(is_first[start_idx + t])
                sequence_is_terminal.append(is_terminal[start_idx + t])
            
            batch_images.append(np.stack(sequence_images, axis=0))
            batch_actions.append(np.stack(sequence_actions, axis=0))
            batch_rewards.append(np.stack(sequence_rewards, axis=0))
            batch_is_first.append(np.stack(sequence_is_first, axis=0))
            batch_is_terminal.append(np.stack(sequence_is_terminal, axis=0))
        
        # Stack sequences
        batch_images = np.stack(batch_images, axis=0)
        batch_actions = np.stack(batch_actions, axis=0)
        batch_rewards = np.stack(batch_rewards, axis=0)
        batch_is_first = np.stack(batch_is_first, axis=0)
        batch_is_terminal = np.stack(batch_is_terminal, axis=0)
        
        return {
            'images': batch_images,      # (B, T, C*stack_n, H, W) = (1, 40, 4, 84, 84)
            'actions': batch_actions,    # (B, T, 6)
            'rewards': batch_rewards,    # (B, T)
            'is_first': batch_is_first,  # (B, T)
            'is_terminal': batch_is_terminal  # (B, T)
        }
    
    def _load_current_file(self):
        """Load the current data file"""
        if self.current_file_idx >= len(self.data_files):
            # Reset to beginning
            self.current_file_idx = 0
            self.current_pos = 0
        
        try:
            file_path = self.data_files[self.current_file_idx]
            self.current_data = np.load(file_path)
            print(f"Loaded file: {file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            self.current_data = None
    
    def next_file(self):
        """Move to next data file"""
        self.current_file_idx += 1
        self.current_pos = 0
        self.current_data = None
        self._load_current_file()
    
    def get_paired_batch(self, batch_size: int = 32) -> Optional[Dict[str, np.ndarray]]:
        """
        Get a batch of paired consecutive observations for BC training
        Uses improved sampling strategy for better file diversity
        
        Returns:
            Dict containing:
            - 'obs': Current observations [B, C*stack_n, H, W]
            - 'next_obs': Next observations [B, C*stack_n, H, W] 
            - 'actions': Actions taken [B, action_dim]
            - 'rewards': Rewards [B]
        """
        return self._get_diverse_batch(batch_size)
    
    def _get_diverse_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Improved batch sampling strategy for maximum diversity:
        - If batch_size <= num_files: sample randomly from different files
        - If batch_size > num_files: distribute evenly across all files
        """
        if len(self.data_files) == 0:
            return None
            
        batch_obs = []
        batch_next_obs = []
        batch_actions = []
        batch_rewards = []
        
        num_files = len(self.data_files)
        
        if batch_size <= num_files:
            # Case 1: More files than batch size - randomly select files
            selected_files = random.sample(range(num_files), batch_size)
            samples_per_file = [1] * batch_size
            file_indices = selected_files
        else:
            # Case 2: More batch size than files - distribute evenly
            base_samples = batch_size // num_files
            extra_samples = batch_size % num_files
            
            samples_per_file = [base_samples] * num_files
            # Distribute extra samples randomly
            extra_file_indices = random.sample(range(num_files), extra_samples)
            for idx in extra_file_indices:
                samples_per_file[idx] += 1
                
            file_indices = list(range(num_files))
        
        
        # Sample from each selected file
        batch_dones = []
        batch_next_actions = []
        for file_idx, num_samples in zip(file_indices, samples_per_file):
            file_samples = self._sample_from_file(file_idx, num_samples)
            if file_samples:
                batch_obs.extend(file_samples['obs'])
                batch_next_obs.extend(file_samples['next_obs'])
                batch_actions.extend(file_samples['actions'])
                batch_rewards.extend(file_samples['rewards'])
                if 'dones' in file_samples:
                    batch_dones.extend(file_samples['dones'])
                if 'next_actions' in file_samples:
                    batch_next_actions.extend(file_samples['next_actions'])

        if len(batch_obs) == 0:
            return None

        batch = {
            'obs': np.stack(batch_obs, axis=0),           # [B, C*stack_n, H, W]
            'next_obs': np.stack(batch_next_obs, axis=0), # [B, C*stack_n, H, W]
            'actions': np.stack(batch_actions, axis=0),   # [B, action_dim]
            'rewards': np.stack(batch_rewards, axis=0)    # [B]
        }
        if batch_dones:
            batch['dones'] = np.stack(batch_dones, axis=0)
        if batch_next_actions:
            batch['next_actions'] = np.stack(batch_next_actions, axis=0)
        return batch

    def _sample_from_file(self, file_idx: int, num_samples: int) -> Optional[Dict[str, list]]:
        """
        Sample specified number of data points from a specific file
        """
        if file_idx >= len(self.data_files):
            return None
            
        # Load the specified file
        try:
            data = np.load(self.data_files[file_idx])
            images = data['image']
            actions = data['action']
            rewards = data['reward']
            is_first = data['is_first']
            is_terminal = data['is_terminal']
        except Exception as e:
            print(f"[WARNING] Failed to load file {self.data_files[file_idx]}: {e}")
            return None
        
        min_length = self.frame_stack_n + 1
        if len(images) < min_length:
            return None
        
        # Find valid indices (ensure sufficient frames)
        valid_indices = []
        for i in range(self.frame_stack_n - 1, len(images) - 1):
            curr_idx = i
            next_idx = i + 1
            valid_indices.append(curr_idx)
        
        if len(valid_indices) == 0:
            return None
        
        # Randomly sample from valid indices
        actual_samples = min(num_samples, len(valid_indices))
        if actual_samples < len(valid_indices):
            sampled_indices = random.sample(valid_indices, actual_samples)
        else:
            sampled_indices = valid_indices[:actual_samples]
        
        file_obs = []
        file_next_obs = []
        file_actions = []
        file_next_actions = []
        file_rewards = []
        file_dones = []

        filename = os.path.basename(self.data_files[file_idx])

        for curr_idx in sampled_indices:
            next_idx = curr_idx + 1

            # Create frame stacks
            curr_obs = self._create_frame_stack(images, curr_idx)
            next_obs = self._create_frame_stack(images, next_idx)

            # Get action, reward, and terminal flag
            action_curr = actions[curr_idx]
            action_next = actions[next_idx] if next_idx < len(actions) else actions[curr_idx]
            if actions.ndim > 1 and actions.shape[-1] > 1:
                action_curr_idx = int(np.argmax(action_curr))
                action_next_idx = int(np.argmax(action_next))
            else:
                action_curr_idx = int(action_curr)
                action_next_idx = int(action_next)
            start_idx = max(0, curr_idx - self.frame_stack_n + 1)
            reward_sum = float(np.sum(rewards[start_idx:curr_idx + 1]))
            done_flag = False
            if curr_idx < len(is_terminal) and is_terminal[curr_idx]:
                done_flag = True
            elif next_idx < len(is_first) and is_first[next_idx]:
                done_flag = True

            file_obs.append(curr_obs)
            file_next_obs.append(next_obs)
            file_actions.append(action_curr_idx)
            file_next_actions.append(action_next_idx)
            file_rewards.append(reward_sum)
            file_dones.append(done_flag)

        return {
            'obs': file_obs,
            'next_obs': file_next_obs,
            'actions': file_actions,
            'next_actions': file_next_actions,
            'rewards': file_rewards,
            'dones': file_dones
        }

    def reset(self):
        """Reset to beginning of dataset"""
        self.current_file_idx = 0
        self.current_pos = 0
        self.current_data = None
        self._load_current_file()
