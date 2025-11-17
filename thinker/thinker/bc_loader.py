import os
import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional, Any
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
                 normalize: bool = True,
                 balance_action_idx: Optional[int] = 0,
                 balance_target_ratio: float = 0.5):
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
            balance_action_idx: Target action index to balance (None disables balancing)
            balance_target_ratio: Desired ratio of samples with ``balance_action_idx`` when BC batches are drawn
        """
        self.base_path = Path(base_path)
        self.subjects = subjects
        self.game_id = game_id
        self.frame_stack_n = frame_stack_n
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.balance_action_idx = balance_action_idx
        self.balance_target_ratio = float(np.clip(balance_target_ratio, 0.0, 1.0))
        self.balance_samples_per_file = 16
        
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
        action_indices_full = self._actions_to_indices(actions)

        T = images.shape[0]

        # Find valid sequence start positions
        valid_starts = []
        starts_by_action = {idx: [] for idx in range(self.num_actions)}
        for i in range(self.frame_stack_n - 1, T - sequence_length + 1):
            # Check if sequence doesn't cross episode boundaries
            if not np.any(is_first[i:i+sequence_length]) and not np.any(is_terminal[i:i+sequence_length]):
                valid_starts.append(i)
                action_idx = int(action_indices_full[i])
                if 0 <= action_idx < self.num_actions:
                    starts_by_action[action_idx].append(i)

        if len(valid_starts) < batch_size:
            return None

        # Sample batch_size sequences
        selected_starts = self._select_starts_by_prior(valid_starts, starts_by_action, batch_size)

        # Create batch
        batch_images = []
        batch_actions = []
        batch_rewards = []
        batch_is_first = []
        batch_is_terminal = []
        batch_prev_actions = []

        for start_idx in selected_starts:
            sequence_images = []
            sequence_actions = []
            sequence_rewards = []
            sequence_is_first = []
            sequence_is_terminal = []
            sequence_prev_actions = []

            for t in range(sequence_length):
                # Create frame stack for this timestep
                stacked_image = self._create_frame_stack(images, start_idx + t)
                sequence_images.append(stacked_image)
                
                # Get corresponding data
                sequence_actions.append(actions[start_idx + t])
                sequence_rewards.append(rewards[start_idx + t])
                sequence_is_first.append(is_first[start_idx + t])
                sequence_is_terminal.append(is_terminal[start_idx + t])
                global_idx = start_idx + t
                prev_idx = global_idx - 1
                if prev_idx < 0 or bool(is_terminal[prev_idx]) or bool(is_first[global_idx]):
                    prev_action_idx = action_indices_full[global_idx]
                else:
                    prev_action_idx = action_indices_full[prev_idx]
                sequence_prev_actions.append(prev_action_idx)

            batch_images.append(np.stack(sequence_images, axis=0))
            batch_actions.append(np.stack(sequence_actions, axis=0))
            batch_rewards.append(np.stack(sequence_rewards, axis=0))
            batch_is_first.append(np.stack(sequence_is_first, axis=0))
            batch_is_terminal.append(np.stack(sequence_is_terminal, axis=0))
            batch_prev_actions.append(np.stack(sequence_prev_actions, axis=0))

        # Stack sequences
        batch_images = np.stack(batch_images, axis=0)
        batch_actions = np.stack(batch_actions, axis=0)
        batch_rewards = np.stack(batch_rewards, axis=0)
        batch_is_first = np.stack(batch_is_first, axis=0)
        batch_is_terminal = np.stack(batch_is_terminal, axis=0)
        batch_prev_actions = np.stack(batch_prev_actions, axis=0)

        return {
            'images': batch_images,      # (B, T, C*stack_n, H, W) = (1, 40, 4, 84, 84)
            'actions': batch_actions,    # (B, T, 6)
            'rewards': batch_rewards,    # (B, T)
            'is_first': batch_is_first,  # (B, T)
            'is_terminal': batch_is_terminal,  # (B, T)
            'prev_actions': batch_prev_actions,  # (B, T)
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
        if self.balance_action_idx is not None:
            balanced = self._sample_balanced_pairs(batch_size)
            if balanced is not None:
                return balanced
            print(
                "[bc_loader] Warning: Failed to satisfy action balance target; "
                "falling back to unbalanced sampling."
            )
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
        batch_curr_rewards = []
        batch_next_rewards = []
        batch_curr_dones = []
        batch_next_dones = []
        batch_curr_action_onehot = []
        batch_next_action_onehot = []
        batch_prev_actions = []
        batch_sequence_starts = []

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
                batch_actions.extend(file_samples['next_action_idx'])
                batch_rewards.extend(file_samples['rewards'])
                batch_curr_rewards.extend(file_samples['curr_rewards'])
                batch_next_rewards.extend(file_samples['next_rewards'])
                batch_curr_dones.extend(file_samples['curr_dones'])
                batch_next_dones.extend(file_samples['next_dones'])
                batch_curr_action_onehot.extend(file_samples['curr_action_onehot'])
                batch_next_action_onehot.extend(file_samples['next_action_onehot'])
                batch_prev_actions.extend(file_samples['prev_actions'])
                batch_sequence_starts.extend(file_samples['sequence_starts'])
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
            'rewards': np.stack(batch_rewards, axis=0),    # [B]
            'curr_rewards': np.array(batch_curr_rewards, dtype=np.float32),
            'next_rewards': np.array(batch_next_rewards, dtype=np.float32),
            'curr_dones': np.array(batch_curr_dones, dtype=np.float32),
            'next_dones': np.array(batch_next_dones, dtype=np.float32),
            'curr_action_onehot': np.stack(batch_curr_action_onehot, axis=0).astype(np.float32),
            'next_action_onehot': np.stack(batch_next_action_onehot, axis=0).astype(np.float32),
            'prev_actions': np.array(batch_prev_actions, dtype=np.int64),
            'sequence_starts': np.array(batch_sequence_starts, dtype=np.bool_),
            'next_action_idx': np.stack(batch_actions, axis=0),
        }
        if batch_dones:
            batch['dones'] = np.stack(batch_dones, axis=0)
        if batch_next_actions:
            batch['next_actions'] = np.stack(batch_next_actions, axis=0)
        return batch

    def _iter_file_samples(self, file_samples: Dict[str, List[Any]]):
        """Yield sample dictionaries from a file-level sample dict."""
        total = len(file_samples.get('obs', []))
        keys = list(file_samples.keys())
        for idx in range(total):
            sample = {key: file_samples[key][idx] for key in keys}
            yield sample

    def _assemble_batch_from_samples(self, samples: List[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
        """Convert a list of sample dicts into the batch structure used by BC."""
        if not samples:
            return None

        next_action_idx = np.stack(
            [
                sample.get('next_action_idx', sample.get('actions'))
                for sample in samples
            ],
            axis=0,
        )
        batch = {
            'obs': np.stack([sample['obs'] for sample in samples], axis=0),
            'next_obs': np.stack([sample['next_obs'] for sample in samples], axis=0),
            'actions': next_action_idx.copy(),
            'rewards': np.asarray([sample['rewards'] for sample in samples], dtype=np.float32),
            'curr_rewards': np.asarray([sample['curr_rewards'] for sample in samples], dtype=np.float32),
            'next_rewards': np.asarray([sample['next_rewards'] for sample in samples], dtype=np.float32),
            'curr_dones': np.asarray([sample['curr_dones'] for sample in samples], dtype=np.float32),
            'next_dones': np.asarray([sample['next_dones'] for sample in samples], dtype=np.float32),
            'curr_action_onehot': np.stack([sample['curr_action_onehot'] for sample in samples], axis=0).astype(np.float32),
            'next_action_onehot': np.stack([sample['next_action_onehot'] for sample in samples], axis=0).astype(np.float32),
            'prev_actions': np.asarray([sample['prev_actions'] for sample in samples], dtype=np.int64),
            'sequence_starts': np.asarray([sample['sequence_starts'] for sample in samples], dtype=np.bool_),
            'next_action_idx': next_action_idx,
        }
        if all('dones' in sample for sample in samples):
            batch['dones'] = np.stack([sample['dones'] for sample in samples], axis=0)
        if all('next_actions' in sample for sample in samples):
            batch['next_actions'] = np.stack([sample['next_actions'] for sample in samples], axis=0)
        return batch

    def _sample_balanced_pairs(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Sample BC pairs while keeping the target action close to the desired ratio."""
        if batch_size <= 0 or len(self.data_files) == 0:
            return None
        target_ratio = float(np.clip(self.balance_target_ratio, 0.0, 1.0))
        target_action = min(batch_size, max(0, int(round(batch_size * target_ratio))))
        other_action = batch_size - target_action
        selected_target: List[Dict[str, Any]] = []
        selected_other: List[Dict[str, Any]] = []

        max_attempts = max(5 * batch_size, 200)
        attempts = 0
        per_file_samples = max(2, min(self.balance_samples_per_file, batch_size))

        while (len(selected_target) < target_action or len(selected_other) < other_action) and attempts < max_attempts:
            file_idx = random.randrange(len(self.data_files))
            file_samples = self._sample_from_file(file_idx, per_file_samples)
            attempts += 1
            if not file_samples:
                continue

            for sample in self._iter_file_samples(file_samples):
                action_val = sample.get('next_actions')
                if action_val is None:
                    action_val = sample.get('next_action_idx')
                if action_val is None:
                    action_val = sample.get('actions')
                if action_val is None:
                    continue
                action_idx = int(action_val)
                if action_idx == self.balance_action_idx:
                    if len(selected_target) < target_action:
                        selected_target.append(sample)
                else:
                    if len(selected_other) < other_action:
                        selected_other.append(sample)
                if len(selected_target) >= target_action and len(selected_other) >= other_action:
                    break

        if len(selected_target) < target_action or len(selected_other) < other_action:
            return None

        combined = selected_target + selected_other
        random.shuffle(combined)
        return self._assemble_batch_from_samples(combined[:batch_size])

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
        file_curr_rewards = []
        file_next_rewards = []
        file_curr_dones = []
        file_next_dones = []
        file_curr_action_onehot = []
        file_next_action_onehot = []
        file_prev_actions = []
        file_sequence_starts = []

        filename = os.path.basename(self.data_files[file_idx])

        for curr_idx in sampled_indices:
            next_idx = curr_idx + 1

            # Create frame stacks
            curr_obs = self._create_frame_stack(images, curr_idx)
            next_obs = self._create_frame_stack(images, next_idx)

            # Get action, reward, and terminal flag
            action_curr = actions[curr_idx]
            next_stack_start = max(0, next_idx - self.frame_stack_n + 1)
            action_next = actions[next_stack_start]
            if actions.ndim > 1 and actions.shape[-1] > 1:
                action_curr_idx = int(np.argmax(action_curr))
                action_next_idx = int(np.argmax(action_next))
            else:
                action_curr_idx = int(action_curr)
                action_next_idx = int(action_next)
            start_idx = max(0, curr_idx - self.frame_stack_n + 1)
            reward_sum = float(np.sum(rewards[start_idx:curr_idx + 1]))
            next_start_idx = max(0, next_idx - self.frame_stack_n + 1)
            next_reward_sum = float(np.sum(rewards[next_start_idx:next_idx + 1]))
            done_flag = False
            if curr_idx < len(is_terminal) and is_terminal[curr_idx]:
                done_flag = True
            elif next_idx < len(is_first) and is_first[next_idx]:
                done_flag = True
            next_done_flag = False
            if next_idx < len(is_terminal) and is_terminal[next_idx]:
                next_done_flag = True
            elif (next_idx + 1) < len(is_first) and is_first[next_idx + 1]:
                next_done_flag = True

            file_obs.append(curr_obs)
            file_next_obs.append(next_obs)
            file_actions.append(action_curr_idx)
            file_next_actions.append(action_next_idx)
            file_rewards.append(reward_sum)
            file_dones.append(done_flag)
            file_curr_rewards.append(reward_sum)
            file_next_rewards.append(next_reward_sum)
            file_curr_dones.append(done_flag)
            file_next_dones.append(next_done_flag)
            curr_onehot = np.zeros(self.num_actions, dtype=np.float32)
            curr_onehot[action_curr_idx] = 1.0
            next_onehot = np.zeros(self.num_actions, dtype=np.float32)
            next_onehot[action_next_idx] = 1.0
            file_curr_action_onehot.append(curr_onehot)
            file_next_action_onehot.append(next_onehot)
            sequence_start_flag = bool(is_first[next_idx]) or curr_idx == 0
            prev_action_idx = action_curr_idx if not sequence_start_flag else 0
            file_prev_actions.append(prev_action_idx)
            file_sequence_starts.append(sequence_start_flag)

        return {
            'obs': file_obs,
            'next_obs': file_next_obs,
            'actions': file_actions,
            'next_actions': file_next_actions,
            'rewards': file_rewards,
            'dones': file_dones,
            'curr_rewards': file_curr_rewards,
            'next_rewards': file_next_rewards,
            'curr_dones': file_curr_dones,
            'next_dones': file_next_dones,
            'curr_action_onehot': file_curr_action_onehot,
            'next_action_onehot': file_next_action_onehot,
            'prev_actions': file_prev_actions,
            'sequence_starts': file_sequence_starts,
            'next_action_idx': file_next_actions,
        }

    def reset(self):
        """Reset to beginning of dataset"""
        self.current_file_idx = 0
        self.current_pos = 0
        self.current_data = None
        self._load_current_file()

    def sample_root_batch(
        self,
        batch_size: int = 32,
        sequence_length: int = 40,
        gamma: float = 0.99,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Convenience helper that returns root observations plus discounted returns."""
        sample = self.get_sequence_batch(batch_size=batch_size, sequence_length=sequence_length)
        if sample is None:
            return None
        images = sample["images"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        obs = images[:, 0].astype(np.float32)
        action_idx = self._to_action_index(actions[:, 0])
        discounts = np.power(gamma, np.arange(rewards.shape[1], dtype=np.float32))
        returns = np.einsum("bt,t->b", rewards, discounts).astype(np.float32)
        return {
            "obs": obs,
            "actions": action_idx,
            "returns": returns,
        }

    def _select_starts_by_prior(
        self,
        valid_starts: List[int],
        starts_by_action: Dict[int, List[int]],
        batch_size: int,
    ) -> np.ndarray:
        """Sample start indices so batch action mix matches the human prior."""
        if len(valid_starts) <= batch_size:
            rng = np.random.default_rng()
            return rng.choice(valid_starts, size=batch_size, replace=False)
        prior = self.action_distribution / np.sum(self.action_distribution)
        target_counts = np.random.multinomial(batch_size, prior)
        selected: List[int] = []
        remaining = set(valid_starts)
        rng = np.random.default_rng()
        for action_idx, target in enumerate(target_counts):
            pool = starts_by_action.get(action_idx, [])
            if not pool or target <= 0:
                continue
            take = min(target, len(pool))
            if take <= 0:
                continue
            chosen = rng.choice(pool, size=take, replace=False)
            for idx in chosen:
                remaining.discard(int(idx))
            selected.extend([int(idx) for idx in chosen])
        if len(selected) < batch_size and remaining:
            needed = batch_size - len(selected)
            extra = rng.choice(sorted(remaining), size=min(needed, len(remaining)), replace=False)
            selected.extend([int(idx) for idx in extra])
        if len(selected) > batch_size:
            selected = rng.choice(selected, size=batch_size, replace=False).tolist()
        return np.array(selected, dtype=np.int64)

    @staticmethod
    def _to_action_index(action_slice: np.ndarray) -> np.ndarray:
        if action_slice.ndim == 1:
            return action_slice.astype(np.int64)
        if action_slice.ndim == 2:
            return np.argmax(action_slice, axis=-1).astype(np.int64)
        raise ValueError(f"Unsupported action slice with shape {action_slice.shape}")

    def _actions_to_indices(self, actions: np.ndarray) -> np.ndarray:
        """Convert raw action tensors to index form."""
        if actions.ndim == 1:
            return actions.astype(np.int64)
        if actions.ndim == 2 and actions.shape[-1] == self.num_actions:
            return np.argmax(actions, axis=-1).astype(np.int64)
        if actions.ndim == 2 and actions.shape[-1] == 1:
            return actions.reshape(-1).astype(np.int64)
        raise ValueError(f"Unsupported actions tensor shape {actions.shape}")
