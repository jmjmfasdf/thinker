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
        self.num_actions = self._num_actions_for_game_id(game_id)
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

    @staticmethod
    def _num_actions_for_game_id(game_id: int) -> int:
        action_counts = {
            0: 9,  # Enduro
            1: 6,  # Pong
            2: 6,  # Space Invaders
        }
        return action_counts.get(game_id, 6)

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

    def _action_index(self, action_entry) -> int:
        """Extract discrete action index from raw action entry (supports one-hot)."""
        if hasattr(action_entry, "shape") and len(action_entry.shape) > 0 and action_entry.shape[-1] == self.num_actions:
            return int(np.argmax(action_entry))
        return int(action_entry)

    def _create_forward_stack(self, images: np.ndarray, start_idx: int) -> np.ndarray:
        """Create a frame stack starting at start_idx using consecutive frames (start_idx ... start_idx+frame_stack_n-1)."""
        frames = []
        for offset in range(self.frame_stack_n):
            frame_idx = start_idx + offset
            if frame_idx >= len(images):
                # Should not happen when caller validates indices
                break
            frames.append(self._preprocess_image(images[frame_idx]))
        return np.concatenate(frames, axis=0)

    def _enumerate_episode_spans(self, is_first: np.ndarray, is_terminal: np.ndarray) -> List[Tuple[int, int]]:
        """Return (start, end) indices (inclusive) for each episode."""
        spans: List[Tuple[int, int]] = []
        start_idx = 0
        T = len(is_first)
        for idx in range(T):
            if idx > start_idx and is_first[idx]:
                spans.append((start_idx, idx - 1))
                start_idx = idx
            if is_terminal[idx]:
                spans.append((start_idx, idx))
                start_idx = idx + 1
        if start_idx < T:
            spans.append((start_idx, T - 1))
        return spans
    
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
    
    def get_sequence_batch(self, batch_size: int = 1, sequence_length: int = 2) -> Optional[Dict[str, np.ndarray]]:
        """
        Get a batch of frame-stack sequences for imitation learning.
        Each observation stack uses consecutive frames with stride=1 between stacked observations.
        Stacks are built from the most recent frame going backwards (t-3..t), and the
        human action is aligned with the most recent frame (action at t).

        Args:
            batch_size: number of sequences to return
            sequence_length: number of stacked observations (L); requires frame_stack_n * L frames

        Returns:
            Dict with keys:
            - 'obs_seq': (B, L, C*stack_n, H, W) stacked observations (float/normalized)
            - 'actions_seq': (B, L) discrete human actions (most recent action at t)
            - 'rewards_seq': (B, L) summed rewards over the last stack_n frames (t-3..t)
            - 'sequence_starts': (B,) bool flags (all True; each sample is a fresh sequence)
        """
        if len(self.data_files) == 0:
            return None

        num_files = len(self.data_files)
        if batch_size <= num_files:
            file_indices = random.sample(range(num_files), batch_size)
            samples_per_file = [1] * batch_size
        else:
            base = batch_size // num_files
            extra = batch_size % num_files
            samples_per_file = [base] * num_files
            for idx in random.sample(range(num_files), extra):
                samples_per_file[idx] += 1
            file_indices = list(range(num_files))

        sequences = []
        for file_idx, num_samples in zip(file_indices, samples_per_file):
            seqs = self._sample_nonoverlap_sequences_from_file(
                file_idx=file_idx,
                num_samples=num_samples,
                sequence_length=sequence_length,
            )
            if seqs:
                sequences.extend(seqs)

        if len(sequences) == 0:
            return None

        if len(sequences) > batch_size:
            sequences = sequences[:batch_size]

        obs_seq = np.stack([s["obs_seq"] for s in sequences], axis=0)
        actions_seq = np.stack([s["actions_seq"] for s in sequences], axis=0)
        rewards_seq = np.stack([s["rewards_seq"] for s in sequences], axis=0)
        sequence_starts = np.ones((len(sequences),), dtype=np.bool_)

        return {
            "obs_seq": obs_seq,
            "actions_seq": actions_seq,
            "rewards_seq": rewards_seq,
            "sequence_starts": sequence_starts,
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

    def _sample_nonoverlap_sequences_from_file(
        self, file_idx: int, num_samples: int, sequence_length: int
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """Sample stacked sequences from a single file (stack starts slide by 1 frame within a sequence)."""
        if file_idx >= len(self.data_files):
            return None
        try:
            data = np.load(self.data_files[file_idx])
            images = data["image"]
            actions = data["action"]
            rewards = data.get("reward", np.zeros(len(images), dtype=np.float32))
            is_first = data["is_first"]
            is_terminal = data["is_terminal"]
        except Exception as e:
            print(f"[WARNING] Failed to load file {self.data_files[file_idx]}: {e}")
            return None

        # Total raw frames needed for a sequence when stacked obs end at t:
        # first stack uses frames [start-(stack_n-1), start], last stack ends at start+(L-1).
        frames_per_seq = (self.frame_stack_n - 1) + sequence_length
        candidates: List[int] = []
        for start, end in self._enumerate_episode_spans(is_first, is_terminal):
            min_start = start + (self.frame_stack_n - 1)
            max_start = end - (sequence_length - 1)
            if max_start < min_start:
                continue
            # Align stacks so each uses fresh frames (stride = frame_stack_n)
            s = min_start
            while s <= max_start:
                window_start = s - (self.frame_stack_n - 1)
                window_end = s + (sequence_length - 1)
                # Ensure stacks stay within episode and don't cross boundaries
                if not np.any(is_first[window_start + 1 : window_end + 1]) and not np.any(is_terminal[window_start:window_end]):
                    candidates.append(s)
                s += self.frame_stack_n

        if not candidates:
            return None

        sampled = random.sample(candidates, min(num_samples, len(candidates)))
        sequences: List[Dict[str, np.ndarray]] = []
        for start_idx in sampled:
            obs_seq = []
            actions_seq = []
            rewards_seq = []
            for t in range(sequence_length):
                # shift stacked observations by 1 frame each step
                stack_end = start_idx + t
                obs_seq.append(self._create_frame_stack(images, stack_end))
                actions_seq.append(self._action_index(actions[stack_end]))
                reward_start = max(0, stack_end - (self.frame_stack_n - 1))
                rewards_seq.append(
                    float(np.sum(rewards[reward_start : stack_end + 1]))
                )
            sequences.append(
                {
                    "obs_seq": np.stack(obs_seq, axis=0),
                    "actions_seq": np.array(actions_seq, dtype=np.int64),
                    "rewards_seq": np.array(rewards_seq, dtype=np.float32),
                }
            )
        return sequences
    
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
                batch_actions.extend(file_samples['actions'])
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
            'actions': np.stack(batch_actions, axis=0),   # [B]
            'rewards': np.stack(batch_rewards, axis=0),    # [B]
            'curr_rewards': np.array(batch_curr_rewards, dtype=np.float32),
            'next_rewards': np.array(batch_next_rewards, dtype=np.float32),
            'curr_dones': np.array(batch_curr_dones, dtype=np.float32),
            'next_dones': np.array(batch_next_dones, dtype=np.float32),
            'curr_action_onehot': np.stack(batch_curr_action_onehot, axis=0).astype(np.float32),
            'next_action_onehot': np.stack(batch_next_action_onehot, axis=0).astype(np.float32),
            'prev_actions': np.array(batch_prev_actions, dtype=np.int64),
            'sequence_starts': np.array(batch_sequence_starts, dtype=np.bool_),
            'next_action_idx': np.stack(batch_next_actions, axis=0) if batch_next_actions else np.stack(batch_actions, axis=0),
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
        
        # Find valid indices within each episode (ensure sufficient history and next step)
        valid_indices = []
        for start, end in self._enumerate_episode_spans(is_first, is_terminal):
            min_idx = start + (self.frame_stack_n - 1)
            max_idx = end - 1
            if max_idx < min_idx:
                continue
            valid_indices.extend(range(min_idx, max_idx + 1))
        
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
            action_next = actions[next_idx]
            if actions.ndim > 1 and actions.shape[-1] > 1:
                action_curr_idx = int(np.argmax(action_curr))
                action_next_idx = int(np.argmax(action_next))
            else:
                action_curr_idx = int(action_curr)
                action_next_idx = int(action_next)
            start_idx = curr_idx - (self.frame_stack_n - 1)
            reward_sum = float(np.sum(rewards[start_idx:curr_idx + 1]))
            next_start_idx = next_idx - (self.frame_stack_n - 1)
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
            sequence_start_flag = bool(is_first[curr_idx]) or bool(is_first[start_idx]) or curr_idx == 0
            if not sequence_start_flag and curr_idx - 1 >= 0:
                prev_action_raw = actions[curr_idx - 1]
                if actions.ndim > 1 and actions.shape[-1] > 1:
                    prev_action_idx = int(np.argmax(prev_action_raw))
                else:
                    prev_action_idx = int(prev_action_raw)
            else:
                prev_action_idx = 0
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
