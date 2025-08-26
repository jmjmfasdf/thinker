import time
import os
import ray
import torch
import numpy as np
from thinker.buffer import ActorBuffer, GeneralBuffer, SelfPlayBuffer
from thinker.self_play import SelfPlayWorker
from thinker.logger import LogWorker
from thinker.main import ray_init
from thinker import util
import sys
sys.path.append('/home/jmme425/thinker/thinker')

if __name__ == "__main__":
    logger = util.logger()
    logger.info("Initializing...")

    st_time = time.time()
    flags = util.create_setting()

    ray.init(
            num_cpus=int(flags.ray_cpu) if flags.ray_cpu > 0 else None,
            num_gpus=int(flags.ray_gpu) if flags.ray_gpu > 0 else None,
            object_store_memory=int(flags.ray_mem * 1024**3)
            if flags.ray_mem > 0
            else None,
        )

    num_gpus_available = torch.cuda.device_count()
    num_cpus_available = ray.cluster_resources()["CPU"]
    logger.info("Detected %d GPU %d CPU" % (num_gpus_available, num_cpus_available))

    gpu_n = min(int(num_gpus_available - 1), 3)    
    if flags.auto_res: flags = util.alloc_res(flags, gpu_n)
    if flags.parallel_actor:
        actor_buffer = ActorBuffer.options(num_cpus=1).remote(
            batch_size=flags.actor_batch_size,
            buffer_save_size=flags.buffer_save_size if hasattr(flags, 'buffer_save_size') else 1
        ) 
        actor_param_buffer = GeneralBuffer.options(num_cpus=1).remote()  
    else:
        actor_buffer = None
        actor_param_buffer = None
    
    # Check if BC training is enabled - BEFORE starting ray workers
    print(f"DEBUG: Checking BC flag - hasattr(flags, 'bc_clone'): {hasattr(flags, 'bc_clone')}")
    if hasattr(flags, 'bc_clone'):
        print(f"DEBUG: flags.bc_clone value: {flags.bc_clone}")
    
    if hasattr(flags, 'bc_clone') and flags.bc_clone:
        logger.info("BC training mode enabled - initializing networks for BC training...")
        
        # Import BC training modules
        from bc_trainer import run_bc_training
        from thinker.model_net import ModelNet
        from thinker.actor_net import ActorNet
        from gymnasium import spaces
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create observation and action spaces for BC
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
        
        # Hidden state space (hs) - from model network
        hs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(flags.env_n, 256, 6, 6),  # typical hidden state dimensions
            dtype=np.float32
        )
        
        # State representation space (xs) - predicted next states
        xs_space = spaces.Box(
            low=0, high=1,
            shape=(flags.env_n, flags.frame_stack_n, 84, 84),
            dtype=np.float32
        )
        
        # Combined observation space
        observation_space = spaces.Dict({
            "real_states": real_state_space,
            "tree_reps": tree_reps_space,
            "hs": hs_space,
            "xs": xs_space
        })
        
        # Action space
        primary_action_space = spaces.Discrete(6)
        reset_action_space = spaces.Discrete(2)
        action_space = spaces.Tuple((
            spaces.Tuple((primary_action_space,)),
            spaces.Tuple((reset_action_space,))
        ))
        
        # Initialize networks
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
            
            # Load model weights and get pretrained settings
            model_path = os.path.join(flags.preload, 'ckp_model.tar')
            if os.path.exists(model_path):
                logger.info(f"Loading model weights from {model_path}")
                model_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Get pretrained model settings
                pretrained_flags = model_checkpoint.get('flags', None)
                if pretrained_flags:
                    # Convert dict to namespace if needed
                    if isinstance(pretrained_flags, dict):
                        pretrained_frame_stack = pretrained_flags.get('frame_stack_n', flags.frame_stack_n)
                        pretrained_grayscale = pretrained_flags.get('grayscale', flags.grayscale)
                        pretrained_env_n = pretrained_flags.get('env_n', flags.env_n)
                    else:
                        pretrained_frame_stack = getattr(pretrained_flags, 'frame_stack_n', flags.frame_stack_n)
                        pretrained_grayscale = getattr(pretrained_flags, 'grayscale', flags.grayscale)
                        pretrained_env_n = getattr(pretrained_flags, 'env_n', flags.env_n)
                        
                    logger.info(f"Pretrained model frame_stack_n: {pretrained_frame_stack}")
                    logger.info(f"Pretrained model grayscale: {pretrained_grayscale}")
                    logger.info(f"Pretrained model env_n: {pretrained_env_n}")
                    logger.info(f"Current model frame_stack_n: {flags.frame_stack_n}")
                    logger.info(f"Current model grayscale: {flags.grayscale}")
                    logger.info(f"Current model env_n: {flags.env_n}")
                    
                    # Update our flags to match pretrained model
                    flags.frame_stack_n = pretrained_frame_stack
                    flags.grayscale = pretrained_grayscale
                    flags.env_n = pretrained_env_n
                    logger.info(f"Updated frame_stack_n to: {flags.frame_stack_n}")
                    logger.info(f"Updated grayscale to: {flags.grayscale}")
                    logger.info(f"Updated env_n to: {flags.env_n}")
                    
                    # Calculate correct channel count
                    channels_per_frame = 1 if flags.grayscale else 3
                    total_channels = channels_per_frame * flags.frame_stack_n
                    logger.info(f"Total input channels: {channels_per_frame} × {flags.frame_stack_n} = {total_channels}")
                    
                    # Recreate model with correct dimensions
                    logger.info("Recreating model with correct dimensions...")
                    model_obs_space = spaces.Box(
                        low=0, high=255,
                        shape=(total_channels, 84, 84),
                        dtype=np.uint8
                    )
                    
                    model_net = ModelNet(
                        obs_space=model_obs_space,
                        action_space=primary_action_space,
                        flags=flags
                    ).to(device)
                    
                    # Also recreate observation space for actor with updated settings
                    real_state_space = spaces.Box(
                        low=0, high=255,
                        shape=(flags.env_n, total_channels, 84, 84),
                        dtype=np.uint8
                    )
                    
                    xs_space = spaces.Box(
                        low=0, high=1,
                        shape=(flags.env_n, total_channels, 84, 84),
                        dtype=np.float32
                    )
                    
                    observation_space = spaces.Dict({
                        "real_states": real_state_space,
                        "tree_reps": tree_reps_space,
                        "hs": hs_space,
                        "xs": xs_space
                    })
                    
                    # Recreate actor net with updated observation space
                    logger.info("Recreating actor net with updated observation space...")
                    actor_net = ActorNet(
                        obs_space=observation_space,
                        action_space=action_space,
                        flags=flags
                    ).to(device)
                
                model_net.load_state_dict(model_checkpoint['model_net_state_dict'])
                logger.info("Model weights loaded successfully")
            else:
                logger.warning(f"Model checkpoint not found at {model_path}")
            
            # Load actor weights  
            actor_path = os.path.join(flags.preload, 'ckp_actor.tar')
            if os.path.exists(actor_path):
                logger.info(f"Loading actor weights from {actor_path}")
                actor_checkpoint = torch.load(actor_path, map_location=device, weights_only=False)
                
                # Get pretrained actor settings
                actor_pretrained_flags = actor_checkpoint.get('flags', None)
                if actor_pretrained_flags:
                    # Update flags with actor-specific settings
                    if isinstance(actor_pretrained_flags, dict):
                        pretrained_rec_t = actor_pretrained_flags.get('rec_t', flags.rec_t)
                        pretrained_has_action_seq = actor_pretrained_flags.get('has_action_seq', getattr(flags, 'has_action_seq', False))
                        pretrained_max_depth = actor_pretrained_flags.get('max_depth', flags.max_depth)
                        pretrained_reset_mode = actor_pretrained_flags.get('reset_mode', getattr(flags, 'reset_mode', 0))
                    else:
                        pretrained_rec_t = getattr(actor_pretrained_flags, 'rec_t', flags.rec_t)
                        pretrained_has_action_seq = getattr(actor_pretrained_flags, 'has_action_seq', getattr(flags, 'has_action_seq', False))
                        pretrained_max_depth = getattr(actor_pretrained_flags, 'max_depth', flags.max_depth)
                        pretrained_reset_mode = getattr(actor_pretrained_flags, 'reset_mode', getattr(flags, 'reset_mode', 0))
                    
                    logger.info(f"Pretrained actor rec_t: {pretrained_rec_t}")
                    logger.info(f"Pretrained actor has_action_seq: {pretrained_has_action_seq}")
                    logger.info(f"Pretrained actor max_depth: {pretrained_max_depth}")
                    logger.info(f"Pretrained actor reset_mode: {pretrained_reset_mode}")
                    
                    # Update flags
                    flags.rec_t = pretrained_rec_t
                    flags.has_action_seq = pretrained_has_action_seq
                    flags.max_depth = pretrained_max_depth
                    flags.reset_mode = pretrained_reset_mode
                    
                    # Recalculate tree rep dimensions (following cenv.pyx logic)
                    num_actions = 6
                    obs_n = 11 + num_actions * 10 + flags.rec_t
                    if flags.has_action_seq:
                        obs_n += flags.max_depth * num_actions
                        if flags.reset_mode == 0:
                            obs_n += num_actions
                    
                    logger.info(f"Recalculated tree rep size: 11 + {num_actions} * 10 + {flags.rec_t}")
                    if flags.has_action_seq:
                        logger.info(f"  + has_action_seq: {flags.max_depth} * {num_actions} = {flags.max_depth * num_actions}")
                        if flags.reset_mode == 0:
                            logger.info(f"  + reset_mode=0: {num_actions}")
                    logger.info(f"  = {obs_n}")
                    
                    # Recreate tree representation space
                    tree_reps_space = spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(flags.env_n, obs_n),
                        dtype=np.float32
                    )
                    
                    # Recreate observation space with correct tree rep dimensions
                    observation_space = spaces.Dict({
                        "real_states": real_state_space,
                        "tree_reps": tree_reps_space,
                        "hs": hs_space,
                        "xs": xs_space
                    })
                    
                    # Recreate actor net with correct tree rep dimensions
                    logger.info("Recreating actor net with correct tree rep dimensions...")
                    actor_net = ActorNet(
                        obs_space=observation_space,
                        action_space=action_space,
                        flags=flags
                    ).to(device)
                # Check what keys are available in actor checkpoint
                logger.info(f"Actor checkpoint keys: {list(actor_checkpoint.keys())}")
                if 'actor_net_state_dict' in actor_checkpoint:
                    actor_net.load_state_dict(actor_checkpoint['actor_net_state_dict'])
                elif 'actor_state_dict' in actor_checkpoint:
                    actor_net.load_state_dict(actor_checkpoint['actor_state_dict'])
                elif 'model_state_dict' in actor_checkpoint:
                    actor_net.load_state_dict(actor_checkpoint['model_state_dict'])
                else:
                    # Try to find the correct state dict key (exclude optimizer keys)
                    state_dict_keys = [k for k in actor_checkpoint.keys() if 'state_dict' in k and 'optimizer' not in k and 'scheduler' not in k]
                    if state_dict_keys:
                        logger.info(f"Using state dict key: {state_dict_keys[0]}")
                        actor_net.load_state_dict(actor_checkpoint[state_dict_keys[0]])
                    else:
                        logger.warning("No suitable state dict found in actor checkpoint")
                logger.info("Actor weights loaded successfully")
            else:
                logger.warning(f"Actor checkpoint not found at {actor_path}")
        
        logger.info("Starting BC training with pretrained networks...")
        
        # Run BC training
        best_loss, epoch_losses = run_bc_training(flags, model_net, actor_net, logger)
        
        logger.info(f"BC training completed with best loss: {best_loss:.4f}")
        print(f"BC training completed! Best loss: {best_loss:.4f}")
        
        # Exit after BC training - skip the rest of the training pipeline
        import sys
        sys.exit(0)

    ray_obj_env = ray_init(flags=flags, save_flags=False, **vars(flags))
    ray_obj_env["actor_param_buffer"] = actor_param_buffer
    ray_obj_actor = {"actor_buffer": actor_buffer,
                     "actor_param_buffer": actor_param_buffer}

    if not flags.train_actor: 
        self_play_buffer = SelfPlayBuffer.options(num_cpus=1).remote(flags=flags)
        ray_obj_actor["self_play_buffer"] = self_play_buffer

    self_play_workers = []
    self_play_workers.extend(
        [
            SelfPlayWorker.options(num_cpus=1, num_gpus=flags.gpu_self_play).remote(
                ray_obj_env=ray_obj_env,
                ray_obj_actor=ray_obj_actor,                
                rank=n,
                env_n=flags.env_n,           
                flags=flags,
            )
            for n in range(flags.self_play_n)
        ]
    )
    r_worker = [x.gen_data.remote() for x in self_play_workers]        

    if flags.use_wandb:
        log_worker = LogWorker.options(num_cpus=1, num_gpus=0).remote(flags)
        r_log_worker = log_worker.start.remote()

    return_codes = ray.get(r_worker)
    if all(return_codes):
        open(os.path.join(flags.ckpdir, 'finish'), 'a').close()
    if flags.use_wandb:
        ray.get(r_log_worker)
    logger.info("Time required: %fs" % (time.time() - st_time))