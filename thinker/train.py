import time
import os
import ray
import torch
from thinker.buffer import ActorBuffer, GeneralBuffer, SelfPlayBuffer
from thinker.self_play import SelfPlayWorker
from thinker.logger import LogWorker
from thinker.main import get_ray_temp_dir, ray_init
from thinker import util
import sys
sys.path.append('/home/jmme425/thinker/thinker')

def _extract_flag_value(source, name, default):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _sync_flags_with_pretrained(flags, logger):
    if getattr(flags, "preload", ''):
        model_ckp = os.path.join(flags.preload, 'ckp_model.tar')
        if os.path.exists(model_ckp):
            try:
                checkpoint = torch.load(model_ckp, map_location='cpu', weights_only=False)
                pretrained_flags = checkpoint.get('flags')
                if pretrained_flags is not None:
                    frame_stack = _extract_flag_value(pretrained_flags, 'frame_stack_n', flags.frame_stack_n)
                    grayscale = _extract_flag_value(pretrained_flags, 'grayscale', flags.grayscale)
                    env_n = _extract_flag_value(pretrained_flags, 'env_n', flags.env_n)
                    if frame_stack != flags.frame_stack_n:
                        logger.info(f"Updating frame_stack_n from {flags.frame_stack_n} to {frame_stack} based on pretrained model")
                        flags.frame_stack_n = frame_stack
                    if grayscale != flags.grayscale:
                        logger.info(f"Updating grayscale from {flags.grayscale} to {grayscale} based on pretrained model")
                        flags.grayscale = grayscale
                    if env_n != flags.env_n:
                        logger.info(f"Updating env_n from {flags.env_n} to {env_n} based on pretrained model")
                        flags.env_n = env_n
            except Exception as exc:
                logger.warning(f"Failed to read model checkpoint flags: {exc}")
    actor_root = getattr(flags, 'preload_actor', '') or getattr(flags, 'preload', '')
    if actor_root:
        actor_ckp = os.path.join(actor_root, 'ckp_actor.tar')
        if os.path.exists(actor_ckp):
            try:
                checkpoint = torch.load(actor_ckp, map_location='cpu', weights_only=False)
                pretrained_flags = checkpoint.get('flags')
                if pretrained_flags is not None:
                    rec_t = _extract_flag_value(pretrained_flags, 'rec_t', flags.rec_t)
                    has_action_seq = _extract_flag_value(pretrained_flags, 'has_action_seq', getattr(flags, 'has_action_seq', True))
                    max_depth = _extract_flag_value(pretrained_flags, 'max_depth', flags.max_depth)
                    reset_mode = _extract_flag_value(pretrained_flags, 'reset_mode', getattr(flags, 'reset_mode', 0))
                    if rec_t != flags.rec_t:
                        logger.info(f"Updating rec_t from {flags.rec_t} to {rec_t} based on pretrained actor")
                        flags.rec_t = rec_t
                    if has_action_seq != getattr(flags, 'has_action_seq', True):
                        logger.info(f"Updating has_action_seq from {getattr(flags, 'has_action_seq', True)} to {has_action_seq} based on pretrained actor")
                        flags.has_action_seq = has_action_seq
                    if max_depth != flags.max_depth:
                        logger.info(f"Updating max_depth from {flags.max_depth} to {max_depth} based on pretrained actor")
                        flags.max_depth = max_depth
                    if reset_mode != getattr(flags, 'reset_mode', 0):
                        logger.info(f"Updating reset_mode from {getattr(flags, 'reset_mode', 0)} to {reset_mode} based on pretrained actor")
                        flags.reset_mode = reset_mode
            except Exception as exc:
                logger.warning(f"Failed to read actor checkpoint flags: {exc}")



if __name__ == "__main__":
    logger = util.logger()
    logger.info("Initializing...")

    st_time = time.time()
    flags = util.create_setting()
    _sync_flags_with_pretrained(flags, logger)

    ray_kwargs = {
        "num_cpus": int(flags.ray_cpu) if flags.ray_cpu > 0 else None,
        "num_gpus": int(flags.ray_gpu) if flags.ray_gpu > 0 else None,
        "object_store_memory": int(flags.ray_mem * 1024**3)
        if flags.ray_mem > 0
        else None,
    }
    ray_temp_dir = get_ray_temp_dir()
    if ray_temp_dir:
        logger.info(f"Using Ray temp dir {ray_temp_dir}")
        ray_kwargs["_temp_dir"] = ray_temp_dir
    ray.init(**ray_kwargs)

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
    
    ray_obj_env = ray_init(flags=flags, save_flags=False, **vars(flags))
    ray_obj_env["actor_param_buffer"] = actor_param_buffer
    ray_obj_actor = {"actor_buffer": actor_buffer,
                     "actor_param_buffer": actor_param_buffer,
                     "model_param_buffer": ray_obj_env.get("param_buffer")}

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
