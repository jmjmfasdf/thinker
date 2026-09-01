import time
import os
import ray
import torch
import numpy as np
import random
from thinker.buffer import ActorBuffer, GeneralBuffer, SelfPlayBuffer
from thinker.self_play import SelfPlayWorker
from thinker.logger import LogWorker
from thinker.main import ray_init
from thinker import util


def set_seed(seed):
    """Seed the driver before it creates the shared initial networks."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cancel_schema6_refs(refs):
    for ref in refs:
        try:
            # These refs are actor-method tasks.  Ray rejects force=True for
            # actor tasks; the actor handle is hard-killed separately below.
            ray.cancel(ref, force=False)
        except Exception:
            pass


def _terminate_schema6_workers(refs, worker_handles):
    for worker in () if worker_handles is None else worker_handles:
        try:
            ray.kill(worker, no_restart=True)
        except Exception:
            # Preserve the originating timeout/malformed/logger exception.
            # A failed cleanup can never turn that path into public success.
            pass
    _cancel_schema6_refs(refs)


def _terminate_schema6_log_worker(
    logger_worker,
    logger_ref,
    *,
    timeout_s=util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
):
    """Kill the logger and confirm its task is terminal before marker cleanup."""

    if logger_worker is None or logger_ref is None:
        raise RuntimeError(
            "schema-6 logger termination requires its actor handle and task ref"
        )
    kill_error = None
    try:
        ray.kill(logger_worker, no_restart=True)
    except Exception as error:
        kill_error = error
    try:
        # Do not cancel first: a cancelled ref could look terminal while an
        # asynchronously killed actor still has time to publish a late ack.
        ray.get(logger_ref, timeout=float(timeout_s))
    except ray.exceptions.GetTimeoutError as error:
        message = "schema-6 logger death could not be confirmed"
        if kill_error is not None:
            message += f" after kill failure: {kill_error}"
        raise TimeoutError(message) from error
    except Exception:
        # RayActorError/TaskCancelledError/application failure all prove the
        # sole start task is terminal and can no longer write private markers.
        pass
    try:
        ray.cancel(logger_ref, force=False)
    except Exception:
        pass
    return True


def _poll_schema6_log_worker(
    logger_ref, worker_refs, *, worker_handles=None, deadline,
    monotonic=time.monotonic,
):
    """Fail immediately if the strict logger exits before its request."""

    if logger_ref is None:
        return
    ready, _ = ray.wait([logger_ref], num_returns=1, timeout=0.0)
    if not ready:
        return
    remaining = deadline - monotonic()
    if remaining <= 0.0:
        _terminate_schema6_workers(worker_refs, worker_handles)
        raise TimeoutError("schema-6 logger health poll exceeded its deadline")
    try:
        result = ray.get(ready[0], timeout=max(remaining, 1e-9))
    except Exception as error:
        _terminate_schema6_workers(worker_refs, worker_handles)
        raise RuntimeError(
            "schema-6 log worker failed before the private finish request"
        ) from error
    _terminate_schema6_workers(worker_refs, worker_handles)
    raise RuntimeError(
        "schema-6 log worker exited before the private finish request: "
        f"result={result!r}"
    )


def wait_for_schema6_workers(
    worker_refs,
    actor_param_buffer,
    *,
    worker_handles=None,
    logger_ref=None,
    timeout_s=util.VOC_ACTOR_POLICY_BARRIER_TIMEOUT_SECONDS,
    monotonic=time.monotonic,
):
    """Bound the sole-worker task by version/enqueue heartbeat progress."""

    pending = list(worker_refs)
    results = []
    last_heartbeat = None
    deadline = monotonic() + float(timeout_s)
    while pending:
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            _terminate_schema6_workers(pending, worker_handles)
            raise TimeoutError(
                "schema-6 self-play worker made no policy heartbeat progress "
                f"for {float(timeout_s):.1f}s"
            )
        _poll_schema6_log_worker(
            logger_ref,
            pending,
            worker_handles=worker_handles,
            deadline=deadline,
            monotonic=monotonic,
        )
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            _terminate_schema6_workers(pending, worker_handles)
            raise TimeoutError(
                "schema-6 self-play worker made no policy heartbeat progress "
                f"for {float(timeout_s):.1f}s"
            )
        ready, pending = ray.wait(
            pending,
            num_returns=1,
            timeout=min(1.0, remaining),
        )
        if ready:
            results.append(ray.get(ready[0], timeout=max(deadline - monotonic(), 1e-9)))
            continue
        heartbeat_ref = actor_param_buffer.get_data.remote(
            util.VOC_ACTOR_POLICY_HEARTBEAT_KEY
        )
        try:
            heartbeat = ray.get(
                heartbeat_ref,
                timeout=max(deadline - monotonic(), 1e-9),
            )
        except ray.exceptions.GetTimeoutError as error:
            _terminate_schema6_workers(pending, worker_handles)
            raise TimeoutError(
                "schema-6 heartbeat RPC exceeded the monotonic deadline"
            ) from error
        if heartbeat is not None:
            if not isinstance(heartbeat, dict) or len(heartbeat) != 1:
                _terminate_schema6_workers(pending, worker_handles)
                raise RuntimeError(
                    "schema-6 heartbeat set must contain sole rank 0"
                )
            heartbeat_rank = next(iter(heartbeat))
            if type(heartbeat_rank) is not int or heartbeat_rank != 0:
                _terminate_schema6_workers(pending, worker_handles)
                raise RuntimeError(
                    "schema-6 heartbeat rank key must be Python integer 0"
                )
            try:
                canonical, progressed = util.validate_actor_policy_heartbeat(
                    heartbeat[heartbeat_rank], previous=last_heartbeat
                )
            except ValueError as error:
                _terminate_schema6_workers(pending, worker_handles)
                raise RuntimeError("schema-6 heartbeat is malformed") from error
            if progressed:
                last_heartbeat = canonical
                deadline = monotonic() + float(timeout_s)
    _poll_schema6_log_worker(
        logger_ref,
        (),
        worker_handles=worker_handles,
        deadline=deadline,
        monotonic=monotonic,
    )
    return results


def finish_schema6_log_worker(
    logger_ref,
    flags,
    *,
    final_bundle,
    logger_worker=None,
    monotonic=time.monotonic,
):
    """Perform the private request/strict result/ack close handshake."""

    if logger_ref is None:
        raise RuntimeError("schema-6 W&B run lacks a log worker task")
    evidence = final_bundle["actor_policy"]
    completion_evidence = final_bundle["completion_evidence"]
    gate_schema = final_bundle.get("resolved_identity", {}).get("gate_schema")
    schema13 = (
        type(gate_schema) is int
        and gate_schema == util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    )
    logger_completion_schema = (
        util.VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
        if schema13
        else util.VOC_ACTOR_POLICY_LOGGER_COMPLETION_SCHEMA_VERSION
    )
    if evidence["voc_actor_policy_terminal"] is not True:
        raise RuntimeError("schema-6 logger close requires terminal checkpoint")
    request = None
    request_identity = None
    logger_task_terminal = False
    try:
        request, request_identity = util.write_actor_policy_logger_finish_request(
            flags.ckpdir,
            evidence,
            completion_evidence,
            return_identity=True,
            schema_version=logger_completion_schema,
        )
        deadline = monotonic() + float(
            flags.voc_actor_policy_barrier_timeout_s
        )
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                "schema-6 log worker close exceeded its monotonic deadline"
            )
        try:
            result = ray.get(logger_ref, timeout=max(remaining, 1e-9))
        except ray.exceptions.GetTimeoutError as error:
            raise TimeoutError(
                "schema-6 log worker did not acknowledge close within "
                f"{float(flags.voc_actor_policy_barrier_timeout_s):.1f}s"
            ) from error
        if type(result) is not bool or result is not True:
            raise RuntimeError(
                "schema-6 log worker close result must be exactly True"
            )
        logger_task_terminal = True
        ack = util.read_actor_policy_logger_finish_ack(
            flags.ckpdir, request
        )
        if ack is None:
            raise RuntimeError("schema-6 log worker returned without an ack")
        post_bundle_validator = (
            util.validate_schema13_final_bundle
            if schema13
            else util.validate_schema6_final_bundle
        )
        post_bundle = post_bundle_validator(
            flags.ckpdir,
            label=(
                "post-logger schema-13 final bundle"
                if schema13
                else "post-logger schema-6 final bundle"
            ),
        )
        post_actor = post_bundle["actor_policy"]
        for name in (
            "voc_actor_policy_version",
            "voc_actor_policy_state_sha256",
            "voc_actor_policy_publication_history_sha256",
        ):
            if post_actor[name] != evidence[name]:
                raise RuntimeError(
                    f"schema-6 actor evidence changed during logger close: {name}"
                )
        if post_bundle["completion_evidence"] != completion_evidence:
            raise RuntimeError(
                "schema-6 completion files changed during logger close"
            )
        return {
            "schema_version": logger_completion_schema,
            "required": True,
            "use_wandb": True,
            "request_sha256": (
                util.actor_policy_logger_finish_request_sha256(request)
            ),
            "ack_verified": True,
            "private_markers_cleaned": True,
            "policy_version": evidence["voc_actor_policy_version"],
            "state_sha256": evidence["voc_actor_policy_state_sha256"],
            "publication_history_sha256": evidence[
                "voc_actor_policy_publication_history_sha256"
            ],
            "checkpoint_files": completion_evidence["checkpoint_files"],
        }
    except BaseException as error:
        try:
            logger_task_terminal = _terminate_schema6_log_worker(
                logger_worker,
                logger_ref,
                timeout_s=flags.voc_actor_policy_barrier_timeout_s,
            )
        except BaseException as termination_error:
            raise error from termination_error
        raise
    finally:
        if logger_task_terminal and request_identity is not None:
            util.clear_actor_policy_logger_completion(
                flags.ckpdir,
                expected_request=request,
                expected_request_identity=request_identity,
            )


def finalize_run(
    flags, *, logger_ref=None, logger_worker=None, monotonic=time.monotonic
):
    """Commit public completion only after schema-6 logger attestation."""

    barrier_runtime = bool(
        getattr(flags, "voc_actor_policy_barrier_runtime", False)
    )
    if barrier_runtime:
        gate_schema = getattr(flags, "voc_gate_policy_schema_version", None)
        if (
            type(gate_schema) is int
            and gate_schema == util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        ):
            final_bundle = util.validate_schema13_final_bundle(flags.ckpdir)
            completion_schema = (
                util.VOC_ACTOR_POLICY_LOGGER_SCHEMA13_COMPLETION_SCHEMA_VERSION
            )
        else:
            final_bundle = util.validate_schema6_final_bundle(flags.ckpdir)
            completion_schema = (
                util.VOC_ACTOR_POLICY_LOGGER_COMPLETION_SCHEMA_VERSION
            )
        if type(flags.use_wandb) is not bool:
            raise ValueError("schema-6 use_wandb must be a strict boolean")
        if final_bundle["config_use_wandb"] != flags.use_wandb:
            raise ValueError(
                "schema-6 runtime use_wandb disagrees with final config"
            )
        actor_evidence = final_bundle["actor_policy"]
        if actor_evidence["voc_actor_policy_terminal"] is not True:
            raise RuntimeError(
                "schema-6 public completion requires terminal actor evidence"
            )
        if flags.use_wandb:
            logger_completion = finish_schema6_log_worker(
                logger_ref,
                flags,
                final_bundle=final_bundle,
                logger_worker=logger_worker,
                monotonic=monotonic,
            )
        else:
            logger_completion = {
                "schema_version": completion_schema,
                "required": False,
                "use_wandb": False,
                "request_sha256": None,
                "ack_verified": False,
                "private_markers_cleaned": True,
                "policy_version": actor_evidence[
                    "voc_actor_policy_version"
                ],
                "state_sha256": actor_evidence[
                    "voc_actor_policy_state_sha256"
                ],
                "publication_history_sha256": actor_evidence[
                    "voc_actor_policy_publication_history_sha256"
                ],
                "checkpoint_files": final_bundle[
                    "completion_evidence"
                ]["checkpoint_files"],
            }
        completion_kwargs = {}
        if (
            type(gate_schema) is int
            and gate_schema == util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
        ):
            completion_kwargs = {
                "completion_schema_version": completion_schema,
                "gate_schema": gate_schema,
            }
        return util.write_run_completion(
            flags.ckpdir,
            expected_evidence=final_bundle["completion_evidence"],
            actor_policy_logger_completion=logger_completion,
            validated_actor_policy=actor_evidence,
            **completion_kwargs,
        )
    payload = util.write_run_completion(flags.ckpdir)
    if flags.use_wandb:
        ray.get(logger_ref)
    return payload


def validate_schema6_worker_results(return_codes, expected_count):
    if (
        type(return_codes) is not list
        or len(return_codes) != int(expected_count)
        or any(type(value) is not bool or value is not True for value in return_codes)
    ):
        raise RuntimeError(
            "schema-6 self-play workers must each return exactly True: "
            f"return_codes={return_codes!r}"
        )
    return True

if __name__ == "__main__":
    logger = util.logger()
    logger.info("Initializing...")

    st_time = time.time()
    flags = util.create_setting()
    if bool(getattr(flags, "voc_actor_policy_barrier_runtime", False)):
        util.validate_schema6_fresh_run_directory(flags.ckpdir)
    else:
        util.clear_run_completion(flags.ckpdir)
    set_seed(flags.base_seed)
    logger.info("Set all random seeds to %d", flags.base_seed)

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
    util.validate_voc_actor_policy_topology(flags)
    strict_ray_options = (
        {"max_restarts": 0, "max_task_retries": 0}
        if bool(getattr(flags, "voc_actor_policy_barrier_runtime", False))
        else {}
    )
    if flags.parallel_actor:
        actor_buffer = ActorBuffer.options(
            num_cpus=1, **strict_ray_options
        ).remote(
            batch_size=flags.actor_batch_size,
            buffer_save_size=flags.buffer_save_size if hasattr(flags, 'buffer_save_size') else 1
        ) 
        actor_param_buffer = GeneralBuffer.options(
            num_cpus=1, **strict_ray_options
        ).remote()
    else:
        actor_buffer = None
        actor_param_buffer = None
    
    ray_obj_env = ray_init(flags=flags, save_flags=False, **vars(flags))
    ray_obj_env["actor_param_buffer"] = actor_param_buffer
    ray_obj_actor = {"actor_buffer": actor_buffer,
                     "actor_param_buffer": actor_param_buffer,
                     # Frozen behavioral planners refresh from the same
                     # authoritative ModelNet weights as self-play workers.
                     "model_param_buffer": ray_obj_env["param_buffer"]}

    if not flags.train_actor: 
        self_play_buffer = SelfPlayBuffer.options(num_cpus=1).remote(flags=flags)
        ray_obj_actor["self_play_buffer"] = self_play_buffer

    self_play_workers = []
    self_play_workers.extend(
        [
            SelfPlayWorker.options(
                num_cpus=1,
                num_gpus=flags.gpu_self_play,
                **strict_ray_options,
            ).remote(
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
        log_worker = LogWorker.options(
            num_cpus=1, num_gpus=0, **strict_ray_options
        ).remote(flags)
        r_log_worker = log_worker.start.remote()

    if bool(getattr(flags, "voc_actor_policy_barrier_runtime", False)):
        return_codes = wait_for_schema6_workers(
            r_worker,
            actor_param_buffer,
            worker_handles=self_play_workers,
            logger_ref=r_log_worker if flags.use_wandb else None,
            timeout_s=flags.voc_actor_policy_barrier_timeout_s,
        )
    else:
        return_codes = ray.get(r_worker)
    if bool(getattr(flags, "voc_actor_policy_barrier_runtime", False)):
        validate_schema6_worker_results(return_codes, flags.self_play_n)
    elif not all(return_codes):
        raise RuntimeError(f"self-play/learner failure: return_codes={return_codes}")
    finalize_run(
        flags,
        logger_ref=r_log_worker if flags.use_wandb else None,
        logger_worker=log_worker if flags.use_wandb else None,
    )
    logger.info("Time required: %fs" % (time.time() - st_time))
