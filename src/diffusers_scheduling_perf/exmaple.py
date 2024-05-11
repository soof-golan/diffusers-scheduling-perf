import gc
import logging
from typing import Optional
import torch
from diffusers import SchedulerMixin, StableDiffusionPipeline
from tqdm import tqdm

from diffusers_scheduling_perf.utils import (
    synchronize_device_and_clear_cache,
    DiskCache,
    DTypeStr,
    DeviceStr,
    _dtype_map,
    DEFAULT_PROMPT,
)

_logger = logging.getLogger(__name__)


def load_model(
    *,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    device: DeviceStr = "cpu",
    torch_dtype: DTypeStr = "float32",
    scheduler: Optional[SchedulerMixin] = None,
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=_dtype_map[torch_dtype],
    )
    pipe = pipe.to(device=torch.device(device))

    # Patch Scheduler
    if scheduler is not None:
        pipe.scheduler = scheduler

    return pipe


def batch_size_before_oom(
    *,
    model_name: str,
    device: DeviceStr,
    dtype: DTypeStr,
    num_images: int,
) -> int:
    _logger.info(
        "Determining batch size before OOM for model=%s, device=%s, dtype=%s",
        model_name,
        device,
        dtype,
    )
    cache = DiskCache("batch_size_cache.db")
    if device == "cuda":
        name = torch.cuda.get_device_name(torch.device(device))
    else:
        name = device
    base_key = f"{model_name=}-{name=}-{dtype=}-{num_images=}-{device=}"
    try:
        return cache[base_key]
    except KeyError:
        pass

    powers_of_two = [1 << i for i in range(6, 0, -1)]
    for batch_size in (pb := tqdm(powers_of_two, desc="Finding batch size before OOM")):
        pb.set_postfix(batch_size=batch_size)
        key = f"{base_key}-{batch_size}"

        try:
            result = cache[key]
            if result == -1:
                _logger.debug("Skipping %s as it is known to OOM", key)
                continue
            return batch_size
        except KeyError:
            pass

        try:
            infer_with_cleanup(
                model_name=model_name,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                num_inference_steps=2,
                num_images=num_images,
                seed=42,
            )
        except RuntimeError:
            _logger.debug("OOM at %s", key)
            cache[key] = -1
            continue
        cache[base_key] = batch_size
        cache[key] = batch_size
        return batch_size


@torch.inference_mode()
def generate(
    *,
    pipe: StableDiffusionPipeline,
    batch_size: int,
    seed: int = 42,
    num_inference_steps: int = 20,
    num_images: int = 1,
):
    return pipe(
        prompt=[DEFAULT_PROMPT] * batch_size,
        generator=[
            torch.Generator(device=pipe.device).manual_seed(seed)
            for _ in range(batch_size * num_images)
        ],
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        return_dict=False,
    )


@torch.inference_mode()
def infer_with_cleanup(
    model_name: str,
    device: DeviceStr,
    dtype: DTypeStr,
    seed: int,
    batch_size: int,
    num_images: int,
    num_inference_steps,
):
    _logger.info(
        "Inference with model=%s, device=%s, dtype=%s",
        model_name,
        device,
        dtype,
    )
    pipe = load_model(model_name=model_name, device=device, torch_dtype=dtype)
    _logger.info(
        "Running Inference with batch_size=%s, num_images=%s num_inference_steps=%s",
        batch_size,
        num_images,
        num_inference_steps,
    )
    _ = generate(
        pipe=pipe,
        batch_size=batch_size,
        seed=seed,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
    )
    synchronize_device_and_clear_cache(device)
    del pipe
    gc.collect()
    synchronize_device_and_clear_cache(device)


def sanity_check(
    *,
    model_name: str,
    device: DeviceStr,
    dtype: DTypeStr,
    seed: int,
    num_images: int,
) -> bool:
    key = f"{model_name=}-{device=}-{dtype=}-{num_images=}"
    _logger.info("Sanity check: Generating %s", key)
    cache = DiskCache("sanity_check_cache.db")
    try:
        is_sane = bool(cache[key])
        _logger.info("found sanity check in cache key=(%s) success=%s", key, is_sane)
        return is_sane
    except KeyError:
        pass
    try:
        infer_with_cleanup(
            model_name=model_name,
            device=device,
            dtype=dtype,
            num_images=num_images,
            seed=seed,
            num_inference_steps=2,
            batch_size=1,
        )
    except Exception as e:
        _logger.exception("Sanity check failed %s", e, exc_info=e)
        cache[key] = 0
        return False
    cache[key] = 1
    return True


@torch.inference_mode()
def run(
    *,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    device: DeviceStr = "cpu",
    dtype: DTypeStr = "float32",
    seed: int = 42,
    num_inference_steps: int = 20,
    num_images: int = 1,
    batch_size: Optional[int] = None,
    timeit_iterations: int = 10,
):
    _logger.info(
        "Running with model=%s, device=%s, dtype=%s",
        model_name,
        device,
        dtype,
    )

    if not sanity_check(
        model_name=model_name,
        device=device,
        dtype=dtype,
        seed=seed,
        num_images=num_images,
    ):
        raise RuntimeError("Sanity check failed")

    batch_size = batch_size or batch_size_before_oom(
        model_name=model_name,
        device=device,
        dtype=dtype,
        num_images=num_images,
    )
    _logger.info(
        "Generation Parameters: seed=%s, num_inference_steps=%s, num_images=%s, batch_size=%s",
        seed,
        num_inference_steps,
        num_images,
        batch_size,
    )

    pipe = load_model(model_name=model_name, device=device, torch_dtype=dtype)
    for _ in tqdm(
        range(timeit_iterations),
        desc="Running Inference",
        total=timeit_iterations,
    ):
        _ = generate(
            pipe=pipe,
            batch_size=batch_size,
            seed=seed,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
        )
