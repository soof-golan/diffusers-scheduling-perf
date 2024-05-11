import gc
import json
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
    model_name: str,
    device: DeviceStr,
    torch_dtype: DTypeStr,
    compile_unet: bool,
    compile_vae: bool,
    scheduler: Optional[SchedulerMixin] = None,
) -> StableDiffusionPipeline:
    torch_device = torch.device(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=_dtype_map[torch_dtype],
    )
    pipe = pipe.to(device=torch_device)
    pipe.unet = torch.compile(
        pipe.unet,
        disable=not compile_unet or torch_device.type != "cuda",
    )

    pipe.vae = torch.compile(
        pipe.vae,
        disable=not compile_vae or torch_device.type != "cuda",
    )

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
    width: int,
    height: int,
    compile_unet: bool,
    compile_vae: bool,
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
    base_key_data = {
        "model_name": model_name,
        "device": name,
        "dtype": dtype,
        "num_images": num_images,
        "compile_unet": compile_unet,
        "compile_vae": compile_vae,
        "width": width,
        "height": height,
    }
    base_key = json.dumps(base_key_data, sort_keys=True)
    try:
        optimal_batch_size = cache[base_key]
        _logger.info("Cache Hit: batch_size=%s for %s", optimal_batch_size, base_key)
        return optimal_batch_size
    except KeyError:
        pass

    powers_of_two = [1 << i for i in range(8, 0, -1)]
    for batch_size in (pb := tqdm(powers_of_two, desc="Finding batch size before OOM")):
        pb.set_postfix(batch_size=batch_size)
        key_data = base_key_data.copy() | {"batch_size": batch_size}
        key = json.dumps(key_data, sort_keys=True)
        try:
            result = cache[key]
            if result == -1:
                _logger.info("Skipping %s as it is known to OOM", key)
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
                width=width,
                height=height,
                compile_unet=compile_unet,
                compile_vae=compile_vae,
            )
        except RuntimeError:
            _logger.info("OOM at %s", key)
            cache[key] = -1
            continue
        _logger.info("Found largest batch_size=%s for %s", batch_size, key)
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
    width: int = 512,
    height: int = 512,
):
    return pipe(
        prompt=[DEFAULT_PROMPT] * batch_size,
        generator=[
            torch.Generator(device=pipe.device).manual_seed(seed)
            for _ in range(batch_size * num_images)
        ],
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        width=width,
        height=height,
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
    width: int,
    height: int,
    compile_unet: bool = False,
    compile_vae: bool = False,
):
    _logger.info(
        "Inference with model=%s, device=%s, dtype=%s",
        model_name,
        device,
        dtype,
    )
    pipe = load_model(
        model_name=model_name,
        device=device,
        torch_dtype=dtype,
        compile_unet=compile_unet,
        compile_vae=compile_vae,
    )
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
        width=width,
        height=height,
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
    num_images: int,
    width: int,
    height: int,
    compile_unet: bool = False,
    compile_vae: bool = False,
) -> bool:
    key = json.dumps(
        {
            "model_name": model_name,
            "device": device,
            "dtype": dtype,
            "num_images": num_images,
            "width": width,
            "height": height,
            "compile_unet": compile_unet,
            "compile_vae": compile_vae,
        },
        sort_keys=True,
    )
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
            seed=42,
            num_inference_steps=2,
            batch_size=1,
            width=width,
            height=height,
            compile_unet=compile_unet,
            compile_vae=compile_vae,
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
    width: int = 512,
    height: int = 512,
    batch_size: Optional[int] = None,
    timeit_iterations: int = 10,
    compile_unet: bool = False,
    compile_vae: bool = False,
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
        num_images=num_images,
        width=width,
        height=height,
        compile_unet=compile_unet,
        compile_vae=compile_vae,
    ):
        raise RuntimeError("Sanity check failed")

    batch_size = batch_size or batch_size_before_oom(
        model_name=model_name,
        device=device,
        dtype=dtype,
        num_images=num_images,
        width=width,
        height=height,
        compile_unet=compile_unet,
        compile_vae=compile_vae,
    )
    params = {
        "model_name": model_name,
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "num_images": num_images,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "compile_unet": compile_unet,
        "compile_vae": compile_vae,
    }
    _logger.info("Generation Parameters: %s", params)

    pipe = load_model(
        model_name=model_name,
        device=device,
        torch_dtype=dtype,
        compile_unet=compile_unet,
        compile_vae=compile_vae,
    )
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
            width=width,
            height=height,
        )
