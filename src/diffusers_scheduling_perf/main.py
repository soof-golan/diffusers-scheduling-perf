import logging
from argparse import ArgumentParser
from typing import Optional

from diffusers_scheduling_perf.exmaple import run


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model Data type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=Optional[int],
        default=None,
        help="Batch size.",
    )
    parser.add_argument(
        "--timeit-iterations",
        type=int,
        default=10,
        help="Number of iterations for timeit.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    return run(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        num_images=args.num_images,
        batch_size=args.batch_size,
        timeit_iterations=args.timeit_iterations,
    )


if __name__ == "__main__":
    main()
