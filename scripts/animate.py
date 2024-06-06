import os
import torch
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path
from decord import VideoReader
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
from diffusers.utils.import_utils import is_xformers_available

from dvd360.models.unet import UNet3DConditionModel
from dvd360.pipelines.pipeline_animation import AnimationPipeline
from dvd360.utils.util import save_videos_grid
from dvd360.utils.util import load_weights
from dvd360.models.adapter360 import Adapter360 as Adapter


def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config = OmegaConf.load(args.config)
    samples = []

    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        motion_modules = model_config.motion_module
        motion_modules = (
            [motion_modules]
            if isinstance(motion_modules, str)
            else list(motion_modules)
        )
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(
                model_config.get("inference_config", args.inference_config)
            )

            ### >>> create validation pipeline >>> ###
            tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_path, subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_path, subfolder="text_encoder"
            )
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_path, subfolder="vae"
            )
            unet = UNet3DConditionModel.from_pretrained_2d(
                args.pretrained_model_path,
                subfolder="unet",
                unet_additional_kwargs=OmegaConf.to_container(
                    inference_config.unet_additional_kwargs
                ),
            )

            # Load 360Adapter
            motion_adapter = Adapter(
                cin=64 * 3,
                channels=[320, 640, 1280, 1280],
                nums_rb=1,
                ksize=1,
                sk=True,
                use_conv=False,
                frame_num=args.L,
            ).to("cuda")
            motion_adapter.dtype = torch.float32
            motion_adapter.device = "cuda"

            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                assert False

            pipeline = AnimationPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                motion_adapter=motion_adapter,
                scheduler=DDIMScheduler(
                    **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
                ),
            ).to("cuda")

            pipeline = load_weights(
                pipeline,
                # motion module
                motion_module_path=motion_module,
                motion_module_lora_configs=model_config.get(
                    "motion_module_lora_configs", []
                ),
                # image layers
                dreambooth_model_path=model_config.get("dreambooth_path", ""),
                lora_model_path=model_config.get("lora_model_path", ""),
                lora_alpha=model_config.get("lora_alpha", 1.0),
                # motion adapter
                motion_adapter=model_config.get("motion_adapter", ""),
            ).to("cuda")

            prompts = model_config.prompt
            n_prompts = (
                list(model_config.n_prompt) * len(prompts)
                if len(model_config.n_prompt) == 1
                else model_config.n_prompt
            )

            random_seeds = model_config.get("seed", [-1])
            random_seeds = (
                [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            )
            random_seeds = (
                random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            )

            flows_path = model_config.flow
            flows_path = (
                list(model_config.flow) * len(prompts)
                if len(model_config.flow) == 1
                else model_config.flow
            )
            flow_videos = []
            for flow_path in flows_path:
                video_reader = VideoReader(flow_path, width=args.W, height=args.H)
                flow_video = video_reader.get_batch(np.linspace(0, 16, 16, dtype=int))
                flow_video = (
                    torch.from_numpy(flow_video.asnumpy().astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .permute(0, 1, 4, 2, 3)
                    .contiguous()
                    .to("cuda")
                )
                flow_video = transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                )(flow_video)
                flow_videos.append(flow_video)

            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed, flow_video) in enumerate(
                zip(prompts, n_prompts, random_seeds, flow_videos)
            ):

                # manually set random seed for reproduction
                if random_seed != -1:
                    torch.manual_seed(random_seed)
                else:
                    torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())

                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    flow=flow_video,
                    negative_prompt=n_prompt,
                    num_inference_steps=model_config.steps,
                    guidance_scale=model_config.guidance_scale,
                    width=args.W,
                    height=args.H,
                    video_length=args.L,
                ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                print(f"save to {savedir}/sample/{prompt}.gif")

                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="ckpts/StableDiffusion/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--inference_config", type=str, default="configs/inference/inference-v1.yaml"
    )
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
