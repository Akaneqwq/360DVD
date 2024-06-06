import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from typing import Dict, Tuple
from decord import VideoReader
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from dvd360.data.dataset import WEB360_flow
from dvd360.models.unet import UNet3DConditionModel
from dvd360.pipelines.pipeline_animation import AnimationPipeline
from dvd360.utils.util import save_videos_grid, zero_rank_print
from dvd360.models.adapter360 import Adapter360 as Adapter


def init_dist(launcher="slurm", backend="nccl", port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(
            f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}"
        )

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


def main(
    name: str,
    use_wandb: bool,
    launcher: str,
    output_dir: str,
    pretrained_model_path: str,
    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs=None,
    motion_module_checkpoint_path: str = "",
    max_train_epoch: int = -1,
    max_train_steps: int = 100000,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),
    learning_rate: float = 1e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",
    num_workers: int = 16,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")
    # Initialize distributed training
    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0
    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = (
        "debug"
        if is_debug
        else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    )
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="360DVD", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )

    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
    )

    # Load 360Adapter
    motion_adapter = Adapter(
        cin=64 * 3,
        channels=[320, 640, 1280, 1280],
        nums_rb=1,
        ksize=1,
        sk=True,
        use_conv=False,
        frame_num=train_data.sample_n_frames,
    )
    motion_adapter.dtype = torch.float32

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path:
            zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = (
            unet_checkpoint_path["state_dict"]
            if "state_dict" in unet_checkpoint_path
            else unet_checkpoint_path
        )
        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        del state_dict, m, u

    # Load pretrained motion module weights
    if motion_module_checkpoint_path != "":
        print(f"load motion module from {motion_module_checkpoint_path}")
        motion_module_state_dict = torch.load(
            motion_module_checkpoint_path, map_location="cpu"
        )
        motion_module_state_dict = (
            motion_module_state_dict["state_dict"]
            if "state_dict" in motion_module_state_dict
            else motion_module_state_dict
        )
        unet_state_dict = unet.state_dict()
        unet_state_dict.update(
            {
                name: param
                for name, param in motion_module_state_dict.items()
                if "motion_modules." in name
            }
        )
        m, u = unet.load_state_dict(unet_state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        del motion_module_state_dict, m, u

    # Freeze vae, text_encoder, unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Set motion adapter trainable parameters
    motion_adapter.requires_grad_(True)

    trainable_params = list(
        filter(lambda p: p.requires_grad, motion_adapter.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(
            f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M"
        )

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    unet.to(local_rank)
    motion_adapter = motion_adapter.to(local_rank)

    # Get the training dataset
    train_dataset = WEB360_flow(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * num_processes
        )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    validation_pipeline = AnimationPipeline(
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=noise_scheduler,
        motion_adapter=motion_adapter,
    ).to("cuda")
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    motion_adapter = DDP(
        motion_adapter, device_ids=[local_rank], output_device=local_rank
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps), disable=not is_main_process
    )
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        motion_adapter.train()

        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch["text"] = [
                    name if random.random() > cfg_random_null_text_ratio else ""
                    for name in batch["text"]
                ]

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, flow_values, texts = (
                    batch["pixel_values"].cpu(),
                    batch["flow_values"].cpu(),
                    batch["text"],
                )

                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                flow_values = rearrange(flow_values, "b f c h w -> b c f h w")
                for idx, (pixel_value, flow_value, text) in enumerate(
                    zip(pixel_values, flow_values, texts)
                ):
                    pixel_value = pixel_value[None, ...]
                    flow_value = flow_value[None, ...]
                    save_videos_grid(
                        pixel_value,
                        f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif",
                        rescale=True,
                    )
                    save_videos_grid(
                        flow_value,
                        f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}_flow.gif",
                        rescale=True,
                    )

            ### >>>> Training >>>> ###

            # zero the flow with a probability of 0.2
            flow_values = (
                batch["flow_values"]
                if random.random() > 0.2
                else torch.zeros_like(batch["flow_values"])
            )
            flow_values = flow_values.to(local_rank)

            # Convert videos to latent space
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]

            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch["text"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                flow_values = rearrange(flow_values, "b f c h w -> (b f) c h w")
                features_adapter = motion_adapter(flow_values)
                for i in range(len(features_adapter)):
                    features_adapter[i] = rearrange(
                        features_adapter[i], "(b f) c h w -> b c f h w", f=video_length
                    )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    features_adapter=features_adapter,
                ).sample

                # Latitude-aware loss
                h = model_pred.shape[-2]
                w = model_pred.shape[-1]
                weight_matrix = torch.zeros((h, w))
                for i in range(h):
                    for j in range(w):
                        weight_matrix[i, j] = abs(
                            torch.cos(
                                ((2 * i - h + 1) / (2 * h)) * torch.tensor(math.pi)
                            )
                        )
                weight_matrix = weight_matrix.to(model_pred.device)
                loss = torch.mean(
                    weight_matrix * (model_pred.float() - target.float()) ** 2
                )
                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    motion_adapter.parameters(), max_grad_norm
                )
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(
                    motion_adapter.parameters(), max_grad_norm
                )
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< Training <<<< ###

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            # Save checkpoint
            if is_main_process and (
                global_step % checkpointing_steps == 0
                or step == len(train_dataloader) - 1
            ):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": motion_adapter.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(
                        state_dict,
                        os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"),
                    )
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (
                global_step % validation_steps == 0
                or global_step in validation_steps_tuple
            ):
                with torch.no_grad():
                    samples = []

                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(global_seed)

                    height = (
                        train_data.sample_size[0]
                        if not isinstance(train_data.sample_size, int)
                        else train_data.sample_size
                    )
                    width = (
                        train_data.sample_size[1]
                        if not isinstance(train_data.sample_size, int)
                        else train_data.sample_size * 2
                    )

                    prompts = (
                        validation_data.prompts[:2]
                        if global_step < 1000
                        else validation_data.prompts
                    )

                    # flow_video = torch.zeros(
                    #     1, train_data.sample_n_frames, 3, height, width
                    # ).to(latents.device)

                    video_reader = VideoReader(
                        "./datasets/WEB360/flows_512x1024x100/100001.mp4",
                        width=width,
                        height=height,
                    )
                    flow_video = video_reader.get_batch(
                        np.linspace(
                            0,
                            train_data.sample_n_frames,
                            train_data.sample_n_frames,
                            dtype=int,
                        )
                    )
                    flow_video = (
                        torch.from_numpy(
                            flow_video.asnumpy().astype(np.float32) / 255.0
                        )
                        .unsqueeze(0)
                        .permute(0, 1, 4, 2, 3)
                        .contiguous()
                        .to(latents.device)
                    )
                    flow_video = transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                    )(flow_video)

                    for idx, prompt in enumerate(prompts):
                        sample = validation_pipeline(
                            prompt,
                            flow=flow_video,
                            generator=generator,
                            video_length=train_data.sample_n_frames,
                            height=height,
                            width=width,
                            **validation_data,
                        ).videos
                        save_videos_grid(
                            sample,
                            f"{output_dir}/samples/sample-{global_step}/{idx}.gif",
                        )
                        samples.append(sample)

                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)

                    logging.info(f"Saved samples to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch"
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
