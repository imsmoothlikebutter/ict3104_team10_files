import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from followyourpose.models.unet import UNet3DConditionModel
from followyourpose.data.hdvila import HDVilaDataset
from followyourpose.pipelines.pipeline_followyourpose import FollowYourPosePipeline
from followyourpose.util import save_videos_grid, ddim_inversion
from einops import rearrange

import sys
sys.path.append('FollowYourPose')

import cv2
import numpy as np

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    validation_data: Dict,
    validation_steps: int = 100,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    selected_folder: Optional[str] = None,
    skeleton_path: Optional[str] = None,
    save_path: Optional[str] = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()


    # Get the validation pipeline
    validation_pipeline = FollowYourPosePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    unet = accelerator.prepare(unet)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    load_path = None
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            load_path = resume_from_checkpoint
            output_dir = os.path.abspath(os.path.join(resume_from_checkpoint, ".."))
        accelerator.print(f"load from checkpoint {load_path}")
        accelerator.load_state(load_path)

        global_step = int(load_path.split("-")[-1])

                
    if accelerator.is_main_process:
        samples = []
        samplesraw = []
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(seed)

        ddim_inv_latent = None

        from datetime import datetime
        selected_folder = selected_folder

        now = str(datetime.now())
        for idx, prompt in enumerate(validation_data.prompts):
            pipeline_output = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent, skeleton_path=skeleton_path, **validation_data)

            video = pipeline_output.videos
            skeleton = pipeline_output.skeleton

            # Create a list to store the modified frames
            combined_video_frames = []

            # Iterate over each frame in the video
            for frame_index in range(video.shape[0]):
                video_frame = video[frame_index].cpu().numpy()
                skeleton_frame = skeleton[frame_index].cpu().numpy()

                # Assuming skeleton_frame is a binary mask, you can overlay it on the video frame
                combined_frame = cv2.addWeighted(video_frame, 0.5, skeleton_frame, 1, 0)

                combined_video_frames.append(combined_frame)

            # Combine the modified frames into a video
            combined_video = np.array(combined_video_frames)

            # Save the combined video with the skeleton
            combined_video_save_path = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}/sample-{global_step}-{str(seed)}-{now}/{prompt}_with_skeleton.mp4"
            frame_height, frame_width = combined_video[0].shape[:2]
            frame_rate = 30  # You may need to adjust this based on your video frame rate
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(combined_video_save_path, fourcc, frame_rate, (frame_width, frame_height))

            for frame in combined_video:
                out.write(frame)

            out.release()

            samples.append(torch.from_numpy(combined_video))
            samplesraw.append(video)

        # Save the video and skeleton (you can customize the saving logic)
        # video_save_path = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}/sample-{global_step}-{str(seed)}-{now}/{prompt}.gif"
        # skeleton_save_path = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}/sample-{global_step}-{str(seed)}-{now}/{prompt}_skeleton.gif"
        # combined_video_save_path_gif = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}/sample-{global_step}-{str(seed)}-{now}/{prompt}_without_skeleton.gif"
        # combined_video_save_path_gif_super_imposed = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}/sample-{global_step}-{str(seed)}-{now}/{prompt}_with_skeleton.gif"
        # Combine and save all generated videos
        samples = torch.cat(samples)
        samplesraw = torch.cat(samplesraw)
        # save_path = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}_result/sample-{global_step}-{str(seed)}-{now}/{selected_folder}.gif"
        save_path_without_skeleton = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}_result/sample-{global_step}-{str(seed)}-{now}/{selected_folder}_combined.gif"
        save_path_super_imposed = f"/content/FollowYourPose/checkpoints/inference/{selected_folder}_result/sample-{global_step}-{str(seed)}-{now}/{selected_folder}_super_imposed.gif"  # Specify the save path for the combined video
        # save_videos_grid(video, video_save_path)
        # save_videos_grid(skeleton, skeleton_save_path)
        # save_videos_grid(samplesraw, combined_video_save_path_gif)
        # save_videos_grid(samples, combined_video_save_path_gif_super_imposed)
        # save_videos_grid(video, save_path)
        save_videos_grid(samplesraw, save_path_without_skeleton)
        save_videos_grid(samples, save_path_super_imposed)
        logger.info(f"Saved samples to {save_path_super_imposed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--selected_folder", type=str)
    parser.add_argument("--skeleton_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main(**OmegaConf.load(args.config), skeleton_path = args.skeleton_path, save_path = args.save_path, selected_folder = args.selected_folder)
