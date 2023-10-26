import os
import random
import math
import pandas as pd
import av
import cv2
import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
import torchvision.transforms._transforms_video as transforms_video
from torchvision.transforms.functional import to_tensor
from collections import OrderedDict
import time
import csv

class CharadesDataset(Dataset):
    """
    Charades Dataset.
    Assumes Charades data is structured as follows.
    Charades_dataset/
        videos/
            video-id.mp4
            ...
        charades_v1.1_train.csv
        charades_v1.1_test.csv
    """
    def __init__(self,
                width=512,
                height=512,
                n_sample_frames=8,
                dataset_set="train",
                sample_frame_rate=2,
                sample_start_idx=0,
                accelerator=None,
                ):        
        try:
            host_gpu_num = accelerator.num_processes
            host_num = 1
            all_rank = host_gpu_num * host_num
            global_rank = accelerator.local_process_index
        except:
            pass
        print('dataset rank:', global_rank, ' / ', all_rank, ' ')
        
        self.data_dir = "/content/charades_dataset/"
        self.csv_path = "/content/charades_dataset/charades-test-v1-8-videos.csv"

        self.global_rank = global_rank
        self.all_rank = all_rank
        self.video_length = n_sample_frames
        self.resolution = [width, height]
        self.frame_stride = sample_frame_rate
        self.load_raw_resolution = False
        self.fps_max = None
        self.load_resize_keep_ratio = False

        self._load_metadata()

        self.spatial_transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms_video.CenterCropVideo(self.resolution),
        ])

    def _load_metadata(self):
        self.metadata = []

        with open(self.csv_path, 'r', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print(row)
                video_id = row['id']
                script = row['script']
                descriptions = row['descriptions']
                actions = row['actions']
                self.metadata.append([video_id, script, descriptions, actions])

    def _get_video_path(self, sample):
        video_id = sample[0]
        video_path = os.path.join(self.data_dir, 'videos', f'{video_id}.mp4')
        return video_path
    
    def __getitem__(self, index):
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = self._get_video_path(sample)

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue

            fps_ori = video_reader.get_avg_fps()

            fs = self.frame_stride
            allf = len(video_reader)
            if self.frame_stride != 1:
                all_frames = list(range(0, len(video_reader), self.frame_stride))
                if len(all_frames) < self.video_length:
                    fs = len(video_reader) // self.video_length
                    assert(fs != 0)
                    all_frames = list(range(0, len(video_reader), fs))
            else:
                all_frames = list(range(len(video_reader)))

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        assert(frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [t, h, w, c] -> [c, t, h, w]

        frames = self.spatial_transform(frames)
        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = frames.byte()

        # fps
        fps_clip = fps_ori // self.frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        # caption
        caption = sample[1]  # You can use script, descriptions, or actions as captions

        frames = frames.permute(1, 0, 2, 3)
        skeleton_final = torch.zeros_like(frames).byte()
        frames = (frames / 127.5 - 1.0)
        skeleton_final = (skeleton_final / 127.5 - 1.0)
        example = {'pixel_values': frames, 'sentence': caption, 'pose': skeleton_final}

        return example

    def __len__(self):
        return len(self.metadata)
