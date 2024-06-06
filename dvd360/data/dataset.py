import os, csv, random
import numpy as np
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from dvd360.utils.util import zero_rank_print


class WEB360_flow(Dataset):
    def __init__(
        self,
        csv_path,
        video_folder,
        flow_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
    ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder = video_folder
        self.flow_folder = flow_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size * 2)
        )

        self.sample_size = sample_size

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name = video_dict["videoid"], video_dict["name"]

        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        flow_dir = os.path.join(self.flow_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        flow_reader = VideoReader(flow_dir)
        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_stride + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        )

        pixel_values = (
            torch.from_numpy(video_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0

        # shape (n_frames, c, h, w)
        flow_values = (
            torch.from_numpy(flow_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        flow_values = flow_values / 255.0

        del video_reader, flow_reader

        return pixel_values, flow_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, flow_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        # resize and augmentation
        pixel_values = transforms.Resize(self.sample_size)(pixel_values)
        flow_values = transforms.Resize(self.sample_size)(flow_values)
        values_concat = torch.cat([pixel_values, flow_values], dim=0)
        values_concat = transforms.RandomHorizontalFlip(0.5)(values_concat)

        # latent rotation mechanism of a random angle during training
        shift = random.randint(0, pixel_values.size(-1))
        values_concat = torch.roll(shifts=shift, input=values_concat, dims=-1)

        pixel_values = values_concat[: self.sample_n_frames]
        flow_values = values_concat[self.sample_n_frames :]

        # normalization
        pixel_values = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )(pixel_values)
        flow_values = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )(flow_values)
        sample = dict(pixel_values=pixel_values, flow_values=flow_values, text=name)

        return sample
