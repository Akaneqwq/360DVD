from time import sleep
from PanoFlowAPI.apis.PanoRaft import PanoRAFTAPI

from PIL import Image
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import random
import torch
import argparse


def abstract_flow(train_video_path, flow_train_video_path, device):

    flow_estimater = PanoRAFTAPI(
        device=device, model_path="./ckpt/PanoFlow-RAFT-wo-CFE.pth"
    )

    cap = cv2.VideoCapture(train_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (1024, 512)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(flow_train_video_path, fourcc, fps, size)

    last_flow_img = None
    last_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            out.write(last_flow_img)
            break
        frame = np.array(frame, dtype=np.float32)
        frame = cv2.resize(frame, (1024, 512))
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
        if last_frame is not None:
            flow = flow_estimater.estimate_flow_cfe(last_frame, frame)
            flow_img = flow_estimater.flow2img(flow, alpha=0.1, max_flow=25)
            flow_img = flow_img[0].numpy()
            flow_img = 255 - flow_img
            last_flow_img = flow_img
            out.write(flow_img)
        last_frame = frame
    cap.release()
    out.release()


if __name__ == "__main__":
    gpus_list = [0, 1, 2, 3]
    cuda_devices = ["cuda:" + str(gpu) for gpu in gpus_list]

    train_video_dir = "datasets/WEB360/videos_512x1024x100"
    flow_train_video_dir = "datasets/WEB360/flows_512x1024x100"
    # make directory
    os.makedirs(flow_train_video_dir, exist_ok=True)

    train_video_names = os.listdir(train_video_dir)
    train_video_names.sort()
    print(len(train_video_names))

    for start in range(0, len(train_video_names), 50):
        end = start + 50
        if end > len(train_video_names):
            end = len(train_video_names)
        for train_video_name in train_video_names[start:end]:
            print(train_video_name)
            train_video_path = os.path.join(train_video_dir, train_video_name)
            flow_train_video_path = os.path.join(flow_train_video_dir, train_video_name)

            device = random.choice(cuda_devices)
            subprocess = mp.Process(
                target=abstract_flow,
                args=(train_video_path, flow_train_video_path, device),
            )
            subprocess.start()
        subprocess.join()
        print(start, end)
    print("done")
