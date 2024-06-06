<div align="center">
<h3>[CVPR2024] 360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model</h3>

[Qian Wang](https://akaneqwq.github.io/), [Weiqi Li](https://github.com/lwq20020127/), [Chong Mou](https://github.com/MC-E/), [Xinhua Cheng](https://cxh0519.github.io/), [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University

[![arXiv](https://img.shields.io/badge/arXiv-2401.06578-b31b1b.svg)](https://arxiv.org/abs/2401.06578)
[![Dataset](https://img.shields.io/badge/Dataset-<WEB360>-green.svg)](https://drive.google.com/file/d/1W1eLmaP16GZOeisAR1q-y9JYP9gT1CRs)
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://akaneqwq.github.io/360DVD/)

This repository is the official implementation of 360DVD, a panorama video generation pipeline based on the given prompts and motion conditions. The main idea is to turn a T2V model into a panoramic T2V model through 360-Adapter and 360 Enhancement Techniques.

</div>

## Gallery

We have showcased some regular videos generated by [AnimateDiff](https://github.com/guoyww/AnimateDiff) and panoramic videos generated by 360DVD below. 

More results can be found on our [Project Page](https://akaneqwq.github.io/360DVD/).

<table>
  <tr>
    <td><img src="__assets__/videos/1.gif" alt="AnimateDiff"></td>
    <td><img src="__assets__/videos/1_1.gif" alt="Ours"></td>
    <td><img src="__assets__/videos/2.gif" alt="AnimateDiff"></td>
    <td><img src="__assets__/videos/2_1.gif" alt="Ours"></td>
  </tr>
  <tr>
    <td colspan="2"><center>"the top of a snow covered mountain range, with the sun shining over it"</center></td>
    <td colspan="2"><center>"a view of fireworks exploding in the night sky over a city, as seen from a plane"</center></td>
  </tr>
  <tr>
    <td><img src="__assets__/videos/3.gif" alt="AnimateDiff"></td>
    <td><img src="__assets__/videos/3_1.gif" alt="Ours"></td>
    <td><img src="__assets__/videos/4.gif" alt="AnimateDiff"></td>
    <td><img src="__assets__/videos/4_1.gif" alt="Ours"></td>
  </tr>
  <tr>
    <td colspan="2"><center>"a desert with sand dunes, blue cloudy sky"</center></td>
    <td colspan="2"><center>"the city under cloudy sky, a car driving down the street with buildings"</center></td>
  </tr>
  <tr>
    <td><img src="__assets__/videos/5.gif" alt="AnimateDiff"></td>
    <td><img src="__assets__/videos/5_1.gif" alt="Ours"></td>
    <td><img src="__assets__/videos/6.gif" alt="AnimateDiff"></td>
    <td><img src="__assets__/videos/6_1.gif" alt="Ours"></td>
  </tr>
  <tr>
    <td colspan="2"><center>"a large mountain lake, the lake surrounded by hills and mountains"</center></td>
    <td colspan="2"><center>"a volcano with smoke coming out, mountains under clouds, at sunset"</center></td>
  </tr>
</table>

Model: [Realistic Vision V5.1](https://civitai.com/models/4201/realistic-vision-v20)

## To Do List
- [ ] Release gradio demo
- [x] Release weights
- [x] Release code
- [x] Release paper
- [x] Release dataset

##  Steps for Inference

### Prepare Environment
```
git clone https://github.com/Akaneqwq/360DVD.git
cd 360DVD

conda env create -f environment.yaml
conda activate 360dvd
```

### Download Pretrained Models
```
git lfs install
mkdir -p ckpts/StableDiffusion/
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 ckpts/StableDiffusion/stable-diffusion-v1-5/

bash download_bashscripts/0-MotionModule.sh
bash download_bashscripts/1-360Adapter.sh
bash download_bashscripts/2-RealisticVision.sh
```

### Generate Panorama Videos
```
python -m scripts.animate --config configs/prompts/0-realisticVision.yaml
```
You can write your own config, then update the path and run it again. We strongly recommend using a personalized T2I model, such as Realistic Vision or Lyriel, for a better performance.

## Steps for Training

### Prepare Dataset
You can directly download WEB360 Dataset.
```
bash download_bashscripts/4-WEB360.sh
unzip /datasets/WEB360.zip -d /datasets
```
Or prepare your own dataset consists of panoramic video clips. 

You can use single [BLIP](https://github.com/salesforce/LAVIS) to caption your videos. For more fine-grained results, modify the code provided in `dvd360/utils/erp2pers.py` and `dvd360/utils/360TextFusion.py` to execute the 360 Text Fusion process.


### Extract Motion Information
Download the pretrained model `PanoFlow(RAFT)-wo-CFE.pth` of Panoflow at [weiyun](https://share.weiyun.com/SIpeQTNE), then put it in `PanoFlowAPI/ckpt/` folder and rename it to `PanoFlow-RAFT-wo-CFE.pth`.

Update `scripts/video2flow.py`.
```
gpus_list = [Replace with available GPUs]
train_video_dir = [Replace with the folder path of panoramic videos]
flow_train_video_dir = [Replace with the folder path you want to save flow videos]
```
Then you can run the below command to obtain corresponding flow videos.
```
python -m scripts.video2flow
```

### Configuration
Update data paths in the config `.yaml` files in `configs/training/` folder.
```
train_data:
  csv_path:     [Replace with .csv Annotation File Path]
  video_folder: [Replace with Video Folder Path]
  flow_folder:  [Replace with Flow Folder Path]
```
Other training parameters (lr, epochs, validation settings, etc.) are also included in the config files.

### Training
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/training.yaml
```

## Contact Us
**Qian Wang**: [qianwang@stu.pku.edu.cn](mailto:qianwang@stu.pku.edu.cn)

## Acknowledgements
Codebase built upon [AnimateDiff](https://github.com/guoyww/AnimateDiff), [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) and [Panoflow](https://github.com/MasterHow/PanoFlow).

## BibTeX
```
@article{wang2024360dvd,
  title={360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model},
  author={Qian Wang and Weiqi Li and Chong Mou and Xinhua Cheng and Jian Zhang},
  journal={arXiv preprint arXiv:2401.06578},
  year={2024}
}
```