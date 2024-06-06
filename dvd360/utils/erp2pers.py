from utils import make_coord, gridy2gridx_erp2pers
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os
from multiprocessing import Pool, cpu_count


def erp_to_pers(erp_path, h_pers, 
                        w_pers, 
                        fov = 90,
                        theta = 0,
                        phi = 0,
                        inter_mode = 'bicubic'):

    img = Image.open(erp_path)
    img_array = np.array(img)/255.0

    img_tensor = torch.from_numpy(img_array).float().permute(2,0,1).unsqueeze(0)
    img_shape = img_tensor.shape[-2:]

    H, W = img_shape
    grid_pers_init = make_coord((h_pers, w_pers))
    grid_pers, mask = gridy2gridx_erp2pers(grid_pers_init, h_pers, w_pers, H, W, fov, theta, phi)
    mask = mask.view(h_pers, w_pers, 1).permute(2, 0, 1).cpu()

    grid_pers = grid_pers.flip(-1).unsqueeze(0).unsqueeze(1)

    res = F.grid_sample(img_tensor, grid_pers, mode=inter_mode, padding_mode='border', align_corners=False)[:, :, 0, :]
    res = res.clamp(0, 1).view(3, h_pers, w_pers).cpu()

    return res.numpy() * 255.


def save_numpy_img(numpy_array, save_path):
    im = Image.fromarray(np.uint8(numpy_array.transpose(1, 2, 0)))
    im.save(save_path)


def process_single_image(img_path, save_path, params_list):

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_folder = os.path.join(save_path, img_name)

    for params in params_list:
        h_pers, w_pers, fov, theta, phi = params
        wfov, hfov = fov, int(h_pers / w_pers * fov)
        converted_img_np = erp_to_pers(img_path, h_pers, w_pers, fov, theta, phi)

        dest_folder = os.path.join(img_folder, f"{h_pers}_{wfov}_{hfov}")
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        dest_path = os.path.join(dest_folder, f"pers_{phi}_{theta}.png")
        save_numpy_img(converted_img_np, dest_path)


def process_images(folder_path, save_path, params_list):
    img_paths = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith('.png')]

    with Pool(cpu_count()//2) as p:
        p.starmap(process_single_image, [(img_path, save_path, params_list) for img_path in img_paths])


if __name__ == '__main__':
    folder_path = "datasets/WEB360/frames_erp"

    save_path = "datasets/WEB360/frames_pers"

    w_pers_size = 512
    h_pers_size = 512
    wFOV = 90

    params_list = [
        # H_PERS, W_PERS, FOV, Theta, Phi
        (h_pers_size, w_pers_size, wFOV, 0, 0),
        (h_pers_size, w_pers_size, wFOV, 90, 0),
        (h_pers_size, w_pers_size, wFOV, 180, 0),
        (h_pers_size, w_pers_size, wFOV, -90, 0),
    ]

    process_images(folder_path, save_path, params_list)