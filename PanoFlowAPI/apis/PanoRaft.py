import torch
import argparse
from PanoFlowAPI.model.panoflow_raft import PanoRAFT
import sys

sys.path.append('../')
from PanoFlowAPI.utils.padder import InputPadder
from PanoFlowAPI.utils.pano_vis import better_flow_to_image

class PanoRAFTAPI:
    def __init__(self, 
                 device='cuda:0' if torch.cuda.is_available() else 'cpu',
                 model_path='../ckpt/PanoFlow-RAFT-wo-CFE.pth',
                 num_iter=12):

        self.args = self._set_args(num_iter)

        self.device = device
        self.model_path = model_path
        self._load_model()

    def estimate_flow(self, img1, img2):
        '''
        img1: [N, C, H, W]
        img2: [N, C, H, W]
        return: [H, W, 2] flow field
        '''

        return self._estimate_flow(img1, img2)

    def estimate_flow_cfe(self, img1, img2): 
        '''
        img1: [N, C, H, W]
        img2: [N, C, H, W]
        return: [H, W, 2] flow field
        '''

        return self._estimate_flow_cfe(img1, img2)

    def _load_model(self):
        self.model = PanoRAFT(self.args)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def _release_model(self):
        del self.model
        torch.cuda.empty_cache()


    def _set_args(self, num_iter):
        parser = argparse.ArgumentParser(description='tmp')
        args = parser.parse_args()
        args.dataset = None
        args.train = False
        args.eval_iters = num_iter

        return args

    def _prepare_img_pair(self, img1, img2):

        img1, img2 = img1.clone().detach().to(self.device), img2.clone().detach().to(self.device)

        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        image_pair = torch.stack((img1, img2))

        del img1, img2
        return image_pair

    def _estimate_flow(self, img1, img2):

        with torch.no_grad():

            padder = InputPadder(img1.shape)

            image_pair = self._prepare_img_pair(img1, img2)
            _, flow_pr = self.model._model(image_pair, test_mode=True)
            
            flow = padder.unpad(flow_pr).permute(0, 2, 3, 1)

            return flow

    def _estimate_flow_cfe(self, img1, img2):

        with torch.no_grad():

            padder = InputPadder(img1.shape)
            image_pair = self._prepare_img_pair(img1, img2)

            # generate fmaps
            fmap1, fmap2, cnet1 = self.model._model(image_pair, test_mode=True, gen_fmap=True)

            img_A1 = fmap1[:, :, :, 0:fmap1.shape[3] // 2]
            img_B1 = fmap1[:, :, :, fmap1.shape[3] // 2:]
            img_A2 = fmap2[:, :, :, 0:fmap2.shape[3] // 2]
            img_B2 = fmap2[:, :, :, fmap2.shape[3] // 2:]

            cnet_A1 = cnet1[:, :, :, 0:fmap1.shape[3] // 2]
            cnet_B1 = cnet1[:, :, :, fmap1.shape[3] // 2:]

            # prepare fmap pairs #
            img11 = torch.cat([img_B1, img_A1], dim=3)
            img21 = torch.cat([img_B2, img_A2], dim=3)
            cnet11 = torch.cat([cnet_B1, cnet_A1], dim=3)
            img_pair_B1A1 = torch.stack((img11, img21, cnet11))

            img12 = torch.cat([img_A1, img_B1], dim=3)
            img22 = torch.cat([img_A2, img_B2], dim=3)
            cnet12 = torch.cat([cnet_A1, cnet_B1], dim=3)
            img_pair_A1B1 = torch.stack((img12, img22, cnet12))

            # flow prediction #
            # skip encoder

            _, flow_pr_B1A1 = self.model._model(img_pair_B1A1, test_mode=True, skip_encode=True)

            _, flow_pr_A1B1 = self.model._model(img_pair_A1B1, test_mode=True, skip_encode=True)

            flow_pr_A1 = flow_pr_B1A1[:, :, :, flow_pr_B1A1.shape[3] // 2:]
            flow_pr_A2 = flow_pr_A1B1[:, :, :, 0:flow_pr_A1B1.shape[3] // 2]

            flow_pr_A = torch.minimum(flow_pr_A1, flow_pr_A2)

            flow_pr_B1 = flow_pr_B1A1[:, :, :, 0:flow_pr_B1A1.shape[3] // 2]
            flow_pr_B2 = flow_pr_A1B1[:, :, :, flow_pr_A1B1.shape[3] // 2:]

            flow_pr_B = torch.minimum(flow_pr_B1, flow_pr_B2)

            # all
            flow_pr = torch.cat([flow_pr_A, flow_pr_B], dim=3)
            flow_pr[:, :, :, flow_pr.shape[3] // 2] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) + 1]
            flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 1] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 2]

            flow = padder.unpad(flow_pr).permute(0, 2, 3, 1)

            return flow

    def flow2img(self, flow, alpha=0.1, max_flow=25):

        flow_img_list = []

        for i in range(flow.shape[0]):
            flow_img = better_flow_to_image(flow[i].detach().cpu().numpy(), alpha, max_flow)
            flow_img = torch.from_numpy(flow_img)
            flow_img_list.append(flow_img)

        flow_img = torch.stack(flow_img_list, 0)
        return flow_img





