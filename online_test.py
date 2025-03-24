import argparse
import os

import cv2

from sensors.gsrobotics import gsdevice, gs3drecon
from sensors import ATISensor

import torch
from torch import nn
from torchvision import transforms as pth_transforms
from PIL import Image
import numpy as np
import json

from args import get_parser
from composed_model import ComposedModel

class Predictor:
    def __init__(self):
        parser = get_parser()
        self.args = parser.parse_args()
        ckpt = torch.load(self.args.checkpoint)
        if "config" in ckpt.keys():
            config = argparse.Namespace(**ckpt["config"])
            self.args = parser.parse_args(namespace=config)

        # Load the model
        self.model = ComposedModel(self.args)

        self.model.cuda()
        self.model.eval()
        self.model.load_state_dict(ckpt["state_dict"])


        with open(self.args.dataset_stats) as convert_file:
            dataset_mean_std = json.load(convert_file)
        mean_rgb, std_rgb = dataset_mean_std[self.args.labels_train][self.args.tactile_mode]

        self.transform_rgb = pth_transforms.Compose([
            pth_transforms.Pad([0, 40, 0, 40]),
            pth_transforms.Resize(self.args.crop_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean_rgb, std_rgb),
        ])

        mean_depth = (0.234, 0.234, 0.234)
        std_depth = (0.138, 0.138, 0.138)
        self.transform_depth = pth_transforms.Compose([
            pth_transforms.Pad([0, 40, 0, 40]),
            pth_transforms.Resize(self.args.crop_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean_depth, std_depth),
        ])
        self.max_val_depth, self.min_val_depth = 27.03226967220051, -8.311515796355053
        self.dev, self.depth_nn = self.init_mini()
        for _ in range(50): self.get_tacdepth(self.dev, self.depth_nn)
        bg, _ = self.get_tacdepth(self.dev, self.depth_nn)
        bg = np.array(bg)
        self.bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    def init_mini(self):
        dev = gsdevice.Camera("GelSight Mini")
        net_file_path = 'nnmini.pt'
        GPU = False
        dev.connect()

        ''' Load neural network '''
        net_path = os.path.join("sensors/gsrobotics", net_file_path)
        print('mini net path = ', net_path)

        if GPU:
            gpuorcpu = "cuda"
        else:
            gpuorcpu = "cpu"

        depth_nn = gs3drecon.Reconstruction3D(dev)
        net = depth_nn.load_nn(net_path, gpuorcpu)

        f0 = dev.get_raw_image()
        roi = (0, 0, f0.shape[1], f0.shape[0])

        return dev, depth_nn

    def get_tacdepth(self, dev, nn):
        for _ in range(1):
            f1 = dev.get_image()
            dm = nn.get_depthmap(f1, False)
        return f1, dm


    def eval(self):
        im, depth = self.get_tacdepth(self.dev, self.depth_nn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(Image.fromarray(im) - self.bg + 127)
        image = self.transform_rgb(im)

        force_vector = torch.FloatTensor(self.ft_sensor.get_ft())[:3]

        image = image.cuda(non_blocking=True)
        force_vector = force_vector.cuda(non_blocking=True)

        with torch.no_grad():
            intermediate_output_rgb = self.model.tactile_backbone(image.unsqueeze(0))
            intermediate_output = intermediate_output_rgb
            output = self.model.regressor(intermediate_output)

        loss = torch.sqrt(nn.MSELoss()(output, force_vector))
        loss = loss.cpu()
        force_vector = force_vector.cpu()
        output = output.cpu()
        cv2.imshow('image_rgb_check.png', np.array(im))
        cv2.waitKey(1)
        print(f"Force predicted: {output}, ground truth {force_vector}, error = {loss}")
        return output, force_vector, im, depth

    def predict(self):
        im, depth = self.get_tacdepth(self.dev, self.depth_nn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(Image.fromarray(im) - self.bg + 127)
        image = self.transform_rgb(im)
        image = image.cuda(non_blocking=True)

        with torch.no_grad():
            intermediate_output_rgb = self.model.tactile_backbone(image.unsqueeze(0))
            intermediate_output = intermediate_output_rgb
            output = self.model.regressor(intermediate_output)

        output = output.cpu()
        cv2.imshow('image_rgb_check', np.array(im))
        cv2.waitKey(1)
        print(f"Force predicted: {output}")
        return output, im, depth


if __name__ == '__main__':
    p = Predictor()
    while True:
        p.predict()