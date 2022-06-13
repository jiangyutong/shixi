# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, utils_io
from model import resnet as pose_resnet
import pdb
import os
import argparse
from utils.vis2d import draw2Dpred_and_gt
import cv2
from utils import AverageMeter
from easydict import EasyDict as edict
import yaml
import numpy as np


LOGGER = ConsoleLogger("Demo", 'test')

def parse_args():
    parser = argparse.ArgumentParser(description="demo script")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--data', default='test', type=str) # "train", "val", "test"
    args = parser.parse_args()

    return args


def main():
    """Main"""
    args = parse_args()
    LOGGER.info('Starting demo...')
    device = torch.device(f"cuda:{args.gpu}")
    LOGGER.info(args)

    # ------------------- Data loader -------------------

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor

    # let's load data from validation set as example
    data = Mocap(
        config.dataset[args.data],
        SetType.VAL,
        transform=data_transform)
    data_loader = DataLoader(
        data,
        batch_size=2,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    # ------------------- Model -------------------
    with open('model/model.yaml') as fin:
        model_cfg = edict(yaml.safe_load(fin))
    resnet = pose_resnet.get_pose_net(model_cfg, False)
    resnet.cuda(device)
    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise ValueError(f"No checkpoint found at {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        resnet.load_state_dict(checkpoint['resnet_state_dict'])
    else:
        raise ValueError("No checkpoint!")

    resnet.eval()
    Loss2D = nn.MSELoss()

    # ------------------- Read dataset frames -------------------
    losses = AverageMeter()
    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):

            print('Iteration: {}'.format(it))
            print('Images: {}'.format(img.shape))
            print('p2ds: {}'.format(p2d.shape))
            print('p3ds: {}'.format(p3d.shape))
            print('Actions: {}'.format(action))

            heatmap = heatmap.to(device)
            img = img.to(device)

            heatmap_hat = resnet(img)
            loss = Loss2D(heatmap_hat, heatmap)
            losses.update(loss.item(), img.size(0))

            # ------------------- visualization -------------------
            if it < 32:
                img_grid = draw2Dpred_and_gt(img, heatmap, (368,368))  # tensor
                img_grid = img_grid.numpy().transpose(1, 2, 0)
                cv2.imwrite(os.path.join(LOGGER.logfile_dir, f'gt_{it}.jpg'), img_grid)

                img_grid_hat = draw2Dpred_and_gt(img, heatmap_hat, (368,368), p2d.clone())  # tensor
                img_grid_hat = img_grid_hat.numpy().transpose(1,2,0)
                cv2.imwrite(os.path.join(LOGGER.logfile_dir, f'pred_{it}.jpg'), img_grid_hat)


    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')


if __name__ == "__main__":
    main()
