# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, utils_io
import argparse
import os
from model import encoder_decoder
from utils.vis2d import draw2Dpred_and_gt, save_batch_heatmaps
import cv2
import pdb
import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from utils.vis3d import show3Dpose
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

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
    print('Starting demo...')
    device = torch.device(f"cuda:{args.gpu}")
    LOGGER.info((args))

    # ------------------- Data loader -------------------

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor

    data = Mocap(
        # config.dataset.test,
        config.dataset[args.data],
        SetType.TEST,
        transform=data_transform)
    data_loader = DataLoader(
        data,
        batch_size=16,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    # ------------------- Evaluation -------------------

    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # ------------------- Model -------------------
    autoencoder = encoder_decoder.AutoEncoder()

    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise ValueError(f"No checkpoint found at {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])

    else:
        raise ValueError("No checkpoint!")
    autoencoder.cuda(device)
    autoencoder.eval()


    # ------------------- Read dataset frames -------------------
    fig = plt.figure(figsize=(19.2, 10.8))
    plt.axis('off')
    subplot_idx = 1

    # ------------------- Read dataset frames -------------------
    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):

            print('Iteration: {}'.format(it))
            print('Images: {}'.format(img.shape))
            print('p2ds: {}'.format(p2d.shape))
            print('p3ds: {}'.format(p3d.shape))
            print('Actions: {}'.format(action))

            p3d = p3d.to(device)
            heatmap = heatmap.to(device)

            p3d_hat, heatmap_hat = autoencoder(heatmap)

            # Evaluate results using different evaluation metrices
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)

            # ------------------- Visualize 3D pose -------------------
            if subplot_idx <= 32:
                # ax1 = plt.subplot(gs1[subplot_idx - 1], projection='3d')
                ax1 = fig.add_subplot(4, 8, subplot_idx, projection='3d')
                show3Dpose(p3d[0].cpu().numpy(), ax1, True)

                # Plot 3d gt
                # ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
                ax2 = fig.add_subplot(4, 8, subplot_idx+1, projection='3d')
                show3Dpose(p3d_hat[0].detach().cpu().numpy(), ax2, False)

                subplot_idx += 2
            if subplot_idx == 33:
                plt.savefig(os.path.join(LOGGER.logfile_dir, 'vis.png'))

            # ------------------- Visualize 2D heatmap -------------------
            if it < 32:
                img_grid = draw2Dpred_and_gt(img, heatmap, (368,368))  # tensor
                img_grid_hat = draw2Dpred_and_gt(img, heatmap_hat, (368,368), p2d)  # tensor
                img_grid = img_grid.numpy().transpose(1,2,0)
                img_grid_hat = img_grid_hat.numpy().transpose(1,2,0)
                cv2.imwrite(os.path.join(LOGGER.logfile_dir, f'gt_{it}.jpg'), img_grid)
                cv2.imwrite(os.path.join(LOGGER.logfile_dir, f'pred_{it}.jpg'), img_grid_hat)

                # save_batch_heatmaps(img[0:1], heatmap_hat[0:1], os.path.join(LOGGER.logfile_dir,"pred_combine.jpg"))
                # save_batch_heatmaps(img[0:1], heatmap[0:1], os.path.join(LOGGER.logfile_dir, "gt_combine.jpg"))


    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')
    res = {'FullBody': eval_body.get_results(),
           'UpperBody': eval_upper.get_results(),
           'LowerBody': eval_lower.get_results()}

    LOGGER.info(pprint.pformat(res))

    print('Done.')


if __name__ == "__main__":
    main()
