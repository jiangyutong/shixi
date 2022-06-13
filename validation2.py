# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger, AverageMeter
from utils import evaluate, utils_io
import os
import pdb
from utils.loss import HeatmapLoss
import pprint
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import encoder_decoder
import matplotlib
from utils.vis3d import show3Dpose
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
matplotlib.use('TkAgg')

def validate(LOGGER, data_loader, autoencoder, device):


    # ------------------- Loss -------------------


    # ------------------- Evaluation -------------------
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # ------------------- validate -------------------
    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            p3d = p3d.to(device)
            heatmap = heatmap.to(device)
            autoencoder=autoencoder.to(device)
            p3d_hat, heatmap2d_recon = autoencoder(heatmap)

            # Evaluate results using different evaluation metrices
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            ax = plt.subplot(111, projection='3d')
            show3Dpose(y_output, ax, True)
            plt.show()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)

        # ------------------- Save results -------------------

        LOGGER.info('===========Evaluation on Val data==========')
        res = {'FullBody': eval_body.get_results(),
               'UpperBody': eval_upper.get_results(),
               'LowerBody': eval_lower.get_results()}
        LOGGER.info(pprint.pformat(res))

        # utils_io.write_json(os.path.join(LOGGER.logfile_dir, f'eval_val_{epoch}'+'.json'), res)

    return eval_body.get_results()['All']
LOGGER = ConsoleLogger('Test3d', 'test')
data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor


test_data = Mocap(
        config.dataset.val,
        SetType.VAL,
        transform=None)
test_data_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=config.data_loader.shuffle,
    num_workers=8)
autoencoder = encoder_decoder.AutoEncoder()
device = torch.device(f"cuda:{0}")
checkpoint = torch.load("experiments/Train3d/2022-06-06-11-14-46/checkpoints/checkpoint_99.tar", map_location=device)
autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
autoencoder.eval()
validate(LOGGER, test_data_loader, autoencoder, device)
if i < 800:
    ax = plt.subplot(111, projection='3d')
    show3Dpose(y_output, ax, True)
    plt.show()
    plt.savefig(output_dir + str(i).zfill(5) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
    i += 1
    output_dir = "egopose_hm36_trainval/"