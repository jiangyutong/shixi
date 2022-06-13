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
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import encoder_decoder
from model import resnet as pose_resnet
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from easydict import EasyDict as edict
import yaml
def validate(LOGGER, data_loader, resnet, autoencoder, device, epoch):


    # ------------------- Loss -------------------

    Loss2D = HeatmapLoss()

    # ------------------- Evaluation -------------------
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # ------------------- validate -------------------
    val_losses = AverageMeter()
    for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
        img = img.to(device)
        p2d = p2d.to(device)
        p3d = p3d.to(device)
        heatmap = heatmap.to(device)
        Loss2D.cuda()
        resnet=resnet.cuda()
        autoencoder=autoencoder.cuda()
        heatmap2d_hat = resnet(img)  # torch.Size([16, 15, 48, 48])
        p3d_hat, heatmap2d_recon = autoencoder(heatmap2d_hat)

        loss2d = Loss2D(heatmap, heatmap2d_hat).mean()


        # Evaluate results using different evaluation metrices
        y_output = p3d_hat.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()

        eval_body.eval(y_output, y_target, action)
        eval_upper.eval(y_output, y_target, action)
        eval_lower.eval(y_output, y_target, action)


    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')
    res = {'FullBody': eval_body.get_results(),
           'UpperBody': eval_upper.get_results(),
           'LowerBody': eval_lower.get_results()}
    LOGGER.info(res)
    utils_io.write_json(os.path.join(LOGGER.logfile_dir, f'eval_res_{epoch}'+'.json'), res)



LOGGER = ConsoleLogger('Test3d', 'test')
data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor
test_data = Mocap(
        config.dataset.test,
        SetType.TEST,
        transform=data_transform)
test_data_loader = DataLoader(
    test_data,
    batch_size=2,
    shuffle=config.data_loader.shuffle,
    num_workers=8)
with open('model/model.yaml') as fin:
    model_cfg = edict(yaml.safe_load(fin))
resnet = pose_resnet.get_pose_net(model_cfg, True)
autoencoder = encoder_decoder.AutoEncoder()
device = torch.device(f"cuda:{0}")
checkpoint = torch.load("/media/zlz422/jyt/xR-EgoPose-change/experiments/finetune/2022-05-27-16-40-24/checkpoints/checkpoint_14.tar", map_location=device)
autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
autoencoder.eval()
validate(LOGGER, test_data_loader,resnet, autoencoder, device,0)
