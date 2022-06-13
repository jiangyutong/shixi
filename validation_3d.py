# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
import glob
from utils.vis3d import show3Dpose
import matplotlib.pyplot as plt
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
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
output_dir = "egopose_hm36_trainval/"
def validate(LOGGER, data_loader, autoencoder, device):


    # ------------------- Loss -------------------


    # ------------------- Evaluation -------------------
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()
    i=0
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
            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)
            # if i < 2000:
            #     ax = plt.subplot(111, projection='3d')
            #     show3Dpose(y_output, ax, True)
            #     # plt.show()
            #     plt.savefig(output_dir + str(i).zfill(5) + '_3Dv.png', dpi=200, format='png', bbox_inches='tight')
            #     i += 1
            print('ok')


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
        transform=data_transform)
test_data_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=config.data_loader.shuffle,
    num_workers=8)
autoencoder = encoder_decoder.AutoEncoder()
device = torch.device(f"cuda:{0}")
checkpoint = torch.load("/media/zlz422/jyt/xR-EgoPose-change/experiments/Train3d/2022-06-08-17-12-58/checkpoints/checkpoint_32.tar", map_location=device)
autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
autoencoder.eval()
validate(LOGGER, test_data_loader, autoencoder, device)
# def img2video(video_path, output_dir):
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     names = sorted(glob.glob(os.path.join(output_dir, '*_3Dv.png')),key=lambda s:int(s[-11:-8]))
#     img = cv2.imread(names[0])
#     size = (img.shape[1], img.shape[0])
#     videoWrite = cv2.VideoWriter(video_path, fourcc, 10,size)
#     for name in names:
#         print(name)
#         img = cv2.imread(name)
#         videoWrite.write(img)
#     videoWrite.release()
# img2video("/media/zlz422/jyt/xR-EgoPose-change/egopose_hm36_trainval/resultv.mp4", "/media/zlz422/jyt/xR-EgoPose-change/egopose_hm36_trainval")
