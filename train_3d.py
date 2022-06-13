# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, utils_io
from utils import arguments
import torch.optim as optim
from model import encoder_decoder

from utils.loss import HeatmapLoss, LimbLoss, PoseLoss
import pprint
import os
from tensorboardX import SummaryWriter
from utils import AverageMeter
import time

from utils.vis2d import draw2Dpred_and_gt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():

    args = arguments.parse_args()
    LOGGER = ConsoleLogger('Train3d', 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    LOGGER.info(config)


    cudnn.benckmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # ------------------- Data loader -------------------

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor

    train_data = Mocap(
        config.dataset.train,
        SetType.TRAIN,
        transform=data_transform)
    train_data_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    test_data = Mocap(
        config.dataset.test,
        SetType.TEST,
        transform=data_transform)
    test_data_loader = DataLoader(
        test_data,
        batch_size=2,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    # ------------------- Model -------------------
    #TODO: add other losses
    autoencoder = encoder_decoder.AutoEncoder(args.batch_norm, args.denis_activation)
    LossHeatmapRecon = HeatmapLoss()
    # Loss3D = nn.MSELoss()
    Loss3D = PoseLoss()
    LossLimb = LimbLoss()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{1}")
        autoencoder = autoencoder.cuda(device)
        LossHeatmapRecon.cuda(device)
        Loss3D.cuda(device)
        LossLimb.cuda(device)

    # ------------------- optimizer -------------------
    if args.optimizer == 'adam':
        optimizer = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(autoencoder.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    # ------------------- load model -------------------
    # if args.load_model:
    #     if not os.path.isfile(args.load_model):
    #         raise FileNotFoundError(f"No checkpoint found at {args.load_model}")
    #     checkpoint = torch.load(args.load_model)
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])


    # ------------------- tensorboard -------------------
    train_global_steps = 0
    writer_dict = {
        'writer': SummaryWriter(log_dir=logdir),
        'train_global_steps': train_global_steps
    }


    best_perf = float('inf')
    best_model = False
    # ------------------- run the model -------------------
    for epoch in range(200):
        with torch.autograd.set_detect_anomaly(True):
            LOGGER.info(f'====Training epoch {epoch}====')
            losses = AverageMeter()
            batch_time = AverageMeter()

            # ------------------- Evaluation -------------------
            eval_body = evaluate.EvalBody()
            eval_upper = evaluate.EvalUpperBody()
            eval_lower = evaluate.EvalLowerBody()

            autoencoder.train()

            end = time.time()
            for it, (img, p2d, p3d, heatmap, action) in enumerate(train_data_loader, 0):

                img = img.to(device)
                p3d = p3d.to(device)
                heatmap = heatmap.to(device)

                p3d_hat, heatmap2d_recon = autoencoder(heatmap)
                loss_recon = LossHeatmapRecon(heatmap2d_recon, heatmap).mean()
                loss_3d = Loss3D(p3d_hat, p3d).mean()
                loss_cos, loss_len = LossLimb(p3d_hat, p3d)
                loss_cos = loss_cos.mean()
                loss_len = loss_len.mean()

                # loss = 0.001 * loss_recon + 0.1 * loss_3d -0.01 * loss_cos + 0.5 * loss_len
                loss = args.lambda_recon * loss_recon + args.lambda_3d * loss_3d - args.lambda_cos * loss_cos + args.lambda_len * loss_len

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                losses.update(loss.item(), img.size(0))

                if it % config.train.PRINT_FREQ == 0:
                    # logging messages
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Batch Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                        epoch, it, len(train_data_loader), batch_time=batch_time,
                        speed=img.size(0) / batch_time.val,  # averaged within batch
                        loss=losses)
                    LOGGER.info(msg)

                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    lr = [group['lr'] for group in scheduler.optimizer.param_groups]
                    print("lr:{}".format(lr))
                    writer.add_scalar('learning_rate', lr, global_steps)
                    writer.add_scalar('train_loss', losses.val, global_steps)
                    writer.add_scalar('batch_time', batch_time.val, global_steps)
                    writer.add_scalar('losses/loss_recon', loss_recon, global_steps)
                    writer.add_scalar('losses/loss_3d', loss_3d, global_steps)
                    writer.add_scalar('losses/loss_cos', loss_cos, global_steps)
                    writer.add_scalar('losses/loss_len', loss_len, global_steps)
                    image_grid = draw2Dpred_and_gt(img, heatmap2d_recon,(368, 368))
                    writer.add_image('predicted_heatmaps', image_grid, global_steps)

                    writer_dict['train_global_steps'] = global_steps + 1

                    # ------------------- evaluation on training data -------------------

                    # Evaluate results using different evaluation metrices
                    y_output = p3d_hat.data.cpu().numpy()
                    y_target = p3d.data.cpu().numpy()

                    eval_body.eval(y_output, y_target, action)
                    eval_upper.eval(y_output, y_target, action)
                    eval_lower.eval(y_output, y_target, action)

                end = time.time()
            scheduler.step()

            # ------------------- Save results -------------------
            checkpoint_dir = os.path.join(logdir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
            states = dict()
            states['autoencoder_state_dict'] = autoencoder.state_dict()
            states['optimizer_state_dict']= optimizer.state_dict()
            states['scheduler'] = scheduler.state_dict()

            torch.save(states, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'))

            res = {'FullBody': eval_body.get_results(),
                   'UpperBody': eval_upper.get_results(),
                   'LowerBody': eval_lower.get_results()}

            LOGGER.info('===========Evaluation on Train data==========')
            LOGGER.info(pprint.pformat(res))
            # utils_io.write_json(config.eval.output_file, res)

            # ------------------- validation -------------------
            # autoencoder.eval()
            # val_loss = validate(LOGGER, test_data_loader, autoencoder, device, epoch)
            # if val_loss < best_perf:
            #     best_perf = val_loss
            #     best_model = True
            #
            # if best_model:
            #     shutil.copyfile(os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'), os.path.join(checkpoint_dir, f'model_best.tar'))
            #     best_model = False

    LOGGER.info('Done.')




if __name__ == "__main__":
    print("ok")
    main()
