from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import sys
import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from models import *


from utils import flow_viz
import datasets
import evaluate

from torch.cuda.amp import GradScaler

# exclude extremly large displacements
MAX_FLOW = 400


def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_args(args):
    print('----------- args ----------')
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    print('---------------------------')


def sequence_loss(flow_preds, flow_gt, valid, gamma):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}
        self.terminal = sys.stdout
        self.log = open(self.args.output+'/log.txt', 'a')

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int32)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        current_time = time.strftime("%Y-%m-%d %H:%M:%S") + "  "
        # print the training status
        print(current_time + training_str + metrics_str + time_left_hms)
        sys.stdout.flush()

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(args):

    model = nn.DataParallel(eval(args.model_name)(args), device_ids=args.gpus)

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)
    sys.stdout = logger
    print_args(args)
    print(f"Parameter Count: {count_parameters(model)}")

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, scaler, args)
        if logger.total_steps >= args.num_steps:
            plot_train(logger, args)
            plot_val(logger, args)
            break

    PATH = args.output+f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)

    # save best model
    metrics_dict = {'chairs':'chairs_epe', 'things':'final', 'sintel':'final', 'kitti':'kitti_epe'}
    key = metrics_dict[args.stage]
    best_result = min(logger.val_results_dict[key])
    best_step = logger.val_steps_list[logger.val_results_dict[key].index(best_result)] + 1
    os.system('cp ' + args.output + f'/{best_step}_{args.name}.pth ' + args.output + f'/best.pth')
    
    return PATH


def train(model, train_loader, optimizer, scheduler, logger, scaler, args):
    for i_batch, data_blob in enumerate(train_loader):
        tic = time.time()
        image1, image2, flow, valid = [x.cuda() for x in data_blob]

        optimizer.zero_grad()

        flow_pred = model(image1, image2)

        loss, metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        toc = time.time()

        metrics['time'] = toc - tic
        logger.push(metrics)
        if logger.total_steps % args.print_freq == args.print_freq - 1 and hasattr(model.module, "corr_lambdas"):
            print(model.module.corr_lambdas)

        # Validate
        if logger.total_steps+1 >= args.num_steps//2 and logger.total_steps % args.val_freq == args.val_freq - 1:
            validate(model, args, logger)
            plot_train(logger, args)
            plot_val(logger, args)
            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            torch.save(model.state_dict(), PATH)

        if logger.total_steps >= args.num_steps:
            break


def validate(model, args, logger):
    model.eval()
    results = {}

    # Evaluate results
    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            results.update(evaluate.validate_chairs(model.module, args.iters, args.chairs_root))
        elif val_dataset == 'sintel':
            results.update(evaluate.validate_sintel(model.module, args.iters, args.sintel_root, tqdm_miniters=100))
        elif val_dataset == 'kitti':
            results.update(evaluate.validate_kitti(model.module, args.iters, args.kitti_root))

    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])

    logger.val_steps_list.append(logger.total_steps)
    
    model.train()


def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        latest_x, latest_result = logger.val_steps_list[-1], logger.val_results_dict[key][-1]
        best_result = min(logger.val_results_dict[key])
        best_x = logger.val_steps_list[logger.val_results_dict[key].index(best_result)] 


        plt.rc('font',family='Times New Roman')
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.annotate('(%s, %s)'%(best_x, round(best_result, 4)), xy=(best_x, best_result), fontproperties='Times New Roman', size=6)
        plt.annotate('(%s, %s)'%(latest_x, round(latest_result, 4)), xy=(latest_x, latest_result), fontproperties='Times New Roman', size=6)
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='RAFTGMA', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--chairs_root', type=str, default=None)
    parser.add_argument('--sintel_root', type=str, default=None)
    parser.add_argument('--things_root', type=str, default=None)
    parser.add_argument('--kitti_root', type=str, default=None)
    parser.add_argument('--hd1k_root', type=str, default=None)
    parser.add_argument('--k_conv', type=int, nargs='+', default=[1, 15])
    parser.add_argument('--UpdateBlock', type=str, default='SKUpdateBlock6')
    parser.add_argument('--PCUpdater_conv', type=int, nargs='+', default=[1, 7])
    


    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    main(args)
