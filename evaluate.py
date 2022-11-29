import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
from tqdm import tqdm


import datasets
from models import *
from utils.utils import InputPadder



@torch.no_grad()
def validate_chairs(model, iters=6, root='/data/FlyingChairs_release/data'):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation', root=root)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=6, root='/data/FlyingThings3D'):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation', root=root)
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel(model, iters=6, root='/data/Sintel', tqdm_miniters=1):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, root=root)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_occ(model, iters=6, root='/data/Sintel', tqdm_miniters=1):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=6, root='/data/kitti15'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', root=root)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--chairs_root', type=str, default=None)
    parser.add_argument('--sintel_root', type=str, default=None)
    parser.add_argument('--things_root', type=str, default=None)
    parser.add_argument('--kitti_root', type=str, default=None)
    parser.add_argument('--hd1k_root', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./submission')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    parser.add_argument('--k_conv', type=int, nargs='+', default=[1, 15])
    parser.add_argument('--UpdateBlock', type=str, default='SKUpdateBlock6')
    parser.add_argument('--PCUpdater_conv', type=int, nargs='+', default=[1, 7])

    args = parser.parse_args()


    model = torch.nn.DataParallel(eval(args.model_name)(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    print('begin evaluation!')
    with torch.no_grad():

        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=args.iters, root=args.chairs_root)

        elif args.dataset == 'things':
            validate_things(model.module, iters=args.iters, root=args.things_root)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, iters=args.iters, root=args.sintel_root)

        elif args.dataset == 'sintel_occ':
            validate_sintel_occ(model.module, iters=args.iters, root=args.sintel_root)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, iters=args.iters, root=args.kitti_root)
