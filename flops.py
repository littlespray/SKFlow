import sys
sys.path.append('core')

import argparse
import torch

from models import *


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
    parser.add_argument('--model_name', default="RAFTGMA", type=str)
    parser.add_argument('--tag', default="RAFTGMA", type=str)
    parser.add_argument('--output_path', default="./submissions", type=str)

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

    model = eval(args.model_name)(args)

    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (6, 432, 1024), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

