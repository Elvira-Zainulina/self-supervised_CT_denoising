import os
import numpy as np
import argparse
import torch
from utils.inference_utils import predict_prob, predict_simple, predict_supervised
from utils.data import LDCT_dataset
from models.Noise2NoiseTD import N2NTDDnCNN
from models.noise_models import Noise
from models.Noise2Void import Noise2VoidDnCNN
from models.Noise2Clean import DnCNN, MedianAbsoluteDeviation
# MIN_V, MAX_V = -0.2022089809179306, 9.114472389221191  # L033
MIN_V, MAX_V = -0.2716837227344513, 8.37  # L134
# MIN_V, MAX_V = -0.23, 8.37  # L004
# MIN_V, MAX_V = (-0.2400284856557846, 8.933542251586914)  # L006


def print_metrics(metrics, name):
    print(name)
    for k in metrics.keys():
        print(f'{k}: {metrics[k]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd_dir', help='Path to clean (full-dose) data, if exists', default=None)
    parser.add_argument('--ld_dir', help='Path to noisy (low-dose) data')
    parser.add_argument('--simulated', help='Pass if the low-dose data is simulated' +
                                            '(projections are npz arrays with names of type file_name.dcm.npz)',
                        action='store_true')
    parser.add_argument('--num_adj_prj',
                        help='Number of adjacent projections from one side to use (>=0)',
                        default=3, type=int)
    parser.add_argument('--approach',
                        help='Approach to use: nc - Noise2Clean, nv - Noise2Void, td - Noise2NoiseTD')
    parser.add_argument('--inv',
                        help='Pass if the model was trained in the projection domain, not in the transmission domain',
                        action='store_false')
    parser.add_argument('--mse',
                        help='Pass if the model was trained using MSE loss (only for the self-supervised mode)',
                        action='store_true')
    parser.add_argument('--dm_path', help='Path to the denoising model', default='./model.pth')
    parser.add_argument('--nm_path', help='Path to the noise model', default='./noise_model.pth')
    parser.add_argument('--dest_path', help='Path to save denoised predictions', default='./predictions.npy')
    args = parser.parse_args()

    ld_path = args.ld_dir
    if args.fd_dir is None:
        fd_path = None
    else:
        fd_path = [args.fd_dir]
    objects = sorted(os.listdir(ld_path))

    # define train and val datasets
    num_add_prj = args.num_adj_prj
    if args.approach == 'td':
        adj_type = 'TS'
    else:
        adj_type = 'new_channels'

    test_dataset = LDCT_dataset([ld_path], [objects], fd_paths=fd_path, cut=False, inv=args.inv,
                                simulated=args.simulated, num_add_prj=num_add_prj,
                                adj_type=adj_type, fill_type='replicate')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.approach == 'nc':
        model = DnCNN((num_add_prj + 1) * 2, 1).to(device)
        model.load_state_dict(torch.load(args.dm_path, map_location=device))
        mad = MedianAbsoluteDeviation(ksize=3, down=False).to(device)
        metrics, predictions = predict_supervised(model, mad, test_dataset, min_v=MIN_V, max_v=MAX_V,
                                                  device=device, return_pred=True, bs=8, get_metrics=True,
                                                  use_roi=True, pad=20)
    else:
        if args.mse:
            out_dim = 1
        else:
            out_dim = 2
            noise_m = Noise().to(device)
            noise_m.load_state_dict(torch.load(args.nm_path, map_location=device))
        if args.approach == 'td':
            model = N2NTDDnCNN(out_dim=out_dim, seq_len=num_add_prj * 2).to(device)
            exclude_cnt = True
        else:
            model = Noise2VoidDnCNN(in_dim=num_add_prj * 2 + 1, out_dim=out_dim).to(device)
            exclude_cnt = False
        model.load_state_dict(torch.load(args.dm_path, map_location=device))

        if args.mse:
            metrics, predictions = predict_simple(model, test_dataset, min_v=MIN_V, max_v=MAX_V,
                                                  device=device, inv=args.inv, exclude_cnt=exclude_cnt,
                                                  return_pred=True, bs=8, get_metrics=True,
                                                  use_roi=True, pad=20)
        else:
            metrics, predictions = predict_prob(model, noise_m, test_dataset, min_v=MIN_V, max_v=MAX_V,
                                                device=device, inv=args.inv, exclude_cnt=exclude_cnt,
                                                return_pred=True, bs=8, get_metrics=True,
                                                use_roi=True, pad=20)
    np.save(args.dest_path, predictions)
    print_metrics(metrics, args.dm_path)
