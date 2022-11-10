import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.train_utils import run_epoch_prob, run_epoch_simple, run_epoch_supervised
from utils.train_utils import setup_experiment, set_random_seeds, print_metrics
from utils.data import LDCT_dataset
from models.Noise2NoiseTD import N2NTDDnCNN
from models.noise_models import Noise
from models.Noise2Void import Noise2VoidDnCNN
from models.Noise2Clean import DnCNN, MedianAbsoluteDeviation
MIN_V, MAX_V = (-0.24, 8.94)


def train_prob(model, noise_m, optimizer, train_dataset, val_dataset=None, inv=True,
               n_epochs=300, bs=16, min_v=MIN_V, max_v=MAX_V, exclude_cnt=False,
               train_max_iter=200, val_max_iter=90, device='cuda', writer=None, chpt_step=10,
               best_model_path='./model.pth', best_noise_path='./noise_model.pth'):

    train_loader = DataLoader(train_dataset, bs, shuffle=True, num_workers=4, drop_last=True)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, bs, shuffle=True, num_workers=4)
        best_val_gmsd = float('+inf')
        best_val_psnr = float('-inf')

    for epoch in range(n_epochs):
        train_loss, train_metrics = run_epoch_prob(model, noise_m, train_loader, optimizer, device,
                                                   min_v, max_v, inv=inv, exclude_cnt=exclude_cnt,
                                                   max_num_iter=train_max_iter, phase='train', epoch=epoch,
                                                   writer=writer)
        if val_dataset is not None:
            val_loss, val_metrics = run_epoch_prob(model, noise_m, val_loader, None, device,
                                                   min_v, max_v, inv=inv, exclude_cnt=exclude_cnt,
                                                   max_num_iter=val_max_iter, phase='val', epoch=epoch,
                                                   writer=writer)

            if val_metrics['PSNR'] > best_val_psnr:
                best_val_psnr = val_metrics['PSNR']
                torch.save(model.state_dict(), f".best_psnr".join(best_model_path.split('.best')))
                torch.save(noise_m.state_dict(), f".best_psnr".join(best_noise_path.split('.best')))

            if val_metrics['GMSD'] < best_val_gmsd:
                best_val_gmsd = val_metrics['GMSD']
                torch.save(model.state_dict(), f".best_gmsd".join(best_model_path.split('.best')))
                torch.save(noise_m.state_dict(), f".best_gmsd".join(best_noise_path.split('.best')))

            print(f'Epoch: {epoch + 1:02}')
            print_metrics(train_metrics, '\tTrain pme')
            print_metrics(val_metrics, '\t Val pme')

        else:
            if epoch % chpt_step == 0:
                torch.save(model.state_dict(), f".epoch_{epoch}".join(best_model_path.split('.best')))
                torch.save(noise_m.state_dict(), f".epoch_{epoch}".join(best_noise_path.split('.best')))
            print(f'Epoch: {epoch + 1:02}')
            print_metrics(train_metrics, '\tTrain pme')


def train_simple(model, loss_fn, optimizer, train_dataset, val_dataset=None, inv=True,
                 n_epochs=300, bs=16, min_v=MIN_V, max_v=MAX_V, exclude_cnt=False,
                 train_max_iter=200, val_max_iter=90, device='cuda', writer=None,
                 chpt_step=10, best_model_path='./model.pth'):

    train_loader = DataLoader(train_dataset, bs, shuffle=True, num_workers=4, drop_last=True)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, bs, shuffle=True, num_workers=4)
        best_val_gmsd = float('+inf')
        best_val_psnr = float('-inf')

    for epoch in range(n_epochs):
        train_loss, train_metrics = run_epoch_simple(model, train_loader, loss_fn, optimizer, device,
                                                     min_v, max_v, inv=inv, exclude_cnt=exclude_cnt,
                                                     max_num_iter=train_max_iter, phase='train', epoch=epoch,
                                                     writer=writer)

        if val_dataset is not None:
            val_loss, val_metrics = run_epoch_simple(model, val_loader, loss_fn, None, device,
                                                     min_v, max_v, inv=inv, exclude_cnt=exclude_cnt,
                                                     max_num_iter=val_max_iter, phase='val', epoch=epoch,
                                                     writer=writer)

            if val_metrics['PSNR'] > best_val_psnr:
                best_val_psnr = val_metrics['PSNR']
                torch.save(model.state_dict(), f".best_psnr".join(best_model_path.split('.best')))

            if val_metrics['GMSD'] < best_val_gmsd:
                best_val_gmsd = val_metrics['GMSD']
                torch.save(model.state_dict(), f".best_gmsd".join(best_model_path.split('.best')))

            print(f'Epoch: {epoch + 1:02}')
            print_metrics(train_metrics, '\tTrain pme')
            print_metrics(val_metrics, '\t Val pme')

        else:
            if epoch % chpt_step == 0:
                torch.save(model.state_dict(), f".epoch_{epoch}".join(best_model_path.split('.best')))
            print(f'Epoch: {epoch + 1:02}')
            print_metrics(train_metrics, '\tTrain pme')


def train_supervised(model, mad, loss_fn, optimizer, train_dataset, val_dataset=None, n_epochs=300,
                     bs=16, min_v=MIN_V, max_v=MAX_V, train_max_iter=200, val_max_iter=90,
                     device='cuda', writer=None, best_model_path='./model.pth'):
    train_loader = DataLoader(train_dataset, bs, shuffle=True, num_workers=4, drop_last=True)
    best_loss = float('+inf')
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, bs, shuffle=True, num_workers=4)

    for epoch in range(n_epochs):
        train_loss, train_metrics = run_epoch_supervised(model, mad, train_loader, loss_fn, optimizer,
                                                         device, min_v, max_v, max_num_iter=train_max_iter,
                                                         phase='train', epoch=epoch, writer=writer)

        if val_dataset is not None:
            val_loss, val_metrics = run_epoch_supervised(model, mad, val_loader, loss_fn, None,
                                                         device, min_v, max_v, max_num_iter=val_max_iter,
                                                         phase='val', epoch=epoch, writer=writer)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            print(f'Epoch: {epoch + 1:02}')
            print_metrics(train_metrics, '\tTrain pme')
            print_metrics(val_metrics, '\t Val pme')

        else:
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), best_model_path)
            print(f'Epoch: {epoch + 1:02}')
            print_metrics(train_metrics, '\tTrain pme')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd_dir', help='Path to the clean (full-dose) projections, if exists', default=None)
    parser.add_argument('--ld_dir', help='Path to the noisy (low-dose) projections')
    parser.add_argument('--val_size',
                        help='Pass the coefficient for splitting data if the ' +
                             'dataset should be splitted into train and validation sets',
                        default=None, type=float)
    parser.add_argument('--simulated', help='Pass if the low-dose data is simulated' +
                                            '(projections are npz arrays with names of type file_name.dcm.npz)',
                        action='store_true')
    parser.add_argument('--num_adj_prj', help='Number of adjacent projections from one side to use (>=0)',
                        default=3, type=int)
    parser.add_argument('--approach',
                        help='Approach to use: nc - Noise2Clean, nv - Noise2Void, td - Noise2NoiseTD')
    parser.add_argument('--inv',
                        help='Pass if to train in the projection domain (p), not in the transmission domain (exp(-p))',
                        action='store_false')
    parser.add_argument('--mse',
                        help='Pass if to train a model using MSE loss (only for the self-supervised mode)',
                        action='store_true')
    parser.add_argument('--epochs', help='Number of epochs', default=500, type=int)
    parser.add_argument('--logpath', help='Path to save logs', default='./')
    parser.add_argument('--models_path', help='Path to save models', default='./')
    parser.add_argument('--nw_path', help='Path to the pre-trained noise weights, if any', default=None)
    args = parser.parse_args()

    ld_path = args.ld_dir
    objects = sorted(os.listdir(ld_path))

    if args.approach == 'nc':
        assert args.fd_dir is not None, 'fd_path should be provided for training using the Noise2Clean approach'
        fd_path = [args.fd_dir]
    else:
        fd_path = None

    num_add_prj = args.num_adj_prj
    if args.approach == 'td':
        adj_type = 'TS'
    else:
        adj_type = 'new_channels'

    if args.val_size is not None:
        # split it into grups of 100 in order to allow random split
        objects = [objects[i * 100: (i + 1) * 100] for i in range(len(objects) // 100)]
        inds = np.arange(len(objects))

        # split data to train and val sets
        train_inds, val_inds = train_test_split(inds, test_size=args.val_size, random_state=67)

        objects = np.asarray(objects)
        train_objects = np.concatenate(objects[train_inds])
        train_objects.sort()
        val_objects = np.concatenate(objects[val_inds])
        val_objects.sort()

        val_dataset = LDCT_dataset([ld_path], [val_objects], fd_paths=fd_path, inv=args.inv,
                                   simulated=args.simulated, num_add_prj=num_add_prj,
                                   adj_type=adj_type, fill_type='replicate')
    else:
        train_objects = objects
        val_dataset = None

    train_dataset = LDCT_dataset([ld_path], [train_objects], fd_paths=fd_path, cut=True, inv=args.inv,
                                 simulated=args.simulated, num_add_prj=num_add_prj,
                                 adj_type=adj_type, fill_type='replicate')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seeds(seed_value=0, device=device)

    if args.approach == 'nc':
        model = DnCNN((num_add_prj + 1) * 2, 1).to(device)
        mad = MedianAbsoluteDeviation(ksize=3, down=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        writer, experiment_name, best_model_path, _ = setup_experiment(
            model.__class__.__name__, models_path=args.models_path, logdir=args.logpath)
        print(f"Experiment name: {experiment_name}")
        train_supervised(model, mad, loss_fn, optimizer, train_dataset, val_dataset=val_dataset,
                         n_epochs=args.epochs, bs=64, device=device, writer=writer,
                         best_model_path=best_model_path)

    else:
        if args.mse:
            out_dim = 1
        else:
            out_dim = 2
            noise_m = Noise().to(device)
            if args.nw_path is not None:
                noise_weights = torch.load(args.nw_path)
                noise_m.emb.weight = nn.Parameter(noise_weights['emb.weight'], requires_grad=False)
                noise_m.map_acq_params[0].weight = nn.Parameter(noise_weights['map_acq_params.0.weight'],
                                                                requires_grad=False)
                noise_m.map_acq_params[0].bias = nn.Parameter(noise_weights['map_acq_params.0.bias'],
                                                              requires_grad=False)
                noise_m.map_acq_params[2].weight = nn.Parameter(noise_weights['map_acq_params.2.weight'],
                                                                requires_grad=False)
                noise_m.map_acq_params[2].bias = nn.Parameter(noise_weights['map_acq_params.2.bias'],
                                                              requires_grad=False)

        if args.approach == 'td':
            model = N2NTDDnCNN(out_dim=out_dim, seq_len=num_add_prj * 2).to(device)
            exclude_cnt = True
        else:
            model = Noise2VoidDnCNN(in_dim=num_add_prj * 2 + 1, out_dim=out_dim).to(device)
            exclude_cnt = False

        if args.mse:
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            writer, experiment_name, best_model_path, _ = setup_experiment(
                model.__class__.__name__, models_path=args.models_path, logdir=args.logpath)
            print(f"Experiment name: {experiment_name}")
            train_simple(model, loss_fn, optimizer, train_dataset, val_dataset=val_dataset, inv=args.inv,
                         n_epochs=args.epochs, bs=16, min_v=MIN_V, max_v=MAX_V, exclude_cnt=exclude_cnt,
                         device=device, writer=writer, best_model_path=best_model_path)
        else:
            optimizer = torch.optim.Adam(list(model.parameters()) + list(noise_m.parameters()), lr=1e-4)
            writer, experiment_name, best_model_path, best_noise_path = setup_experiment(
                model.__class__.__name__ + '_' + noise_m.__class__.__name__,
                models_path=args.models_path, logdir=args.logpath)
            print(f"Experiment name: {experiment_name}")
            train_prob(model, noise_m, optimizer, train_dataset, val_dataset=val_dataset, inv=args.inv,
                       n_epochs=args.epochs, bs=16, min_v=MIN_V, max_v=MAX_V, exclude_cnt=exclude_cnt,
                       device=device, writer=writer, chpt_step=10,
                       best_model_path=best_model_path, best_noise_path=best_noise_path)
