import os
import numpy as np
from datetime import datetime
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from .ss_funcs import loss_fn, posterior_mean_est
from piq import psnr, ssim, gmsd


def plot_results(input_prj, output_prj, gt_prj=None):
    """
    Plots first 2 projections of the input_prj, output_prj and GT_prj if given.
    Used for visualization in tensorboard.

    Args:
        input_prj: tensor of noisy projections of shape (B, 1, H, W).
        output_prj: tensor of denoised projections of shape (B, 1, H, W).
        gt_prj: tensor of clean projection of shape (B, 1, H, W).
    """
    num_ims = 3
    if gt_prj is None:
        num_ims = 2
    f, axes = plt.subplots(2 * num_ims, 1, figsize=(15, 4 * num_ims))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(2):
        axes[i * num_ims].imshow(input_prj[i, 0], cmap='gray')
        axes[i * num_ims + 1].imshow(output_prj[i, 0], cmap='gray')
        axes[i * num_ims].set_xticks(())
        axes[i * num_ims].set_yticks(())
        axes[i * num_ims + 1].set_xticks(())
        axes[i * num_ims + 1].set_yticks(())
        axes[i * num_ims].set_ylabel('Noisy prj', fontsize=12)
        axes[i * num_ims + 1].set_ylabel('Cleaned prj', fontsize=12)

        if gt_prj is not None:
            axes[i * num_ims + 2].imshow(gt_prj[i, 0], cmap='gray')
            axes[i * num_ims + 2].set_yticks(())
            axes[i * num_ims + 2].set_xticks(())
            axes[i * num_ims + 2].set_ylabel('Full dose prj', fontsize=12)
    return f


def print_metrics(metrics, phase):
    """
    Function for printing all available metrics for specific phase.

    Args:
        metrics: dictionary of calculated metrics;
        phase: specifies phase for which metrics were calculated.
    """
    outputs = []
    for name in metrics.keys():
        outputs.append("{}: {:4f}".format(name, metrics[name]))

    print("{}: {}".format(phase, ", ".join(outputs)))


def setup_experiment(title, models_path, logdir="./tb"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    best_model_path = os.path.join(models_path, f"{title}.best.pth")
    print('Model path:', best_model_path)
    best_noise_path = os.path.join(models_path, title+"_Noise.best.pth")
    return writer, experiment_name, best_model_path, best_noise_path


def set_random_seeds(seed_value=0, device='cpu'):
    """
    source https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(prediction, target, min_v, max_v):
    metrics = dict()
    pred = prediction.clamp(min_v, max_v)
    pred = (pred - min_v) / (max_v - min_v)
    trgt = target.clamp(min_v, max_v)
    trgt = (trgt - min_v) / (max_v - min_v)
    data_range = 1.0

    metrics['SSIM'] = ssim(pred, trgt, data_range=data_range, kernel_size=5).item()
    metrics['PSNR'] = psnr(pred, trgt, data_range=data_range).item()
    metrics['GMSD'] = gmsd(pred, trgt, data_range=data_range).item()

    return metrics


def run_epoch_prob(model, noise_m, loader, optimizer, device, min_v, max_v,
                   inv=True, exclude_cnt=False, max_num_iter=None,
                   phase='train', epoch=0, writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    epoch_metrics = defaultdict(float)

    if max_num_iter:
        num_iter = min(len(loader), max_num_iter)
    else:
        num_iter = len(loader)

    with torch.set_grad_enabled(is_train):
        for i, sample in enumerate(tqdm(loader, total=num_iter)):
            if i == num_iter:
                break
            global_i = num_iter * epoch + i

            if len(sample) > 3:
                # for validation dataset
                prj, fd_prj, idx, mA = sample
                fd_prj = fd_prj.to(device)
            else:
                prj, idx, mA = sample
            prj = prj.to(device)
            idx = idx.to(device)
            mA = mA.to(device)

            # exclude central projections for the Noise2NoiseTD approach
            if exclude_cnt:
                input = torch.cat([prj[:, :prj.shape[1] // 2], prj[:, prj.shape[1] // 2 + 1:]], 1)
                mu_x, std_x = torch.chunk(model(input), 2, dim=1)
            else:
                mu_x, std_x = torch.chunk(model(prj), 2, dim=1)

            sigma_x = std_x ** 2
            noise_std = noise_m(mu_x, idx, mA)

            # get the noisy projection to be denoised from the sequence
            if len(prj.shape) > 4:
                if prj.shape[1] > 1:
                    prj1d = prj[:, prj.shape[1] // 2]
                else:
                    prj1d = prj[:, :, prj.shape[2] // 2]
            elif prj.shape[1] > 1:
                prj1d = prj[:, prj.shape[1] // 2, :, :].unsqueeze(1)
            else:
                prj1d = prj

            loss = loss_fn(prj1d, noise_std, mu_x, sigma_x)
            exp_out = posterior_mean_est(prj1d, noise_std, mu_x, sigma_x)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if working with transmission data, transform it to projection data
            if inv:
                exp_out[exp_out <= 0] = 1e-6
                exp_out = -exp_out.log()
                prj1d = -prj1d.log()
                if not is_train:
                    fd_prj = -fd_prj.log()

            if is_train:
                metrics = calculate_metrics(exp_out.detach(), prj1d.detach(), min_v, max_v)
            else:
                metrics = calculate_metrics(exp_out.detach(), fd_prj.detach(), min_v, max_v)

            # dump train metrics to tensorboard
            if writer is not None:
                if is_train:
                    writer.add_scalar(f"loss/{phase}", loss.item(), global_i)
                    for k in metrics.keys():
                        writer.add_scalar(f"{k}/{phase}", metrics[k], global_i)
                if is_train and (i + 1) % 50 == 0:
                    writer.add_figure(f"Training progress/{phase}",
                                      plot_results(prj1d.data.cpu().numpy(),
                                                   exp_out.data.cpu().numpy()),
                                      global_step=global_i)
                if not is_train and (i + 1) % 30 == 0:
                    writer.add_figure(f"Training progress/{phase}",
                                      plot_results(prj1d.data.cpu().numpy()[..., 200:600],
                                                   exp_out.data.cpu().numpy()[..., 200:600],
                                                   fd_prj.data.cpu().numpy()[..., 200:600]),
                                      global_step=global_i)

            # free cache
            if exclude_cnt:
                del input
            if not is_train:
                del fd_prj
            del prj, exp_out, mu_x, std_x, sigma_x, noise_std, prj1d, idx, mA
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            for k in metrics.keys():
                epoch_metrics[k] += metrics[k]

        for k in epoch_metrics.keys():
            epoch_metrics[k] /= num_iter

        # dump epoch metrics to tensorboard
        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / num_iter, epoch)
            for k in epoch_metrics.keys():
                writer.add_scalar(f"{k}_epoch/{phase}", epoch_metrics[k], epoch)

        return epoch_loss / num_iter, epoch_metrics


# for the case of training using MSE loss function (simple_loss)
def run_epoch_simple(model, loader, simple_loss, optimizer, device, min_v, max_v,
                     inv=True, exclude_cnt=False, max_num_iter=None,
                     phase='train', epoch=0, writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    epoch_metrics = defaultdict(float)

    if max_num_iter:
        num_iter = min(len(loader), max_num_iter)
    else:
        num_iter = len(loader)

    with torch.set_grad_enabled(is_train):
        for i, sample in enumerate(tqdm(loader, total=num_iter)):
            if i == num_iter:
                break
            global_i = num_iter * epoch + i

            if len(sample) > 3:
                # for validation dataset
                prj, fd_prj, _, _ = sample
                fd_prj = fd_prj.to(device)
            else:
                prj, idx, _ = sample
            prj = prj.to(device)

            # exclude central projections for the Noise2NoiseTD approach
            if exclude_cnt:
                input = torch.cat([prj[:, :prj.shape[1] // 2], prj[:, prj.shape[1] // 2 + 1:]], 1)
                exp_out = model(input)
            else:
                exp_out = model(prj)

            # get the noisy projection to be denoised from the sequence
            if len(prj.shape) > 4:
                if prj.shape[1] > 1:
                    prj1d = prj[:, prj.shape[1] // 2]
                else:
                    prj1d = prj[:, :, prj.shape[2] // 2]
            elif prj.shape[1] > 1:
                prj1d = prj[:, prj.shape[1] // 2, :, :].unsqueeze(1)
            else:
                prj1d = prj

            loss = simple_loss(exp_out, prj1d)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if working with transmission data, transform it to projection data
            if inv:
                exp_out[exp_out <= 0] = 1e-6
                exp_out = -exp_out.log()
                prj1d = -prj1d.log()
                if not is_train:
                    fd_prj = -fd_prj.log()

            if is_train:
                metrics = calculate_metrics(exp_out.detach(), prj1d.detach(), min_v, max_v)
            else:
                metrics = calculate_metrics(exp_out.detach(), fd_prj.detach(), min_v, max_v)

            # dump train metrics to tensorboard
            if writer is not None:
                if is_train:
                    writer.add_scalar(f"loss/{phase}", loss.item(), global_i)
                    for k in metrics.keys():
                        writer.add_scalar(f"{k}/{phase}", metrics[k], global_i)
                if is_train and (i + 1) % 50 == 0:
                    writer.add_figure(f"Training progress/{phase}",
                                      plot_results(prj1d.data.cpu().numpy(),
                                                   exp_out.data.cpu().numpy()),
                                      global_step=global_i)
                if not is_train and (i + 1) % 30 == 0:
                    writer.add_figure(f"Training progress/{phase}",
                                      plot_results(prj1d.data.cpu().numpy()[..., 200:600],
                                                   exp_out.data.cpu().numpy()[..., 200:600],
                                                   fd_prj.data.cpu().numpy()[..., 200:600]),
                                      global_step=global_i)

            # free cache
            if exclude_cnt:
                del input
            if not is_train:
                del fd_prj
            del prj, exp_out, prj1d
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            for k in metrics.keys():
                epoch_metrics[k] += metrics[k]

        for k in epoch_metrics.keys():
            epoch_metrics[k] /= num_iter

        # dump epoch metrics to tensorboard
        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / num_iter, epoch)
            for k in epoch_metrics.keys():
                writer.add_scalar(f"{k}_epoch/{phase}", epoch_metrics[k], epoch)

        return epoch_loss / num_iter, epoch_metrics


def run_epoch_supervised(model, mad, loader, simple_loss, optimizer, device, min_v, max_v,
                         max_num_iter=None, phase='train', epoch=0, writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    epoch_metrics = defaultdict(float)

    if max_num_iter:
        num_iter = min(len(loader), max_num_iter)
    else:
        num_iter = len(loader)

    with torch.set_grad_enabled(is_train):
        for i, sample in enumerate(tqdm(loader, total=num_iter)):
            if i == num_iter:
                break
            global_i = num_iter * epoch + i

            prj, fd_prj, _, _ = sample
            fd_prj = fd_prj.to(device)
            prj = prj.to(device)

            out = model(torch.cat([prj, mad(prj)], dim=1))

            # get the noisy projection to be denoised from the sequence
            if prj.shape[1] > 1:
                prj1d = prj[:, prj.shape[1] // 2, :, :].unsqueeze(1)
            else:
                prj1d = prj

            residual = prj1d - fd_prj

            loss = simple_loss(out, residual)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = calculate_metrics((prj1d - out).detach(), fd_prj.detach(), min_v, max_v)

            # dump train metrics to tensorboard
            if writer is not None:
                if is_train:
                    writer.add_scalar(f"loss/{phase}", loss.item(), global_i)
                    for k in metrics.keys():
                        writer.add_scalar(f"{k}/{phase}", metrics[k], global_i)
                if is_train and (i + 1) % 50 == 0:
                    writer.add_figure(f"Training progress/{phase}",
                                      plot_results(prj1d.data.cpu().numpy(),
                                                   (prj1d - out).data.cpu().numpy(),
                                                   fd_prj.data.cpu().numpy()),
                                      global_step=global_i)
                if not is_train and (i + 1) % 30 == 0:
                    writer.add_figure(f"Training progress/{phase}",
                                      plot_results(prj1d.data.cpu().numpy()[..., 200:600],
                                                   (prj1d - out).data.cpu().numpy()[..., 200:600],
                                                   fd_prj.data.cpu().numpy()[..., 200:600]),
                                      global_step=global_i)

            # free cache
            del prj, fd_prj, out, prj1d, residual
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            for k in metrics.keys():
                epoch_metrics[k] += metrics[k]

        for k in epoch_metrics.keys():
            epoch_metrics[k] /= num_iter

        # dump epoch metrics to tensorboard
        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / num_iter, epoch)
            for k in epoch_metrics.keys():
                writer.add_scalar(f"{k}_epoch/{phase}", epoch_metrics[k], epoch)

        return epoch_loss / num_iter, epoch_metrics
