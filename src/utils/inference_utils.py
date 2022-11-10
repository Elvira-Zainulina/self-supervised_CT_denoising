import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import cv2
import skimage.morphology as sm
from piq import psnr, ssim, gmsd
from .ss_funcs import posterior_mean_est


def get_rect(img):
    mask = sm.dilation(img[0] > np.mean(img[0]), sm.disk(10)).astype(np.uint8)
    ext_contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(ext_contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def calculate_metrics(prediction, target, min_v, max_v):
    metrics = dict()
    pred = prediction.clamp(min_v, max_v)
    pred = (pred - min_v) / (max_v - min_v)
    trgt = target.clamp(min_v, max_v)
    trgt = (trgt - min_v) / (max_v - min_v)
    data_range = 1.0

    metrics['SSIM'] = ssim(pred, trgt, data_range=data_range, kernel_size=5, reduction='none')
    metrics['PSNR'] = psnr(pred, trgt, data_range=data_range, reduction='none')
    metrics['GMSD'] = gmsd(pred, trgt, data_range=data_range, reduction='none')

    return metrics


# this function denoises images from the "dataset" and estimates metrics
def predict_prob(model, noise_m, dataset, min_v, max_v, device='cpu',
                 inv=True, exclude_cnt=False, return_pred=True, bs=8,
                 get_metrics=True, use_roi=True, pad=20):
    model = model.to(device)
    model.eval()
    noise_m = noise_m.to(device)
    noise_m.eval()
    if use_roi:
        bs = 1  # cannot calculate ROI for stack of images
    loader = DataLoader(dataset, bs, shuffle=False, num_workers=4)
    if return_pred:
        pred = []
    res = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(loader):
            if len(sample) > 3:
                ld_prj, fd_prj, idx, mA = sample
                fd_prj = fd_prj.to(device)
            else:
                ld_prj, idx, mA = sample
            ld_prj = ld_prj.to(device)
            idx = idx.to(device)
            mA = mA.to(device)

            if exclude_cnt:
                input = torch.cat([ld_prj[:, :ld_prj.shape[1] // 2], ld_prj[:, ld_prj.shape[1] // 2 + 1:]], 1)
                mu_x, std_x = torch.chunk(model(input), 2, dim=1)
            else:
                mu_x, std_x = torch.chunk(model(ld_prj), 2, dim=1)

            sigma_x = std_x ** 2
            noise_std = noise_m(mu_x, idx, mA)

            if len(ld_prj.shape) > 4:
                if ld_prj.shape[1] > 1:
                    ld_prj = ld_prj[:, ld_prj.shape[1] // 2]
                else:
                    ld_prj = ld_prj[:, :, ld_prj.shape[2] // 2]
            elif ld_prj.shape[1] > 1:
                ld_prj = ld_prj[:, ld_prj.shape[1] // 2, :, :].unsqueeze(1)

            exp_out = posterior_mean_est(ld_prj, noise_std, mu_x, sigma_x)

            if inv:
                exp_out[exp_out <= 0] = 1e-6
                exp_out = -exp_out.log()
                ld_prj = -ld_prj.log()
                if len(sample) > 3:
                    fd_prj = -fd_prj.log()

            if return_pred:
                pred += [exp_out.data.cpu()]

            if get_metrics:
                if use_roi:
                    if len(sample) > 3:
                        x, y, w, h = get_rect(fd_prj[0].detach().cpu().numpy())
                        metrics = calculate_metrics(exp_out[..., x-pad:x+w+pad].detach(),
                                                    fd_prj[..., x-pad:x+w+pad].detach(), min_v, max_v)
                    else:
                        x, y, w, h = get_rect(ld_prj[0].detach().cpu().numpy())
                        metrics = calculate_metrics(exp_out[..., x-pad:x+w+pad].detach(),
                                                    ld_prj[..., x-pad:x+w+pad].detach(), min_v, max_v)
                else:
                    if len(sample) > 3:
                        metrics = calculate_metrics(exp_out.detach(), fd_prj.detach(), min_v, max_v)
                    else:
                        metrics = calculate_metrics(exp_out.detach(), ld_prj.detach(), min_v, max_v)
                for k in metrics.keys():
                    res[k] += [metrics[k].detach().cpu()]

            if len(sample) > 3:
                del fd_prj
            if exclude_cnt:
                del input
            del ld_prj, exp_out, mu_x, std_x, sigma_x, noise_std, idx, mA
            torch.cuda.empty_cache()

    if return_pred:
        pred = torch.cat(pred, dim=0).numpy()

    if get_metrics:
        for k in res.keys():
            res[k] = torch.cat(res[k]).numpy()
            res[k] = f'{np.round(np.mean(res[k]), 5)} \pm {np.round(np.std(res[k]), 5)}'
    else:
        return pred
    if return_pred:
        return res, pred
    return res


# for the case when simple loss function was used for model training (MSE loss),
# i.e., the simple train-inference scheme is used
def predict_simple(model, dataset, min_v, max_v, device='cpu',
                   inv=True, exclude_cnt=False, return_pred=True, bs=8,
                   get_metrics=True, use_roi=True, pad=20):
    model = model.to(device)
    model.eval()
    if use_roi:
        bs = 1  # cannot calculate ROI for stack of images
    loader = DataLoader(dataset, bs, shuffle=False, num_workers=4)
    if return_pred:
        pred = []
    res = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(loader):
            if len(sample) > 3:
                ld_prj, fd_prj, _, _ = sample
                fd_prj = fd_prj.to(device)
            else:
                ld_prj, _, _ = sample
            ld_prj = ld_prj.to(device)

            if exclude_cnt:
                input = torch.cat([ld_prj[:, :ld_prj.shape[1] // 2], ld_prj[:, ld_prj.shape[1] // 2 + 1:]], 1)
                exp_out = model(input)
            else:
                exp_out = model(ld_prj)

            if len(ld_prj.shape) > 4:
                if ld_prj.shape[1] > 1:
                    ld_prj = ld_prj[:, ld_prj.shape[1] // 2]
                else:
                    ld_prj = ld_prj[:, :, ld_prj.shape[2] // 2]
            elif ld_prj.shape[1] > 1:
                ld_prj = ld_prj[:, ld_prj.shape[1] // 2, :, :].unsqueeze(1)

            if inv:
                exp_out[exp_out <= 0] = 1e-6
                exp_out = -exp_out.log()
                ld_prj = -ld_prj.log()
                if len(sample) > 3:
                    fd_prj = -fd_prj.log()

            if return_pred:
                pred += [exp_out.data.cpu()]

            if get_metrics:
                if use_roi:
                    if len(sample) > 3:
                        x, y, w, h = get_rect(fd_prj[0].detach().cpu().numpy())
                        metrics = calculate_metrics(exp_out[..., x-pad:x+w+pad].detach(),
                                                    fd_prj[..., x-pad:x+w+pad].detach(), min_v, max_v)
                    else:
                        x, y, w, h = get_rect(ld_prj[0].detach().cpu().numpy())
                        metrics = calculate_metrics(exp_out[..., x-pad:x+w+pad].detach(),
                                                    ld_prj[..., x-pad:x+w+pad].detach(), min_v, max_v)
                else:
                    if len(sample) > 3:
                        metrics = calculate_metrics(exp_out.detach(), fd_prj.detach(), min_v, max_v)
                    else:
                        metrics = calculate_metrics(exp_out.detach(), ld_prj.detach(), min_v, max_v)
                for k in metrics.keys():
                    res[k] += [metrics[k].detach().cpu()]

            if len(sample) > 3:
                del fd_prj
            if exclude_cnt:
                del input
            del ld_prj, exp_out
            torch.cuda.empty_cache()

    if return_pred:
        pred = torch.cat(pred, dim=0).numpy()

    if get_metrics:
        for k in res.keys():
            res[k] = torch.cat(res[k]).numpy()
            res[k] = f'{np.round(np.mean(res[k]), 5)} \pm {np.round(np.std(res[k]), 5)}'
    else:
        return pred
    if return_pred:
        return res, pred
    return res


def predict_supervised(model, mad, dataset, min_v, max_v, device='cpu',
                       return_pred=True, bs=8, get_metrics=True, use_roi=True, pad=20):
    model = model.to(device)
    model.eval()
    if use_roi:
        bs = 1  # cannot calculate ROI for stack of images
    loader = DataLoader(dataset, bs, shuffle=False, num_workers=4)
    if return_pred:
        pred = []
    res = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(loader):
            if len(sample) > 3:
                ld_prj, fd_prj, _, _ = sample
                fd_prj = fd_prj.to(device)
            else:
                ld_prj, _, _ = sample
            ld_prj = ld_prj.to(device)

            out = model(torch.cat([ld_prj, mad(ld_prj)], dim=1))

            if ld_prj.shape[1] > 1:
                ld_prj = ld_prj[:, ld_prj.shape[1] // 2, :, :].unsqueeze(1)

            if return_pred:
                pred += [(ld_prj - out).data.cpu()]

            if get_metrics:
                if use_roi:
                    if len(sample) > 3:
                        x, y, w, h = get_rect(fd_prj[0].detach().cpu().numpy())
                        metrics = calculate_metrics((ld_prj - out)[..., x-pad:x+w+pad].detach(),
                                                    fd_prj[..., x-pad:x+w+pad].detach(), min_v, max_v)
                    else:
                        x, y, w, h = get_rect(ld_prj[0].detach().cpu().numpy())
                        metrics = calculate_metrics((ld_prj - out)[..., x-pad:x+w+pad].detach(),
                                                    ld_prj[..., x-pad:x+w+pad].detach(), min_v, max_v)
                else:
                    if len(sample) > 3:
                        metrics = calculate_metrics((ld_prj - out).detach(), fd_prj.detach(), min_v, max_v)
                    else:
                        metrics = calculate_metrics((ld_prj - out).detach(), ld_prj.detach(), min_v, max_v)
                for k in metrics.keys():
                    res[k] += [metrics[k].detach().cpu()]

            if len(sample) > 3:
                del fd_prj
            del ld_prj, out
            torch.cuda.empty_cache()

    if return_pred:
        pred = torch.cat(pred, dim=0).numpy()

    if get_metrics:
        for k in res.keys():
            res[k] = torch.cat(res[k]).numpy()
            res[k] = f'{np.round(np.mean(res[k]), 5)} \pm {np.round(np.std(res[k]), 5)}'
    else:
        return pred
    if return_pred:
        return res, pred
    return res
