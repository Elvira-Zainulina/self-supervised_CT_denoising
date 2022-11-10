import torch


def loss_fn(y, noise_std, mu_x, sigma_x):
    """
    Maximizes log-likelihood of the noisy data y

    Input tensors of shape `(B, C, H, W)`.

    Args:
        y: noisy input image;
        noise_std: standard deviation of the noise;
        mu_x: mean values of the distributions of clean images;
        sigma_x: variance values of the distributions of clean images.
    """
    sigma_n = noise_std ** 2
    sigma_y = sigma_x + sigma_n
    loss = (y - mu_x) ** 2 / sigma_y + torch.log(sigma_y) - 0.1 * noise_std
    return loss.mean()


def posterior_mean_est(y, noise_std, mu_x, sigma_x):
    """
        Returns posterior mean estimation of the clean image given its noisy version,
        noise std, and parameters of its normal distribution.

        Input tensors of shape `(B, C, H, W)`.
        Output tensor of shape `(B, C, H, W)`.

        Args:
            y: noisy input image;
            noise_std: standard deviation of the noise;
            mu_x: mean of the distribution of clean images;
            sigma_x: variance of the distribution of clean images.
    """
    sigma_n = noise_std ** 2
    out = (mu_x * sigma_n + y * sigma_x) / (sigma_x + sigma_n)
    return out
