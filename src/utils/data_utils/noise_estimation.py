import numpy as np
from numba import njit
from numpy.linalg import matrix_rank
from scipy.stats import gamma
from scipy.ndimage import correlate


# python realization of the noise estimation tool from
# https://www.mathworks.com/matlabcentral/fileexchange/36921-noise-level-estimation-from-a-single-image
@njit
def my_convmtx2(H, m, n):
    s = H.shape
    t = np.zeros(((m-s[0]+1) * (n-s[1]+1), m*n))
    k = 0
    for i in range(m-s[0]+1):
        for j in range(n-s[1]+1):
            for p in range(s[0]):
                t[k, (i+p)*n+j:(i+p)*n+j+s[1]] = H[p]
            k = k + 1
    return t


@njit
def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    for i in range(sy):
        for j in range(sx):
            for k in range(block_size[0]):
                for l in range(block_size[1]):
                    result[l * block_size[0] + k, i * sx + j] = mtx[j + k, i + l]
    return result


@njit
def get_mask(p, shape, patchsize):
    mask = np.zeros(shape)
    ind = 0
    for j in range(shape[1] - patchsize + 1):
        for i in range(shape[0] - patchsize + 1):
            if p[ind] > 0:
                mask[i:i+patchsize, j:j+patchsize] = 1
            ind += 1
    return mask


def noise_level(im, patchsize=7, conf=1-1e-6, itr=5,
                return_mask=True, return_history=False):
    kH = np.array([[-1 / 2, 0, 1 / 2]])
    imH = correlate(im, kH, mode='nearest')
    imH = imH[:, 1:imH.shape[1] - 1]
    imH = imH * imH

    kV = kH.T
    imV = correlate(im, kV, mode='nearest')
    imV = imV[1:imV.shape[0] - 1, :]
    imV = imV * imV

    Dh = my_convmtx2(kH, patchsize, patchsize)
    Dv = my_convmtx2(kV, patchsize, patchsize)
    DD = Dh.T @ Dh + Dv.T @ Dv
    r = matrix_rank(DD)
    Dtr = np.trace(DD)
    tau0 = gamma.ppf(conf, r / 2, scale=2 * Dtr / r)
    
    # costyl
    _ = im2col(np.zeros((2, 2)), (1, 1))
        
    X = im2col(im, (patchsize, patchsize))
    Xh = im2col(imH, (patchsize, patchsize - 2))
    Xv = im2col(imV, (patchsize - 2, patchsize))

    Xtr = np.sum(np.vstack((Xh, Xv)), axis=0)
    
    if return_mask:
        Xtr0 = Xtr.copy()
    
    # noise level estimation
    tau = np.inf
    if X.shape[1] < X.shape[0]:
        sig2 = 0
    else:
        cov = X @ X.T / (X.shape[1] - 1)
        d = np.linalg.eig(cov)[0]
        d[d <= 0] = 0.
        sig2 = np.min(d)
        
    if return_history:
        std_history = [sig2 ** 0.5]
        if return_mask:
            mask_history = [np.ones(im.shape, dtype=bool)]
        else:
            mask_history = None
            
    sig2_old = sig2
        
    for i in range(itr):
        tau = sig2 * tau0
        p = (Xtr < tau)
        Xtr = Xtr[p]
        X = X[:, p]

        # noise level estimation
        if X.shape[1] < X.shape[0]:
            break
        cov = X @ X.T / (X.shape[1] - 1)
        d = np.linalg.eigvals(cov)
        d[d <= 0] = 0.
        sig2 = np.min(d)
        
        if return_history:
            std_history += [sig2 ** 0.5]
            if return_mask:
                mask_history += [get_mask(Xtr < tau, im.shape, patchsize).astype(np.bool)]
                
        if np.abs(sig2 - sig2_old) < 1e-6:
            break
        sig2_old = sig2
                
    if not return_history and return_mask:
        mask = get_mask(Xtr0 < tau, im.shape, patchsize).astype(np.bool)
        
    nlevel = np.sqrt(sig2)  # estimated noise levels
    th = tau  # threshold to extract weak texture patches at the last iteration.
    num = X.shape[1]  # number of extracted weak texture patches at the last iteration.
    
    if return_history:
        return nlevel, th, num, std_history, mask_history
    elif return_mask:
        return nlevel, th, num, mask
    return nlevel, th, num
