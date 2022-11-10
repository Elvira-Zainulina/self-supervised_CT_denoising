import numpy as np
import pydicom
import os
import struct
import torch
from torch.utils.data import Dataset


VR2type = {'FL': 'f', 'US': 'H', 'CS': 'str', 'DS': 'str'}
VRsizes = {'FL': 4, 'US': 2, 'CS': 16, 'DS': 16}


def read_private_data(proj, group, elem, VR):
    private_creator = proj.private_creators(group)[0]
    b_val = proj.get_private_item(group, elem, private_creator).value

    if VR == 'CS':
        return b_val.decode()
    if VR == 'DS':
        return float(b_val.decode())

    count = len(b_val) // VRsizes[VR]
    if count == 1: return struct.unpack(VR2type[VR], b_val)[0]
    return struct.unpack(VR2type[VR] * count, b_val)


def random_crop(im1, im2=None, crop_size=64):
    i = torch.randint(0, im1.shape[-2] - crop_size, (1,)).item() if im1.shape[-2] > crop_size else 0
    j = torch.randint(0, im1.shape[-1] - crop_size, (1,)).item() if im1.shape[-1] > crop_size else 0
    im1 = im1[..., i:i+crop_size, j:j+crop_size]
    if im2 is None:
        return im1, j
    im2 = im2[..., i:i+crop_size, j:j+crop_size]
    return im1, im2, j


class LDCT_dataset(Dataset):
    def __init__(self, ld_paths, objects, fd_paths=None, cut=False,
                 inv=True, simulated=False,
                 num_add_prj=0, adj_type='TS', fill_type='replicate'):

        assert adj_type in ['TS', 'new_channels', 'depth']
        assert fill_type in ['replicate', 'circular']
        # circular mode is used for the case of the axial projection geometry

        self.ld_paths = ld_paths
        self.fd_paths = fd_paths
        self.objects = objects
        self.inv = inv
        self.transform = None
        if cut:
            self.transform = random_crop
        self.adj_type = adj_type
        self.num_add_prj = num_add_prj
        self.fill_type = fill_type
        if simulated:
            self.load_prj = self.load_prj_numpy
        else:
            self.load_prj = self.load_prj_dicom

        self.ld_prj_paths = []
        for i, ld_path in enumerate(self.ld_paths):
            self.ld_prj_paths += [os.path.join(ld_path, current_obj) for current_obj in objects[i]]
        if fd_paths is not None:
            self.fd_prj_paths = []
            for i, fd_path in enumerate(self.fd_paths):
                if simulated:
                    # simulated low-dose projections are of type "file_name.dcm.npz"
                    # full-dose projections are of type "file_name.dcm"
                    self.fd_prj_paths += [os.path.join(fd_path, '.'.join(current_obj.split('.')[:-1]))
                                          for current_obj in objects[i]]
                else:
                    self.fd_prj_paths += [os.path.join(fd_path, current_obj)
                                          for current_obj in objects[i]]

        if self.fill_type == 'circular':
            self.num_prjs = read_private_data(pydicom.dcmread(self.ld_prj_paths[0]),
                                              0x7033, 0x0013, 'US')

    def __len__(self):
        return len(self.ld_prj_paths)

    def load_prj_numpy(self, prj_path, return_mA=False):
        im = np.load(prj_path)['ld_im'].astype(np.float32).T
        if self.inv:
            im = np.exp(-im)
        if return_mA:
            return im[None], np.load(prj_path)['XRayTubeCurrent']
        return im[None]

    def load_prj_dicom(self, prj_path, return_mA=False):
        prj = pydicom.dcmread(prj_path)
        im = prj.pixel_array.T.astype(np.float32) * float(prj.RescaleSlope) \
             + float(prj.RescaleIntercept)
        if self.inv:
            im = np.exp(-im)
        if return_mA:
            return im[None], prj.XRayTubeCurrent
        return im[None]

    def replicate_fill_packs(self, pack0, pack1, side='left'):
        assert len(pack0) > 0 or len(pack1) == self.num_add_prj
        if len(pack0) > 0:
            add_ims = [pack0[-1].copy() for _ in range(self.num_add_prj - len(pack0))]
            if side == 'left':
                return add_ims + pack0[::-1]
            return pack0 + add_ims
        return pack1[::-1]

    def circular_fill_pack(self, pack, circle, side='left'):
        l = self.num_add_prj - len(pack)
        if side == 'left':
            add_ims = [self.load_prj(self.ld_prj_paths[circle * self.num_prjs - i - 1])
                       for i in range(l - 1, -1, -1)]
            return add_ims + pack[::-1]
        add_ims = [self.load_prj(self.ld_prj_paths[(circle - 1) * self.num_prjs + i])
                   for i in range(l)]
        return pack + add_ims

    def __getitem__(self, idx):

        prj_num = int(self.ld_prj_paths[idx].split('-')[-1].split('.')[0])
        ld_im, mA = self.load_prj(self.ld_prj_paths[idx], return_mA=True)

        if idx < 0: idx = len(self.ld_prj_paths) + idx

        if self.num_add_prj > 0:

            left_ims = []
            for i in range(1, min(self.num_add_prj, idx) + 1):
                adj_prj_num = int(self.ld_prj_paths[idx - i].split('-')[-1].split('.')[0])
                if adj_prj_num == prj_num - i:
                    left_ims.append(self.load_prj(self.ld_prj_paths[idx - i]))
                else:
                    break

            right_ims = []
            for i in range(1, min(self.num_add_prj + 1, len(self.ld_prj_paths) - idx)):
                adj_prj_num = int(self.ld_prj_paths[idx + i].split('-')[-1].split('.')[0])
                if adj_prj_num == prj_num + i:
                    right_ims.append(self.load_prj(self.ld_prj_paths[idx + i]))
                else:
                    break

            if self.fill_type == 'circular':
                circle = idx // self.num_prjs + 1
                left_ims = self.circular_fill_pack(left_ims, circle)
                right_ims = self.circular_fill_pack(right_ims, circle, side='right')
            else:
                if len(left_ims) == 0:
                    right_ims = self.replicate_fill_packs(right_ims, left_ims, side='right')
                    left_ims = self.replicate_fill_packs(left_ims, right_ims)
                else:
                    left_ims = self.replicate_fill_packs(left_ims, right_ims)
                    right_ims = self.replicate_fill_packs(right_ims, left_ims, side='right')
            ld_ims = left_ims + [ld_im] + right_ims

            # concatenate
            if self.adj_type == 'TS':
                ld_im = np.stack(ld_ims, axis=0)
            elif self.adj_type == 'depth':
                ld_im = np.stack(ld_ims, axis=1)
            else:
                ld_im = np.concatenate(ld_ims, axis=0)

        ld_im = torch.from_numpy(ld_im).float()
        mA = torch.tensor(mA, dtype=torch.float32).unsqueeze(0)

        if self.fd_paths is None:
            if self.transform is None:
                return ld_im, 0, mA
            ld_im, ix = self.transform(ld_im)
            return ld_im, ix, mA

        fd_im = self.load_prj_dicom(self.fd_prj_paths[idx])
        fd_im = torch.from_numpy(fd_im).float()

        if self.transform is not None:
            ld_im, fd_im, ix = self.transform(ld_im, fd_im)
            return ld_im, fd_im, ix, mA

        return ld_im, fd_im, 0, mA
