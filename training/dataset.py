"""Import dataset specified in the config and wrap it into a PyTorch Dataset subclass object."""

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision

# Bring tomography_utils packages onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'tomography_utils')))
from tomography_utils.projector import get_and_save_projector

class MnistDataset(Dataset):
    """MNIST dataset with a single digit class - created just for tests"""
    def __init__(self, digit, path):
        _dataset = torchvision.datasets.MNIST(path, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))
        idx = _dataset.targets==digit
        self._dataset = (_dataset.data[idx].unsqueeze(1) / 127.5) - 1

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
    
    @property
    def num_channels(self): # C
        assert self._dataset.shape[1] == 1
        return 1

    @property
    def resolution(self):
        return [self._dataset.shape[-2], self._dataset.shape[-1]]
   
class BrainWebDatasetLowHighCountPair(Dataset):
    """PET images constructed from BrainWeb dataset with concatenated pair of (low-count, high-count)
    photons reconstructed images along width dimension. Images format N x C x H x W and  here N = 20*281, H = W = 64, C = 1.
    We assume that path directory give access to two directories, named 'recon-noisy' and 
    'recon-noisy-restricted' containing npz files of high-count and low count reconstructed images respectively.
    Each npz file rapresents a single slice of a single patient so here we have 20 patients x 281 slices.
    To be specific here npz file are named : 'low_res_recon_idxslicex.npz' as we are dealing with 64 x 64 pet images.
    Need to be preprocessed the first time we run the code, so put to false if it's first run."""
    def __init__(self,
                 path,                          # path to the dataset 
                 preprocessed=True
    ):
        super().__init__()
        self._path = path
        self._high_dose = os.path.join(path, 'recon-noisy')
        self._low_dose = os.path.join(path, 'recon-noisy-restricted')
        if not preprocessed :
            self._preprocess()
        # Here we save it as "low_res_data bcs we are dealing with 64x64 images
        _data_path = os.path.join(self._path, 'low_res_data.npz')
        self._data = np.load(_data_path)['data'] 
        self._data = self._data / (self._data.max() / 2) - 1

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]
        
    @property
    def num_channels(self): # C
        assert self._data.shape[1] == 1
        return 1

    @property
    def resolution(self):
        return [self._data.shape[-2], self._data.shape[-1]]
        
    def _preprocess(self):
        data = []
        for f_high, f_low in zip(os.listdir(self._high_dose), os.listdir(self._low_dose)):
            # Check both low count and high count coresponds to same patient and slice
            assert f_high[-14:] == f_low[-14:]
            file_high = os.path.join(self._high_dose, f_high)
            file_low = os.path.join(self._low_dose, f_low)
            data.append(np.concatenate((np.load(file_high)['arr_0'], np.load(file_low)['arr_0']), axis=0))
        data = np.stack(data)[:, None, ...]
        data_path = os.path.join(self._path, 'low_res_data.npz')
        np.savez(data_path, data=data)        

class BrainWebSinogramsClean(Dataset):
    """Clean sinograms constructed from BrainWeb dataset and renormalized by dose_mean_int. 
    Images format N x H x W and here N = 20*281 and H = W = 64.
    We assume that path directory give access to a directory 'true-phantom' 
    containing directory 'train' and 'infer' containing npz files of synthetic phantoms.
    Each npz file rapresents a single slice of a single patient so here we have 20 patients x 281 slices.
    To be specific npz file are named : 'low_res_ph_id0slice0.npz' as we are dealing with 64 x 64 phantoms.
    If mode = train then select the first 19 patients out of 20. If mode = infer then selct the last patient. 
    Need to be preprocessed the first time we run the code in both train and infer mode, so put to false if it's first run."""
    def __init__(self,
                path,
                dose_mean_int=1e6, 
                mode='train',
                preprocessed=True,
        ):
        assert mode in ['train', 'infer']
        self._mode = mode
        super().__init__()
        # Give a name to the dataset from the data path
        self._name = os.path.splitext(os.path.basename(path))[0]
        # Do it once first time you run the code
        if not preprocessed :
            self._path = path
            self._preprocess()
        # load phantoms, numpy array of shape N x H x W where N = 19*281 or 281 and H = W = 64
        _phantoms_path = os.path.join(path, f'low_res_phantoms_{mode}.npz')
        _phantoms = np.load(_phantoms_path)['phantoms'] 
        if mode == 'train' : 
            assert _phantoms.shape == (281*19, 64*64), f"_phantoms.shape{_phantoms.shape}"
        else :
            assert _phantoms.shape == (281, 64*64), f"_phantoms.shape{_phantoms.shape}"
        # Projection matrix 
        _projector_dir_path = os.path.join('tomography_utils', 'projector_matrices')
        _projector_filepath = os.path.join(_projector_dir_path, 'projector_64x64.npz')
        # Need to be done once first time you run the code
        if not os.path.exists(_projector_dir_path):
            os.mkdir(_projector_dir_path)
        if not os.path.isfile(_projector_filepath) : 
            get_and_save_projector(_projector_filepath, 64)
        _projector = np.load(_projector_filepath)['projector']
        assert _projector.shape == (2*64*64, 64*64)
        self._sinograms_clean = np.matmul(_phantoms, _projector.T)
        assert self._sinograms_clean.shape == ((281*19, 2*64*64) if mode == 'train' else (281, 2*64*64))
        self._sinogram_clean = self._sinograms_clean/self._sinograms_clean.sum(axis=-1)[:, None] * dose_mean_int
    
    def __len__(self):
        return len(self._sinogram_clean)
    
    def __getitem__(self, idx):
        return self._sinogram_clean[idx]
    
    @property
    def num_channels(self): # C
        return 1

    @property
    def resolution(self):
        return [64, 64]
        
    def _preprocess(self):
        phantoms_path = os.path.join(self._path, 'true-phantoms', self._mode)
        phantoms = []
        for file in os.listdir(phantoms_path):
            file_path = os.path.join(phantoms_path, file)
            phantoms.append(np.load(file_path)['arr_0'].ravel())
        phantoms = np.stack(phantoms)
        phantoms_path = os.path.join(self._path, f'low_res_phantoms_{self._mode}.npz')
        np.savez(phantoms_path, phantoms=phantoms)

    @property
    def name(self):
        return self._name




