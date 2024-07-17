import sys
from .dataloader import read_file

import os
import random
from math import floor
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform
sys.path.append("../../")
from utils import set_seed
from utils import download_url, extract_tar, decide_download 

class JetClassTransform(BaseTransform):
    def __call__(self, data):
        eta = data.x[1, :]  # Index 1 for `part_eta`
        phi = data.x[2, :]  # Index 2 for `part_phi`
        
        data.pos = torch.stack([eta, phi])
        data.coords = torch.cat([data.pos, data.x[[0, 3], :]]).T # indexes 0 and 3 represent part_pt, part_energy
        data.pos = data.pos.T
        data.x = data.x.T
        return data

class JetClass(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, seed=42):
        set_seed(seed)
        self.url_raw = "https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar"
        super(JetClass, self).__init__(root, transform, pre_transform)
        loaded_data = torch.load(self.processed_paths[0])
        if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
            self.data, self.slices, self.idx_split = loaded_data
        else:
            raise ValueError("Unexpected data format in the processed file")
        self.x_dim = 4
        self.coords_dim = 2 + 2 # (pt, eta, phi, energy)

    def download(self):
        warning = "This dataset will need 7.6 GB of space. Do you want to proceed? (y/n) \n"
        if osp.exists(self.processed_paths[0]):
            return
        if decide_download(self.url_raw) and input(warning).lower() == "y":
            path = download_url(self.url_raw, self.root)
            extract_tar(path, self.root+"/raw")
            os.unlink(path)
        else:
            print("stop downloading")
            shutil.rmtree(self.root)
            exit(-1)
    
    def process(self):
        all_data = []
        for filepath in self.raw_paths:
            x_particles, x_jet, y = read_file(filepath)
            # x_jet = torch.from_numpy(x_jet).unsqueeze(-1)
            x_particles = torch.from_numpy(x_particles)
            y = torch.from_numpy(y)
            # x_combined = torch.cat((x_particles, x_jet), dim=2) # uncomment if we want to use jets in dataset
            for i in range(floor(x_particles.shape[0]/2500)):
                sample_data = Data(x=x_particles[i], y=y[i])
                all_data.append(sample_data)

        #shuffle data
        random.shuffle(all_data)
        num_samples = len(all_data)
        train_size = int(num_samples * 0.6)
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]

        idx_split = self.get_idx_split(train_data, val_data)

        data, slices = self.collate(train_data + val_data)  
        torch.save((data, slices, idx_split), self.processed_paths[0])

    def get_idx_split(self, train_data, val_test_data):
        n_train = len(train_data)
        n_valid = len(val_test_data) // 2
        n_test = len(val_test_data) - n_valid
        dataset_len = n_train + n_valid + n_test

        idx = np.arange(dataset_len)
        np.random.shuffle(idx[n_train:])
        train_idx = idx[:n_train]
        valid_idx = idx[n_train : n_train + n_valid]
        test_idx = idx[n_train + n_valid :]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def raw_file_names(self):
        raw_dir = os.path.join(self.root, 'raw', 'val_5M')
        return [os.path.join('val_5M', f) for f in os.listdir(raw_dir) if f.endswith('.root')]
        # return ["val_5M/HToBB_120.root", "val_5M/TTBarLep_121.root"]

    def processed_file_names(self):
        return ["data.pt"]
        

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Now construct the relative path
    relative_path = '../../../data/jetclass'
    absolute_path = os.path.abspath(relative_path)
    root = absolute_path
    dataset = JetClass(root)
    