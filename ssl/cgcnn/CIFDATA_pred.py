import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain

from .data_pred import *


class CIF_Cry_Dataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):
        """
        :param root: selection of the materials dataset to use (default: CGCNN)
        The directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the CIF, and the
        processed dir can either empty or a previously processed file
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """

        self.root = root
        super(CIF_Cry_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_list = []
        dataset = pd.read_csv(f'{self.root}/raw/id_prop.csv', names=['material_ids', 'label']).sample(frac=1, random_state=456)
        CRYSTAL_DATA = CIF_Dataset(dataset, root_dir=f'{self.root}/')
        print("-------- CRYSTAL_DATA Ready!!!!! ------------------")
        print("This is the length of original dataset")
        print(len(dataset))
        idx_list = list(range(len(dataset)))
        # print(idx_list)
        datalist = CIF_Lister(idx_list, CRYSTAL_DATA, df=dataset, src='CGCNN')

        for idx in range(len(datalist)):
            data_list.append(datalist[idx])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
