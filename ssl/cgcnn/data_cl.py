import numpy as np
import pandas as pd
import functools
import torch
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as torch_Dataset
from torch_geometric.data import Data, DataLoader as torch_DataLoader
import sys, json, os
from pymatgen.core.structure import Structure
import warnings
import random

from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class ELEM_Encoder:
    def __init__(self):
        self.elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                         'Ar', 'K',
                         'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                         'Kr', 'Rb',
                         'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                         'Xe', 'Cs',
                         'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                         'Hf', 'Ta',
                         'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                         'Th', 'Pa',
                         'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        self.e_arr = np.array(self.elements)

    def encode(self, composition_dict):
        answer = [0] * len(self.elements)

        elements = [str(i) for i in composition_dict.keys()]
        counts = [j for j in composition_dict.values()]
        total = sum(counts)

        for idx in range(len(elements)):
            elem = elements[idx]
            ratio = counts[idx] / total
            idx_e = self.elements.index(elem)
            answer[idx_e] = ratio
        return torch.tensor(answer).float().view(1, -1)

    def decode_pymatgen_num(tensor_idx):
        idx = (tensor_idx - 1).cpu().tolist()
        return self.e_arr[idx]


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIF_Lister(Dataset):
    def __init__(self, crystals_ids, full_dataset, df=None, src='CGCNN'):
        self.crystals_ids = crystals_ids
        self.full_dataset = full_dataset
        self.material_ids = df.iloc[crystals_ids].values[:, 0].squeeze()
        self.src = src

    def __len__(self):
        return len(self.crystals_ids)

    def extract_ids(self, original_dataset):
        names = original_dataset.iloc[self.crystals_ids]
        return names

    def __getitem__(self, idx):
        i = self.crystals_ids[idx]
        material = self.full_dataset[i]
        n_features = material[0][0]
        e_features = material[0][1]
        e_features = e_features.view(-1, 41)
        a_matrix = material[0][2]
        y = material[2]
        atom_specie_number = material[3]

        graph_crystal = Data(x=n_features, y=y, edge_attr=e_features, edge_index=a_matrix,
                             num_atoms=torch.tensor([len(n_features)]).float(),
                             the_idx=torch.tensor([float(i)]), atom_specie_number=atom_specie_number)

        return graph_crystal


class CIF_Dataset(Dataset):
    def __init__(self, part_data=None, norm_obj=None, normalization=None, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 root_dir='DATA/CIF-DATA/', cell_size=None):

        self.root_dir = root_dir
        self.cell_size = cell_size
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.normalizer = norm_obj
        self.normalization = normalization
        self.full_data = part_data
        self.ari = AtomCustomJSONInitializer(self.root_dir + 'atom_init.json')
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.encoder_elem = ELEM_Encoder()

    def __len__(self):
        return len(self.partial_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.full_data.iloc[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, str(cif_id) + '.cif'))
        if self.cell_size is not None:
            if self.cell_size == 1:
                crystal.make_supercell([1, 1, 1])
            elif self.cell_size == 2:
                crystal.make_supercell([1, 1, 2])
            elif self.cell_size == 4:
                crystal.make_supercell([1, 2, 2])
            elif self.cell_size == 8:
                crystal.make_supercell([2, 2, 2])
            elif self.cell_size == 77:
                aug_list = [[1, 1, 2], [1, 1, 1], [1, 2, 2], [2, 2, 2], 'None']
                random_choice = random.choice(aug_list)
                # print(random_choice)
                if random_choice != "None":
                    crystal.make_supercell(random_choice)
                else:
                    pass
            else:
                pass

        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        atom_index = np.vstack([crystal[i].specie.number for i in range(len(crystal))])
        atom_index = np.squeeze(atom_index)
        atom_index = torch.LongTensor(atom_index)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))
        target = torch.Tensor([float(target)])

        g_coords = crystal.cart_coords
        coordinates = torch.tensor(g_coords)

        return (atom_fea, nbr_fea, nbr_fea_idx), coordinates, target, atom_index, \
               [crystal[i].specie for i in range(len(crystal))]

    def format_adj_matrix(self, adj_matrix):
        size = len(adj_matrix)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x] * adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes, all_dst_nodes), dim=0)
