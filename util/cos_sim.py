import argparse
import os
from random import random
import torch
import torch.nn as nn
import pandas as pd
import sys
import csv
import numpy as np
from pymatgen.core.structure import Structure

from cgcnn.data import *
from cgcnn.model import CGCNN

parser = argparse.ArgumentParser(description='Get embeddings from a pretrained CGCNN')
parser.add_argument('root', help='The path where you save your structure files')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
# ----------------------------  Model Params-----------------------------------------
parser.add_argument('--model_type', type=str, default="cgcnn", choices=["cgcnn", "gatgnn"])
parser.add_argument('--num_layer', default=5, type=int,
                    help='number of AGAT layers to use in model (default:5)')
parser.add_argument('--use_hidden_layers', default=True, type=bool,
                    help='option to use hidden layers following global feature summation (default:True)')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimensions (default: 64)')
parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio (default: 0)')
# ------------------------------------------------------------------------------
parser.add_argument('--input_model_file', type=str, default='', help='filename to output the model')
parser.add_argument('--lth_emb', type=str, default='01', choices=["0", "1", "2", "3", "4", "5", "01", "012",
                                                                  "0123", "01234", "012345"],
                    help='generate embeddings under lth convolutional layer(default:01)')
parser.add_argument('--partial_csv', type=str, default=None, help='the path to the csv file with specified atoms')

args = parser.parse_args(sys.argv[1:])


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


# gpu_id = 0
# device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

if args.partial_csv is not None:
    with open(args.partial_csv) as f:
        reader = csv.reader(f)
        id_prop = [row for row in reader]
        id = []
        for i in range(len(id_prop)):
            id.append(id_prop[i][0])

dataset = pd.read_csv(f'{args.root}/id_prop.csv', names=['material_ids', 'label'])
CRYSTAL_DATA = CIF_Dataset(dataset, root_dir=args.root)
idx_list = list(range(len(dataset)))
testing_set = CIF_Lister(crystals_ids=idx_list, full_dataset=CRYSTAL_DATA, df=dataset)

test_param = {'batch_size': args.batch_size, 'shuffle': False}
loader = torch_DataLoader(dataset=testing_set, **test_param)
print("This is length of dataset: ", len(dataset))

if args.model_type == 'cgcnn':
    model = CGCNN(n_conv=args.num_layer,
                  atom_fea_len=args.emb_dim,
                  drop_ratio=args.dropout_ratio,
                  stage='get_emb').to(device)

elif args.model_type == 'gatgnn':
    raiseNotImplementedError("Model Type Not Implemented !!")
else:
    raiseNotImplementedError("Model Type Not Implemented !!")

if args.model_type == 'cgcnn':
    try:
        checkpoint = torch.load(f'{args.input_model_file}', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        model_stat_dict = model.state_dict()
        pretrained_model_stat_dict = torch.load(f'{args.input_model_file}', map_location='cpu')

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in pretrained_model_stat_dict['model_state_dict'].items():
            string = k.split(".")
            if string[0] == 'gnn':
                name = string[1]
                for item in string[2:]:
                    name += ("." + item)
            new_state_dict[name] = v

        model_stat_dict.update(new_state_dict)
        model.load_state_dict(model_stat_dict)

print(f'> Forward Pass ...')

model.eval()

count = 0

id_cs = {}

for data in loader:

    all_atom_embedding = {}

    count += 1
    data = data.to(device)
    batch_cif_ids = dataset.iloc[data.the_idx].material_ids.values
    batch_atom_name = []
    for cif_id in batch_cif_ids:
        crystal = Structure.from_file(os.path.join(args.root, str(cif_id) + '.cif'))
        for i in range(len(crystal)):
            batch_atom_name += [str(crystal[i].specie) +
                                '_' + str(crystal[i].specie.number)
                                + '_' + str(i + 1)]

    with torch.no_grad():
        atom_embedding = model(data)
        index = -1
        for atom_idx in range(len(batch_atom_name)):
            count1 = int(batch_atom_name[atom_idx].split('_')[-1])
            if count1 == 1:
                index += 1
            list1 = [atom_embedding[i][atom_idx] for i in range(args.num_layer + 1)]
            uni_atom_name = str(batch_cif_ids[index]) + '_' + batch_atom_name[atom_idx]
            if len(args.lth_emb) == 1:
                tmp = list1[int(args.lth_emb)]
                if args.partial_csv is None:
                    all_atom_embedding[uni_atom_name] = tmp.detach().numpy()
                else:
                    if uni_atom_name in id:
                        all_atom_embedding[uni_atom_name] = tmp.detach().numpy()
            else:
                tmp = torch.cat([list1[i] for i in range(len(args.lth_emb))])
                if args.partial_csv is None:
                    all_atom_embedding[uni_atom_name] = tmp.detach().numpy()
                else:
                    if uni_atom_name in id:
                        all_atom_embedding[uni_atom_name] = tmp.detach().numpy()

    if len(all_atom_embedding) > 1:
        sum_cs = 0
        sum_count = 0
        for i in range(len(all_atom_embedding)):
            for j in range(i + 1, len(all_atom_embedding)):
                cs = cos_sim(all_atom_embedding[list(all_atom_embedding.keys())[i]],
                             all_atom_embedding[list(all_atom_embedding.keys())[j]])
                sum_cs += cs
                sum_count += 1
        sum_cs_avg = sum_cs / sum_count
        id_cs[batch_cif_ids[0]] = sum_cs_avg

        np.save("id_cs.npy", id_cs)
