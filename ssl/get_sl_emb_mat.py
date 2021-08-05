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

parser = argparse.ArgumentParser(description='get embedding from pretrained CGCNN/GATGNN')
parser.add_argument('root', help='The path where you define raw and processed dir')
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--emb_dim', type=int, default=64,
                    help='embedding dimensions (default: 64)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers for dataset loading')
parser.add_argument('--model_type', type=str, default="cgcnn", choices=["cgcnn", "gatgnn"])
parser.add_argument('--num_layer', default=5, type=int,
                    help='number of AGAT layers to use in model (default:5)')
parser.add_argument('--use_hidden_layers', default=True, type=bool,
                    help='option to use hidden layers following global feature summation (default:True)')
parser.add_argument('--dropout_ratio', type=float, default=0,
                    help='dropout ratio (default: 0)')
parser.add_argument('--lth_emb', type=str, default='01', choices=["0", "1", "2", "3", "4", "5", "01", "012",
                                                                  "0123", "01234", "012345"],
                    help='generate embeddings under lth convolutional layer(default:01)')
parser.add_argument('--input_model_file', type=str, default='', help='filename to output the model')

args = parser.parse_args(sys.argv[1:])

material_embedding = {}

# gpu_id = 0
# device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

dataset = pd.read_csv(args.root + '/id_prop.csv', names=['material_ids', 'label'])
CRYSTAL_DATA = CIF_Dataset(dataset, root_dir=args.root)
idx_list = list(range(len(dataset)))

testing_set = CIF_Lister(crystals_ids=idx_list, full_dataset=CRYSTAL_DATA, df=dataset)
test_param = {'batch_size': args.batch_size, 'shuffle': False}
loader = torch_DataLoader(dataset=testing_set, **test_param)

if args.model_type == 'cgcnn':
    model = CGCNN(n_conv=args.num_layer,
                  atom_fea_len=args.emb_dim,
                  drop_ratio=args.dropout_ratio,
                  stage='get_emb').to(device)

if args.model_type == 'gatgnn':
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
        for k, v in pretrained_model_stat_dict.items():
            string = k.split(".")
            if string[0] == 'gnn':
                name = string[1]
                for item in string[2:]:
                    name += ("." + item)
            new_state_dict[name] = v

        model_stat_dict.update(new_state_dict)
        model.load_state_dict(model_stat_dict)

if args.model_type == 'gatgnn':
    model.load_state_dict(torch.load(f'gatgnn.pth', map_location=device))

print(f'> Forward Pass ...')

model.eval()

count = 0

for data in loader:
    count += 1
    data = data.to(device)
    batch_cif_ids = dataset.iloc[data.the_idx].material_ids.values
    if count == 1:
        print("This is the 1st batch material id : ", batch_cif_ids)

    with torch.no_grad():
        atom_embedding = model(data)
        base_idx = 0
        for cif_id in batch_cif_ids:
            crystal = Structure.from_file(
                os.path.join(args.root, str(cif_id) + '.cif'))
            length = len(crystal)
            material_embedding[cif_id] = 0
            for idx in range(length):
                batch_atom_idx = base_idx + idx
                if len(args.lth_emb) == 1:
                    tmp = atom_embedding[int(args.lth_emb)][batch_atom_idx]
                    tmp = tmp.detach().numpy()
                else:
                    tmp = torch.cat([atom_embedding[i][batch_atom_idx] for i in range(len(args.lth_emb))])
                    tmp = tmp.detach().numpy()
                material_embedding[cif_id] += tmp
            base_idx += length

if args.model_type == 'cgcnn':
    np.save('sl_cgcnn_mat_' + str(args.lth_emb) + 'embedding_dict.npy', material_embedding)

if args.model_type == 'gatgnn':
    np.save('sl_gatgnn_mat_embedding_dict.npy', material_embedding)
