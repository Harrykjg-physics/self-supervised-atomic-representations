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

from gatgnn2.data import *
from gatgnn2.utils import *

from model import GATGNN, CGCNN

# MOST CRUCIAL DATA PARAMETERS

parser = argparse.ArgumentParser(description='get embedding from pretrained CGCNN/GATGNN')
parser.add_argument('root', help='The path where you define raw and processed dir')

parser.add_argument('--data_src', default='CGCNN', choices=['CGCNN', 'MEGNET', 'NEW'],
                    help='selection of the materials dataset to use (default: CGCNN)')

parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--emb_dim', type=int, default=64,
                    help='embedding dimensions (default: 64)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers for dataset loading')
parser.add_argument('--model_type', type=str, default="cgcnn", choices=["cgcnn", "gatgnn"])
# ---------------------------- GATGNN model -----------------------------------------
parser.add_argument('--num_layer', default=5, type=int,
                    help='number of AGAT layers to use in model (default:5)')
parser.add_argument('--num_heads', default=4, type=int,
                    help='number of Attention-Heads to use  per AGAT Layer (default:4)')
parser.add_argument('--use_hidden_layers', default=True, type=bool,
                    help='option to use hidden layers following global feature summation (default:True)')
parser.add_argument('--global_attention', default='composition', choices=['composition', 'cluster']
                    , help='selection of the unpooling method as referenced in paper GI M-1 to GI M-4 ('
                           'default:composition)')
parser.add_argument('--cluster_option', default='fixed', choices=['fixed', 'random', 'learnable'],
                    help='selection of the cluster unpooling strategy referenced in paper GI M-1 to GI M-4 (default: '
                         'fixed)')
parser.add_argument('--concat_comp', default=False, type=bool,
                    help='option to re-use vector of elemental composition after global summation of crystal '
                         'feature.(default: False)')
parser.add_argument('--dropout_ratio', type=float, default=0,
                    help='dropout ratio (default: 0)')
parser.add_argument('--input_model_file', type=str, default='', help='filename to output the model')

args = parser.parse_args(sys.argv[1:])

global all_atom_embedding

# ----------------------------- 3/31 -----------------------------
all_atom_embedding = {}

# SETTING UP CODE TO RUN ON GPU
gpu_id = 0
# device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# DATA PARAMETERS
random_num = 456
random.seed(random_num)

# DATALOADER/ TARGET NORMALIZATION

dataset = pd.read_csv(f'{args.root}/raw/id_prop.csv',
                      names=['material_ids', 'label']).sample(frac=1, random_state=random_num)

# dataset = pd.read_csv(f'D:\\learn\\1\\D\\Pycode\\Project\\pretrain-m\\Material\\DATA\\raw\\id_prop.csv', names=['material_ids', 'label']).sample(frac=1, random_state=random_num)
# here the root dir contains CIF files of materials, i.e. the raw folder
CRYSTAL_DATA = CIF_Dataset(dataset, root_dir=args.root)
idx_list = list(range(len(dataset)))
testing_set = CIF_Lister(crystals_ids=idx_list, full_dataset=CRYSTAL_DATA, df=dataset)

test_param = {'batch_size': args.batch_size, 'shuffle': False}
loader = torch_DataLoader(dataset=testing_set, **test_param)
print("This is length of dataset: ", len(dataset))

# loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
# NEURAL-NETWORK

if args.model_type == 'cgcnn':
    model = CGCNN(n_conv=args.num_layer,
                  atom_fea_len=args.emb_dim,
                  drop_ratio=args.dropout_ratio,
                  stage='get_emb').to(device)

elif args.model_type == 'gatgnn':
    model = GATGNN(heads=args.num_heads,
                   neurons=args.emb_dim,
                   nl=args.num_layer,
                   # global_attention=args.global_attention,
                   # unpooling_technique=args.cluster_option,
                   stage='get_emb').to(device)
else:
    raiseNotImplementedError("Model Type Not Implemented !!")

# LOADING MODEL
if args.model_type == 'cgcnn':
    try:
        checkpoint = torch.load(f'{args.input_model_file}', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        model_stat_dict = model.state_dict()
        pretrained_model_stat_dict = torch.load(f'{args.input_model_file}', map_location='cpu')
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        # ---------------------- 6/4 fix key not match problem for contrastive ----------------------
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

if args.model_type == 'gatgnn':
    model.load_state_dict(torch.load(f'gatgnn.pth', map_location=device))

# ---------------------------- 4/1 ------------------------
# print("This is the number of AGAT layers: ", args.num_layer)
# ----------------------------------------------------------

# METRICS-OBJECT INITIALIZATION

print(f'> Forward Pass ...')

# TESTING PHASE
model.eval()
# ------------------- 3/31 -----------------------
count = 0
# ------------------------------------------------

# ------------------------- dataset in CIFDATA.py ---4/24------------------------------------
# D:\\learn\\1\\D\Pycode\\Project\\pretrain-m\\Material\\DATA\\raw\\id_prop.csv
# /home/zhujiaji/github_git_clone/Local-Env-Descriptor-of-Material-for-Physics-Discovery/DATA/raw/id_prop.csv

# dataset1 = pd.read_csv(f'/home/zhujiaji/github_git_clone/Local-Env-Descriptor-of-Material-for-Physics-Discovery/DATA/raw/id_prop.csv',
#                       names=['material_ids', 'label']).sample(frac=1, random_state=456)
# -----------------------------------------------------------------------------------------------

for data in loader:
    count += 1
    print("------------------------------------------")
    print("This is the", count, "th batch")
    print("------------------------------------------")
    data = data.to(device)
    # print("This is the batch data index: ", data.the_idx)
    batch_cif_ids = dataset.iloc[data.the_idx].material_ids.values
    if count == 1:
        print("This is the 1st batch material id : ", batch_cif_ids)

    # ---------------------- 4/24 ----------------------------------
    batch_atom_name = []
    for cif_id in batch_cif_ids:
        crystal = Structure.from_file(
            os.path.join(args.root, 'raw/' + cif_id + '.cif'))
        for i in range(len(crystal)):
            batch_atom_name += [str(crystal[i].specie) +
                                '_' + str(crystal[i].specie.number)
                                + '_' + str(i + 1)]
    # if count == 1:
    #    print("This is the batch_atom_name : ", len(batch_atom_name), batch_atom_name)
    # -------------------------------------------------------------

    # --------------------- 4/28 Add unique magmom ---------------------------------
    # D:\\learn\\1\D\\Pycode\\Project\\cgcnn-m\\id_prop_non_zero_1.csv

    with open("/home/zhujiaji/github_git_clone/Local-Env-Descriptor-of-Material-for-Physics-Discovery/cgcnn-m/id_prop_non_zero_1.csv") as f:
        reader = csv.reader(f)
        magmom_data = [row for row in reader]
    magmom = []
    for i in range(len(magmom_data)):
        magmom.append(magmom_data[i][0])

    # --------------------- 4/28 Add unique magmom ---------------------------------

    with torch.no_grad():
        atom_embedding = model(data)

        # --------------------------- 4/24 -------------------------------------------------
        index = -1
        for n_atoms in range(len(batch_atom_name)):
            count1 = int(batch_atom_name[n_atoms].split('_')[-1])
            if count1 == 1:
                index += 1
            list1 = [atom_embedding[i][n_atoms] for i in range(args.num_layer+1)]
            uni_atom_name = batch_cif_ids[index] + '_' + batch_atom_name[n_atoms]
            if uni_atom_name in magmom:
                all_atom_embedding[uni_atom_name] = list1
    # -----------------------------------------------------------------------

if args.model_type == 'cgcnn':
    np.save('sl_cgcnn_embedding_dict_7_19.npy', all_atom_embedding)

if args.model_type == 'gatgnn':
    np.save('sl_gatgnn_embedding_dict.npy', all_atom_embedding)
