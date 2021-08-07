import numpy as np
import re
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.structure import Structure
import math
import os
import csv
import argparse
import sys
import time

parser = argparse.ArgumentParser(description='Get material representation of Orbital Field Matrix')
parser.add_argument('root', help='path to the directory of CIF files.')
args = parser.parse_args(sys.argv[1:])
cif_path = args.root

start_all = time.time()
# ----------------------------- Pre-defined dictionary -------------------------------------------
elements = {'H': ['1s2'], 'Li': ['[He] 1s2'], 'Be': ['[He] 2s2'], 'B': ['[He] 2s2 2p1'], 'N': ['[He] 2s2 2p3'],
            'O': ['[He] 2s2 2p4'],
            'C': ['[He] 2s2 2p2'], 'I': ['[Kr] 4d10 5s2 5p5'],
            'F': ['[He] 2s2 2p5'], 'Na': ['[Ne] 3s1'], 'Mg': ['[Ne] 3s2'], 'Al': ['[Ne] 3s2 3p1'],
            'Si': ['[Ne] 3s2 3p2'],
            'P': ['[Ne] 3s2 3p3'], 'S': ['[Ne] 3s2 3p4'], 'Cl': ['[Ne] 3s2 3p5'], 'K': ['[Ar] 4s1'],
            'Ca': ['[Ar] 4s2'], 'Sc': ['[Ar] 3d1 4s2'],
            'Ti': ['[Ar] 3d2 4s2'], 'V': ['[Ar] 3d3 4s2'], 'Cr': ['[Ar] 3d5 4s1'], 'Mn': ['[Ar] 3d5 4s2'],
            'Fe': ['[Ar] 3d6 4s2'], 'Co': ['[Ar] 3d7 4s2'], 'Ni': ['[Ar] 3d8 4s2'], 'Cu': ['[Ar] 3d10 4s1'],
            'Zn': ['[Ar] 3d10 4s2'],
            'Ga': ['[Ar] 3d10 4s2 4p2'], 'Ge': ['[Ar] 3d10 4s2 4p2'], 'As': ['[Ar] 3d10 4s2 4p3'],
            'Se': ['[Ar] 3d10 4s2 4p4'], 'Br': ['[Ar] 3d10 4s2 4p5'], 'Rb': ['[Kr] 5s1'],
            'Sr': ['[Kr] 5s2'], 'Y': ['[Kr] 4d1 5s2'], 'Zr': ['[Kr] 4d2 5s2'], 'Nb': ['[Kr] 4d4 5s1'],
            'Mo': ['[Kr] 4d5 5s1'],
            'Ru': ['[Kr] 4d7 5s1'], 'Rh': ['[Kr] 4d8 5s1'], 'Pd': ['[Kr] 4d10'], 'Ag': ['[Kr] 4d10 5s1'],
            'Cd': ['[Kr] 4d10 5s2'],
            'In': ['[Kr] 4d10 5s2 5p1'], 'Sn': ['[Kr] 4d10 5s2 5p2'], 'Sb': ['[Kr] 4d10 5s2 5p3'],
            'Te': ['[Kr] 4d10 5s2 5p4'], 'Cs': ['[Xe] 6s1'], 'Ba': ['[Xe] 6s2'],
            'La': ['[Xe] 5d1 6s2'], 'Ce': ['[Xe] 4f1 5d1 6s2'], 'Hf': ['[Xe] 4f14 5d2 6s2'],
            'Ta': ['[Xe] 4f14 5d3 6s2'],
            'W': ['[Xe] 4f14 5d5 6s1'], 'Re': ['[Xe] 4f14 5d5 6s2'], 'Os': ['[Xe] 4f14 5d6 6s2'],
            'Ir': ['[Xe] 4f14 5d7 6s2'], 'Pt': ['[Xe] 4f14 5d10'], 'Au': ['[Xe] 4f14 5d10 6s1'],
            'Hg': ['[Xe] 4f14 5d10 6s2'],
            'Tl': ['[Xe] 4f14 5d10 6s2 6p2'], 'Pb': ['[Xe] 4f14 5d10 6s2 6p2'],
            'Bi': ['[Xe] 4f14 5d10 6s2 6p3'],
            'Tc': ['[Kr] 4d5 5s2'], 'Fr': ['[Rn]7s1'], 'Ra': ['[Rn]7s2'], 'Pr': ['[Xe]4f3 6s2'],
            'Nd': ['[Xe] 4f4 6s2'], 'Pm': ['[Xe] 4f5 6s2'], 'Sm': ['[Xe] 4f6 6s2'],
            'Eu': ['[Xe] 4f7 6s2'], 'Gd': ['[Xe] 4f7 5d1 6s2'], 'Tb': ['[Xe] 4f9 6s2'],
            'Dy': ['[Xe] 4f10 6s2'], 'Ho': ['[Xe] 4f11 6s2'], 'Er': ['[Xe] 4f12 6s2'],
            'Tm': ['[Xe] 4f13 6s2'], 'Yb': ['[Xe] 4f14 6s2'], 'Lu': ['[Xe] 4f14 5d1 6s2'],
            'Po': ['[Xe] 4f14 5d10 6s2 6p4'], 'At': ['[Xe] 4f14 5d10 6s2 6p5'],
            'Ac': ['[Rn] 6d1 7s2'], 'Th': ['[Rn] 6d2 7s2'], 'Pa': ['[Rn] 5f2 6d1 7s2'],
            'U': ['[Rn] 5f3 6d1 7s2'], 'Np': ['[Rn] 5f4 6d1 7s2'], 'Pu': ['[Rn] 5f6 7s2'],
            'Am': ['[Rn] 5f7 7s2'], 'Cm': ['[Rn] 5f7 6d1 7s2'], 'Bk': ['[Rn] 5f9 7s2'],
            'Cf': ['[Rn] 5f10 7s2'], 'Es': ['[Rn] 5f11 7s2'], 'Fm': ['[Rn] 5f12 7s2'],
            'Md': ['[Rn] 5f13 7s2'], 'No': ['[Rn] 5f14 7s2'], 'Lr': ['[Rn] 5f14 6d1 7s2'],
            'Rf': ['[Rn] 5f14 6d2 7s2'], 'Db': ['[Rn] 5f14 6d3 7s2'],
            'Sg': ['[Rn] 5f14 6d4 7s2'], 'Bh': ['[Rn] 5f14 6d5 7s2'],
            'Hs': ['[Rn] 5f14 6d6 7s2'], 'Mt': ['[Rn] 5f14 6d7 7s2'], 'Xe': ['[Kr] 4d10 5s2 5p6'],
            'He': ['1s2'], 'Kr': ['[Ar] 3d10 4s2 4p6'], 'Ar': ['[Ne] 3s2 3p6'], 'Ne': ['[He] 2s2 2p6']}

orbitals = {"s1": 0, "s2": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5, "p5": 6, "p6": 7, "d1": 8, "d2": 9, "d3": 10,
            "d4": 11,
            "d5": 12, "d6": 13, "d7": 14, "d8": 15, "d9": 16, "d10": 17, "f1": 18, "f2": 19, "f3": 20, "f4": 21,
            "f5": 22, "f6": 23, "f7": 24, "f8": 25, "f9": 26, "f10": 27, "f11": 28, "f12": 29, "f13": 30,
            "f14": 31}

# ------------------------- hvs ----------------------------------------------------
# hvs define a dictionary , which map a element to a 32 vector representation
# according to its electronic configurations

hvs = {}

for key in elements.keys():
    element = key
    hv = np.zeros(shape=(32, 1))
    s = elements[key][0]
    sp = (re.split('(\s+)', s))
    if key == "H":
        hv[0] = 1
    if key != "H":
        for j in range(1, len(sp)):
            if sp[j] != ' ':
                n = sp[j][:1]
                orb = sp[j][1:]
                hv[orbitals[orb]] = 1
    hvs[element] = hv


# --------------------------- pre-defined functions ----------------------------------

def make_hot_for_atom_i(crystal, i, hvs):
    EP = str(crystal[i].specie)
    # nan_to_num: Replace nan with zero and inf with finite numbers.
    HV_P = np.nan_to_num(hvs[EP])
    # reshape from (32,1) to (1,32)
    AA = HV_P.reshape((HV_P.shape[1], 32))
    A = np.array(AA)
    # get the Voronoi nearest neighbours(VNN) of atom indexed by i
    b = VoronoiNN().get_nn_info(crystal, i)
    angles = []
    # store the solid angles between central atom i with VNN's in angles
    for nb in b:
        angle_K = nb['poly_info']['solid_angle']
        angles.append(angle_K)
    max_angle = max(angles)
    X_P = np.zeros(shape=(32, 32))
    tmp_X = []
    for nb in b:
        # check VNN b's specie type
        EK = str(nb['site'].specie)
        # check the solid angle between b and central atom
        angle_K = nb['poly_info']['solid_angle']
        index_K = nb['site_index']
        # calculate the distance square between b and central atom
        r_pk = ((calculateDistance(nb['site'].coords, crystal[i].coords)) * (
            calculateDistance(nb['site'].coords, crystal[i].coords)))
        # map b to the vector representation and reshape
        HV_K = hvs[EK]
        HV_K = HV_K.reshape((HV_K.shape[1], 32))
        # weight coeficients of b-central atom pair
        coef_K = (angle_K / max_angle) * ((1 / ((r_pk) ** 2)))
        HV_K_new = np.nan_to_num(coef_K * HV_K)
        # matrix product between b and central atom ---> (32, 32)
        X_PT = np.matmul(HV_P, HV_K_new)
        tmp_X.append(X_PT)
    X0 = np.zeros(shape=(32, 32))
    # el stands for one matrix(of VNN-central_atom pair)
    # weighted sum over all pairs
    for el in tmp_X:
        X0 = [[sum(x) for x in zip(el[i], X0[i])] for i in range(len(el))]
    # the size of X0 is (32, 33),not flattened
    # the first column is central atom
    X0 = np.concatenate((A.T, X0), axis=1)
    X0 = np.asarray(X0)
    # flatten arranged by column
    X0 = X0.flatten(order='F')
    return X0


def calculateDistance(a, b):  # Atom-wise OFM
    dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return dist


# ---------------------- Loading id-data(cif-id) of magnetic material-----------------------------
id_prop_file = os.path.join(cif_path, 'id_prop.csv')
with open(id_prop_file) as f:
    reader = csv.reader(f)
    id_prop_data = [row for row in reader]

id_data = []
for i in range(len(id_prop_data)):
    id_data.append(id_prop_data[i][0])

# ----------------------- Consruct OFM representation for given material-----------------------
# store in a dictionary called all_atom_embedding
material_embedding = {}
start = time.time()

for ids in id_data:
    material_embedding[ids] = 0
    crystal = Structure.from_file(os.path.join(cif_path, ids + '.cif'))
    for idx in range(len(crystal)):
        material_embedding[ids] += make_hot_for_atom_i(crystal, idx, hvs)
    material_embedding[ids] = material_embedding[ids]/len(crystal)


print("Spend ", time.time()-start, ' s to store OFM rep of materials in a dictionary material_embeddings')
print("*********************************************************************")
# --------------------------- Save as .npy file ----------------------------------

np.save('OFM_mat.npy', material_embedding)
print("Over !!!!! This script takes: ", time.time()-start_all, ' s')
