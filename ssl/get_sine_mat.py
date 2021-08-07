import csv
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import os
import dscribe
from dscribe.descriptors import SineMatrix
import argparse
import sys
import numpy as np
import time

parser = argparse.ArgumentParser(description='generate 1d sine matrix description of materials')
parser.add_argument('root', help='path to the directory of CIF files.')
args = parser.parse_args(sys.argv[1:])

start_all = time.time()

cif_path = args.root

# ---------------------- Loading id-data -----------------------------
id_prop_file = os.path.join(cif_path, 'id_prop.csv')
with open(id_prop_file) as f:
    reader = csv.reader(f)
    id_prop_data = [row for row in reader]

id_data = []
for i in range(len(id_prop_data)):
    id_data.append(id_prop_data[i][0])

# ----------------------- bulk_dict, position_dict, atom_name_dict -----------------------------------------------------
# convert Structure type to ASE type
Ada = AseAtomsAdaptor()
bulk_dict = {}  # store cif-id and corresponding ASE type structure

start = time.time()

for ids in id_data:
    crystal = Structure.from_file(os.path.join(cif_path, ids + '.cif'))
    bulk = Ada.get_atoms(crystal)
    bulk_dict[ids] = bulk

all_species = ['Hg', 'He', 'Lu', 'I', 'Zr', 'Sc', 'Na', 'Bi', 'Cl', 'Ir', 'Tl',
               'Cr', 'S', 'B', 'Tb', 'Ag', 'F', 'Sr', 'Li', 'Ba', 'Sb', 'Hf',
               'N', 'Os', 'K', 'Mn', 'Ge', 'O', 'Tc', 'Cs', 'Sn', 'Mg', 'Ru',
               'Pt', 'Cu', 'C', 'La', 'Ca', 'Au', 'Al', 'H', 'Mo', 'Nd', 'Ti',
               'W', 'Re', 'Cd', 'Pb', 'P', 'Be', 'Co', 'Xe', 'In', 'Pd', 'Nb',
               'Ta', 'Br', 'As', 'Ga', 'V', 'Ni', 'Kr', 'Rb', 'Fe', 'Y', 'Se',
               'Rh', 'Si', 'Te', 'Zn']

sm = SineMatrix(
    n_atoms_max=50,
    permutation="sorted_l2",
    sparse=False,
    flatten=True
)

print("The length of the feature vector is: ", sm.get_number_of_features())
print("*********************************************************************")
# --------------------- Store sine rep of materials in a dictionary sine_material ---------------------------
start = time.time()
sine_material = {}
for cif_id in list(bulk_dict.keys()):
    sine_material[cif_id] = sm.create(bulk_dict[cif_id])

print("Spend ", time.time() - start, ' s to store sine rep of materials in a dictionary sine_material')
print("*********************************************************************")
# --------------------------- Save as .npy file ----------------------------------
np.save('Sine_mat.npy', sine_material)

print("Over !!!!! This script takes: ", time.time() - start_all, ' s')
