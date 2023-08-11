import csv
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import site_is_of_motif_type
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import argparse
import sys
import os

parser = argparse.ArgumentParser(description='t-SNE for visualization of GNN embeddings')
parser.add_argument('--perplexity',
                    help='The perplexity is related to the number of nearest neighbors that is used in other manifold '
                         'learning algorithms. ',
                    default=30.0, type=float)
parser.add_argument('--early_exaggeration',
                    help='Controls how tight natural clusters in the original space are in the embedded space and how '
                         'much space will be between them. For larger values, the space between natural clusters will '
                         'be larger in the embedded space.',
                    type=float, default=12.0)
parser.add_argument('--learning_rate', help='The learning rate for t-SNE is usually in the range [10.0, 1000.0].',
                    type=float, default=200.0)
args = parser.parse_args(sys.argv[1:])


def plot_tsne(xy, y, colors=None, alpha=0.5, figsize=(6, 6), s=35, cmap='viridis'):
# def plot_tsne(xy, y, element_list, figsize=(8, 8), cmap='viridis'):
    """
    plt.figure(figsize=figsize, facecolor='white')
    plt.margins(0)
    plt.axis('off')
    for i in range(xy.shape[0]):
        color = plt.cm.Set1(y[i])
        plt.scatter(xy[i, 0], xy[i, 1], color=color, cmap=cmap,
                    label=element_list[y[i]], marker='o', alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    new_dict = dict(zip(["Mn", "Fe", "Co", "Ni", "Cu", "Cr", "Nd", "Tb", "Others"],
                        [by_label["Mn"], by_label["Fe"],
                         by_label["Co"], by_label["Ni"],
                         by_label["Cu"], by_label["Cr"],
                         by_label["Nd"], by_label["Tb"],
                         by_label["U"]]))

    plt.legend(new_dict.values(), new_dict.keys(), loc="lower right", fontsize=11)

    """
    print(xy.shape)
    plt.figure(figsize=figsize, facecolor='white')
    plt.margins(0)
    plt.axis('on')
    fig = plt.scatter(xy[:, 0], xy[:, 1],
                      c=colors,  # set colors of markers
                      cmap=cmap,  # set color map of markers
                      alpha=alpha,  # set alpha of markers
                      marker='o',  # use smallest available marker (square)
                      s=s,  # set marker size. single pixel is 0.5 on retina, 1.0 otherwise
                      lw=0,  # don't use edges
                      edgecolor='',
                      label=y)  # don't use edges
    # remove all axes and whitespace / borders
    # plt.colorbar(fig)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

"""
# load non-zero-magmom-atoms
with open("id_prop_non_zero_1.csv") as f:
    reader = csv.reader(f)
    magmom_data = [row for row in reader]
mag_atom = []
for i in range(len(magmom_data)):
    mag_atom.append(magmom_data[i][0])

local_envs = {}
for magatom in mag_atom:
    id_mag = magatom.split("_")[0] + "_" + magatom.split("_")[1]
    idx_mag = int(magatom.split("_")[-1])
    mag_mat = Structure.from_file(os.path.join(
        "D:\\MLDataset\\New",
        id_mag + '.cif'))
    local_env = site_is_of_motif_type(mag_mat, idx_mag, approach="voronoi")
    local_envs[magatom] = local_env

local_envs_recog = {}
for magatom in list(local_envs.keys()):
    if local_envs[magatom] != "unrecognized":
        local_envs_recog[magatom] = local_envs[magatom]

np.save("local_envs_recog.npy", local_envs_recog)

# count the number of Local-Envs in each type

count = {"bcc": 0, "cp": 0, "octahedral": 0, "tetrahedral": 0, "trigonal bipyramidal": 0}
for magatom in list(local_envs_recog.keys()):
    if local_envs[magatom] == "bcc":
        count["bcc"] += 1
    if local_envs[magatom] == "cp":
        count["cp"] += 1
    if local_envs[magatom] == "octahedral":
        count["octahedral"] += 1
    if local_envs[magatom] == "tetrahedral":
        count["tetrahedral"] += 1
    if local_envs[magatom] == "trigonal bipyramidal":
        count["trigonal bipyramidal"] += 1

np.save("count.npy", count)

"""
count = np.load("count.npy", allow_pickle=True).item()
local_envs_recog = np.load("local_envs_recog.npy", allow_pickle=True).item()

loc_env_list = list(count.keys())
print("loc_env_list: ", loc_env_list)

cg_2nd = np.load(
    'C:\\Users\\Harry\\Desktop\\Research_PPT\\Version\\Paper_plot\\sl_elem01.npy',
    allow_pickle=True).item()

count = 0
for ids in list(local_envs_recog.keys()):
    fea_2nd = cg_2nd[ids]
    if count == 0:
        X_2nd = fea_2nd
        y_site = loc_env_list.index(local_envs_recog[ids])
    else:
        X_2nd = np.vstack((X_2nd, fea_2nd))
        y_site = np.vstack((y_site, loc_env_list.index(local_envs_recog[ids])))
    count += 1

y_site = np.squeeze(y_site)

# t-SNE for visualization
tsne = manifold.TSNE(n_components=2, init='pca', perplexity=args.perplexity,
                     early_exaggeration=args.early_exaggeration,
                     learning_rate=args.learning_rate,
                     random_state=501)
X_tsne = tsne.fit_transform(X_2nd)

print("Org data dimension is {}.Embedded data dimension is {}".format(X_2nd.shape[-1], X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
"""
plt.figure(figsize=(6, 6))
for i in range(X_norm.shape[0]):
    color = plt.cm.Set1(y_site[i])
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y_site[i]), color=color, fontdict={'weight': 'bold', 'size': 9})

plt.xticks([])
plt.yticks([])
# plt.show()
"""
color = plt.cm.Set1(y_site)
print("y: ", y_site.shape)
print("color: ", color.shape)
plot_tsne(X_norm, y_site, colors=color)
# plot_tsne(X_norm, y_site, local_list_sorted)
plt.savefig("tsne" + "local" + "_" + str(args.perplexity) + ".eps")
