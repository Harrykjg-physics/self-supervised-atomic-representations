import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import argparse
import os
import sys
import torch

parser = argparse.ArgumentParser(description='t-SNE for visualization of GNN embeddings')
parser.add_argument('--perplexity',
                    help='The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. ',
                    default=30.0, type=float)
parser.add_argument('--early_exaggeration',
                    help='Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space.',
                    type=float, default=12.0)
parser.add_argument('--learning_rate', help='The learning rate for t-SNE is usually in the range [10.0, 1000.0].',
                    type=float, default=200.0)
args = parser.parse_args(sys.argv[1:])


def plot_tsne(xy, y, element_list, figsize=(8, 8), cmap='viridis'):
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


cg_2nd = np.load('C:\\Users\\Harry\\Desktop\\Research_PPT\\Version\\Paper_plot\\sl_elem01.npy', allow_pickle=True).item()

# ------------------- elements -----------------------
count = 0
for ids in list(cg_2nd.keys()):
    element = ids.split("_")[-3]
    if count == 0:
        y = element
    else:
        y = np.vstack((y, element))
    count += 1

y = np.squeeze(y)
all_ele = list(y)

ele_num_dict = {}
for ele in all_ele:
    ele_num_dict[ele] = all_ele.count(ele)

ele_num_dict_sorted = sorted(ele_num_dict.items(), key=lambda item: item[1], reverse=True)
ele_list_sorted = []
for idx in range(len(ele_num_dict_sorted)):
    ele_list_sorted.append(ele_num_dict_sorted[idx][0])

# print("ele_num_dict_sorted: ", ele_num_dict_sorted)
# print("ele_list_sorted: ", ele_list_sorted)
# ---------------------- keep elements with #>50-------------------------
count = 0

for ids in list(cg_2nd.keys()):
    fea_2nd = cg_2nd[ids]
    element = ids.split("_")[-3]
    print(ids)
    if count == 0:
        X_2nd = fea_2nd
        y = ele_list_sorted.index(element)
    else:
        X_2nd = np.vstack((X_2nd, fea_2nd))
        y = np.vstack((y, ele_list_sorted.index(element)))
    count += 1

y = np.squeeze(y)
print(y)

print("This is the shape of feature matrix: ", fea_2nd.shape)

# t-SNE for visualization
tsne = manifold.TSNE(n_components=2, init='pca', perplexity=args.perplexity,
                     early_exaggeration=args.early_exaggeration,
                     learning_rate=args.learning_rate,
                     random_state=501)

X_tsne = tsne.fit_transform(X_2nd)

print("Org data dimension is {}.Embedded data dimension is {}".format(X_2nd.shape[-1], X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

print("y: ", y.shape)
plot_tsne(X_norm, y, ele_list_sorted)
plt.savefig("tsne" + "perplexity" + '_' + str(args.perplexity) + '_' + "rand" + ".eps")
