import numpy as np

cg_mat = np.load("sl_cgcnn_embedding_dict_7_19.npy", allow_pickle=True).item()
magmom_order = {}
for i in list(cg_mat.keys()):
    magmom_order[i] = 0

np.save("magmom_order.npy", magmom_order)
