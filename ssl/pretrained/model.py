import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, CGConv
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
import numpy as np


class CGCNN(nn.Module):
    """
    Self-supervised training of a CGCNN Encoder(No fully connected layers)
    """
    def __init__(self, orig_atom_fea_len=92, nbr_fea_len=41,
                 atom_fea_len=64, n_conv=3,
                 drop_ratio=0, stage='train'):

        """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        """
        super(CGCNN, self).__init__()
        self.stage = stage
        self.drop_ratio = drop_ratio
        self.embedding1 = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embedding2 = nn.Linear(nbr_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([CGConv(atom_fea_len, dim=atom_fea_len, batch_norm=False) for _ in range(n_conv)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(atom_fea_len) for _ in range(n_conv)])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.embedding1(x)
        edge_attr = self.embedding2(edge_attr)
        # save node representations under different layers
        atom_embedding = [x]
        for a_idx in range(len(self.convs)):
            x = self.convs[a_idx](x, edge_index, edge_attr)
            x = self.batch_norm[a_idx](x)
            if a_idx == (len(self.convs) - 1):
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            if self.stage == 'get_emb':
                atom_embedding.append(x)

        if self.stage == 'get_emb':
            return atom_embedding
        else:
            return x


class CGCNN_pred(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total material properties.
    """

    def __init__(self, num_task=1, atom_fea_len=64, n_conv=3, xtra_layers=True,
                 JK="last", drop_ratio=0, graph_pooling="mean", classification=False):

        """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
        """
        super(CGCNN_pred, self).__init__()
        self.num_task = num_task
        self.atom_fea_len = atom_fea_len
        self.classification = classification
        self.additional = xtra_layers
        self.drop_ratio = drop_ratio
        self.n_conv = n_conv
        self.JK = JK

        if self.n_conv < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.cgcnn = CGCNN(n_conv=self.n_conv,
                           atom_fea_len=self.atom_fea_len,
                           drop_ratio=self.drop_ratio,
                           stage="get_emb")

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * atom_fea_len, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(atom_fea_len, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * atom_fea_len, set2set_iter)
            else:
                self.pool = Set2Set(atom_fea_len, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.additional:
            if self.JK == "concat":
                self.linear1 = torch.nn.Linear((self.n_conv + 1) * self.atom_fea_len, self.atom_fea_len)
                self.linear2 = torch.nn.Linear((self.n_conv + 1) * self.atom_fea_len, self.atom_fea_len)
            else:
                self.linear1 = nn.Linear(self.atom_fea_len, self.atom_fea_len)
                self.linear2 = nn.Linear(self.atom_fea_len, self.atom_fea_len)

        if self.JK == "concat":
            self.out = torch.nn.Linear((self.n_conv + 1) * self.atom_fea_len, self.num_task)
        else:
            self.out = torch.nn.Linear(self.atom_fea_len, self.num_task)

    def from_pretrained(self, model_file, froze):
        print("I am enter from pretrained file")
        model_stat_dict = self.cgcnn.state_dict()
        pretrained_model_stat_dict = torch.load(model_file, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrained_model_stat_dict.items():
            string = k.split(".")
            if string[0] == 'gnn':
                name = string[1]
                for item in string[2:]:
                    name += ("." + item)
            if froze:
                v.requires_grad = False  # froze the pretrained model
            new_state_dict[name] = v
        model_stat_dict.update(new_state_dict)
        self.cgcnn.load_state_dict(model_stat_dict)

    def forward(self, data):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        h_list = self.cgcnn(data)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        y = self.pool(node_representation, batch).unsqueeze(1).squeeze()

        if self.additional:
            y = F.softplus(self.linear1(y))
            y = F.softplus(self.linear2(y))

        if self.classification:
            y = self.out(y)
        else:
            y = self.out(y).squeeze()

        return y


if __name__ == "__main__":
    pass
