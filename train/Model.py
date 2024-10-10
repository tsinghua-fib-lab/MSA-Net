import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np


class Attn(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attn, self).__init__()
        self.mlp = nn.Linear(input_size, output_size)
        self.sigma = nn.ReLU()

    def forward(self, ht, zt):
        x = torch.cat([ht, zt], dim=2)
        x = self.mlp(x)
        x = self.sigma(x)
        x = x * ht
        return x


class Net(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, history_window, device
    ):
        super(Net, self).__init__()
        self.gcn = GCNConv(input_size, hidden_size)
        self.macro_gcn = GCNConv(3, hidden_size)
        self.macro_lstm = nn.LSTM(2 * hidden_size, 2, num_layers)
        self.attn = Attn(3 * hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.gcn_out = GCNConv(hidden_size * history_window, output_size)
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.output_size = output_size
        self.history_window = history_window

    def forward(
        self,
        micro_dynamic,
        micro_edge_index,
        micro_node_degrees,
        com,
        com_edges,
        com_weights,
        macro_dynamic,
        macro_static,
        k_in,
        h=None,
    ):
        user_num, timestep, n_feat = micro_dynamic.size()
        com_num, _, _, _ = macro_dynamic.size()
        Xtk_list = []
        Qtk_list = []
        for each_step in range(timestep):
            com_pool_list = {}
            for i in list(set(com)):
                com_pool_list[i] = []
            batch_x = micro_dynamic[:, each_step, :]
            batch_macro_x = macro_dynamic[:, each_step, :, :]
            Ht_list = []
            Yt_list = []
            Zt_list = []
            Qt_list = []
            Zst_list = []
            Yst_list = []
            Pt_list = []
            for day in range(batch_x.shape[1]):
                com_pool = []
                x = batch_x[:, day].view(-1, 1)
                x = torch.cat((x, micro_node_degrees.unsqueeze(1)), dim=1)
                x.requires_grad = True
                Ht = self.gcn(x, micro_edge_index)
                Ht_list.append(Ht)

                macro_x = batch_macro_x[:, day, :]
                macro_x = torch.cat((macro_x, macro_static.unsqueeze(1)), dim=1)
                macro_x.requires_grad = True
                Yt = self.macro_gcn(macro_x, com_edges, com_weights)
                Yt_list.append(Yt)

                for user in range(Ht.shape[0]):
                    user_ht = Ht[user, :]
                    com_pool_list[com[user]].append(user_ht)
                for key in com_pool_list:
                    stacked_tensor = torch.stack(com_pool_list[key])
                    n = stacked_tensor.shape[0]
                    sum_tensor = torch.sum(stacked_tensor, dim=0)
                    mean_tensor = sum_tensor / n
                    com_pool.append(mean_tensor)
                Zt = torch.stack(com_pool, dim=0)
                Zt_list.append(Zt)

                Qt = torch.cat((Zt, Yt), dim=1)
                Qt_list.append(Qt)

                Zst = []
                for user_com in com:
                    Zst.append(Zt[user_com, :])
                Zst = torch.stack(Zst, dim=0)
                Zst_list.append(Zst)

                Yst = []
                for user_com in com:
                    Yst.append(Yt[user_com, :])
                Yst = torch.stack(Yst, dim=0)
                Yst_list.append(Yst)

                Pt = torch.cat((Zst, Yst), dim=1)
                Pt_list.append(Pt)

            H = torch.stack(Ht_list).to(self.device)
            Z = torch.stack(Zst_list).to(self.device)
            Y = torch.stack(Yst_list).to(self.device)
            variables = {"H": H.clone(), "Z": Z.clone(), "Y": Y.clone()}
            Q = torch.stack(Qt_list).to(self.device)
            P = torch.stack(Pt_list).to(self.device)
            H = F.relu(H)
            Q = F.relu(Q)

            H_v = self.attn(H, P)
            k_in = k_in.unsqueeze(0)
            k_in = k_in.repeat(10, 32, 1)
            k_in = k_in.permute(0, 2, 1)
            H_v = self.attn(H_v, k_in)
            variables["Xt"] = H_v.clone()
            Htk, _ = self.lstm(H_v)  # input:hidden_dim2 of cur_h，output:gru_dim of h
            variables["Xt+1"] = Htk.clone()
            Qtk, _ = self.macro_lstm(Q)
            Htk = Htk.view(user_num, -1)
            Qtk = Qtk[-1]
            Htk = F.relu(Htk)
            Xtk = self.gcn_out(Htk, micro_edge_index)
            Xtk_list.append(F.softmax(Xtk, dim=1))
            Qtk_list.append(Qtk)
        micro_pred = torch.stack(Xtk_list).to(self.device)
        macro_pred = torch.stack(Qtk_list).to(self.device)
        return micro_pred, macro_pred, variables
