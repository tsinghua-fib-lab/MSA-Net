import pandas as pd
import pickle as pkl
from scipy import sparse
import torch.nn as nn
from sklearn.utils import class_weight
from Model import Net
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler

random.seed(10)
np.random.seed(10)

writer = SummaryWriter(log_dir="save", flush_secs=30)
data_path = "../data/higg_sample"
data = sparse.load_npz("{0}/com_feature_graphs.npz".format(data_path))
data = data.toarray()[:, 1:]
edges = pd.read_csv("{0}/com_edges.csv".format(data_path)).to_numpy().astype(int)
node_degrees = np.load("{0}/k_out.npy".format(data_path))
k_in = np.load("{0}/k_in.npy".format(data_path))
scaler = MinMaxScaler()
node_degrees = scaler.fit_transform(node_degrees.reshape(-1, 1)).flatten()
k_in = scaler.fit_transform(k_in.reshape(-1, 1)).flatten()
with open("{0}/com_community.pkl".format(data_path), "rb") as f:
    com = pkl.load(f)
com = list(com.values())
macro_dynamic_feat = np.load("{0}/dynamic_feat.npy".format(data_path))
macro_static_feat = np.load("{0}/N.npy".format(data_path))
macro_static_feat = scaler.fit_transform(macro_static_feat.reshape(-1, 1)).flatten()
com_edges = np.load("{0}/com_edges.npy".format(data_path))
com_weights = np.load("{0}/com_weights.npy".format(data_path))

num_communities = macro_dynamic_feat.shape[0]
num_nodes = data.shape[0]
timestep = data.shape[1]
history_window = 10
pred_window = 30
slide_step = 5
train_prop = 0.9
test_prop = 0.1
input_size = 2  # dim of input vector
hidden_size = 16  # hidden dim of LSTM and GCN
num_layers = 1
output_size = 2

x, y = [], []
macro_x, macro_y = [], []
for i in range(0, timestep, slide_step):
    if i + history_window + pred_window >= timestep or i + history_window >= timestep:
        break
    x_data = data[:, i : i + history_window].reshape((num_nodes, history_window))
    macro_x_data = macro_dynamic_feat[:, i : i + history_window, :].reshape(
        (num_communities, history_window, 2)
    )
    x.append(x_data)
    macro_x.append(macro_x_data)
    y.append(data[:, i + history_window + pred_window].reshape(num_nodes, 1))
    macro_y.append(
        macro_dynamic_feat[:, i + history_window + pred_window, :].reshape(
            num_communities, 2
        )
    )

x = np.array(x, dtype=np.float32).transpose((1, 0, 2))
y = np.array(y, dtype=np.float32).transpose((1, 0, 2))
macro_x = np.array(macro_x, dtype=np.float32).transpose((1, 0, 2, 3))
macro_y = np.array(macro_y, dtype=np.float32).transpose((1, 0, 2))
one_hot = np.eye(2)
y = one_hot[y.squeeze().astype(int)]
idx = np.random.permutation(x.shape[0])
x, y = x[idx], y[idx]
x = x[:, 10:11, :].reshape(num_nodes, -1, history_window)
y = y[:, 10:11, :].reshape(num_nodes, -1, 2)
macro_x = macro_x[:, 10:11, :, :].reshape(num_communities, -1, history_window, 2)
macro_y = macro_y[:, 10:11, :].reshape(num_communities, -1, 2)

dataset_length = x.shape[0]
train_begin = 0
train_end = test_begin = int(dataset_length * train_prop)
test_end = int(dataset_length) - 1
train_mask = np.zeros(dataset_length)
test_mask = train_mask.copy()
train_mask[idx[train_begin:train_end]] = 1
test_mask[idx[test_begin:test_end]] = 1
train_mask = train_mask.astype(bool)
test_mask = test_mask.astype(bool)

device = torch.device("cuda:3")  # Use GPU

inputs = x[:, :, -1]
ce_loss = torch.tensor(0, dtype=torch.float32).to(device)
all_m_y = []
for batch in range(inputs.shape[1]):
    inp = inputs[:, batch]
    outputs = y[:, batch][:, 1]
    mask = inp == 0
    m_y = outputs[mask]
    all_m_y += list(m_y)
all_m_y = np.array(all_m_y).reshape(-1)
weight = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(all_m_y), y=all_m_y
)
print(weight)
weight = torch.tensor(weight, dtype=torch.float32)


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def forward(
        self, micro_inputs, micro_outputs, micro_targets, macro_targets, macro_outputs
    ):
        micro_inputs = micro_inputs[:, :, -1]
        batch_num = micro_inputs.shape[1]
        ce_loss = torch.tensor(0, dtype=torch.float32).to(device)
        mse_loss = torch.tensor(0, dtype=torch.float32).to(device)
        for batch in range(batch_num):
            # micro_loss
            x = micro_inputs[:, batch]
            y = micro_targets[:, batch]
            y_hat = micro_outputs[:, :, batch]
            mask = x == 0
            m_y = y[mask]
            m_y_hat = y_hat[mask]
            ce_loss += nn.CrossEntropyLoss(weight=weight.to(device))(m_y_hat, m_y)

            # macro_loss
            macro_y = macro_targets[:, batch, :]
            macro_y_hat = macro_outputs[:, batch, :]
            mse_loss += nn.MSELoss()(macro_y, macro_y_hat)
        ce_loss = ce_loss / batch_num
        mse_loss = mse_loss / batch_num
        loss = (ce_loss + mse_loss) / 2
        return loss


model = Net(
    input_size, hidden_size, num_layers, output_size, history_window, device
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = CustomLoss(alpha=1)
print(model)

node_degrees = torch.tensor(node_degrees, dtype=torch.float32).to(device)
k_in = torch.tensor(k_in, dtype=torch.float32).to(device)
x = torch.tensor(x).to(device)
y = torch.tensor(y).to(device)
y = torch.transpose(y, 1, 2)
edges = torch.tensor(edges).T.to(device)
macro_x = torch.tensor(macro_x, dtype=torch.float32).to(device)
macro_y = torch.tensor(macro_y, dtype=torch.float32).to(device)
com_edges = torch.tensor(com_edges).to(device)
com_weights = torch.tensor(com_weights, dtype=torch.float32).to(device)
macro_static_feat = torch.tensor(macro_static_feat, dtype=torch.float32).to(device)

epoch = 5000000
all_loss = []
all_test_loss = []
(
    all_train_Acc_0,
    all_train_Acc_1,
    all_train_Precision,
    all_train_Recall,
    all_train_F1,
) = ([], [], [], [], [])
all_test_Acc_0, all_test_Acc_1, all_test_Precision, all_test_Recall, all_test_F1 = (
    [],
    [],
    [],
    [],
    [],
)


def cal_0_to_1(input, result, label):
    input = input.cpu().numpy()[:, :, -1]
    result = torch.argmax(result, dim=1).cpu().numpy()
    label = torch.argmax(label, dim=1).cpu().numpy()
    input, result, label = input.T, result.T, label.T
    label_0, label_1, pred_00, pred_01, pred_10, pred_11 = [], [], [], [], [], []
    Precison, Recall, F1 = [], [], []
    for day in range(input.shape[0]):
        day_input, day_result, day_label = input[day], result[day], label[day]
        if np.sum((day_input == 0) & (day_label == 1)) == 0:
            continue
        label_0.append(np.sum((day_input == 0) & (day_label == 0)))
        label_1.append(np.sum((day_input == 0) & (day_label == 1)))
        pred_00.append(np.sum((day_input == 0) & (day_label == 0) & (day_result == 0)))
        pred_01.append(np.sum((day_input == 0) & (day_label == 0) & (day_result == 1)))
        pred_10.append(np.sum((day_input == 0) & (day_label == 1) & (day_result == 0)))
        pred_11.append(np.sum((day_input == 0) & (day_label == 1) & (day_result == 1)))
        Precison.append(pred_11[-1] / (pred_11[-1] + pred_01[-1] + 1e-7))
        Recall.append(pred_11[-1] / (pred_11[-1] + pred_10[-1] + 1e-7))
        F1.append(2 * (Precison[-1] * Recall[-1]) / (Precison[-1] + Recall[-1] + 1e-7))
    Acc_0 = np.mean(np.array(pred_00) / (np.array(label_0) + 1e-7))
    Acc_1 = np.mean(np.array(pred_11) / (np.array(label_1) + 1e-7))
    Precison = np.mean(Precison)
    Recall = np.mean(Recall)
    F1 = np.mean(F1)
    return (
        label_0,
        label_1,
        pred_00,
        pred_01,
        pred_10,
        pred_11,
        Acc_0,
        Acc_1,
        Precison,
        Recall,
        F1,
    )


F1_best = 0
for epoch in range(epoch):
    print("epoch:", epoch)
    model.train()

    optimizer.zero_grad()

    micro_output, macro_output, varibales = model(
        x,
        edges,
        node_degrees,
        com,
        com_edges,
        com_weights,
        macro_x,
        macro_static_feat,
        k_in,
    )
    micro_output = micro_output.permute(1, 2, 0)
    macro_output = macro_output.permute(1, 0, 2)
    micro_y = torch.argmax(y, dim=1)
    train_loss = criterion(
        x[train_mask],
        micro_output[train_mask],
        micro_y[train_mask],
        macro_y,
        macro_output,
    )
    print("train loss", train_loss)

    (
        label_0,
        label_1,
        pred_00,
        pred_01,
        pred_10,
        pred_11,
        train_Acc_0,
        train_Acc_1,
        train_Precision,
        train_Recall,
        train_F1,
    ) = cal_0_to_1(x[train_mask], micro_output[train_mask], y[train_mask])

    model.eval()

    test_loss = criterion(
        x[test_mask], micro_output[test_mask], micro_y[test_mask], macro_y, macro_output
    )
    all_test_loss.append(test_loss.item())
    print("test loss", test_loss)
    (
        label_0,
        label_1,
        pred_00,
        pred_01,
        pred_10,
        pred_11,
        test_Acc_0,
        test_Acc_1,
        test_Precision,
        test_Recall,
        test_F1,
    ) = cal_0_to_1(x[test_mask], micro_output[test_mask], y[test_mask])

    writer.add_scalars(
        "Loss", {"train": train_loss.item(), "test": test_loss.item()}, epoch
    )
    writer.add_scalars("Acc_0", {"train": train_Acc_0, "test": test_Acc_0}, epoch)
    writer.add_scalars("Acc_1", {"train": train_Acc_1, "test": test_Acc_1}, epoch)
    writer.add_scalars(
        "Precision", {"train": train_Precision, "test": test_Precision}, epoch
    )
    writer.add_scalars("Recall", {"train": train_Recall, "test": test_Recall}, epoch)
    writer.add_scalars(
        "F1", {"train": np.mean(train_F1), "test": np.mean(test_F1)}, epoch
    )
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    if np.mean(test_F1) >= F1_best:
        H = varibales["H"].cpu().detach().numpy()
        Z = varibales["Z"].cpu().detach().numpy()
        Y = varibales["Y"].cpu().detach().numpy()
        Xt = varibales["Xt"].cpu().detach().numpy()
        Xt1 = varibales["Xt+1"].cpu().detach().numpy()
        np.save("H/{0}.npy".format(epoch), H)
        np.save("Y/{0}.npy".format(epoch), Y)
        np.save("Z/{0}.npy".format(epoch), Z)
        np.save("Xt/{0}.npy".format(epoch), Xt)
        np.save("Xt1/{0}.npy".format(epoch), Xt1)
    train_loss.backward(retain_graph=True)
    optimizer.step()
