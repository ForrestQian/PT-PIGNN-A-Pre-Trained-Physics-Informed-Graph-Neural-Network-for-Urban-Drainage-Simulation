import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable


# GCN dataset
class PINN_Dataset(Dataset):
    def __init__(self, file_path, device, require_grad):
        self.file_path = file_path
        self.data = pd.read_excel(file_path)
        self.data.iloc[:, 1] = self.data.iloc[:, 1].astype(float)
        self.x = self.data.iloc[:, 0].values
        self.t = self.data.iloc[:, 1].values
        self.q = self.data.iloc[:, 2].values
        self.node_id = self.data.iloc[:, 3].values
        self.D = self.data.iloc[:, 4].values
        self.S0 = self.data.iloc[:, 5].values
        self.n = self.data.iloc[:, 6].values
        self.before_h = self.data.iloc[:, 7].values
        self.before_u = self.data.iloc[:, 8].values
        self.output_data = self.data.iloc[:, 9:].values
        self.device = device
        self.require_grad = require_grad

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, index):
        out_x = torch.tensor(self.x[index], dtype=torch.float32).unsqueeze(dim=0)
        out_t = torch.tensor(self.t[index], dtype=torch.float32).unsqueeze(dim=0)
        x = Variable(out_x, requires_grad=self.require_grad).to(self.device)
        t = Variable(out_t, requires_grad=self.require_grad).to(self.device)
        q = torch.tensor(self.q[index], dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        node_id = torch.tensor(self.node_id[index], dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        D = torch.tensor(self.D[index], dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        S0 = torch.tensor(self.S0[index], dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        n = torch.tensor(self.n[index], dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        before_h = torch.tensor(self.before_h[index], dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        before_u = torch.tensor(self.before_u[index], dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        output = torch.tensor(self.output_data[index, :], dtype=torch.float32).to(self.device)
        return [x, t, q, node_id, D, S0, n, before_h, before_u, output]

#PINN model
class PINN_Model(nn.Module):
    def __init__(self, device):
        super(PINN_Model, self).__init__()
        self.nn = nn.ModuleList()
        self.nn.append(nn.Linear(6, 64))
        self.nn.append(nn.Linear(64, 64))
        self.nn.append(nn.Linear(64, 32))
        self.nn.append(nn.Linear(32, 2))
        self.device = device
        self.lambda1 = torch.nn.Parameter(torch.tensor([0.50]))
        self.lambda2 = torch.nn.Parameter(torch.tensor([0.50]))
    def forward(self, x, t, q, node_id, before_h, before_u):
        lambda1 = torch.clamp(self.lambda1, 0.01, 0.99)
        lambda2 = torch.clamp(self.lambda2, 0.01, 0.99)
        input = torch.concatenate((x, t, q, node_id, before_h, before_u), dim=1)
        for i in range(len(self.nn)):
            if i == 0:
                output = self.nn[i](input)
            else:
                output = torch.tanh(output)
                output = self.nn[i](output)
        h_w = output[:, 0].unsqueeze(dim=1)
        u = output[:, 1].unsqueeze(dim=1)
        return h_w, u, lambda1, lambda2



