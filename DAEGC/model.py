import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GATLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class pseudo_gat(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(pseudo_gat, self).__init__()
        self.w1 = nn.Linear(num_features, hidden_size)
        self.iden = nn.Parameter(data = torch.randn((num_points, hidden_dims),dtype=torch.float).to(device), requires_grad=True)

    def forward(self, x, adj, M):
        z = self.w1(x) + iden
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred, z
