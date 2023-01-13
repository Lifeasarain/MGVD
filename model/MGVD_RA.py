import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNN(nn.Module):
    def __init__(self, arch: object, num_classes=2) -> object:
        super(CNN, self).__init__()
        self.in_channels = 2
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.se = SELayer(channel=64)
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.fc1 = nn.Linear(16 * 16 * 256, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.se(out)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc1(out)
        print(out.shape)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        print(out.shape)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out


def MakeCNN():
    return CNN([1, 1, 2, 2, 2], num_classes=2)


class MGVD(torch.nn.Module):
    def __init__(self, num_features, g_hidden_channels, g_out_channels, gheads):
        super(MGVD, self).__init__()
        self.raw_graph_GAT = GAT(num_features, g_hidden_channels, g_out_channels, gheads)
        self.abstract_graph_GAT = GAT(num_features, g_hidden_channels, g_out_channels, gheads)
        # self.cnn = CNN(nlabels, base_width)
        self.cnn = MakeCNN()

    def forward(self, raw_data, abstract_data, sequence, device):
        # graphs_one = enumerate(data)[0][0].to(device)
        # graphs_matrix = self.graph(graphs_one.x, graphs_one.edge_index, graphs_one.batch)
        raw_graphs_matrix = torch.empty(1, 1, 128, 128)
        for i, raw_graph_loader in enumerate(raw_data):
            if i == 0:
                for raw_graphs in raw_graph_loader:
                    raw_graphs.to(device)
                    raw_graphs_matrix = self.raw_graph_GAT(raw_graphs.x, raw_graphs.edge_index, raw_graphs.batch)
                    raw_graphs_matrix = raw_graphs_matrix.reshape(1, 1, 128, 128)
                continue
            for raw_graphs in raw_graph_loader:
                raw_graphs.to(device)
                x = self.raw_graph_GAT(raw_graphs.x, raw_graphs.edge_index, raw_graphs.batch)
                x = x.reshape(1, 1, 128, 128)
                raw_graphs_matrix = torch.cat((raw_graphs_matrix, x), dim=0)

        abstract_graphs_matrix = torch.empty(1, 1, 128, 128)
        for i, abstract_graph_loader in enumerate(abstract_data):
            if i == 0:
                for abstract_graphs in abstract_graph_loader:
                    abstract_graphs.to(device)
                    abstract_graphs_matrix = self.abstract_graph_GAT(abstract_graphs.x, abstract_graphs.edge_index, abstract_graphs.batch)
                    abstract_graphs_matrix = abstract_graphs_matrix.reshape(1, 1, 128, 128)
                continue
            for abstract_graphs in abstract_graph_loader:
                abstract_graphs.to(device)
                x = self.raw_graph_GAT(abstract_graphs.x, abstract_graphs.edge_index, abstract_graphs.batch)
                x = x.reshape(1, 1, 128, 128)
                abstract_graphs_matrix = torch.cat((abstract_graphs_matrix, x), dim=0)

        feature_matrix = torch.cat((raw_graphs_matrix, abstract_graphs_matrix), dim=0)
        out = self.cnn(feature_matrix)
        # print(out.shape)
        return out