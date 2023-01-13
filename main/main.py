import numpy as np
import torch
import argparse
from dataset_util.dataset_util import get_dataset, loda_dataset, FunctionDataset
from prettytable import PrettyTable
import os
import datetime
from dpu_utils.utils import RichPath
from model.MGVD import MGVD
from tqdm import tqdm

def read_args():

    parser = argparse.ArgumentParser(description='MGVD')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1, metavar='N',
                        help='eval batch size')
    parser.add_argument('--eval_batch_num', type=int, default=1, metavar='N',
                        help='eval batch num')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('-no-cuda', action='store_true', default=False,
                        help='disable the GPU')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--flag', type=str, default='train',
                        help='train the model or calculate the entropy')
    parser.add_argument('-save_dir', type=str, default='./snapshot',
                        help='where to save the snapshot')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='hidden channels in GCN conv')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='the number of classifier class')
    parser.add_argument('--num_features', type=int, default=128,
                        )

    parser.add_argument('--g_hidden_channels', type=int, default=16,
                        help='graph hidden channels')
    parser.add_argument('--g_out_channels', type=int, default=100,
                        help='graph out channels')
    parser.add_argument('--max_line_length', type=int, default=128,
                        help='max line in func')
    return parser


def save_own(model, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}.pt'.format(save_prefix)
    torch.save(model.state_dict(), save_path)


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # Accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)

        # Precision, Recall, F1-score
        table = PrettyTable()
        table.field_names = ["", "Accuracy", "Precision", "Recall", "F1-score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
            recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
            f1 = round(2 * (precision * recall) / (precision + recall), 4) if precision + recall != 0 else 0.
            table.add_row([i, acc, precision, recall, f1])
        print(table)


def train_model(params, train_data, valid_data):
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model = MGVD(num_features=128, g_hidden_channels=16, g_out_channels=128, gheads=4)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, params.epochs + 1):
        total_loss = 0.
        for data in tqdm(train_data):
            raw_graph, abstract_graph, sequence, labels = data
            labels = torch.tensor(labels).to(params.device)
            # data = data.to(params.device)
            model.zero_grad()
            out = model(raw_graph, abstract_graph, sequence, params.device)
            preds = torch.argmax(out, dim=1).flatten()
            loss = criterion(out, labels)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'acc : {(torch.sum(preds == labels)/len(labels)):.3f}')

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.epochs, total_loss))

    model.eval()
    correct = 0
    total = 0
    count = 0
    test_labels = torch.tensor(labels).to(params.device)
    labels = test_labels.tolist()
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    with torch.no_grad():
        for data in tqdm(valid_data):
            raw_graph, abstract_graph, sequence, label = data
            label = torch.tensor(label).to(params.device)
                # data = data.to(params.device)
            model.zero_grad()
            out = model(raw_graph, abstract_graph, sequence, params.device)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += label.size(0)
            count += 1
            confusion.update(pred.cpu().numpy(), label.cpu().numpy())

    confusion.summary()
    save_own(model, params.save_dir, params.project)


def test_model(params, test_data):
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model = MGVD(num_features=128, g_hidden_channels=16, g_out_channels=128, gheads=4)
    model.load_state_dict(torch.load(params.load_model))
    model.eval()
    correct = 0
    total = 0
    count = 0
    test_labels = torch.tensor(labels).to(params.device)
    labels = test_labels.tolist()
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    with torch.no_grad():
        for data in tqdm(test_data):
            raw_graph, abstract_graph, sequence, label = data
            label = torch.tensor(label).to(params.device)
                # data = data.to(params.device)
            model.zero_grad()
            out = model(raw_graph, abstract_graph, sequence, params.device)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += label.size(0)
            count += 1
            confusion.update(pred.cpu().numpy(), label.cpu().numpy())

    confusion.summary()


def train():
    params = read_args().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.cuda.set_device(1)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = RichPath.create("")
    loaded_data_hyperparameters = dict(
        train_save_path="",
        valid_save_path="",
        test_save_path="",
        num_fwd_edge_types="1",
        tie_fwd_bkwd_edges=True,
        add_self_loop_edges="0",
        batch_size=16
    )
    get_dataset(data_path=data_path,
                loaded_data_hyperparameters=loaded_data_hyperparameters)
    train_loader, valid_loader, test_loader = loda_dataset(loaded_data_hyperparameters)
    train_model(params, train_loader, valid_loader)


def test():
    params = read_args().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.cuda.set_device(1)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaded_data_hyperparameters = dict(
        train_save_path="",
        valid_save_path="",
        test_save_path="",
        num_fwd_edge_types="1",
        tie_fwd_bkwd_edges=True,
        add_self_loop_edges="0",
        batch_size=16
    )
    train_loader, valid_loader, test_loader = loda_dataset(loaded_data_hyperparameters)
    test_model(params, test_loader)


if __name__ == "__main__":
    flage = "train"
    if flage == "train":
        train()
    elif flage == "test":
        test()