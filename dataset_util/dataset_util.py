import pickle
import logging
from enum import Enum
from typing import Any, Dict, List, Iterator, Tuple, TypeVar, Generic, NamedTuple, Set, Optional, Type
from dpu_utils.utils import RichPath
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

import torch_geometric.transforms as transforms
# from graph_dataset import GraphDataset, DataFold


logger = logging.getLogger(__name__)
class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2



class GraphSample(object):
    """Data structure holding information for a single graph."""
    def __init__(
        self,
        adjacency_list: np.ndarray,
        edge_features: np.ndarray,
        node_features: np.ndarray,
    ):
        super().__init__()
        self._adjacency_list = adjacency_list
        self._edge_features = edge_features
        self._node_features = node_features

    @property
    def adjacency_list(self) -> np.ndarray:
        """Adjacency information by edge type as list of ndarrays of shape [E, 2]"""
        return self._adjacency_list

    @property
    def edge_features(self) -> np.ndarray:
        """Number of edge by edge type as ndarray of shape [E, edge_type]"""
        return self._edge_features

    @property
    def node_features(self) -> np.ndarray:
        """Initial node features as ndarray of shape [V, ...]"""
        return self._node_features


GraphSampleType = TypeVar("GraphSampleType", bound=GraphSample)


class LineSample(Generic[GraphSampleType]):
    def __init__(
        self,
        raw_graph: GraphSampleType,
        normalize_graph: GraphSampleType,
        sequence: List
    ):
        self._raw_graph = raw_graph
        self._normalize_graph = normalize_graph
        self._sequence = sequence

    @property
    def raw_graph(self):
        return self._raw_graph

    def normalize_graph(self):
        return self._normalize_graph

    def sequence(self):
        return self._sequence


LineSampleType = TypeVar("LineSampleType", bound=LineSample)


class FunctionSample(Generic[LineSampleType]):
    def __init__(
        self,
        label,
        raw_graph: List[GraphSampleType],
        abstract_graph: List[GraphSampleType],
        text
    ):
        self._label = label
        self._raw_graph = raw_graph
        self._abstract_graph = abstract_graph
        self._text = text

    @property
    def label(self):
        return self._label

    @property
    def raw_graph(self):
        return self._raw_graph

    @property
    def abstract_graph(self):
        return self._abstract_graph

    @property
    def text(self):
        return self._text


FunctionSampleType = TypeVar("FunctionSampleType", bound=FunctionSample)


class Function(Generic[FunctionSampleType]):
    """Data structure holding information for a single function."""
    def __init__(
        self,
        params: Dict[str, Any],
    ):
        self.params = params
        self.loaded_data: Dict[DataFold, List[FunctionSampleType]] = {}

        self.num_edge_types = 1

    def load_data(self, path:RichPath, folds_to_load: Optional[Set[DataFold]]=None) -> None:
        logger.info(f"Starting to load data from {path}.")

        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}

        if DataFold.TRAIN in folds_to_load:
            print("loading training data")
            self.loaded_data[DataFold.TRAIN] = self._load_data(path.join("train.jsonl.gz"))
            logger.debug("Done loading training data.")
        if DataFold.VALIDATION in folds_to_load:
            print("loading validation data")
            self.loaded_data[DataFold.VALIDATION] = self._load_data(path.join("valid.jsonl.gz"))
            logger.debug("Done loading validation data.")
        if DataFold.TEST in folds_to_load:
            print("loading test data")
            self.loaded_data[DataFold.TEST] = self._load_data(path.join("test.jsonl.gz"))
            logger.debug("Done loading test data.")


    def _load_data(self, data_file: RichPath) -> List[FunctionSample]:
        print(data_file)
        return [
            self._process_raw_datapoint(datapoint) for datapoint in data_file.read_by_file_suffix()
        ]

    def _process_raw_datapoint(self, datapoint: Dict[str, Any]) -> FunctionSample:
        label = datapoint["label"]
        raw_line_list = datapoint["raw"]
        abstract_line_list = datapoint["abstract_graph"]

        raw_function_graphs = []
        abstract_function_graphs = []

        for line in raw_line_list:
            node_features = line["node_features"]
            node_features_np = np.array(node_features)
            adjacency_list = line["edges"]
            adj_np, edge_feature_np = self._process_raw_adjacency_lists(
                raw_adjacency_lists=adjacency_list,
                num_nodes=len(node_features),
            )
            graph = GraphSample(
                adjacency_list=adj_np,
                edge_features=edge_feature_np,
                node_features=node_features_np,
            )
            raw_function_graphs.append(graph)

        for line in abstract_line_list:
            node_features = line["node_features"]
            node_features_np = np.array(node_features)
            adjacency_list = line["edges"]
            adj_np, edge_feature_np = self._process_raw_adjacency_lists(
                raw_adjacency_lists=adjacency_list,
                num_nodes=len(node_features),
            )
            graph = GraphSample(
                adjacency_list=adj_np,
                edge_features=edge_feature_np,
                node_features=node_features_np,
            )
            abstract_function_graphs.append(graph)

        return FunctionSample(
            label=label,
            raw_graph=raw_function_graphs,
            abstract_graph=abstract_function_graphs
        )

    def _process_raw_adjacency_lists(self, raw_adjacency_lists: List[Tuple], num_nodes: int):
        type_to_num_incoming_edges = np.zeros(shape=(self.num_edge_types, num_nodes))
        edge_feature_list = []
        adj_list = []
        # for edges in raw_adjacency_lists:
        for src, dest in raw_adjacency_lists:
            fwd_edge_type = 0
            adj_list.append((src, dest))
            edge_feature = [0 for _ in range(self.num_edge_types)]
            edge_feature[fwd_edge_type] += 1
            edge_feature_list.append(edge_feature)
            try:
                type_to_num_incoming_edges[fwd_edge_type, dest] += 1
            except:
                type_to_num_incoming_edges[fwd_edge_type, dest - 1] += 1
        adj_np = np.array(adj_list, dtype=np.long)
        edge_feature_np = np.array(edge_feature_list, dtype=float)

        return adj_np, edge_feature_np

    def get_pyg_dataset(self):
        raw_data = self.loaded_data
        valid_data = raw_data[DataFold.VALIDATION]
        pyg_dataset_valid = []
        valid_labels = []
        for func in valid_data:
            label = torch.tensor(np.array([1]), dtype=torch.long)
            if func.label == "clean":
                label = torch.tensor(np.array([0]), dtype=torch.long)
            valid_labels.append(label)
            func_graph = []
            for sgq in func.graph_list:
                edges = torch.tensor(sgq.adjacency_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(sgq.edge_features, dtype=torch.float)
                nodes = torch.tensor(sgq.node_features, dtype=torch.float)
                graph = Data(x=nodes, edge_index=edges, edge_attr=edge_attr)
                func_graph.append(graph)

        #     pyg_dataset_valid.append(func_graph)
        # valid_set = FunctionDataset(labels=valid_labels, func_graphs=pyg_dataset_valid)
        # torch.save(valid_set, self.params["valid_save_path"])
            valid_graph_loader = pyg_DataLoader(func_graph, batch_size=100, shuffle=False, drop_last=False)
            pyg_dataset_valid.append(valid_graph_loader)

        valid_set = FunctionDataset(labels=valid_labels, func_graphs=pyg_dataset_valid)
        torch.save(valid_set, self.params["valid_save_path"])



class FunctionDataset(Dataset):
    def __init__(self, labels, raw_graphs, abstract_graphs, texts, transform=None, target_transform=None):
        self.labels = labels
        self.raw_graphs = raw_graphs
        self.abstract_graphs = abstract_graphs
        self.texts = texts
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_graph = self.raw_graphs[idx]
        abstract_graph = self.abstract_graphs[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        if self.transform:
            raw_graph = self.transform(raw_graph)
            abstract_graph = self.transform(abstract_graph)
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)
        return raw_graph, abstract_graph, text, label


def my_collate_fn(batch):
    raw_func_list = []
    abstract_func_list = []
    text_list = []
    labels_list = []
    for data in batch:
        raw_func_list.append(data[0])
        abstract_func_list.append(data[1])
        text_list.append(data[2])
        labels_list.append(data[3])
    del batch
    return raw_func_list, abstract_func_list, text_list, labels_list


def get_dataset(
    data_path: RichPath,
    loaded_data_hyperparameters: Dict[str, Any],
    folds_to_load: Optional[Set[DataFold]] = None,
):

    dataset_params = loaded_data_hyperparameters
    dataset_cls = Function
    dataset = dataset_cls(dataset_params)
    print(f"Loading data from {data_path}.")
    dataset.load_data(data_path, folds_to_load)
    dataset.get_pyg_dataset()


def loda_dataset(dataset_params):
    train_dataset = torch.load(dataset_params["train_save_path"])
    train_loader = DataLoader(train_dataset, batch_size=dataset_params["batch_size"],
                              shuffle=True, drop_last=False, collate_fn=my_collate_fn)

    valid_dataset = torch.load(dataset_params["valid_save_path"])
    valid_loader = DataLoader(valid_dataset, batch_size=dataset_params["batch_size"],
                              shuffle=True, drop_last=False, collate_fn=my_collate_fn)

    test_dataset = torch.load(dataset_params["test_save_path"])
    test_loader = DataLoader(test_dataset, batch_size=dataset_params["batch_size"],
                             shuffle=False, drop_last=False, collate_fn=my_collate_fn)

    return train_loader, valid_loader, test_loader




# if __name__ == "__main__":
#     data_path = RichPath.create("/home/qiufangcheng/workspace/SGS/data/sard1/graph_dataset")
#     loaded_data_hyperparameters = dict(label="clean",
#                                        train_save_path="/home/qiufangcheng/workspace/SGS/data/sard1/graph_dataset/train_test3.pkl",
#                                        valid_save_path="/home/qiufangcheng/workspace/SGS/data/sard1/graph_dataset/valid3.pkl",
#                                        test_save_path="/home/qiufangcheng/workspace/SGS/data/sard1/graph_dataset/test3.pkl",
#                                        num_fwd_edge_types="1",
#                                        tie_fwd_bkwd_edges=True,
#                                        add_self_loop_edges="0",
#                                        batch_size=64
#                                        )
#     get_dataset(data_path=data_path,
#                 loaded_data_hyperparameters=loaded_data_hyperparameters)
    # loda_dataset(loaded_data_hyperparameters)
