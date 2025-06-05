"""
Loads the BioBridge dataset and prepares it for training and evaluation
using LightningDataModule from PyTorch Lightning, with optional caching.
"""

import os
import pickle
from typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from .biobridge_primekg import BioBridgePrimeKG

class BioBridgeDataModule(LightningDataModule):
    """
    LightningDataModule for the BioBridge dataset.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BioBridgeDataModule.

        Args:
            cfg (DictConfig): Configuration object with dataset parameters.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.primekg_dir = cfg.data.primekg_dir
        self.biobridge_dir = cfg.data.biobridge_dir
        self.batch_size = cfg.data.batch_size
        self.cache_path = cfg.data.cache_path
        self.biobridge = None
        self.mapper = {}
        self.data = {}

    def _load_biobridge_data(self) -> None:
        """
        Private method to load related files of PrimeKG dataset.

        Returns:
            None
        """
        self.biobridge = BioBridgePrimeKG(self.cfg)
        self.biobridge.load_data()

        self.data['nt2ntid'] = self.biobridge.get_data_config()["node_type"]
        self.data['ntid2nt'] = {v: k for k, v in self.data['nt2ntid'].items()}

    def _filter_triplets(self):
        """
        Private method to filter valid triplets from the PrimeKG edges.

        Returns:
            triplets (pd.DataFrame): Filtered triplet data.
        """
        node_index_list = []
        for node_type in self.biobridge.preselected_node_types:
            df_node = pd.read_csv(os.path.join(self.biobridge.local_dir,
                                               "processed", f"{node_type}.csv"))
            node_index_list.extend(df_node["node_index"].tolist())

        triplets = self.biobridge.primekg.get_edges().copy()
        triplets = triplets[
            triplets["head_index"].isin(node_index_list) &
            triplets["tail_index"].isin(node_index_list)
        ].reset_index(drop=True)

        triplets = triplets[
            triplets["head_index"].isin(self.biobridge.emb_dict) &
            triplets["tail_index"].isin(self.biobridge.emb_dict)
        ].reset_index(drop=True)
        return triplets

    def _add_node_features(self, nodes: pd.DataFrame, node_types: np.ndarray) -> None:
        """
        Private method to add node features to HeteroData.

        Args:
            nodes (pd.DataFrame): DataFrame containing node metadata.
            node_types (np.ndarray): Array of unique node types.

        Returns:
            None
        """
        self.data["init"] = HeteroData()

        for nt in node_types:
            self.mapper[nt] = {}
            nt_nodes = nodes[nodes['node_type'] == nt].reset_index(drop=True)

            self.mapper[nt]['to_nidx'] = nt_nodes["node_index"].to_dict()
            self.mapper[nt]['from_nidx'] = {v: k for k, v in self.mapper[nt]['to_nidx'].items()}

            self.data["init"][nt].num_nodes = len(self.mapper[nt]['from_nidx'])

            keys = list(self.mapper[nt]['from_nidx'].keys())
            emb_ = np.array([self.biobridge.emb_dict[i] for i in keys])
            self.data["init"][nt].x = torch.tensor(emb_, dtype=torch.float32)

            node_names = nt_nodes["node_name"].tolist()
            self.data["init"][nt]["node_name"] = node_names

    def _add_edge_indices(self, triplets: pd.DataFrame) -> None:
        """
        Private method to add edge indices to the graph.

        Args:
            triplets (pd.DataFrame): DataFrame containing filtered triplet edges.

        Returns:
            None
        """
        for ht, rt, tt in triplets[["head_type",
        "display_relation",
        "tail_type"]].drop_duplicates().values:
            t_ = triplets[
                (triplets['head_type'] == ht) &
                (triplets['display_relation'] == rt) &
                (triplets['tail_type'] == tt)
            ]
            src_ids = t_['head_index'].map(self.mapper[ht]['from_nidx']).values
            dst_ids = t_['tail_index'].map(self.mapper[tt]['from_nidx']).values

            self.data["init"][(ht, rt, tt)].edge_index = torch.tensor([src_ids,
            dst_ids], dtype=torch.long)

    def prepare_data(self) -> None:
        """
        Loads and processes the data, optionally using cached data.

        Returns:
            None
        """
        if os.path.exists(self.cache_path):
            print(f"🔁 Loading cached data from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.data = pickle.load(f)
            return

        self._load_biobridge_data()
        triplets = self._filter_triplets()

        nodes = self.biobridge.primekg.get_nodes().copy()
        nodes = nodes[nodes["node_index"].isin(
            np.unique(np.concatenate([triplets.head_index.unique(),
                                      triplets.tail_index.unique()])))
        ].reset_index(drop=True)
        node_types = np.unique(nodes['node_type'].tolist())

        self._add_node_features(nodes, node_types)
        self._add_edge_indices(triplets)

        with open(self.cache_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"✅ Cached processed data to {self.cache_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up training, validation, and test splits using RandomLinkSplit.

        Args:
            stage (Optional[str]): Optional stage indicator.

        Returns:
            None
        """
        if "train" in self.data:
            return
        with hydra.initialize(version_base=None, config_path="../../../configs"):
            cfg: DictConfig = hydra.compose(config_name="config")

        transform = RandomLinkSplit(
            num_val=cfg.random_link_split.num_val,
            num_test=cfg.random_link_split.num_test,
            is_undirected=cfg.random_link_split.is_undirected,
            add_negative_train_samples=cfg.random_link_split.add_negative_train_samples,
            neg_sampling_ratio=cfg.random_link_split.neg_sampling_ratio,
            split_labels=cfg.random_link_split.split_labels,
            edge_types=self.data["init"].edge_types,
        )

        self.data["train"], self.data["val"], self.data["test"] = transform(self.data["init"])

        with open(self.cache_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"✅ Cached train/val/test splits to {self.cache_path}")

    def train_dataloader(self) -> GeoDataLoader:
        """
        Returns the training dataloader.

        Returns:
            GeoDataLoader: DataLoader for training set.
        """
        if "train" not in self.data:
            raise RuntimeError("Please run `setup()` before calling train_dataloader().")
        return GeoDataLoader([self.data["train"]], batch_size=1, shuffle=False)

    def val_dataloader(self) -> GeoDataLoader:
        """
        Returns the validation dataloader.

        Returns:
            GeoDataLoader: DataLoader for validation set.
        """
        if "val" not in self.data:
            raise RuntimeError("Please run `setup()` before calling val_dataloader().")
        return GeoDataLoader([self.data["val"]], batch_size=1, shuffle=False)

    def test_dataloader(self) -> GeoDataLoader:
        """
        Returns the test dataloader.

        Returns:
            GeoDataLoader: DataLoader for test set.
        """
        if "test" not in self.data:
            raise RuntimeError("Please run `setup()` before calling test_dataloader().")
        return GeoDataLoader([self.data["test"]], batch_size=1, shuffle=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Optional cleanup after training/testing.

        Args:
            stage (Optional[str]): Current stage of execution.

        Returns:
            None
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Returns the internal state of the data module.

        Returns:
            dict: Empty dictionary (no state to save).
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the internal state of the data module.

        Args:
            state_dict (dict): State dictionary.

        Returns:
            None
        """
        pass
