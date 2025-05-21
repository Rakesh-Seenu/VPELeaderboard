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
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from .biobridge_primekg import BioBridgePrimeKG

class BioBridgeDataModule(LightningDataModule):
    """
    `LightningDataModule` for the BioBridge dataset.
    """
    def __init__(self,
                 primekg_dir: str = "../../../data/primekg/",
                 biobridge_dir: str = "../../../data/biobridge_primekg/",
                 batch_size: int = 64,
                 cache_path: str = "./biobridge_cache.pkl") -> None:
        """
        Initializes the BioBridgeDatadm.

        Args:
            primekg_dir (str): Directory where the PrimeKG dataset is stored.
            biobridge_dir (str): Directory where the BioBridge dataset is stored.
            batch_size (int): Batch size for training and evaluation.
            cache_path (str): Path to cache the processed HeteroData splits.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.primekg_dir = primekg_dir
        self.biobridge_dir = biobridge_dir
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.biobridge = None
        self.mapper = {}
        self.data = {}
        self.data = {}

    def _load_biobridge_data(self) -> None:
            # Load BioBridge and PrimeKG data
        self.biobridge = BioBridgePrimeKG(local_dir=self.biobridge_dir)
        self.biobridge.load_data()

        self.data['nt2ntid'] = self.biobridge.get_data_config()["node_type"]
        self.data['ntid2nt'] = {v: k for k, v in self.data['nt2ntid'].items()}

    def _filter_triplets(self):
        node_index_list = []
        for node_type in self.biobridge.preselected_node_types:
            df_node = pd.read_csv(os.path.join(self.biobridge.local_dir,
                                               "processed", f"{node_type}.csv"))
            node_index_list.extend(df_node["node_index"].tolist())

        # self.primekg = PrimeKG(local_dir=self.primekg_dir)
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

    def prepare_data(self) -> None:
        """
        Prepare the data by downloading and processing it, 
        or loading from cache if available.
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
                                      triplets.tail_index.unique()]))
        )].reset_index(drop=True)

        node_types = np.unique(nodes['node_type'].tolist())
        self.data["init"] = HeteroData()

        for nt in node_types:
            self.mapper[nt] = {}
            self.mapper[nt]['to_nidx'] = nodes[nodes['node_type'] == nt]["node_index"]
            self.mapper[nt]['to_nidx']= self.mapper[nt]['to_nidx'].reset_index(drop=True).to_dict()
            self.mapper[nt]['from_nidx'] = {v: k for k, v in self.mapper[nt]['to_nidx'].items()}
            self.data["init"][nt].num_nodes = len(self.mapper[nt]['from_nidx'])
            keys = list(self.mapper[nt]['from_nidx'].keys())
            emb_ = np.array([self.biobridge.emb_dict[i] for i in keys])
            self.data["init"][nt].x = torch.tensor(emb_, dtype=torch.float32)

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
            self.data["init"][(ht, rt, tt)].edge_index = torch.tensor([src_ids, dst_ids],
                                                                      dtype=torch.long)


        # Cache the initial processed HeteroData to disk
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"✅ Cached processed data to {self.cache_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for training, validation, and testing.
        """
        if "train" in self.data:
            print("🔁 Reusing previously split train/val/test data")
            return

        transform = RandomLinkSplit(num_val=0.1,
                                    num_test=0.2,
                                    is_undirected=False,
                                    add_negative_train_samples=True,
                                    neg_sampling_ratio=1.0,
                                    split_labels=True,
                                    edge_types=self.data["init"].edge_types)

        self.data["train"], self.data["val"], self.data["test"] = transform(self.data["init"])

        with open(self.cache_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"✅ Cached train/val/test splits to {self.cache_path}")
    def train_dataloader(self) -> DataLoader:
        if "train" not in self.data:
            raise RuntimeError("Please run `setup()` before calling train_dataloader().")
        return DataLoader([self.data["train"]], batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        if "val" not in self.data:
            raise RuntimeError("Please run `setup()` before calling val_dataloader().")
        return DataLoader([self.data["val"]], batch_size=1, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if "test" not in self.data:
            raise RuntimeError("Please run `setup()` before calling test_dataloader().")
        return DataLoader([self.data["test"]], batch_size=1, shuffle=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
